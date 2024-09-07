import sys
import os

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import torchaudio

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import BasePredictionWriter
import wandb

import dac

from utils import audio_spectrogram_image, load_state_dict_partial
from data.dac_encodec_clap_dataset import DacEncodecClapDatasetH5
from module.unet_cond_large import Unet1DParallelPattern, Unet1DVALLEPatternPrimary, Unet1DVALLEPatternSecondary
from module.cond_net import CondFuser

from diffusion.gaussian_ddpm import dac_latent_normalize_heterogeneous
# from diffusion.ddpm_diffusers import dac_latent_normalize_heterogeneous

class LatentDiffusionCondModule(pl.LightningModule):
    def __init__(
        self,
        config,
        ckpt_path = None,
        only_train_primary = False,
        only_train_secondary = False,
        only_train_chroma = False,
        no_clap_feat = False,
    ):
        super().__init__()
        self.config = config
        if hasattr(config, "scheduler"):
            if config.scheduler == "handcrafted":
                from diffusion.gaussian_ddpm import DACLatentDDPM, DACLatentDDPMVALLE
                print("Using handcrafted diffusion scheduler codebase")
            else:
                from diffusion.ddpm_diffusers import DACLatentDDPM, DACLatentDDPMVALLE
                print("Using diffusers scheduler codebase")
        else:
            from diffusion.gaussian_ddpm import DACLatentDDPM, DACLatentDDPMVALLE
            print("Scheduler type not found. Using handcrafted diffusion scheduler codebase")

        if config.rvq_pattern == "parallel":
            DenoiseModelClass = Unet1DParallelPattern
        elif config.rvq_pattern == "flattened":
            # DenoiseModelClass = Unet1DFlatteningPattern
            assert 0, "Flattening pattern disabled"
        elif config.rvq_pattern == "VALL-E":
            DenoiseModelClass = Unet1DVALLEPatternPrimary
            DenoiseModelClassSecondary = Unet1DVALLEPatternSecondary
        else:
            assert 0

        # DEBUG: if using T5, replace meta cond with T5
        # self.meta_cond_encoder_decoder = FMADummyCondNet(
        #     out_emb_dim = config.meta_cond_dim,
        #     feature_attr_dims = config.in_emb_dims_fma
        # )
        self.meta_cond_encoder_decoder = CondFuser(feature_emb_dim = config.meta_cond_dim)

        # define vq net and diffusion model
        if config.rvq_pattern == "VALL-E":
            denoise_model = DenoiseModelClass(
                input_dim=config.model_dim_in,
                feature_cond_dim=config.clap_dim + config.meta_cond_dim,
                chroma_cond_dim=config.chroma_dim,
                text_cond_dim=self.meta_cond_encoder_decoder.text_emb_dim,
                num_codebooks=config.num_codebooks,
                num_attn_heads=config.num_codebooks,
                dim=config.inner_dim_primary,
                dim_mults=config.dim_mults_primary,
                attn_dim_head=config.head_dim_primary,
                cond_drop_prob=config.cond_drop_prob,
            )
            denoise_model_secondary = DenoiseModelClassSecondary(
                input_dim = config.model_dim_in,
                feature_cond_dim=config.clap_dim + config.meta_cond_dim,
                chroma_cond_dim=config.chroma_dim,
                text_cond_dim=self.meta_cond_encoder_decoder.text_emb_dim,
                num_codebooks = config.num_codebooks,
                num_attn_heads=config.num_codebooks,
                dim = config.inner_dim,
                dim_mults = config.dim_mults,
                attn_dim_head = config.head_dim,
                cond_drop_prob = config.cond_drop_prob,
            )
        else:
            denoise_model = DenoiseModelClass(
                input_dim=config.model_dim_in,
                feature_cond_dim=config.clap_dim + config.meta_cond_dim,
                chroma_cond_dim=config.chroma_dim,
                text_cond_dim=self.meta_cond_encoder_decoder.text_emb_dim,
                num_codebooks=config.num_codebooks,
                num_attn_heads=config.num_codebooks,
                dim=config.inner_dim,
                dim_mults=config.dim_mults,
                attn_dim_head=config.head_dim,
                cond_drop_prob=config.cond_drop_prob,
            )

        chroma_gen_model = Unet1DParallelPattern(
            input_dim=config.chroma_dim,
            feature_cond_dim=config.clap_dim + config.meta_cond_dim,
            chroma_cond_dim=config.chroma_dim,
            text_cond_dim=self.meta_cond_encoder_decoder.text_emb_dim,
            num_codebooks=config.chroma_num_heads,
            num_attn_heads=config.chroma_num_heads,
            dim=config.chroma_inner_dim,
            dim_mults=config.dim_mults,
            attn_dim_head=config.chroma_head_dim,
            cond_drop_prob=config.cond_drop_prob
        )

        self.obj_shape = (config.codebook_dim, config.frame_len_dac)

        self.chroma_diffusion = DACLatentDDPM(
            chroma_gen_model,
            config.chroma_frame_len,
            latent_dim = config.chroma_dim,
            timesteps = config.diffusion_steps,
            prediction_type = config.prediction_type,
            normalize = False
        )
        print("Prediction type:", config.prediction_type)
        if config.rvq_pattern != "VALL-E":
            self.diffusion = DACLatentDDPM(
                denoise_model,
                config.frame_len_dac,
                latent_dim = config.model_dim_in,
                timesteps = config.diffusion_steps,
                prediction_type = config.prediction_type
            )
        else:
            self.diffusion = DACLatentDDPMVALLE(
                denoise_model,
                denoise_model_secondary,
                config.frame_len_dac,
                latent_dim = config.model_dim_in,
                timesteps = config.diffusion_steps,
                prediction_type = config.prediction_type
            )
            self.only_train_primary = only_train_primary
            self.only_train_secondary = only_train_secondary
            self.only_train_chroma = only_train_chroma

        # load checkpoint
        if ckpt_path is not None:
            load_state_dict_partial(
                self.state_dict(),
                torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
            )
            print("Loaded checkpoint", ckpt_path)

        # get dac model
        dac_model_path = dac.utils.download(model_type="44khz")
        self.dac_model = dac.DAC.load(dac_model_path)

        self.no_clap_feat = no_clap_feat
        if self.no_clap_feat:
            print("Not using CLAP and meta features")

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        return self.diffusion(*args, **kwargs)

    def configure_optimizers(self):
        return optim.Adam(
            list(self.diffusion.parameters()) + list(self.chroma_diffusion.parameters()),
            lr=self.config.lr
        )

    def training_step(self, batch, batch_idx):
        latents = batch["dac_latents"]
        if "text_clap" in batch:
            use_text_clap = torch.rand(1) < self.config.text_clap_load_prob
            if use_text_clap:
                clap_emb = batch["text_clap"]
            else:
                clap_emb = batch["audio_clap"]
        else:
            clap_emb = batch["audio_clap"]

        if "chroma" in batch:
            chroma = batch["chroma"]
        else:
            chroma = None

        # DEBUG: T5
        with torch.no_grad():
            feat_cond_emb, text_cond_emb_seq = self.meta_cond_encoder_decoder.encode(batch)

        feat_cond_emb.to(torch.float32).to(latents.device)
        text_cond_emb_seq.to(torch.float32).to(latents.device)

        # DEBUG
        # print("latents dtype", latents.dtype)
        # print("T5 emb mean:", text_cond_emb_seq.mean())
        # print("T5 emb dtype", text_cond_emb_seq.dtype)

        if not self.no_clap_feat:
            # vec_cond = torch.cat([clap_emb, feat_cond_emb], dim=-1)
            vec_cond = feat_cond_emb # debug: T5 embedding replacing CLAP embedding
        else:
            vec_cond = None
        seq_conds = [chroma, text_cond_emb_seq]

        if self.only_train_primary or self.only_train_secondary:
            loss_chroma = 0
        else:
            loss_chroma = self.chroma_diffusion(chroma, vec_cond = vec_cond, seq_conds = [None, text_cond_emb_seq])
            self.log('chroma_loss', loss_chroma)

        if self.only_train_chroma:
            # print("chroma loss:", loss_chroma)

            return loss_chroma

        if self.config.rvq_pattern != "VALL-E":
            loss = self.diffusion(latents, vec_cond = vec_cond, seq_conds = seq_conds)
            self.log('train_loss', loss)
        else:
            rand_flag = (torch.rand(1) < 0.5).item()
            if (self.only_train_primary or rand_flag) and not self.only_train_secondary:
                loss_primary, _ = self.diffusion(
                    latents, vec_cond=vec_cond, seq_conds=seq_conds,
                    only_secondary = False,
                    only_primary = True,
                )
                loss_secondary = 0

            if (self.only_train_secondary or (not rand_flag)) and not self.only_train_primary:
                _, loss_secondary = self.diffusion(
                    latents, vec_cond=vec_cond, seq_conds=seq_conds,
                    only_secondary = True,
                    only_primary = False,
                )
                loss_primary = 0

            loss = loss_secondary + loss_primary

            self.log('primary_loss', loss_primary)
            self.log('secondary_loss', loss_secondary)
            self.log('train_loss', loss)

        loss = loss + loss_chroma
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        latents = batch["dac_latents"]
        # DEBUG: T5
        feat_cond_emb, text_cond_emb_seq = self.meta_cond_encoder_decoder.encode(batch)
        feat_cond_emb.to(torch.float32).to(latents.device)
        text_cond_emb_seq.to(torch.float32).to(latents.device)

        # DEBUG
        # print("T5 emb mean:", text_cond_emb_seq.mean())

        audio_clap_emb = batch["audio_clap"]
        if not self.no_clap_feat:
            # vec_cond_audio_clap = torch.cat([audio_clap_emb, feat_cond_emb], dim = -1)
            vec_cond_audio_clap = feat_cond_emb  # debug: T5 embedding replacing CLAP embedding
            if "text_clap" in batch:
                text_clap_emb = batch["text_clap"]
                # vec_cond_text_clap = torch.cat([text_clap_emb, feat_cond_emb], dim = -1)
                vec_cond_text_clap = feat_cond_emb  # debug: T5 embedding replacing CLAP embedding
        else:
            vec_cond_audio_clap = None
            vec_cond_text_clap = None

        if "chroma" in batch:
            chroma = batch["chroma"]
        else:
            chroma = None

        seq_condas_chroma = [None, text_cond_emb_seq]
        seq_conds = [chroma, text_cond_emb_seq]

        num_demos = audio_clap_emb.shape[0]

        if "inpaint_mask" not in batch:
            chroma_sampled = self.chroma_diffusion.sample(
                batch_size=num_demos,
                vec_cond=vec_cond_audio_clap,
                seq_conds=seq_condas_chroma,
                cond_scale=self.config.cond_scale,
            )
            latents_samples_audio_clap = self.diffusion.sample(
                batch_size=num_demos,
                vec_cond=vec_cond_audio_clap,
                seq_conds=seq_conds,
                cond_scale=self.config.cond_scale,
            )  # [B, K, L] of integers
            if "text_clap" in batch:
                # seq_conds_sampled = [chroma_sampled, text_cond_emb_seq] # debug
                seq_conds_sampled = [chroma, text_cond_emb_seq]
                latents_samples_text_clap = self.diffusion.sample(
                    batch_size=num_demos,
                    vec_cond=vec_cond_text_clap,
                    seq_conds=seq_conds_sampled,
                    cond_scale=self.config.cond_scale,
                )  # [B, K, L] of integers
            else:
                latents_samples_text_clap = latents_samples_audio_clap

            if self.config.rvq_pattern == "VALL-E":
                noise_given_primary = torch.randn_like(latents)
                latents_normalized = dac_latent_normalize_heterogeneous(latents)
                noise_given_primary[:, :self.diffusion.latent_dim_primary] *= 0
                noise_given_primary[:, :self.diffusion.latent_dim_primary] += latents_normalized[:, :self.diffusion.latent_dim_primary]

                latents_samples_given_primary = self.diffusion.sample_given_primary(
                    noise_given_primary = noise_given_primary,
                    vec_cond = vec_cond_audio_clap,
                    seq_conds = seq_conds,
                    cond_scale = self.config.cond_scale,
                )

        else:
            latents_samples_audio_clap = self.diffusion.sample_repaint(
                source = latents,
                mask = batch["inpaint_mask"],
                vec_cond = vec_cond_audio_clap,
                seq_conds = seq_conds,
                cond_scale = self.config.cond_scale,
            ) # [B, K, L] of integers

        if self.config.rvq_pattern != "VALL-E":
            if "text_clap" in batch:
                latents_all = torch.cat(
                    [latents_samples_audio_clap,
                     latents_samples_text_clap,
                     latents], dim = 0
                )
                meanings = ["sampled_given_audio_clap", "sampled_given_text_clap", "gt_data"]
            else:
                latents_all = torch.cat(
                    [latents_samples_audio_clap,
                     latents], dim=0
                )
                meanings = ["sampled_given_audio_clap", "gt_data"]
        else:
            if "text_clap" in batch:
                latents_all = torch.cat(
                    [latents_samples_audio_clap,
                     latents_samples_text_clap,
                     latents_samples_given_primary,
                     latents], dim=0
                )
                meanings = ["sampled_given_audio_clap", "sampled_given_text_clap",
                            "sampled_given_text_clap_and_primary_latent", "gt_data"]
            else:
                latents_all = torch.cat(
                    [latents_samples_audio_clap,
                     latents_samples_given_primary,
                     latents], dim=0
                )
                meanings = ["sampled_given_audio_clap", "sampled_given_text_clap_and_primary_latent", "gt_data"]

            print("VALL-E secondary-only sampling enabled")

        z = self.dac_model.quantizer.from_latents(latents_all)[0]
        wav_recons = self.dac_model.decode(z)
        if torch.abs(wav_recons).max() > 1:
            wav_recons = wav_recons / torch.abs(wav_recons).max()

        wav_return_dict = {}
        for i_meaning, key in enumerate(meanings):
            id_start = i_meaning * num_demos
            id_end = (i_meaning+1) * num_demos
            wav_return_dict[key] = wav_recons[id_start:id_end]

        wav_return_dict["gt_chroma"] = chroma
        wav_return_dict["sampled_chroma"] = chroma_sampled

        return wav_return_dict

    @torch.no_grad()
    def custom_pred(self, batch, use_audio_clap = False):
        if not use_audio_clap:
            clap_emb = batch["text_clap"]
        else:
            clap_emb = batch["audio_clap"]

        if "chroma" in batch:
            chroma = batch["chroma"].to(torch.float32)
        else:
            chroma = None

        # DEBUG: T5
        feat_cond_emb, text_cond_emb_seq = self.meta_cond_encoder_decoder.encode(batch)
        feat_cond_emb.to(self.dac_model.device).to(torch.float32)
        text_cond_emb_seq.to(self.dac_model.device).to(torch.float32)

        num_demos = clap_emb.shape[0]
        if not self.no_clap_feat:
            # vec_cond = torch.cat([clap_emb, feat_cond_emb], dim=-1)
            vec_cond = feat_cond_emb  # debug: T5 embedding replacing CLAP embedding
        else:
            vec_cond = None
        seq_conds = [chroma, text_cond_emb_seq]

        latents_samples = self.diffusion.sample(
            batch_size = num_demos,
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = self.config.cond_scale,
        ) # [B, K, L] of integers

        z = self.dac_model.quantizer.from_latents(latents_samples)[0]
        wav_recon = self.dac_model.decode(z)

        return wav_recon

    def audio_edit(self, batch, use_audio_clap = False, start_diffusion_step = 50):
        if not use_audio_clap:
            clap_emb = batch["text_clap"]
        else:
            clap_emb = batch["audio_clap"]

        if "chroma" in batch:
            chroma = batch["chroma"]
        else:
            chroma = None

        latents_normalized = batch["latents_normalized"]

        # DEBUG: T5
        feat_cond_emb, text_cond_emb_seq = self.meta_cond_encoder_decoder.encode(batch)
        feat_cond_emb.to(torch.float32).to(clap_emb.device)
        text_cond_emb_seq.to(torch.float32).to(clap_emb.device)

        num_demos = clap_emb.shape[0]
        # vec_cond = torch.cat([clap_emb, feat_cond_emb], dim=-1)
        vec_cond = feat_cond_emb  # debug: T5 embedding replacing CLAP embedding
        seq_conds = [chroma, text_cond_emb_seq]

        latents_samples = self.diffusion.sample_editing(
            input_latents_normalized = latents_normalized,
            start_diffusion_step = start_diffusion_step,
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = self.config.cond_scale,
        ) # [B, K, L] of integers

        z = self.dac_model.quantizer.from_latents(latents_samples)[0]
        wav_recon = self.dac_model.decode(z)

        return wav_recon

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.predict_step(batch, batch_idx)

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.predict_step(batch, batch_idx)

    def on_before_zero_grad(self, *args, **kwargs):
        pass

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}", file=sys.stderr)

def save_demo(outputs, batch, save_path, sample_rate):
    num_demos = len(batch['name'])
    print("Showing {} demos".format(num_demos))

    for i_batch in range(num_demos):
        log_dict = {}
        if 'madmom_tempo' in batch:
            log_dict[f'madmom_tempo_{i_batch}'] = batch['madmom_tempo'][i_batch]
        if 'madmom_key' in batch:
            log_dict[f'madmom_key_{i_batch}'] = batch['madmom_key'][i_batch]

        audio_name = str(batch['name'][i_batch])
        meanings = list(outputs.keys())

        for i_audio in range(len(meanings) - 2):
            filename_sample = f'{audio_name}.wav'
            audio_type = meanings[i_audio]
            dir_sample = os.path.join(save_path, audio_type)
            if not os.path.exists(dir_sample):
                os.makedirs(dir_sample)
            filepath_sample = os.path.join(dir_sample, filename_sample)

            this_audio = outputs[audio_type][i_batch]
            if len(this_audio.shape) < 2:
                this_audio = this_audio.unsqueeze(0)
            this_wave_sample = this_audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filepath_sample, this_wave_sample, sample_rate)
            log_dict[f"{audio_type}_{i_batch}"] = wandb.Audio(
                filepath_sample,
                sample_rate=sample_rate,
                caption=f'Audio {audio_name}, {audio_type}'
            )
            log_dict[f'melspec_{audio_type}_{i_batch}'] = wandb.Image(
                audio_spectrogram_image(this_audio)
            )

        log_dict[f'gt_chroma'] = wandb.Image(outputs["gt_chroma"][0])
        log_dict[f'sampled_chroma'] = wandb.Image(outputs["sampled_chroma"][0])

        if 'text' in batch:
            text_caption = str(batch['text'][i_batch])
            log_dict[f'text_{i_batch}'] = text_caption
            filename_text = f'{audio_name}.txt'
            dir_text = os.path.join(save_path, "text")
            if not os.path.exists(dir_text):
                os.makedirs(dir_text)
            filepath_text = os.path.join(dir_text, filename_text)
            with open(filepath_text, 'w') as f:
                f.write(text_caption)

    return log_dict

class DemoCallback(BasePredictionWriter):
    def __init__(self, config, save_path=''):
        super().__init__()
        self.demo_every = config.demo_every
        self.num_demos = config.num_demos
        self.sample_rate = config.sample_rate
        self.sample_skip_step = config.sample_skip_step
        self.last_demo_step = -1
        self.save_path = save_path

    @rank_zero_only
    @torch.no_grad()
    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
        log_dict = save_demo(outputs, batch, self.save_path, self.sample_rate)
        trainer.logger.experiment.log(log_dict, step=trainer.global_step)

    @rank_zero_only
    @torch.no_grad()
    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.on_validation_batch_end(trainer, module, outputs, batch, batch_idx)


def pad_last_dim(tensor, max_len, pad_val = 0):
    pad_shape = list(tensor.shape)
    assert max_len >= pad_shape[-1]
    pad_shape[-1] = max_len - pad_shape[-1]
    pad_tensor = np.zeros(pad_shape, dtype=tensor.dtype) + pad_val
    return np.concatenate([tensor, pad_tensor], axis=-1)

def padding_collate_func(batch):
    max_len = 0
    for i, this_dict in enumerate(batch):
        this_len = this_dict['t5_input_ids'].shape[-1]
        assert this_len == this_dict['t5_attention_mask'].shape[-1]
        if this_len > max_len:
            max_len = this_len

    for i, this_dict in enumerate(batch):
        this_len = this_dict['t5_input_ids'].shape[-1]
        if this_len < max_len:
            batch[i]["t5_input_ids"] = pad_last_dim(this_dict['t5_input_ids'], max_len, pad_val = 0)
            batch[i]["t5_attention_mask"] = pad_last_dim(this_dict['t5_attention_mask'], max_len, pad_val=0)

    return default_collate(batch)


class DacCLAPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_dir,
        config,
        custum_dataset_size = None,
        val_h5_dir = None,
        test_h5_dir = None,
        another_h5_dir = None
    ):
        super().__init__()
        self.h5_dir = h5_dir
        self.val_h5_dir = val_h5_dir
        self.test_h5_dir = test_h5_dir
        self.another_h5_dir = another_h5_dir
        self.batch_size = config.batch_size
        self.num_demos = config.num_demos
        self.frame_len_dac = config.frame_len_dac
        self.frame_len_encodec = config.frame_len_encodec
        self.chroma_frame_len = config.chroma_frame_len
        self.chroma_frame_len = self.chroma_frame_len - (self.chroma_frame_len % 8)

        print("dac and chroma frame len: ", self.frame_len_dac, self.chroma_frame_len)

        if custum_dataset_size is not None:
            self.dataset_size = custum_dataset_size
        else:
            self.dataset_size = config.dataset_size

        self.dataset = DacEncodecClapDatasetH5(
            self.h5_dir,
            self.frame_len_dac,
            self.frame_len_encodec,
            chroma_frame_len = self.chroma_frame_len,
            dataset_size = self.dataset_size,
            random_load = True,
        )
        if self.another_h5_dir is not None:
            self.another_dataset = DacEncodecClapDatasetH5(
                self.another_h5_dir,
                self.frame_len_dac,
                self.frame_len_encodec,
                chroma_frame_len = self.chroma_frame_len,
                dataset_size = self.dataset_size,
                random_load = True
            )
            self.dataset = torch.utils.data.ConcatDataset([self.dataset, self.another_dataset])
    # prepare dataset
    def setup(self, stage):
        pass

    # create train loader
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=padding_collate_func)

    def val_dataloader(self, sel_list = [0]):
        val_h5_dir = self.h5_dir if self.val_h5_dir is None else self.val_h5_dir
        self.val_dataset = DacEncodecClapDatasetH5(
            val_h5_dir,
            self.frame_len_dac,
            self.frame_len_encodec,
            chroma_frame_len=self.chroma_frame_len,
            random_load=False,
        )
        if sel_list is not None:
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, sel_list)

        return DataLoader(self.val_dataset, batch_size=self.num_demos, collate_fn=padding_collate_func)

    def test_dataloader(self):
        test_h5_dir = self.h5_dir if self.test_h5_dir is None else self.test_h5_dir
        self.test_dataset = DacEncodecClapDatasetH5(
            test_h5_dir,
            self.frame_len_dac,
            self.frame_len_encodec,
            chroma_frame_len=self.chroma_frame_len,
            random_load=False,
        )
        return DataLoader(self.test_dataset, batch_size=1, collate_fn=padding_collate_func)
