import sys
import os
import argparse

import torch
import torchaudio
import random

import numpy as np
import laion_clap
from encodec.utils import convert_audio
from train_conditional import sel_config, set_save_path, set_device_accelerator, config_adjustments
from pl_module_callbacks import LatentDiffusionCondModule, DacCLAPDataModule
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from diffusion.gaussian_ddpm import dac_latent_normalize_heterogeneous
from utils import KEY_LABELS

def main(args):
    # determine config type according to pattern
    config = sel_config(args.model_size, args.rvq_pattern)
    print(config)

    save_dir = set_save_path(args.save_dir)
    device, accelerator = set_device_accelerator()
    # torch.manual_seed(0)

    config = config_adjustments(
        config,
        prediction_type = args.prediction_type,
        scheduler = args.scheduler,
        frame_len_dac = int(16 * ((args.dur_sec * 86.1) // 16))
    )

    # get CLAP audio encoder
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base') # need pip install transformers==4.30.0; if later version is installed, downgrade it to 4.30.0
    clap_model.load_ckpt(
        "./music_audioset_epoch_15_esc_90.14.pt"
    ) # download the default pretrained checkpoint.
    clap_model.to(device)
    print("CLAP model loaded")

    # get T5 model
    t5_model_string = "google/flan-t5-large"
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(
        t5_model_string,
        torch_dtype=torch.float16
    )  # .to(self.device) # , device_map="auto"
    # self.t5_model.parallelize()
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_string)

    # get pl diffusion module (model defined inside)
    diffusion_pl_module = LatentDiffusionCondModule(config, ckpt_path = args.load_ckpt_path)
    print("Diffusion model created")

    text_prompts = args.text_prompt

    # for audio editing: get normalized dac latents
    if args.edit_audio_path is not None:
        import dac
        dac_model_path = dac.utils.download(model_type="44khz")
        dac_model = dac.DAC.load(dac_model_path)
        dac_model.to(device)

        audio_name = os.path.splitext(os.path.basename(args.edit_audio_path))[0]

        wav, sr = torchaudio.load(args.edit_audio_path)
        wav_dac = convert_audio(
            wav, sr, 44100, 1
        ).unsqueeze(0).to(dac_model.device)
        _, codes, latents, _, _ = dac_model.encode(wav_dac)
        latents_normalized = dac_latent_normalize_heterogeneous(latents)
        frame_len = latents_normalized.shape[-1]
        rectified_frame_len = int(16 * (frame_len // 16))
        latents_normalized = latents_normalized[..., :rectified_frame_len]

    for text_prompt in text_prompts:
        with torch.no_grad():
            if args.key is None:
                args.key = random.randint(0, 23)
            if args.tempo is None:
                args.tempo = random.randint(72, 160)
            data_dict = {
                'madmom_key': args.key,
                'madmom_tempo': args.tempo,
            }
            key_str = KEY_LABELS[int(data_dict["madmom_key"])]
            text_clap = clap_model.get_text_embedding([text_prompt, ""], use_tensor=True)[0:1]
            data_dict["text_clap"] = text_clap

            enc = t5_tokenizer(
                text_prompt,
                return_tensors="pt", truncation=True, padding='longest'
            )
            data_dict["t5_input_ids"] = enc["input_ids"].squeeze(0).to(torch.long).cpu().numpy()
            data_dict["t5_attention_mask"] = enc["attention_mask"].squeeze(0).to(torch.long).cpu().numpy()
            print(data_dict["t5_input_ids"].shape, data_dict["t5_attention_mask"].shape)
            if args.chroma_path is not None:
                data_dict["chroma"] = np.load(args.chroma_path).to(np.float32)

            for key in data_dict:
                data_dict[key] = torch.tensor(data_dict[key]).to(device).unsqueeze(0)
            data_dict["randomized_text"] = text_prompt

            diffusion_pl_module.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                if args.edit_audio_path is None:
                    waveform_generated = diffusion_pl_module.custom_pred(data_dict).cpu()[0]
                    filename_sample = f'sampled_prompt[{text_prompt}]_tempo[{args.tempo}]_key[{key_str}].wav'
                else:
                    data_dict["latents_normalized"] = latents_normalized
                    waveform_generated = diffusion_pl_module.audio_edit(
                        data_dict,
                        start_diffusion_step = args.edit_audio_start_step
                    ).cpu()[0]
                    filename_sample = f'edited_audio[{audio_name}]_prompt[{text_prompt}]_tempo[{args.tempo}]_key[{key_str}].wav'

            filepath_sample = os.path.join(save_dir, filename_sample)
            torchaudio.save(filepath_sample, waveform_generated, config.sample_rate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in training DAC latent diffusion.')
    parser.add_argument(
        '--save-dir', type=str, default='',
        help='the directory that model results are saved'
    )
    parser.add_argument(
        '--load-ckpt-path', type=str, nargs='?',
        help='the checkpoint path to load'
    )
    parser.add_argument(
        '--edit-audio-path', type=str, nargs='?',
        help='the audio to edit. If specified, sampling will become the editing task on the given audio.'
    )
    parser.add_argument(
        '--edit-audio-start-step', type=int, default=50,
        help='the start step of sampling the audio editing loop.'
    )
    parser.add_argument(
        '--tempo', type=int, default=120,
        help='the tempo input, give non-negative integers'
    )
    parser.add_argument(
        '--key', type=int, default=None,
        help='the key input, take integers from 0 to 23'
    )
    parser.add_argument(
        '--chroma-path', type=str, nargs='?',
        help='the numpy file providing chroma feature'
    )
    parser.add_argument(
        '--text-prompt', type=str, default=[''], nargs='+'
    )
    parser.add_argument(
        '--rvq-pattern', type=str, default='parallel',
        help='choose from "parallel", "flattened" and "VALL-E"; default: "parallel"'
    )
    parser.add_argument(
        '--model-size', type=str, default='large',
        help='choose from "large" and "small"; default: "large"'
    )
    parser.add_argument(
        '--prediction-type', type=str, default='sample',
        help='choose from "epsilon", "sample", "v_prediction"; default: "sample"'
    )
    parser.add_argument(
        '--scheduler', type=str, default='handcrafted',
        help='choose from "handcrafted", "diffusers"; default: "handcrafted"'
    )
    parser.add_argument(
        '--dur-sec', type=float, default=10.0,
        help='for audio editing, the duration is set to be the same as input audio length instead of this input'
    )
    args = parser.parse_args()
    main(args)
