# this is the encoder decoder for meta condition embedding

import os
from typing import Any, Dict, Optional, Tuple, List
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import math
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Model
# from accelerate import dispatch_model, infer_auto_device_map
# from accelerate.utils import get_balanced_memory

class CondNetBase(pl.LightningModule):
    def __init__(
        self,
        feature_attr_dims: Dict = {
            'acousticness': 1, 'danceability': 1,
            'energy': 1, 'instrumentalness': 1,
            'key': 12, 'liveness': 1, 'loudness': 1, 'mode': 2,
            'speechiness': 1, 'tempo': 1, 'time_signature': 6,
            'valence': 1
        },
    ):
        super().__init__()

        self.NUM_KEYS = 12
        self.NUM_MODES = 2
        self.NUM_TIME_SIGNATURES = 6

        # determine dimensions
        self.feature_attr_dims = feature_attr_dims
        self.feature_attrs = list(feature_attr_dims.keys())
        self.feature_dims = list(feature_attr_dims.values())

        self.feature_dims_cumsum = torch.cumsum(torch.tensor(self.feature_dims), dim=0)
        self.feature_dims_total = sum(self.feature_dims)

        # input embeddings
        if 'key' in self.feature_attrs:
            self.key_embedding = nn.Embedding(self.NUM_KEYS, feature_attr_dims['key'])
        if 'mode' in self.feature_attrs:
            self.mode_embedding = nn.Embedding(self.NUM_MODES, feature_attr_dims['mode'])
        if 'time_signature' in self.feature_attrs:
            self.time_signature_embedding = nn.Embedding(self.NUM_TIME_SIGNATURES, feature_attr_dims['time_signature'])

    def default_normalize(self, x):
        return 2 * (x - 0.5)

    def default_denormalize(self, x):
        return torch.clamp(x / 2 + 0.5, min = 0, max = 1)

    def tempo_normalize(self, x):
        return (x - 132) / 50

    def tempo_denormalize(self, x):
        return torch.clamp(x * 50 + 132, min = 0)

    def loudness_normalize(self, x):
        return (x + 20) / 20

    def loudness_denormalize(self, x):
        return x * 20 - 20

    def get_emb_from_input_dict(self, batch_dict):
        normalized_cond_list = []
        for attr in self.feature_attrs:
            if attr not in batch_dict:
                assert 0, f"attr {attr} is not defined in the model input"
            if attr == 'key':
                batch_dict[attr] = batch_dict[attr].to(int)
                normalized_cond_list.append(self.key_embedding(batch_dict[attr]))
            elif attr == 'mode':
                batch_dict[attr] = batch_dict[attr].to(int)
                normalized_cond_list.append(self.mode_embedding(batch_dict[attr]))
            elif attr == 'time_signature':
                batch_dict[attr] = batch_dict[attr].to(int)
                normalized_cond_list.append(self.time_signature_embedding(batch_dict[attr]))
            elif attr == 'tempo':
                batch_dict[attr] = batch_dict[attr].to(torch.float32)
                normalized_cond_list.append(self.tempo_normalize(batch_dict[attr]))
            elif attr == 'loudness':
                batch_dict[attr] = batch_dict[attr].to(torch.float32)
                normalized_cond_list.append(self.loudness_normalize(batch_dict[attr]))
            else:
                batch_dict[attr] = batch_dict[attr].to(torch.float32)
                normalized_cond_list.append(self.default_normalize(batch_dict[attr]))

        for i_emb in range(len(normalized_cond_list)):
            if len(normalized_cond_list[i_emb].shape) == 1:
                normalized_cond_list[i_emb] = normalized_cond_list[i_emb].unsqueeze(-1)

        return torch.cat(normalized_cond_list, dim = -1)

    def denormalize(self, output_dict):
        for attr in self.feature_attrs:
            if attr not in output_dict:
                assert 0, f"attr {attr} is not defined in the output dict"
            if attr == 'key' or attr == 'mode' or attr == 'time_signature':
                output_dict[attr] = torch.argmax(output_dict[attr], dim = -1)
            elif attr == 'tempo':
                output_dict[attr] = self.tempo_denormalize(output_dict[attr])
            elif attr == 'loudness':
                output_dict[attr] = self.loudness_normalize(output_dict[attr])
            else:
                output_dict[attr] = self.default_denormalize(output_dict[attr])
        return output_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self.forward(batch)
        self.log_dict(loss_dict)

        final_loss = 0
        for x in loss_dict.values():
            final_loss = x + final_loss
        final_loss = final_loss / len(loss_dict)
        self.log("final loss", final_loss)

        return final_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict = self.forward(batch)

        final_loss = 0
        for x in loss_dict.values():
            final_loss = x + final_loss
        final_loss = final_loss / len(loss_dict)
        self.log("val final loss", final_loss)

        return final_loss

    def load_state_dict_partial(self, ckpt_state_dict):
        for name, param in ckpt_state_dict.items():
            if name not in self.state_dict():
                print(f"{name} not in the model state dict")
                continue
            elif isinstance(param, torch.nn.Parameter):
                try:
                    param = param.data
                    self.state_dict()[name].copy_(param)
                    print(f"{name} loaded into model state dict")
                except Exception as e:
                    print(f"error encountered in loading param {name}")
                    print(e)
            elif torch.is_tensor(param):
                try:
                    self.state_dict()[name].copy_(param)
                    print(f"{name} loaded into model state dict")
                except Exception as e:
                    print(f"error encountered in loading param {name}")
                    print(e)

class DISCO200kMetaEncoderDecoder(CondNetBase):
    def __init__(
        self,
        feature_emb_dim: int = 512,
        feature_attr_dims: Dict = {
            'acousticness': 1, 'danceability': 1,
            'energy': 1, 'instrumentalness': 1,
            'key': 12, 'liveness': 1, 'loudness': 1, 'mode': 2,
            'speechiness': 1, 'tempo': 1, 'time_signature': 6,
            'valence': 1
        },
    ):
        super().__init__(feature_attr_dims = feature_attr_dims)

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dims_total, feature_emb_dim),
            nn.GELU(),
            nn.Linear(feature_emb_dim, feature_emb_dim)
        )

        # decoders
        self.base_decoder = nn.Sequential(
            nn.Linear(feature_emb_dim, self.feature_dims_total)
        )
        if 'key' in self.feature_attrs:
            self.key_decoder = nn.Linear(feature_attr_dims['key'], self.NUM_KEYS)
        if 'mode' in self.feature_attrs:
            self.mode_decoder = nn.Linear(feature_attr_dims['mode'], self.NUM_MODES)
        if 'time_signature' in self.feature_attrs:
            self.time_signature_decoder = nn.Linear(feature_attr_dims['time_signature'], self.NUM_TIME_SIGNATURES)

        # losses
        self.discrete_loss = nn.CrossEntropyLoss()
        self.continuous_loss = nn.L1Loss()

        self.save_hyperparameters()

    def encode(self, batch_dict):
        normalized_cond = self.get_emb_from_input_dict(batch_dict)
        cond_emb = self.encoder(normalized_cond)
        return cond_emb

    def decode(self, cond_emb):
        decoded_cond = self.base_decoder(cond_emb)

        output_dict = {}
        dim_start = 0
        for i, attr in enumerate(self.feature_attrs):
            dim_end = self.feature_dims_cumsum[i]
            output_dict[attr] = decoded_cond[..., dim_start:dim_end]
            # get logits for discrete-valued attributes
            if attr == 'key':
                output_dict[attr] = self.key_decoder(output_dict[attr])
            elif attr == 'mode':
                output_dict[attr] = self.mode_decoder(output_dict[attr])
            elif attr == 'time_signature':
                output_dict[attr] = self.time_signature_decoder(output_dict[attr])

            dim_start = dim_end

        return output_dict

    # encode and decode, getting reconstruction loss
    def forward(self, batch_dict):
        normalized_cond = self.get_emb_from_input_dict(batch_dict)
        normalized_cond = normalized_cond.to(torch.float32)
        cond_emb = self.encoder(normalized_cond)
        decoded_cond_dict = self.decode(cond_emb)

        loss_dict = {}
        dim_start = 0
        for i, attr in enumerate(self.feature_attrs):
            dim_end = self.feature_dims_cumsum[i]
            # get logits for discrete-valued attributes
            if attr == 'key' or attr == 'mode' or attr == 'time_signature':
                this_target = batch_dict[attr]
                loss_dict[attr] = self.discrete_loss(decoded_cond_dict[attr], this_target)
            else:
                this_target = normalized_cond[..., dim_start:dim_end]
                loss_dict[attr] = self.continuous_loss(decoded_cond_dict[attr], this_target)

            dim_start = dim_end

        return loss_dict

    def configure_optimizers(self, lr = 0.001):
        return optim.Adam(self.parameters(), lr = lr)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.encode(batch)


class DISCO200kDecoderFromCLAP(CondNetBase):
    def __init__(
        self,
        emb_dim: int = 512,
        feature_attr_dims: Dict = {
            'acousticness': 1, 'danceability': 1,
            'energy': 1, 'instrumentalness': 1,
            'key': 12, 'liveness': 1, 'loudness': 1, 'mode': 2,
            'speechiness': 1, 'tempo': 1, 'time_signature': 6,
            'valence': 1
        },
    ):
        super().__init__(feature_attr_dims = feature_attr_dims)

        self.emb_dim = emb_dim
        # decoders
        self.base_decoder = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(emb_dim, self.feature_dims_total),
        )
        if 'key' in self.feature_attrs:
            self.key_decoder = nn.Linear(feature_attr_dims['key'], self.NUM_KEYS)
        if 'mode' in self.feature_attrs:
            self.mode_decoder = nn.Linear(feature_attr_dims['mode'], self.NUM_MODES)
        if 'time_signature' in self.feature_attrs:
            self.time_signature_decoder = nn.Linear(feature_attr_dims['time_signature'], self.NUM_TIME_SIGNATURES)

        # losses
        self.discrete_loss = nn.CrossEntropyLoss()
        self.continuous_loss = nn.L1Loss()

        self.save_hyperparameters()

    def decode(self, cond_emb):
        decoded_cond = self.base_decoder(cond_emb)

        output_dict = {}
        dim_start = 0
        for i, attr in enumerate(self.feature_attrs):
            dim_end = self.feature_dims_cumsum[i]
            output_dict[attr] = decoded_cond[..., dim_start:dim_end]
            # get logits for discrete-valued attributes
            if attr == 'key':
                output_dict[attr] = self.key_decoder(output_dict[attr])
            elif attr == 'mode':
                output_dict[attr] = self.mode_decoder(output_dict[attr])
            elif attr == 'time_signature':
                output_dict[attr] = self.time_signature_decoder(output_dict[attr])

            dim_start = dim_end

        return output_dict

    # encode and decode, getting reconstruction loss
    def forward(self, batch_dict):
        normalized_cond = self.get_emb_from_input_dict(batch_dict)
        normalized_cond = normalized_cond.to(torch.float32)
        cond_emb = batch_dict["audio_clap"]
        decoded_cond_dict = self.decode(cond_emb)

        loss_dict = {}
        dim_start = 0
        for i, attr in enumerate(self.feature_attrs):
            dim_end = self.feature_dims_cumsum[i]
            # get logits for discrete-valued attributes
            if attr == 'key' or attr == 'mode' or attr == 'time_signature':
                this_target = batch_dict[attr]
                loss_dict[attr] = self.discrete_loss(decoded_cond_dict[attr], this_target)
            else:
                this_target = normalized_cond[..., dim_start:dim_end]
                loss_dict[attr] = self.continuous_loss(decoded_cond_dict[attr], this_target)

            dim_start = dim_end

        return loss_dict

    def configure_optimizers(self, lr = 0.001):
        return optim.Adam(self.parameters(), lr = lr)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        cond_emb = batch["audio_clap"]
        decoded_cond_dict = self.decode(cond_emb)
        return self.denormalize(decoded_cond_dict)

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.predict_step(batch, batch_idx)


class FMAKeyEmb24(nn.Module):
    def __init__(
        self,
        num_keys: int = 24,
        key_emb_dim: int = 27,
        init_with_spiral_emb: bool = True
    ):
        super().__init__()
        self.key_embedding = nn.Embedding(num_keys, key_emb_dim)
        if init_with_spiral_emb:
            assert key_emb_dim == num_keys + 3

            weight_matrix_onehots = np.eye(num_keys) * 2 # (24, 24)
            key_index_vector = np.arange(12)
            cos_vector = np.cos(2 * np.pi * key_index_vector / 12)
            cos_vector = np.concatenate([cos_vector, cos_vector], axis=0)[np.newaxis, :]
            sin_vector = np.sin(2 * np.pi * key_index_vector / 12)
            sin_vector = np.concatenate([sin_vector, sin_vector], axis=0)[np.newaxis, :]
            maj_min_key_vector = np.ones(12)
            maj_min_key_vector = np.concatenate([maj_min_key_vector, -maj_min_key_vector], axis=0)[np.newaxis, :]
            weight_matrix = np.concatenate([maj_min_key_vector, cos_vector, sin_vector, weight_matrix_onehots], axis=0)

            with torch.no_grad():
                self.key_embedding.weight = torch.nn.Parameter(torch.from_numpy(weight_matrix).to(torch.float32), requires_grad=False)

    def forward(self, key_int_tensor: torch.Tensor):
        return self.key_embedding(key_int_tensor)

class TempoEncoding(nn.Module):

    def __init__(self, d_model: int, max_tempo: int = 300):
        super().__init__()
        position = torch.arange(max_tempo).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(max_tempo/2) / d_model))

        pe = torch.zeros(max_tempo, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, tempo) -> torch.Tensor:
        """
        Arguments:
            tempo: shape [batch_size,]
        """
        return self.pe[tempo]

MODELS_DIMS = {
    "t5-small": 512,
    "t5-base": 768,
    "t5-large": 1024,
    "t5-3b": 1024,
    "t5-11b": 1024,
    "google/flan-t5-small": 512,
    "google/flan-t5-base": 768,
    "google/flan-t5-large": 1024,
    "google/flan-t5-3b": 1024,
    "google/flan-t5-11b": 1024,
}
class CondFuser(nn.Module):
    def __init__(
        self,
        feature_emb_dim: int = 512,
        feature_attr_dims: Dict = {
            'madmom_key':27, 'madmom_tempo':27
        },
        t5_model: str = "google/flan-t5-large"
    ):
        super().__init__()

        self.NUM_KEYS = 24
        self.feature_attr_dims = feature_attr_dims
        self.feature_attrs = list(feature_attr_dims.keys())
        self.feature_dims = list(feature_attr_dims.values())

        self.feature_dims_cumsum = torch.cumsum(torch.tensor(self.feature_dims), dim=0)
        self.feature_dims_total = sum(self.feature_dims)

        if 'madmom_key' in self.feature_attrs:
            self.key_embedding = FMAKeyEmb24(self.NUM_KEYS, feature_attr_dims['madmom_key'])
        elif 'key' in self.feature_attrs:
            self.key_embedding = FMAKeyEmb24(self.NUM_KEYS, feature_attr_dims['key'])
        if 'madmom_tempo' in self.feature_attrs:
            self.tempo_embedding = TempoEncoding(d_model = feature_attr_dims['madmom_tempo']-1)
            self.tempo_dim = feature_attr_dims['madmom_tempo']
        elif 'tempo' in self.feature_attrs:
            self.tempo_embedding = TempoEncoding(d_model=feature_attr_dims['tempo'] - 1)
            self.tempo_dim = feature_attr_dims['tempo']

        self.feature_emb_dim = feature_emb_dim
        self.text_emb_dim = MODELS_DIMS[t5_model]
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model, torch_dtype=torch.float16)# .to(self.device) # , device_map="auto"
        # self.t5_model.parallelize()
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model)

        # max_memory = get_balanced_memory(
        #     self.t5_model,
        #     max_memory=None,
        #     dtype=torch.float32,
        #     low_zero=False
        # )
        # device_map = infer_auto_device_map(
        #     self.t5_model,
        #     max_memory=max_memory,
        #     dtype=torch.float32
        # )
        # self.t5_model = dispatch_model(self.t5_model, device_map=device_map)

    def default_normalize(self, x):
        return 2 * (x - 0.5)

    def default_denormalize(self, x):
        return torch.clamp(x / 2 + 0.5, min = 0, max = 1)

    def tempo_normalize(self, x):
        return (x - 132) / 50

    def tempo_denormalize(self, x):
        return torch.clamp(x * 50 + 132, min = 0)

    def get_feature_emb_from_input_dict(self, batch_dict):
        normalized_cond_list = []
        for attr in self.feature_attrs:
            if attr not in batch_dict:
                assert 0, f"attr {attr} is not defined in the model input"
            if attr == 'madmom_key' or attr == 'key':
                batch_dict[attr] = batch_dict[attr].to(int)
                normalized_cond_list.append(self.key_embedding(batch_dict[attr]))
            elif attr == 'madmom_tempo' or attr == 'tempo':
                batch_dict[attr] = batch_dict[attr].to(torch.float32)
                normalized_tempo = self.tempo_normalize(batch_dict[attr])
                tempo_int = batch_dict[attr].to(int)
                tempo_pe = self.tempo_embedding(tempo_int)

                if len(normalized_tempo.shape) == 1:
                    normalized_tempo = normalized_tempo.unsqueeze(-1)
                assert len(normalized_tempo.shape) == 2
                # normalized_tempo_repeat = normalized_tempo.repeat(1, self.tempo_dim).to(torch.float32)
                tempo_emb = torch.cat([normalized_tempo, tempo_pe], dim = 1)
                normalized_cond_list.append(tempo_emb)

        for i_emb in range(len(normalized_cond_list)):
            if len(normalized_cond_list[i_emb].shape) == 1:
                normalized_cond_list[i_emb] = normalized_cond_list[i_emb].unsqueeze(-1)

        default_emb = torch.cat(normalized_cond_list, dim = -1)
        zero_emb = torch.zeros_like(default_emb).sum(-1, keepdim = True)
        fill_dim = self.feature_emb_dim - default_emb.shape[-1]
        assert fill_dim >= 0, "Feature dim is larger than feature_emb_dim"
        zero_pad = zero_emb.repeat(1, fill_dim)

        return torch.cat([default_emb, zero_pad], dim = -1)

    @torch.no_grad()
    def get_text_emb_from_input_dict(self, batch_dict):
        text = batch_dict["randomized_text"]
        if self.t5_model.training:
            self.t5_model.eval()
            torch.cuda.empty_cache()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # enc = self.tokenizer(
            #     text,
            #     return_tensors="pt", truncation=True, padding='longest'
            # )#.to(self.device) #.to(self.t5_model.device)
            # print(enc)
            # print(self.t5_model.device)
            # print("In fuser t5_input_ids", batch_dict["t5_input_ids"])
            # print(batch_dict["t5_attention_mask"])
            text_emb_seq = self.t5_model.encoder(
                input_ids=batch_dict["t5_input_ids"],
                attention_mask=batch_dict["t5_attention_mask"],
                return_dict=True,
            ).last_hidden_state
            # print(emb.shape)
            # print("In fuser T5 emb mean:", text_emb_seq.mean())
        return text_emb_seq.permute(0, 2, 1) # [B, D, L]

    def denormalize(self, output_dict):
        for attr in self.feature_attrs:
            if attr not in output_dict:
                assert 0, f"attr {attr} is not defined in the output dict"
            if attr == 'madmom_key' or attr == 'key':
                output_dict[attr] = torch.argmax(output_dict[attr], dim = -1) - 3 # there is dim-3 spiral emb
            elif attr == 'madmom_tempo' or attr == 'tempo':
                output_dict[attr] = self.tempo_denormalize(output_dict[attr][..., 0])
            else:
                output_dict[attr] = self.default_denormalize(output_dict[attr])
        return output_dict

    def encode(self, batch_dict):
        # feature_cond_orig = self.get_feature_emb_from_input_dict(batch_dict)
        text_cond = self.get_text_emb_from_input_dict(batch_dict)
        feature_cond = text_cond[..., 0] # debug: not using CLAP, but Flan-T5 CLS
        # print(feature_cond.shape)
        # print(feature_cond_orig.shape)
        # text_cond = batch_dict["t5_emb"]
        return feature_cond, text_cond

    def decode_feature_cond(self, feature_cond):
        decoded_cond = self.base_decoder(feature_cond)

        output_dict = {}
        dim_start = 0
        for i, attr in enumerate(self.feature_attrs):
            dim_end = self.feature_dims_cumsum[i]
            output_dict[attr] = decoded_cond[..., dim_start:dim_end]
            dim_start = dim_end

        return self.denormalize(output_dict)

    # encode and decode, getting reconstruction loss
    def forward(self, batch_dict):
        return self.encode(batch_dict)

