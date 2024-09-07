# this is the encoder decoder for meta condition embedding

from typing import Any, Dict, Optional, Tuple, List
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np

from .cond_net import CondNetBase

from laion_clap.training.data import get_audio_features

class DISCO200kWav2featNet(CondNetBase):
    def __init__(
        self,
        clap_model,
        clap_emb_dim,
        fix_clap = False,
        feature_attr_dims: Dict = {
            'acousticness': 1, 'danceability': 1,
            'energy': 1, 'instrumentalness': 1,
            'key': 12, 'liveness': 1, 'loudness': 1, 'mode': 2,
            'speechiness': 1, 'tempo': 1, 'time_signature': 6,
            'valence': 1
        },
    ):
        super().__init__(feature_attr_dims)

        self.clap_model = clap_model
        self.clap_emb_dim = clap_emb_dim
        self.fix_clap = fix_clap

        # decoders
        self.base_decoder = nn.Sequential(
            nn.Linear(clap_emb_dim, clap_emb_dim),
            nn.Dropout(p = 0.3),
            nn.ReLU(),
            nn.Linear(clap_emb_dim, clap_emb_dim),
            nn.Dropout(p = 0.3),
            nn.ReLU(),
            nn.Linear(clap_emb_dim, self.feature_dims_total),
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

    def decode(self, clap_emb):
        decoded_cond = self.base_decoder(clap_emb)

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

    def get_htsat_audio_input(self, x):
        device = x.device

        audio_input = []
        for audio_waveform in x:
            # quantize
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict, audio_waveform, 480000,
                data_truncating='rand_trunc',
                data_filling='repeatpad',
                audio_cfg=self.clap_model.model_cfg['audio_cfg'],
                require_grad=audio_waveform.requires_grad
            )
            audio_input.append(temp_dict)

        # clap_audio_encoder_input_dict = {}
        # keys = audio_input[0].keys()
        # for k in keys:
        #     clap_audio_encoder_input_dict[k] = torch.cat(
        #         [d[k].unsqueeze(0) for d in audio_input], dim=0
        #     ).to(device)

        return audio_input # clap_audio_encoder_input_dict

    def get_audip_clap(self, waveforms):
        clap_audio_encoder_input = self.get_htsat_audio_input(waveforms)

        input_dict = {}
        keys = clap_audio_encoder_input[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in clap_audio_encoder_input], dim=0)

        audio_embeds = self.clap_model.model.audio_branch(input_dict, mixup_lambda=None)["embedding"]
        # print("audio_embeds shape: ", audio_embeds.shape)
        # print("audio_embeds sum: ", audio_embeds.sum())
        audio_embeds = self.clap_model.model.audio_projection(audio_embeds)
        # print("audio_embeds sum2: ", audio_embeds.sum())
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        return audio_embeds

    # encode and decode, getting reconstruction loss
    def forward(self, batch_dict):
        waveforms = batch_dict["waveform"]
        # device = batch_dict["waveform"].device
        # print("waveform sum: ", waveforms.sum())

        normalized_cond = self.get_emb_from_input_dict(batch_dict)
        normalized_cond = normalized_cond.to(torch.float32)
        # print("normalized_cond sum: ", normalized_cond.sum())
        if torch.isnan(normalized_cond.sum()):
            print("NaN encountered in input normalized cond")
            normalized_cond = torch.nan_to_num(normalized_cond, nan=0, neginf=-4)
            print("Using nan_to_num to reduce nan to zero")

        if not self.fix_clap:
            self.clap_model.model.train()
            clap_emb = self.get_audip_clap(waveforms)
        else:
            with torch.no_grad():
                self.clap_model.model.eval()
                clap_emb = self.get_audip_clap(waveforms)

        if torch.isnan(clap_emb.sum()):
            print("NaN encountered in clap calculation")
        # clap_emb = self.clap_model.get_audio_embedding_from_data(x=waveforms, use_tensor=True)
        # print("clap_emb sum: ", clap_emb.sum())

        decoded_cond_dict = self.decode(clap_emb)
        for attr in decoded_cond_dict:
            if torch.isnan(decoded_cond_dict[attr].sum()):
                print(f"NaN encountered in {attr} after decoding")

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
        if self.fix_clap:
            params = []
            params += list(self.base_decoder.parameters())
            params += list(self.key_decoder.parameters())
            params += list(self.mode_decoder.parameters())
            params += list(self.time_signature_decoder.parameters())
            return optim.Adam(params, lr = lr)
        else:
            return optim.Adam(self.parameters(), lr = lr)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        waveforms = batch["waveform"]
        # device = waveforms.device
        self.clap_model.model.eval()
        clap_emb = self.get_audip_clap(waveforms)
        # clap_audio_encoder_input = self.get_htsat_audio_input(waveforms)
        # clap_emb = self.clap_model.model.get_audio_embedding(clap_audio_encoder_input)
        # clap_emb = self.clap_model.get_audio_embedding_from_data(x=waveforms, use_tensor=True)
        decoded_cond_dict = self.decode(clap_emb)
        decoded_cond_dict = self.denormalize(decoded_cond_dict)
        if "name" in batch:
            decoded_cond_dict["name"] = batch["name"]
        return decoded_cond_dict

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.predict_step(batch, batch_idx, dataloader_idx = dataloader_idx)