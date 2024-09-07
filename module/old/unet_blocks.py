from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .attention_processor import Attention
from .resnet import Downsample1D, Upsample1D
from .lora import LoRACompatibleLinear
from .activations import get_activation


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

class ResConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        mid_channels: int, 
        out_channels: int, 
        temb_channels: Optional[int] = None, 
        dropout: float = 0.0,
        up: bool = False,
        down: bool = False,
        is_last: bool = False
    ):
        super().__init__()
        self.is_last = is_last
        self.has_conv_skip = in_channels != out_channels

        if self.has_conv_skip:
            # self.conv_skip = nn.Conv1d(in_channels, out_channels, 1, bias=False)
            self.conv_skip = nn.Linear(in_channels, out_channels)

        self.up = up
        self.down = down
        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample1D(in_channels, use_conv=False)
        elif self.down:
            self.downsample = Downsample1D(in_channels, use_conv=False)

        # self.conv_1 = nn.Conv1d(in_channels, mid_channels, 5, padding=2)
        # self.conv_1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.conv_1 = nn.Linear(in_channels, mid_channels)

        self.nonlinearity = nn.GELU()
        if temb_channels is not None:
            self.time_emb_proj = LoRACompatibleLinear(temb_channels, mid_channels)
        else:
            self.time_emb_proj = None

        self.group_norm_1 = nn.GroupNorm(1, mid_channels)
        self.gelu_1 = nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
        # self.conv_2 = nn.Conv1d(mid_channels, out_channels, 5, padding=2)
        # self.conv_2 = nn.Conv1d(mid_channels, out_channels, 1)
        self.conv_2 = nn.Linear(mid_channels, out_channels)

        if not self.is_last:
            self.group_norm_2 = nn.GroupNorm(1, out_channels)
            self.gelu_2 = nn.GELU()

        init_layer(self.conv_1)
        init_layer(self.conv_2)

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None):
        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                hidden_states = hidden_states.contiguous()
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        residual = self.conv_skip(hidden_states.transpose(-2,-1)).transpose(-2,-1) if self.has_conv_skip else hidden_states
        hidden_states = self.conv_1(hidden_states.transpose(-2,-1)).transpose(-2,-1)

        if temb is not None:
            if self.time_emb_proj is not None:
                temb = self.nonlinearity(temb)
                temb = self.time_emb_proj(temb)[:, :, None]
            hidden_states = hidden_states + temb

        hidden_states = self.group_norm_1(hidden_states)
        hidden_states = self.gelu_1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv_2(hidden_states.transpose(-2,-1)).transpose(-2,-1)

        if not self.is_last:
            hidden_states = self.group_norm_2(hidden_states)
            hidden_states = self.gelu_2(hidden_states)

        output = hidden_states + residual
        return output


class DownBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        downsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResConvBlock(in_channels, out_channels, out_channels, temb_channels = temb_channels, dropout = dropout)
            )

        self.resnets = nn.ModuleList(resnets)
        if downsample:
            self.downsamplers = nn.ModuleList([Downsample1D(out_channels, use_conv=False, name="op")])
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None):
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

class UpBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResConvBlock(resnet_in_channels + res_skip_channels, out_channels, out_channels, temb_channels = temb_channels, dropout = dropout)
            )

        self.resnets = nn.ModuleList(resnets)
        if upsample:
            self.upsamplers = nn.ModuleList([Upsample1D(out_channels, use_conv=False, name="op")])
        else:
            self.upsamplers = None

    def forward(
        self, 
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor], 
        temb: Optional[torch.FloatTensor] = None
    ):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class UNetMidBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        attention_head_dim: int = 1,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        resnet_time_scale_shift: str = "default",  # default, spatial
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResConvBlock(in_channels, in_channels, in_channels, temb_channels = temb_channels, dropout = dropout)
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            attentions.append(
                Attention(
                    in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups if resnet_time_scale_shift == "default" else None,
                    spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )
            resnets.append(
                ResConvBlock(in_channels, in_channels, in_channels, temb_channels = temb_channels, dropout = dropout)
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states



class AttnDownBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        attention_head_dim: int = 1,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        resnet_time_scale_shift: str = "default",
        output_scale_factor: float = 1.0,
        downsample: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.downsample = downsample

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResConvBlock(in_channels, out_channels, out_channels, temb_channels = temb_channels, dropout = dropout)
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if downsample:
            self.downsamplers = nn.ModuleList([Downsample1D(out_channels, use_conv=False, name="op")])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

class AttnUpBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        attention_head_dim: int = 1,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        resnet_time_scale_shift: str = "default",
        output_scale_factor: float = 1.0,
        upsample: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.upsample = upsample

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResConvBlock(resnet_in_channels + res_skip_channels, out_channels, out_channels, temb_channels = temb_channels, dropout = dropout)
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if upsample:
            self.upsamplers = nn.ModuleList([Upsample1D(out_channels, use_conv=False, name="op")])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states



def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    attention_head_dim: int, # should be divisible to out_channels
    add_downsample: bool,
    resnet_eps: float = 1e-6,
    resnet_groups: Optional[int] = None,
    resnet_time_scale_shift: str = "default",
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock1D":
        return DownBlock1D(
            in_channels=in_channels, 
            out_channels=out_channels, 
            temb_channels=temb_channels,
            downsample = add_downsample,
        )
    elif down_block_type == "AttnDownBlock1D":
        return AttnDownBlock1D(
            in_channels = in_channels,
            out_channels = out_channels,
            temb_channels = temb_channels,
            num_layers = num_layers,
            resnet_eps = resnet_eps,
            resnet_time_scale_shift = resnet_time_scale_shift,
            resnet_groups = resnet_groups,
            attention_head_dim = attention_head_dim,
            downsample = add_downsample,
        )
    raise ValueError(f"{down_block_type} does not exist.")

def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    prev_output_channel: int,
    out_channels: int,
    temb_channels: int,
    attention_head_dim: int, # should be divisible to out_channels
    add_upsample: bool,
    resnet_eps: float = 1e-6,
    resnet_groups: Optional[int] = None,
    resnet_time_scale_shift: str = "default",
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock1D":
        return UpBlock1D(
            in_channels=in_channels, 
            prev_output_channel=prev_output_channel, 
            out_channels=out_channels, 
            temb_channels=temb_channels,
            upsample = add_upsample,
        )
    elif up_block_type == "AttnUpBlock1D":
        return AttnUpBlock1D(
            in_channels = in_channels,
            prev_output_channel = prev_output_channel,
            out_channels = out_channels,
            temb_channels = temb_channels,
            num_layers = num_layers,
            resnet_eps = resnet_eps,
            resnet_time_scale_shift = resnet_time_scale_shift,
            resnet_groups = resnet_groups,
            attention_head_dim = attention_head_dim,
            upsample = add_upsample,
        )
    raise ValueError(f"{up_block_type} does not exist.")

