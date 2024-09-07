from typing import Any, Dict, Optional, Tuple
import torch
from torch import nn

from random import random
from functools import partial

from .utils import prob_mask_like
from .unet_blocks import ResnetBlock, SinusoidalPosEmb, LinearAttention, Residual, PreNorm, Downsample, Attention, Upsample


class Unet1D(nn.Module):
    def __init__(
        self,
        input_dim: int = 72,
        input_cond_dim: int = 512,
        num_codebooks: int = 9,
        dim: int = 144,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        sinusoidal_pos_emb_theta: float = 10000,
        attn_dim_head: int = 32,
        cond_drop_prob: float = 0.5,
    ):
        super().__init__()

        # determine dimensions

        self.init_conv = nn.Conv1d(input_dim, dim, 3, padding = 1)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = num_codebooks)

        # time embeddings

        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # cond embeddings

        self.cond_drop_prob = cond_drop_prob
        cond_dim = dim * 4
        self.null_cond_emb = nn.Parameter(torch.randn(input_cond_dim), requires_grad=True)
        self.cond_mlp = nn.Sequential(
            nn.Linear(input_cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, cond_emb_dim = cond_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, cond_emb_dim = cond_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in, heads = num_codebooks))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, cond_emb_dim = cond_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = num_codebooks)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, cond_emb_dim = cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, cond_emb_dim = cond_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, cond_emb_dim = cond_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out, heads = num_codebooks))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.out_dim = input_dim

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, cond_emb_dim = cond_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(
        self,
        latent: torch.FloatTensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cond_drop_prob: Optional[float] = None,
    ):
        B, D_in, L = latent.shape

        if cond_drop_prob is None:
            cond_drop_prob = self.cond_drop_prob

        if encoder_hidden_states is not None:
            assert len(encoder_hidden_states.shape) == 2
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((B,), 1 - self.cond_drop_prob, device=encoder_hidden_states.device)  # (B,)
                keep_mask = keep_mask.unsqueeze(-1) # (B, 1)
                null_cond_emb = encoder_hidden_states*0 + self.null_cond_emb.unsqueeze(0)
                encoder_hidden_states = torch.where(
                    keep_mask, encoder_hidden_states, null_cond_emb
                )
        else:
            encoder_hidden_states = self.null_cond_emb.unsqueeze(0) # (1, D2)
            encoder_hidden_states = encoder_hidden_states.repeat(B, 1) # (B, D2)

        c = self.cond_mlp(encoder_hidden_states)

        x = self.init_conv(latent)
        r = x.clone()

        t = self.time_mlp(timestep)

        h = []

        for block1, block2, attn, block3, block4, attn2, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, block3, block4, attn2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        pred_cond = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return pred_cond

        pred_uncon = self.forward(*args, cond_drop_prob = 1., **kwargs)
        pred_cond = pred_uncon + (pred_cond - pred_uncon) * cond_scale

        return pred_cond