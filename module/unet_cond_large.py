from typing import Any, Dict, Optional, Tuple, List
import torch
from torch import nn

from random import random
from functools import partial

from .utils import prob_mask_like
from .unet_blocks import ResnetBlock, SinusoidalPosEmb, LinearAttention, CrossAttention, Residual, PreNorm, PreNormCrsattn, Downsample, Attention, Upsample
from .unet_blocks import ResnetBlock2D, Upsample2D, Downsample2D

class Unet1DCondBase(nn.Module): # without init and final conv
    def __init__(
        self,
        input_dim: int = 72,
        feature_cond_dim: int = 512,
        chroma_cond_dim: int = 12,
        text_cond_dim: int = 1024,
        num_codebooks: int = 9,
        num_attn_heads: int = 9,
        dim: int = 144,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        sinusoidal_pos_emb_theta: float = 10000,
        attn_dim_head: int = 32,
        cond_drop_prob: float = 0.5,
    ):
        super().__init__()

        # determine dimensions

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
        self.null_cond_emb = nn.Parameter(torch.randn(feature_cond_dim, dtype=self.time_mlp[1].weight.dtype), requires_grad=True)
        self.null_chroma_seq_cond_emb = nn.Parameter(torch.zeros(chroma_cond_dim, 1, dtype=self.time_mlp[1].weight.dtype), requires_grad=True)
        self.null_text_seq_cond_emb = nn.Parameter(torch.zeros(text_cond_dim, 1, dtype=self.time_mlp[1].weight.dtype), requires_grad=True)
        self.cond_mlp = nn.Sequential(
            nn.Linear(feature_cond_dim, cond_dim),
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
                Residual(PreNorm(dim_in, LinearAttention(dim_in, heads=num_attn_heads))),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, cond_emb_dim=cond_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in, heads=num_attn_heads))),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, cond_emb_dim=cond_dim),
                Residual(PreNormCrsattn(dim_in, CrossAttention(dim_in, context_dim=chroma_cond_dim, heads=num_attn_heads, dim_head=attn_dim_head))),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, cond_emb_dim=cond_dim),
                Residual(PreNormCrsattn(dim_in, CrossAttention(dim_in, context_dim=text_cond_dim, heads=num_attn_heads, dim_head=attn_dim_head))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, cond_emb_dim = cond_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = num_attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, cond_emb_dim = cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, cond_emb_dim = cond_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, cond_emb_dim = cond_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out, heads = num_attn_heads))),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, cond_emb_dim=cond_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out, heads=num_attn_heads))),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, cond_emb_dim=cond_dim),
                Residual(PreNormCrsattn(dim_out, CrossAttention(dim_out, context_dim=chroma_cond_dim, heads=num_attn_heads, dim_head=attn_dim_head))),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, cond_emb_dim=cond_dim),
                Residual(PreNormCrsattn(dim_out, CrossAttention(dim_out, context_dim=text_cond_dim, heads=num_attn_heads, dim_head=attn_dim_head))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.out_dim = input_dim

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, cond_emb_dim = cond_dim)

    def forward_embeddings(self, x, c, c_seqs, t):
        r = x.clone()
        h = []

        assert type(c_seqs) == list or type(c_seqs) == tuple

        for block1, block2, attn, block3, attn2, block4, crs_attn, block5, crs_attn2, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = block3(x, t, c)
            x = attn2(x)
            h.append(x)

            x = block4(x, t, c)
            x = crs_attn(x, context=c_seqs[0])
            h.append(x)

            x = block5(x, t, c)
            x = crs_attn2(x, context=c_seqs[1])
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, block3, attn2, block4, crs_attn, block5, crs_attn2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = attn(x)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block3(x, t, c)
            x = attn2(x)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block4(x, t, c)
            x = crs_attn(x, context=c_seqs[0])

            x = torch.cat((x, h.pop()), dim = 1)
            x = block5(x, t, c)
            x = crs_attn2(x, context=c_seqs[1])

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return x

    def form_condition_emb(self, batch_size, vec_cond, seq_conds, cond_drop_prob):
        B = batch_size

        if cond_drop_prob is None:
            cond_drop_prob = self.cond_drop_prob

        if vec_cond is not None:
            assert len(vec_cond.shape) == 2
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((B,), 1 - self.cond_drop_prob, device=vec_cond.device)  # (B,)
                keep_mask = keep_mask.unsqueeze(-1) # (B, 1)
                null_cond_emb = vec_cond*0 + self.null_cond_emb.unsqueeze(0)
                vec_cond = torch.where(
                    keep_mask, vec_cond, null_cond_emb
                )
        else:
            vec_cond = self.null_cond_emb.unsqueeze(0) # (1, D2)
            vec_cond = vec_cond.repeat(B, 1) # (B, D2)

        if seq_conds is None:
            seq_conds = [None, None]
        for i, seq_cond in enumerate(seq_conds):
            this_null_seq_cond_emb = self.null_chroma_seq_cond_emb if i == 0 else self.null_text_seq_cond_emb
            if seq_cond is not None:
                assert len(seq_cond.shape) == 3
                if cond_drop_prob > 0:
                    keep_mask = prob_mask_like((B,), 1 - self.cond_drop_prob, device=seq_cond.device)  # (B,)
                    keep_mask = keep_mask.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                    this_null_seq_cond_emb = seq_cond*0 + this_null_seq_cond_emb.unsqueeze(0)
                    seq_cond = torch.where(
                        keep_mask, seq_cond, this_null_seq_cond_emb
                    )
            else:
                seq_cond = this_null_seq_cond_emb.unsqueeze(0) # (1, D2, 1)
                seq_cond = seq_cond.repeat(B, 1, 1) # (B, D2, 1)

            seq_conds[i] = seq_cond

        return vec_cond, seq_conds

    def forward(
        self,
        latent: torch.FloatTensor,
        timestep: torch.LongTensor,
        vec_cond: Optional[torch.Tensor] = None,
        seq_conds: Optional[List[torch.Tensor]] = None,
        cond_drop_prob: Optional[float] = None,
    ):
        pass

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

class Unet1DParallelPattern(Unet1DCondBase):
    def __init__(
        self,
        input_dim: int = 72,
        feature_cond_dim: int = 512,
        chroma_cond_dim: int = 12,
        text_cond_dim: int = 1024,
        num_codebooks: int = 9,
        num_attn_heads: int = 9,
        dim: int = 144,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        sinusoidal_pos_emb_theta: float = 10000,
        attn_dim_head: int = 32,
        cond_drop_prob: float = 0.5,
    ):
        super().__init__(
            input_dim = input_dim,
            feature_cond_dim = feature_cond_dim,
            chroma_cond_dim = chroma_cond_dim,
            text_cond_dim = text_cond_dim,
            num_codebooks = num_codebooks,
            num_attn_heads = num_attn_heads,
            dim = dim,
            dim_mults = dim_mults,
            sinusoidal_pos_emb_theta = sinusoidal_pos_emb_theta,
            attn_dim_head = attn_dim_head,
            cond_drop_prob = cond_drop_prob
        )
        assert input_dim % num_attn_heads == 0
        self.num_codebooks = num_codebooks
        self.num_attn_heads = num_attn_heads

        self.init_conv = nn.Conv1d(input_dim, dim, 3, padding = 1)
        self.final_conv = nn.Conv1d(dim, input_dim, 1)

    def forward(
        self,
        latent: torch.FloatTensor,
        timestep: torch.LongTensor,
        vec_cond: Optional[torch.Tensor] = None,
        seq_conds: Optional[List[torch.Tensor]] = None,
        cond_drop_prob: Optional[float] = None,
    ):
        B, D_in, L = latent.shape

        x = self.init_conv(latent)
        c, c_seqs = self.form_condition_emb(B, vec_cond, seq_conds, cond_drop_prob)
        c = self.cond_mlp(c)
        t = self.time_mlp(timestep)
        x = self.forward_embeddings(x, c, c_seqs, t)
        return self.final_conv(x)

class Unet1DVALLEPatternPrimary(Unet1DCondBase):
    def __init__(
        self,
        input_dim: int = 72,
        feature_cond_dim: int = 512,
        chroma_cond_dim: int = 12,
        text_cond_dim: int = 1024,
        num_codebooks: int = 9,
        num_attn_heads: int = 9,
        dim: int = 144,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        sinusoidal_pos_emb_theta: float = 10000,
        attn_dim_head: int = 32,
        cond_drop_prob: float = 0.5,
    ):
        super().__init__(
            input_dim = input_dim,
            feature_cond_dim = feature_cond_dim,
            chroma_cond_dim = chroma_cond_dim,
            text_cond_dim = text_cond_dim,
            num_codebooks = num_codebooks,
            num_attn_heads = num_attn_heads,
            dim = dim,
            dim_mults = dim_mults,
            sinusoidal_pos_emb_theta = sinusoidal_pos_emb_theta,
            attn_dim_head = attn_dim_head,
            cond_drop_prob = cond_drop_prob
        )
        assert input_dim % num_attn_heads == 0
        self.input_dim_primary = input_dim // num_codebooks
        self.num_codebooks = num_codebooks
        self.num_attn_heads = num_attn_heads
        self.out_dim_primary = self.input_dim_primary

        self.init_conv = nn.Conv1d(self.input_dim_primary, dim, 3, padding = 1)
        self.final_conv = nn.Conv1d(dim, self.out_dim_primary, 1)
    def forward(
        self,
        latent: torch.FloatTensor,
        timestep: torch.LongTensor,
        vec_cond: Optional[torch.Tensor] = None,
        seq_conds: Optional[List[torch.Tensor]] = None,
        cond_drop_prob: Optional[float] = None,
    ):
        B, D_in, L = latent.shape
        primary_latent = latent[:,:self.input_dim_primary] # (B, 72, L) -> (B, 8, L)

        x = self.init_conv(primary_latent)
        c, c_seqs = self.form_condition_emb(B, vec_cond, seq_conds, cond_drop_prob)
        # print(c.dtype, self.cond_mlp[0].weight.dtype)  # debug
        c = self.cond_mlp(c)
        t = self.time_mlp(timestep)
        x = self.forward_embeddings(x, c, c_seqs, t)
        return self.final_conv(x)

class Unet1DVALLEPatternSecondary(Unet1DCondBase):
    def __init__(
        self,
        input_dim: int = 72,
        feature_cond_dim: int = 512,
        chroma_cond_dim: int = 12,
        text_cond_dim: int = 1024,
        num_codebooks: int = 9,
        num_attn_heads: int = 9,
        dim: int = 144,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        sinusoidal_pos_emb_theta: float = 10000,
        attn_dim_head: int = 32,
        cond_drop_prob: float = 0.5,
    ):
        super().__init__(
            input_dim = input_dim,
            feature_cond_dim = feature_cond_dim,
            chroma_cond_dim = chroma_cond_dim,
            text_cond_dim = text_cond_dim,
            num_codebooks = num_codebooks,
            num_attn_heads = num_attn_heads,
            dim = dim,
            dim_mults = dim_mults,
            sinusoidal_pos_emb_theta = sinusoidal_pos_emb_theta,
            attn_dim_head = attn_dim_head,
            cond_drop_prob = cond_drop_prob
        )
        assert input_dim % num_codebooks == 0
        self.input_dim_primary = input_dim // num_codebooks
        self.input_dim_secondary = input_dim
        self.num_codebooks = num_codebooks
        self.num_attn_heads = num_attn_heads
        self.out_dim_secondary = self.input_dim_primary * (num_codebooks - 1)

        self.init_conv = nn.Conv1d(self.input_dim_secondary, dim, 3, padding = 1)
        self.final_conv = nn.Conv1d(dim, self.out_dim_secondary, 1)

    def forward(
        self,
        latent: torch.FloatTensor,
        timestep: torch.LongTensor,
        vec_cond: Optional[torch.Tensor] = None,
        seq_conds: Optional[List[torch.Tensor]] = None,
        cond_drop_prob: Optional[float] = None,
    ):
        B, D_in, L = latent.shape

        x = self.init_conv(latent)
        c, c_seqs = self.form_condition_emb(B, vec_cond, seq_conds, cond_drop_prob)
        c = self.cond_mlp(c)
        t = self.time_mlp(timestep)
        x = self.forward_embeddings(x, c, c_seqs, t)
        return self.final_conv(x)
