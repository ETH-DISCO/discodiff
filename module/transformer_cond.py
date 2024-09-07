from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm

from .attention import BasicTransformerBlock
from .embeddings import RVQCodebookEmbeddings, RVQEmbeddingsWithPosition
from .utils import prob_mask_like

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class TransformerCondRVQ(nn.Module):
    """
    A Transformer model for RVQ data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
    """

    def __init__(
        self,
        codebook_size: int = 1024,
        num_codebooks: int = 8,
        attention_head_dim: int = 128,
        max_frame_len: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.0,
        cond_drop_prob: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        num_embeds_ada_norm: Optional[int] = None, # for timestep emb
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        norm_type: str = "layer_norm",
        attention_type: str = "default",
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.class_num = codebook_size + 1
        self.num_codebooks = num_codebooks
        self.max_frame_len = max_frame_len
        self.attention_head_dim = attention_head_dim
        num_attention_heads = num_codebooks
        inner_dim = num_attention_heads * attention_head_dim

        self.cond_drop_prob = cond_drop_prob

        # 1. Define input layers
        self.content_with_pos_emb = RVQCodebookEmbeddings( # RVQEmbeddingsWithPosition(
            num_classes = self.class_num, 
            num_codebooks = self.num_codebooks, 
            max_frame_len = self.max_frame_len, 
            embed_dim = attention_head_dim
        )

        # 2. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    attention_type=attention_type,
                )
                for d in range(num_layers)
            ]
        )

        # 2.1. Define default conditions
        self.default_encoder_hidden_state = nn.Parameter(torch.randn(cross_attention_dim, 1), requires_grad=True)

        self.norm_out = nn.LayerNorm(inner_dim)
        # TODO: define num_codebooks different proj_out layers, for different heads
        # self.proj_out = nn.Linear(attention_head_dim, codebook_size)
        self.W_proj_out = nn.Parameter(
            torch.randn(num_codebooks, attention_head_dim, codebook_size), requires_grad=True
        )
        self.b_proj_out = nn.Parameter(
            torch.randn(num_codebooks, codebook_size), requires_grad=True
        )

    def forward(
        self,
        indices: torch.LongTensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        cond_drop_prob: Optional[float] = None,
    ):
        """
        Args:
            indices: torch.LongTensor, in shape (B, K, L) where K is num of codebooks, and L is frame length
            encoder_hidden_states: torch.FloatTensor of shape (B, D2, L2) for cross attention, for text prompt
            timestep: torch.LongTensor in shape (B, )
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels: torch.LongTensor in shape (B, )
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.

        Returns:
            logp(x_0), in shape (B, C, K, L), where C is the codebook size
        """
        B, K, L = indices.shape
        # 0. CFG condition drop
        if cond_drop_prob is None:
            cond_drop_prob = self.cond_drop_prob
        if encoder_hidden_states is not None:
            if len(encoder_hidden_states.shape) == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(-1)
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((B,), 1 - self.cond_drop_prob, device=encoder_hidden_states.device)  # (B,)
                keep_mask = keep_mask.unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)
                default_encoder_hidden_states = encoder_hidden_states*0 + self.default_encoder_hidden_state.unsqueeze(0)
                encoder_hidden_states = torch.where(
                    keep_mask, encoder_hidden_states, default_encoder_hidden_states
                )
        else:
            encoder_hidden_states = self.default_encoder_hidden_state.unsqueeze(0) # (1, D2, L2=1)
            encoder_hidden_states = encoder_hidden_states.repeat(B, 1, 1) # (B, D2, L2=1)

        # 1. Input
        hidden_states = self.content_with_pos_emb(indices) # (B, K, L) -> (B, K, L, dhead)
        hidden_states = hidden_states.reshape(B, L, -1) # (B, K, L, dhead) -> (B, L, K*dhead)

        encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1) # (B, D2, L2) -> (B, L2, D2)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            ) # (B, L, K*dhead) -> (B, L, K*dhead)

        # 3. Output
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states.reshape(B, L, K, -1).permute(0,2,1,3) # (B, L, K*dhead) -> (B, K, L, dhead)
        # logits = self.proj_out(hidden_states) # (B, K, L, dhead) -> (B, K, L, C), logp(x_0)
        # logits = logits.permute(0, 3, 1, 2) # (B, K, L, C) -> (B, C, K, L)

        logits = torch.einsum('bkld,kdc->bklc', hidden_states, self.W_proj_out) # (B, K, L, dhead) -> (B, K, L, C)
        logits = logits + self.b_proj_out.unsqueeze(0).unsqueeze(-2)  # (K, C) -> (1, K, 1, C)
        logits = logits.permute(0, 3, 1, 2)  # (B, K, L, C) -> (B, C, K, L)
        return logits

    def forward_with_cond_scale(self, *args, cond_scale = 1., **kwargs):
        pred_cond = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return pred_cond

        pred_uncon = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return pred_uncon + (pred_cond - pred_uncon) * cond_scale

class TransformerCondLatent(nn.Module):
    """
    A Transformer model for RVQ data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
    """

    def __init__(
        self,
        input_dim: int = 72,
        num_codebooks: int = 9,
        attention_head_dim: int = 128,
        max_frame_len: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.0,
        cond_drop_prob: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        num_embeds_ada_norm: Optional[int] = None, # for timestep emb
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        norm_type: str = "layer_norm",
        attention_type: str = "default",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_codebooks = num_codebooks
        self.max_frame_len = max_frame_len
        self.cross_attention_dim = cross_attention_dim
        self.attention_head_dim = attention_head_dim
        num_attention_heads = num_codebooks
        inner_dim = num_attention_heads * attention_head_dim

        self.cond_drop_prob = cond_drop_prob

        # 1. Define input layers
        self.codebook_dim = self.input_dim // self.num_codebooks
        assert self.input_dim % self.num_codebooks == 0, "input dim should be multiples of num_codebooks"

        # self.W_proj_in = nn.Parameter(
        #     torch.randn(num_codebooks, self.codebook_dim, attention_head_dim), requires_grad=True
        # )
        # self.b_proj_in = nn.Parameter(
        #     torch.randn(num_codebooks, attention_head_dim), requires_grad=True
        # )
        self.proj_in = nn.Linear(num_codebooks * self.codebook_dim, inner_dim)

        # 2. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    attention_type=attention_type,
                )
                for d in range(num_layers)
            ]
        )

        # 2.1. Define default conditions
        if cross_attention_dim is not None:
            self.default_encoder_hidden_state = nn.Parameter(torch.randn(cross_attention_dim, 1), requires_grad=True)

        self.norm_out = nn.LayerNorm(inner_dim)

        # self.W_proj_out = nn.Parameter(
        #     torch.randn(num_codebooks, attention_head_dim, self.codebook_dim), requires_grad=True
        # )
        # self.b_proj_out = nn.Parameter(
        #     torch.randn(num_codebooks, self.codebook_dim), requires_grad=True
        # )
        self.proj_out = nn.Linear(num_codebooks*attention_head_dim, num_codebooks*self.codebook_dim)

    def multi_head_linear(self, x, W, b):
        """
            x in shape [B, K, d_in, L]
            W in shape [K, d_in, d_out]
            b in shape [K, d_out]
            out in shape [B, K, d_out, L]
        """
        x = torch.einsum('bkdl,kdc->bkcl', x, W) # (B, K, d_in, L) -> (B, K, d_out, L)
        x = x + b.unsqueeze(0).unsqueeze(-1)  # (K, d_out) -> (1, K, d_out, 1)
        return x

    def forward(
        self,
        latent: torch.FloatTensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        cond_drop_prob: Optional[float] = None,
    ):
        """
        Args:
            latent: torch.FloatTensor, in shape (B, D_in, L) where L is frame length
            timestep: torch.LongTensor in shape (B, )
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            encoder_hidden_states: torch.FloatTensor of shape (B, D2, L2) for cross attention, for text prompt
            class_labels: torch.LongTensor in shape (B, )
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.

        Returns:
            latent_0, in shape (B, D_in, L)
        """
        B, D_in, L = latent.shape
        # 0. CFG condition drop
        if cond_drop_prob is None:
            cond_drop_prob = self.cond_drop_prob

        if self.cross_attention_dim is not None:
            if encoder_hidden_states is not None:
                if len(encoder_hidden_states.shape) == 2:
                    encoder_hidden_states = encoder_hidden_states.unsqueeze(-1)
                if cond_drop_prob > 0:
                    keep_mask = prob_mask_like((B,), 1 - self.cond_drop_prob, device=encoder_hidden_states.device)  # (B,)
                    keep_mask = keep_mask.unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)
                    default_encoder_hidden_states = encoder_hidden_states*0 + self.default_encoder_hidden_state.unsqueeze(0)
                    encoder_hidden_states = torch.where(
                        keep_mask, encoder_hidden_states, default_encoder_hidden_states
                    )
            else:
                encoder_hidden_states = self.default_encoder_hidden_state.unsqueeze(0) # (1, D2, L2=1)
                encoder_hidden_states = encoder_hidden_states.repeat(B, 1, 1) # (B, D2, L2=1)
        else:
            encoder_hidden_states = None

        # 1. Input
        # latent = latent.reshape(B, self.num_codebooks, self.codebook_dim, L) # (B, K, d, L)
        # hidden_states = self.multi_head_linear(latent, self.W_proj_in, self.b_proj_in) # (B, K, d, L) -> (B, K, dhead, L)
        # hidden_states = hidden_states.reshape(B, L, -1) # (B, K*dhead, L) -> (B, L, K*dhead)

        hidden_states = latent.reshape(B, L, -1) # (B, K*d, L) -> (B, L, K*d)
        hidden_states = self.proj_in(hidden_states) # (B, K*d, L) -> (B, L, K*dhead)

        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1)  # (B, D2, L2) -> (B, L2, D2)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            ) # (B, L, K*dhead) -> (B, L, K*dhead)

        # 3. Output
        hidden_states = self.norm_out(hidden_states)
        # hidden_states = hidden_states.reshape(B, L, self.num_codebooks, -1).permute(0,2,3,1) # (B, L, K*dhead) -> (B, K, dhead, L)
        # hidden_states = self.multi_head_linear(hidden_states, self.W_proj_out, self.b_proj_out) # (B, K, dhead, L) -> (B, K, d, L)
        # out_latent = hidden_states.reshape(B, D_in, L) # (B, K, d, L) -> (B, D_in=K*d, L)

        hidden_states = self.proj_out(hidden_states) # (B, L, K*d)
        out_latent = hidden_states.reshape(B, D_in, L)
        return out_latent

    def forward_with_cond_scale(self, *args, cond_scale = 1., **kwargs):
        pred_cond = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return pred_cond

        pred_uncon = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return pred_uncon + (pred_cond - pred_uncon) * cond_scale