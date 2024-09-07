from typing import Any, Dict, Optional

import torch
from torch import nn

from .attention import BasicTransformerBlock
from .embeddings import RVQEmbeddingsWithPosition


class TransformerRVQ(nn.Module):
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
        cross_attention_dim: Optional[int] = None,
        num_embeds_ada_norm: Optional[int] = None, # for timestep emb
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        sample_size: Optional[int] = None,
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

        # 1. Define input layers
        self.content_with_pos_emb = RVQEmbeddingsWithPosition(
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

        self.norm_out = nn.LayerNorm(inner_dim)
        # TODO: define num_codebooks different proj_out layers, for different heads
        self.proj_out = nn.Linear(attention_head_dim, codebook_size)

    def forward(
        self,
        indices: torch.LongTensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
    ):
        """
        Args:
            indices: torch.LongTensor, in shape (B, K, L) where K is num of codebooks, and L is frame length
            encoder_hidden_states: torch.FloatTensor of shape (B, L2, D2) for cross attention, not used currently
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
        # 1. Input
        hidden_states = self.content_with_pos_emb(indices) # (B, K, L) -> (B, K, L, dhead)
        hidden_states = hidden_states.reshape(B, L, -1) # (B, K, L, dhead) -> (B, L, K*dhead)

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
        logits = self.proj_out(hidden_states) # (B, K, L, dhead) -> (B, K, L, C), logp(x_0)
        logits = logits.permute(0, 3, 1, 2) # (B, K, L, C) -> (B, C, K, L)

        return logits