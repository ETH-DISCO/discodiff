# adapted from https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/models/unet_2d.py
# changed conv to 1d for RVQ

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from .embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import UNetMidBlock1D, get_down_block, get_up_block

class UNet1DModel(nn.Module):
    """Multi-layer perceptron model.

    It uses leaky ReLU as activation function and the last layer is a
    linear layer.

    Parameters
    ----------
    input_dim : int
        Dimension of the input.
    hidden_dim : int
        Dimension of the hidden layers.
    output_dim : int
        Dimension of the output.
    n_layers : int
        Number of hidden layers.
    use_softmax : bool
        Whether to use softmax as the activation function of the last layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_codebooks: int = 8,
        
        down_block_types: Tuple[str] = ("DownBlock1D", "AttnDownBlock1D", "AttnDownBlock1D", "AttnDownBlock1D"),
        up_block_types: Tuple[str] = ("AttnUpBlock1D", "AttnUpBlock1D", "AttnUpBlock1D", "UpBlock1D"),
        block_out_channels: Tuple[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        time_embed_dim = block_out_channels[0] * 4
        self.num_codebooks = num_codebooks
        self.codebook_size = in_channels // num_codebooks - 1
        assert in_channels % num_codebooks == 0

        # input
        # self.conv_in = nn.Conv1d(in_channels, block_out_channels[0], kernel_size=3, padding=1) # in: (B, C*H, L)
        self.conv_in = nn.Linear(in_channels, block_out_channels[0])  # in: (B, C*H, L)

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos = True, downscale_freq_shift = 0)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # TODO: class embedding
        self.class_embedding = None

        # init unet blocks
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            assert output_channel % self.num_codebooks == 0
            attention_head_dim = output_channel // self.num_codebooks

            is_final_block = i == len(block_out_channels) - 1

            # print(is_final_block) # debug

            down_block = get_down_block(
                down_block_type,
                num_layers = layers_per_block,
                in_channels = input_channel,
                out_channels = output_channel,
                temb_channels = time_embed_dim,
                attention_head_dim = attention_head_dim, # should be divisible to out_channels
                add_downsample = not is_final_block,
                resnet_eps = norm_eps,
                resnet_groups = norm_num_groups,
                resnet_time_scale_shift = "default",
            )
            self.down_blocks.append(down_block)

        # mid, one layer
        assert block_out_channels[-1] % self.num_codebooks == 0
        attention_head_dim = block_out_channels[-1] // self.num_codebooks
        self.mid_block = UNetMidBlock1D(
            in_channels = block_out_channels[-1],
            temb_channels = time_embed_dim,
            attention_head_dim = attention_head_dim,
            resnet_eps = norm_eps,
            resnet_groups = norm_num_groups,
            resnet_time_scale_shift = "default",
            output_scale_factor=1.0,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            assert output_channel % self.num_codebooks == 0
            attention_head_dim = output_channel // self.num_codebooks

            is_final_block = i == len(block_out_channels) - 1

            # print(is_final_block) # debug

            up_block = get_up_block(
                up_block_type,
                num_layers = layers_per_block + 1,
                in_channels = input_channel,
                prev_output_channel = prev_output_channel,
                out_channels = output_channel,
                temb_channels = time_embed_dim,
                attention_head_dim = attention_head_dim, # should be divisible to out_channels
                add_upsample = not is_final_block,
                resnet_eps = norm_eps,
                resnet_groups = norm_num_groups,
                resnet_time_scale_shift = "default",
            )
            self.up_blocks.append(up_block)

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        # self.conv_out = nn.Conv1d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Linear(block_out_channels[0], out_channels)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
    ) -> Tuple:
        r"""
        The [`UNet1DModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.FloatTensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.

        Returns:
            the sample tensor.
        """
        # print("debug: sample shape", sample.shape)

        # 0. reshape
        if len(sample.shape) == 4:
            B, codebooks_size, codebook_num, frame_len = sample.shape
            sample = sample.reshape(B, codebooks_size*codebook_num, frame_len)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.float()
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).float()
            emb = emb + class_emb

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample.transpose(-2, -1)).transpose(-2, -1)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

            # print("down:", sample.shape) # debug

        # 4. mid
        sample = self.mid_block(sample, emb)
        # print("mid:", sample.shape) # debug

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples, emb)

            # print("up:", sample.shape) # debug

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample.transpose(-2, -1)).transpose(-2, -1)

        if skip_sample is not None:
            sample += skip_sample

        # print("convout:", sample.shape) # debug
        sample = sample.reshape(B, codebooks_size - 1, codebook_num, frame_len)
        # print("reshaped:", sample.shape) # debug
        return sample
