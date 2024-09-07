import copy
from .config_base import AttrDict, config_base

config_denoise_nn_chroma = {
    "chroma_dim": 12,
    "chroma_num_heads": 4,
    "chroma_inner_dim": 128,
    "chroma_head_dim": 32,
    "chroma_dim_mults": (1, 2, 4, 8),
    "chroma_frame_len": 240,
    # when DAC frame len is 896, chroma frame len is 240
    # when DAC frame len is 2320, chroma frame len is 624
    # factor is 3.687
}
config_base.update(config_denoise_nn_chroma)

config_dac_large = copy.deepcopy(config_base)
config_dac_small = copy.deepcopy(config_base)
config_denoise_nn_large_dac = {
    "model_dim_in": 72,
    "model_dim_out": 72,
    "inner_dim": 576,  # 128, # DEBUG
    "head_dim": 128,
    "dim_mults": (1, 1, 2, 2, 4),
    "frame_len_dac": 896, # 2500
    "frame_len_encodec": 650, # 1440
}
config_denoise_nn_small_dac = {
    "model_dim_in": 72,
    "chroma_dim": 12,
    "model_dim_out": 72,
    "inner_dim": 144,
    "head_dim": 128,
    "dim_mults": (1, 2, 4, 8),
    "frame_len_dac": 896,
    "frame_len_encodec": 650,
}
# config_denoise_nn_small_dac = {
#     "model_dim_in": 72,
#     "model_dim_out": 72,
#     "inner_dim": 18,
#     "head_dim": 18,
#     "dim_mults": (1, 2), # (1, 2, 4, 8),
#     "frame_len_dac": 896,
#     "frame_len_encodec": 93,
# }
config_dac_large.update(config_denoise_nn_large_dac)
config_dac_small.update(config_denoise_nn_small_dac)

config_dac_base = {
    "sample_rate": 44100,
    "num_codebooks": 9,
    "codebook_size": 1024,
    "codebook_dim": 8,
    "clap_dim": 512,
    "meta_cond_dim": 512,
}
config_dac_large.update(config_dac_base)
config_dac_small.update(config_dac_base)

config_dac_parallel_large = copy.deepcopy(config_dac_large)
config_dac_parallel_small = copy.deepcopy(config_dac_small)
config_dac_parallel_bonus = {
    "rvq_pattern": "parallel",
    "batch_size": 32,
}
config_dac_parallel_large.update(config_dac_parallel_bonus)
config_dac_parallel_small.update(config_dac_parallel_bonus)


config_dac_flattened_large = copy.deepcopy(config_dac_large)
config_dac_flattened_small = copy.deepcopy(config_dac_small)
config_dac_flattened_bonus = {
    "rvq_pattern": "flattened",
    "batch_size": 16,
    "frame_len": 672
}
config_dac_flattened_large.update(config_dac_flattened_bonus)
config_dac_flattened_small.update(config_dac_flattened_bonus)

config_dac_valle_large = copy.deepcopy(config_dac_large)
config_dac_valle_small = copy.deepcopy(config_dac_small)
config_dac_valle_bonus_large = {
    "rvq_pattern": "VALL-E",
    "batch_size": 16,
    "inner_dim_primary": 576,  # now we change into the same size as the secondary model
    "dim_mults_primary": (1, 1, 2, 2, 4),
    "head_dim_primary": 128,
}
config_dac_valle_bonus_small = {
    "rvq_pattern": "VALL-E",
    "batch_size": 64,
    "inner_dim_primary": 144,
    "dim_mults_primary": (1, 2, 4, 8),
    "head_dim_primary": 128,
}
# config_dac_valle_bonus_small = {
#     "rvq_pattern": "VALL-E",
#     "batch_size": 1,
#     "inner_dim_primary": 18,
#     "dim_mults_primary": (1, 2), # (1, 2, 4, 8),
#     "head_dim_primary": 18,
# }
config_dac_valle_large.update(config_dac_valle_bonus_large)
config_dac_valle_small.update(config_dac_valle_bonus_small)

# configs to export

config_dac_parallel_large = AttrDict(config_dac_parallel_large)
config_dac_parallel_small = AttrDict(config_dac_parallel_small)

config_dac_flattened_large = AttrDict(config_dac_flattened_large)
config_dac_flattened_small = AttrDict(config_dac_flattened_small)

config_dac_valle_large = AttrDict(config_dac_valle_large)
config_dac_valle_small = AttrDict(config_dac_valle_small)