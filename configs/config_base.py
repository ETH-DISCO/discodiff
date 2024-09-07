import copy

# configs
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

config_base = {}

config_spectrogram = {
    "n_mels": 80,
    "n_fft": 1024,
    "hop_samples": 256,
    "crop_mel_frames": 62,
}
config_base.update(config_spectrogram)

config_data = {
    "in_emb_dims_fma": {
        'madmom_key': 27, 'madmom_tempo': 27
    },
    "text_clap_load_prob": 0.00,
    "dataset_size": 51200,
}
config_base.update(config_data)

config_diffusion = {
    "diffusion_steps": 100, # 100, # DEBUG
    "cond_scale": 3.0,
    "cond_drop_prob": 0.2,
    "prediction_type": "sample", # choose from ["epsilon", "sample", "v_prediction"]
    "scheduler": "diffusers", # choose from ["handcrafted", "diffusers"]
}
config_base.update(config_diffusion)

config_learning_dac = {
    "ema_decay": 0.995,
    "lr": 4e-5,
    "checkpoint_every": 2000,  # DEBUG
    "max_epochs": 500,
    "batch_size": 64,

    "demo_every": 4000,  # DEBUG
    "num_demos": 4, # 1
    "sample_skip_step": 1,
}
config_base.update(config_learning_dac)

