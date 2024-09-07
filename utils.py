from typing import Type
import logging
from pathlib import Path
import pytorch_lightning as pl

import importlib
import math
import numpy as np
import torch
import torchaudio
import librosa 

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image

KEY_LABELS = ['A major', 'Bb major', 'B major', 'C major', 'Db major',
              'D major', 'Eb major', 'E major', 'F major', 'F# major',
              'G major', 'Ab major', 'A minor', 'Bb minor', 'B minor',
              'C minor', 'C# minor', 'D minor', 'D# minor', 'E minor',
              'F minor', 'F# minor', 'G minor', 'G# minor']

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def load_model(type: Type[pl.LightningModule], path: Path, **kwargs):
    """Load the autoencoder model from the save directory.

    Parameters
    ----------
    type : Type[pl.LightningModule]
        The type of the model to load.
    path : Path
        The checkpoint path.
    **kwargs : dict
        The keyword arguments to pass to the load_from_checkpoint method.
        Instead one can add `self.save_hyperparameters()` to the init method
        of the model.

    Returns
    -------
    model : pl.LightningModule
        The trained model.

    """
    if not path.exists():
        logging.info("Model not found. Returning None.")
        return None
    model = type.load_from_checkpoint(path, **kwargs)
    return model

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def extract(a, t, x_shape):
# get a[t] and put into x_shape num dims, for torch tensors
# e.g., a = [0,1,2,3], t = 2, x_shape = (2,3); then return [[2]]
# e.g., a = [0,1,2,3], t = 2, x_shape = (2,3,4); then return [[[2]]]
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# utils of https://github.com/CompVis/stable-diffusion/blob/main/ldm/util.py

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


'''
    encodec decoding
    rvq_tokens: in shape [B, K, L]
    encodec: the encodec encoder-decoder
'''
def encodec_decoding_given_rvq_tensor(rvq_tokens, encodec_model):
    encoded_frames =  [(rvq_tokens, None)]
    wav_recon = encodec_model.decode(encoded_frames)
    return wav_recon

'''
    discript audio codec (dac) decoding
    rvq_tokens: in shape [B, K, L]
    dac_model: the dac encoder-decoder
'''
def dac_decoding_given_rvq_tensor(rvq_tokens, dac_model):
    z = dac_model.quantizer.from_codes(rvq_tokens.to(dac_model.device))[0]
    wav_recon = dac_model.decode(z) # (1, 1, t*fs)
    return wav_recon.squeeze(0)

def dac_decoding_given_rvq_tensor_no_device_convert(rvq_tokens, dac_model):
    z = dac_model.quantizer.from_codes(rvq_tokens)[0]
    wav_recon = dac_model.decode(z) # (1, 1, t*fs)
    return wav_recon.squeeze(0)

'''
    discript audio codec (dac) decoding
    latents: in shape [B, num_codebook x d_codebook, L]
    dac_model: the dac encoder-decoder
'''
def dac_decoding_given_latents(latents, dac_model):
    z = dac_model.quantizer.from_latents(latents.to(dac_model.device))[0]
    wav_recon = dac_model.decode(z) # (1, 1, t*fs)
    return wav_recon.squeeze(0)

def dac_get_latents_from_rvq(rvq_tokens, dac_model):
    rvq_tokens = rvq_tokens.to(dac_model.device)
    z_e = []
    n_codebooks = rvq_tokens.shape[1]
    for i in range(n_codebooks):
        z_e.append(
            dac_model.quantizer.quantizers[i].decode_code(rvq_tokens[:, i, :])
        )
    return torch.cat(z_e, dim = 1)

"""
    cf. https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html
"""
def spectrogram_image(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):

    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axs = fig.add_subplot()
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba)

def audio_spectrogram_image(waveform, power=2.0, sample_rate=48000):
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 80

    mel_spectrogram_op = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, 
        hop_length=hop_length, center=True, pad_mode="reflect", power=power, 
        norm='slaney', onesided=True, n_mels=n_mels, mel_scale="htk")

    melspec = mel_spectrogram_op(waveform.float().cpu())
    melspec = melspec[0] # TODO: only left channel for now
    return spectrogram_image(melspec, title="MelSpectrogram", ylabel='mel bins (log freq)')

@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)

def load_state_dict_partial(
    target_state_dict, ckpt_state_dict,
    must_contain = None, dont_contain = None,
    verbose=False
):
    for name, param in ckpt_state_dict.items():
        if name not in target_state_dict:
            if verbose: print(f"{name} not in the model state dict")
            continue

        if isinstance(param, torch.nn.Parameter):
            param = param.data
        elif torch.is_tensor(param):
            pass
        else:
            if verbose: print(f"{name} has unrecognized type {str(type(param))}")
            continue

        if must_contain is not None:
            if must_contain not in name:
                print(must_contain, "should be contained. Skipped", name)
                continue

        if dont_contain is not None:
            if dont_contain in name:
                print(dont_contain, "should not be contained. Skipped", name)
                continue

        try:
            if target_state_dict[name].shape == param.shape:
                target_state_dict[name].copy_(param)
                if verbose: print(f"{name} loaded into model state dict")
            else:
                shape_self = target_state_dict[name].shape
                shape_to_load = param.shape
                if len(shape_self) != len(shape_to_load):
                    raise ValueError(
                        f'Shape {shape_to_load} of loaded param {name} is different from {shape_self}.'
                    )
                elif shape_self < shape_to_load:
                    raise ValueError(
                        f'Shape {shape_to_load} of loaded param {name} is larger than {shape_self}.'
                    )

                if len(shape_self) == 1:
                    target_state_dict[name][:shape_to_load[0]].copy_(param)
                elif len(shape_self) == 2:
                    target_state_dict[name][:shape_to_load[0], :shape_to_load[1]].copy_(param)
                elif len(shape_self) == 3:
                    target_state_dict[name][:shape_to_load[0], :shape_to_load[1], :shape_to_load[2]].copy_(param)
                elif len(shape_self) == 4:
                    target_state_dict[name][:shape_to_load[0], :shape_to_load[1], :shape_to_load[2],
                    :shape_to_load[3]].copy_(param)
                else:
                    raise ValueError(
                        f'Shape {shape_to_load} of loaded param {name} is different from {shape_self}.'
                    )

                if verbose:
                    print(f"{name} with shape {shape_to_load} partially loaded into model, with shape {shape_self}")

        except Exception as e:
            print(f"error encountered in loading param {name}")
            print(e)

def load_state_dict_partial_primary_secondary(
        target_state_dict, ckpt_state_dict_primary, ckpt_state_dict_secondary, verbose=False
):
    load_state_dict_partial(
        target_state_dict, ckpt_state_dict_primary, must_contain = None, verbose=verbose
    )
    load_state_dict_partial(
        target_state_dict, ckpt_state_dict_secondary, must_contain = "secondary", verbose=verbose
    )

def load_state_dict_partial_chroma(
        target_state_dict,
        ckpt_state_dict_chroma,
        verbose=False
):
    load_state_dict_partial(
        target_state_dict, ckpt_state_dict_chroma, must_contain = "chroma", verbose=verbose
    )