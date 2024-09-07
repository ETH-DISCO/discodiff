import sys
sys.path.append("..")
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce

from tqdm.auto import tqdm

from utils import exists, default, identity, extract
from diffusers import DDPMScheduler

CODEBOOK_MEANS = [0.58, -0.11, -0.18, -0.04, -0.23, -0.06, 0.08, 0, 0.03]
CODEBOOK_STDS = [4.95, 4.00, 3.61, 3.41, 3.24, 3.16, 3.06, 2.93, 2.79]
DIM_SINGLE = 8
def dac_latent_normalize_heterogeneous(latents):
    if len(latents.shape) == 2:
        for i_codebook in range(len(CODEBOOK_MEANS)):
            latents[i_codebook*DIM_SINGLE:(i_codebook+1)*DIM_SINGLE] = (
                latents[i_codebook*DIM_SINGLE:(i_codebook+1)*DIM_SINGLE] - CODEBOOK_MEANS[i_codebook]
            ) / CODEBOOK_STDS[i_codebook]
    else:
        for i_codebook in range(len(CODEBOOK_MEANS)):
            latents[:, i_codebook*DIM_SINGLE:(i_codebook+1)*DIM_SINGLE] = (
                latents[:, i_codebook*DIM_SINGLE:(i_codebook+1)*DIM_SINGLE] - CODEBOOK_MEANS[i_codebook]
            ) / CODEBOOK_STDS[i_codebook]
    return latents

def dac_latent_denormalize_heterogeneous(latents):
    if len(latents.shape) == 2:
        for i_codebook in range(len(CODEBOOK_MEANS)):
            latents[i_codebook*DIM_SINGLE:(i_codebook+1)*DIM_SINGLE] = (
                latents[i_codebook*DIM_SINGLE:(i_codebook+1)*DIM_SINGLE] * CODEBOOK_STDS[i_codebook]
            ) + CODEBOOK_MEANS[i_codebook]
    else:
        for i_codebook in range(len(CODEBOOK_MEANS)):
            latents[:, i_codebook*DIM_SINGLE:(i_codebook+1)*DIM_SINGLE] = (
                latents[:, i_codebook*DIM_SINGLE:(i_codebook+1)*DIM_SINGLE] * CODEBOOK_STDS[i_codebook]
            ) + CODEBOOK_MEANS[i_codebook]
    return latents

class DACLatentDDPM(nn.Module):
    def __init__(
        self, 
        denoise_model,
        signal_len,
        latent_dim = 72,
        timesteps = 1000, 
        loss_type = 'l1', 
        prediction_type = 'v_prediction', # choose from ["epsilon", "sample", "v_prediction"]
    ):
        super().__init__()
        
        self.denoise_model = denoise_model
        self.signal_len = signal_len
        self.latent_dim = latent_dim
        self.obj_shape = (latent_dim, signal_len)

        self.objective = prediction_type
        self.num_timesteps = int(timesteps)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_timesteps,
            prediction_type=prediction_type
        )

        self.loss_type = loss_type

    def q_sample(self, x_start, t, noise=None): # q(x_t|x_0)
        noise = default(noise, lambda: torch.randn_like(x_start))
        return self.noise_scheduler.add_noise(x_start, noise, t)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def model_predictions(self, x, t, cond_emb = None, cond_scale = 3.):
        b, device = x.shape[0], x.device
        batched_times = torch.full((b,), t, device=device)
        return self.denoise_model.forward_with_cond_scale(
            latent = x,
            timestep = batched_times,
            cond_scale = cond_scale,
            encoder_hidden_states = cond_emb,
        )

    @torch.no_grad()
    def p_sample(self, x, t: int, cond_emb = None, cond_scale = 3.):
        model_output = self.model_predictions(x, t, cond_emb=cond_emb, cond_scale=cond_scale)
        preds = self.noise_scheduler.step(
            model_output = model_output,
            timestep = t,
            sample = x,
        )
        return preds.prev_sample, preds.pred_original_sample

    @torch.no_grad()
    def p_sample_loop(self, shape, cond_emb = None, cond_scale = 3.):
        device = next(self.parameters()).device

        pred = torch.randn(shape, device = device)
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            pred, x_start = self.p_sample(pred, t, cond_emb, cond_scale)

        ret = dac_latent_denormalize_heterogeneous(pred)
        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, cond_emb = None, cond_scale = 3., return_all_timesteps = False):
        if exists(cond_emb):
            text_batch_size = cond_emb.shape[0]
            if text_batch_size != batch_size:
                batch_size = text_batch_size
                print("Warning: the desired batch size is different from the input text batch size, using text batch size instead")
        signal_len = self.signal_len
        latent_dim = self.latent_dim
        sample_fn = self.p_sample_loop
        return sample_fn(
            (batch_size, latent_dim, signal_len), 
            cond_emb = cond_emb,
            cond_scale = cond_scale,
            return_all_timesteps = return_all_timesteps
        )

    def get_training_target(self, x_start, t, noise):
        if self.objective == 'epsilon':
            target = noise
        elif self.objective == 'sample':
            target = x_start
        elif self.objective == 'v_prediction':
            if type(t) == int:
                b, *_, device = *x_start.shape, x_start.device
                batched_times = torch.full((b,), t, device=device).to(int)
            else:
                batched_times = t.to(int)
            target = self.noise_scheduler.get_velocity(
                sample = x_start,
                noise = noise,
                timesteps = batched_times,
            )
        else:
            raise ValueError(f'unknown objective {self.objective}')
        return target

    def p_losses(self, x_start, t, cond_emb = None, noise = None):
    # given x0, calculate ||x0 - f_\theta(a' x0 + b' eta)|| (for x0 predictive net)
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
        model_out = self.denoise_model(x, t, encoder_hidden_states = cond_emb)
        target = self.get_training_target(x_start, t, noise)

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, latent, *args, **kwargs):
        b, latent_dim, l, device = *latent.shape, latent.device, 
        assert l == self.signal_len, f'length of signal must be {self.signal_len}'
        assert latent_dim == self.latent_dim, f'dim of signal must be {self.latent_dim}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        latent = dac_latent_normalize_heterogeneous(latent)
        return self.p_losses(latent, t, *args, **kwargs)


class DACLatentDDPMVALLE(DACLatentDDPM):
    def __init__(
        self,
        denoise_model_primary,
        denoise_model_secondary,
        signal_len,
        latent_dim = 72,
        timesteps = 1000,
        loss_type = 'l1',
        prediction_type = 'v_prediction',
    ):
        super().__init__(
            denoise_model = denoise_model_primary,
            signal_len = signal_len,
            latent_dim = latent_dim,
            timesteps = timesteps,
            loss_type = loss_type,
            prediction_type = prediction_type,
        )
        self.denoise_model_secondary = denoise_model_secondary
        assert latent_dim % denoise_model_primary.num_codebooks == 0
        self.latent_dim_primary = latent_dim // denoise_model_primary.num_codebooks
        self.obj_shape_primiary = (self.latent_dim_primary, signal_len)
        self.obj_shape_secondary = (latent_dim - self.latent_dim_primary, signal_len)

    def model_predictions_primary(self, x, t, cond_emb = None, cond_scale = 3.):
        b, device = x.shape[0], x.device
        batched_times = torch.full((b,), t, device=device)
        return self.denoise_model.forward_with_cond_scale(
            latent = x[:, :self.latent_dim_primary],
            timestep = batched_times,
            cond_scale=cond_scale,
            encoder_hidden_states = cond_emb,
        )

    def model_predictions_secondary(self, x, t, cond_emb = None, cond_scale = 3.):
        b, device = x.shape[0], x.device
        batched_times = torch.full((b,), t, device=device)
        return self.denoise_model_secondary.forward_with_cond_scale(
            latent = x,
            timestep = batched_times,
            cond_scale=cond_scale,
            encoder_hidden_states = cond_emb,
        )
    @torch.no_grad()
    def p_sample_primary(self, x, t: int, cond_emb = None, cond_scale = 3.):
        model_output_primary = self.model_predictions_primary(
            x, t, cond_emb=cond_emb, cond_scale=cond_scale
        )
        preds = self.noise_scheduler.step(
            model_output = model_output_primary,
            timestep = t,
            sample = x[:, :self.latent_dim_primary],
        )
        return preds.prev_sample, preds.pred_original_sample

    @torch.no_grad()
    def p_sample_secondary(self, x, t: int, cond_emb = None, cond_scale = 3.):
        model_output_secondary = self.model_predictions_secondary(
            x, t, cond_emb=cond_emb, cond_scale=cond_scale
        )
        preds = self.noise_scheduler.step(
            model_output = model_output_secondary,
            timestep = t,
            sample = x[:, self.latent_dim_primary:],
        )
        return preds.prev_sample, preds.pred_original_sample

    @torch.no_grad()
    def p_sample_loop_primary(self, shape, cond_emb=None, cond_scale=3.):
        device = next(self.parameters()).device
        pred = torch.randn(shape, device=device)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            pred_primary, x_start_primary = self.p_sample_primary(pred, t, cond_emb, cond_scale)
            pred[:, :self.latent_dim_primary] = pred_primary

        return pred

    @torch.no_grad()
    def p_sample_loop_secondary(self, noise_given_primary, cond_emb=None, cond_scale=3.):
        pred = noise_given_primary
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            pred_secondary, x_start_secondary = self.p_sample_secondary(noise_given_primary, t, cond_emb, cond_scale)
            pred[:, self.latent_dim_primary:] = pred_secondary

        pred = dac_latent_denormalize_heterogeneous(pred)
        return pred

    @torch.no_grad()
    def sample(self, batch_size = 16, cond_emb = None, cond_scale = 3.):
        if exists(cond_emb):
            text_batch_size = cond_emb.shape[0]
            if text_batch_size != batch_size:
                batch_size = text_batch_size
                print("Warning: the desired batch size is different from the input text batch size, using text batch size instead")
        signal_len = self.signal_len
        latent_dim = self.latent_dim
        pred = self.p_sample_loop_primary(
            (batch_size, latent_dim, signal_len),
            cond_emb = cond_emb,
            cond_scale = cond_scale,
        )
        pred = self.p_sample_loop_secondary(
            pred,
            cond_emb = cond_emb,
            cond_scale = cond_scale,
        )
        return pred

    @torch.no_grad()
    def sample_given_primary(self, noise_given_primary, cond_emb = None, cond_scale = 3.):
        if exists(cond_emb):
            assert noise_given_primary.shape[0] == cond_emb.shape[0]
        pred = self.p_sample_loop_secondary(
            noise_given_primary,
            cond_emb = cond_emb,
            cond_scale = cond_scale,
        )
        return pred

    def p_losses(self, x_start, t, cond_emb = None, noise = None, only_secondary = False, only_primary = False):
    # given x0, calculate ||x0 - f_\theta(a' x0 + b' eta)|| (for x0 predictive net)
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
        if not only_secondary:
            model_out_primary = self.denoise_model(
                x[:, :self.latent_dim_primary], t, encoder_hidden_states = cond_emb
            )
            target_primary = self.get_training_target(
                x_start[:, :self.latent_dim_primary], t, noise[:, :self.latent_dim_primary]
            )

            loss_primary = self.loss_fn(model_out_primary, target_primary, reduction='none')
            loss_primary = reduce(loss_primary, 'b ... -> b (...)', 'mean').mean()
        else:
            # print("only secondary")
            loss_primary = 0

        if not only_primary:
            x_secondary = x[:]
            x_secondary[:, :self.latent_dim_primary] = x_start[:, :self.latent_dim_primary]

            model_out_secondary = self.denoise_model_secondary(
                x_secondary, t, encoder_hidden_states=cond_emb
            )
            target_secondary = self.get_training_target(
                x_start[:, self.latent_dim_primary:], t, noise[:, self.latent_dim_primary:]
            )

            loss_secondary = self.loss_fn(model_out_secondary, target_secondary, reduction='none')
            loss_secondary = reduce(loss_secondary, 'b ... -> b (...)', 'mean').mean()
        else:
            # print("only primary")
            loss_secondary = 0

        return loss_primary, loss_secondary

    def forward(self, latent, only_secondary = False, only_primary = False, *args, **kwargs):
        b, latent_dim, l, device = *latent.shape, latent.device,
        assert l == self.signal_len, f'length of signal must be {self.signal_len}'
        assert latent_dim == self.latent_dim, f'dim of signal must be {self.latent_dim}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        latent = dac_latent_normalize_heterogeneous(latent)
        return self.p_losses(
            latent, t,
            only_secondary = only_secondary,
            only_primary = only_primary,
            *args, **kwargs
        )