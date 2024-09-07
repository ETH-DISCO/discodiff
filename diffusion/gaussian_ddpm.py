# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
import sys
sys.path.append("..")

from collections import namedtuple

import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce

from tqdm.auto import tqdm

from utils import exists, default, identity, extract
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# dac latent statistics: mean, std, min, max =(apprx)= -0.08513347, 3.1341403, -18.098831, 15.562451
def dac_latent_normalize(latents, std = 6):
    return latents / std

def dac_latent_denormalize(latents, std = 6):
    return latents * std

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
        prediction_type = 'sample', # choose from ["epsilon", "sample", "v_prediction"]
        schedule_fn_kwargs = dict(),
        normalize = True
    ):
        super().__init__()
        
        self.denoise_model = denoise_model
        self.signal_len = signal_len
        self.latent_dim = latent_dim
        self.obj_shape = (latent_dim, signal_len)

        self.prediction_type = prediction_type

        beta_schedule_fn = cosine_beta_schedule
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        timesteps, = betas.shape

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.sampling_timesteps = int(timesteps) - 1
        p2_loss_weight_gamma = 0. # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1
        # self.is_ddim_sampling = True
        self.is_ddim_sampling = False
        self.ddim_sampling_eta = 0.

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32)) # to register param for the model, this is the method of parent nn.Module

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        self.normalize = normalize

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_output_to_prediction(self, model_output, x, t):
        if self.prediction_type == 'epsilon':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.prediction_type == 'sample':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.prediction_type == 'v_prediction':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        else:
            ValueError(f'unknown objective {self.prediction_type}')
        
        return ModelPrediction(pred_noise, x_start)

    def q_posterior(self, x_start, x_t, t): #q(x_{t-1}|x_t,x_0)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None): # q(x_t|x_0)
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
    
    def get_training_target(self, x_start, t, noise):
        if self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'sample':
            target = x_start
        elif self.prediction_type == 'v_prediction':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.prediction_type}')
        return target

    def model_predictions(self, x, t, vec_cond = None, seq_conds = None, cond_scale = 3.):
        model_output = self.denoise_model.forward_with_cond_scale(
            latent = x,
            timestep = t,
            cond_scale=cond_scale,
            vec_cond = vec_cond,
            seq_conds = seq_conds
        )
        # returning a tuple (epsilon_0|t, x_0)
        return self.model_output_to_prediction(model_output, x, t)

    def p_mean_variance(self, x, t, vec_cond = None, seq_conds = None, cond_scale = 3.): # p_\theta(x_{t-1}|x_t)?
        preds = self.model_predictions(x, t, vec_cond = vec_cond, seq_conds = seq_conds, cond_scale = cond_scale) # epsilon_0|t, x_0
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start 

    @torch.no_grad()
    def p_sample(self, x, t: int, vec_cond = None, seq_conds = None, cond_scale = 3.):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x = x, 
            t = batched_times, 
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = cond_scale,
        )
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred = model_mean + (0.5 * model_log_variance).exp() * noise

        # print(f"at diffusion step {t}")
        # print("generated mean and var:", pred.mean().cpu().item(), pred.var().cpu().item())
        # print(" ")

        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, vec_cond = None, seq_conds = None, cond_scale = 3., return_all_timesteps = False):
        batch, device = shape[0], self.betas.device

        pred = torch.randn(shape, device = device)
        preds = [pred]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            pred, x_start = self.p_sample(pred, t, vec_cond, seq_conds, cond_scale)
            preds.append(pred)

        ret = pred if not return_all_timesteps else torch.stack(preds, dim = 1)

        if self.normalize:
            ret = dac_latent_denormalize_heterogeneous(ret)
        return ret

    @torch.no_grad()
    def p_sample_repaint(self, x, source, mask, t: int, last_resample: bool, vec_cond = None, seq_conds = None, cond_scale = 3.):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x = x,
            t = batched_times,
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = cond_scale,
        )
        if not last_resample or t == 0:
            source_noisy = self.q_sample(source, batched_times)
        else:
            batched_times_prev = torch.full((b,), t - 1, device = x.device, dtype = torch.long)
            source_noisy = self.q_sample(source, batched_times_prev)

        # print(f"at diffusion step {t}")
        # print("source mean and var:", source_noisy.mean().cpu().item(), source_noisy.var().cpu().item())

        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred = model_mean + (0.5 * model_log_variance).exp() * noise

        # print("generated mean and var:", pred.mean().cpu().item(), pred.var().cpu().item())
        # print(" ")

        pred = mask * source_noisy + (~mask) * pred
        x_start = mask * source + (~mask) * x_start

        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_repaint(self, source, mask, vec_cond = None, seq_conds = None, num_resamples = 20, cond_scale = 3.):
        shape = source.shape
        batch, device = shape[0], self.betas.device
        if self.normalize:
            source = dac_latent_normalize_heterogeneous(source)
        pred = torch.randn_like(source)
        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            for r in range(num_resamples):
                last_resample = False
                if r == num_resamples - 1:
                    last_resample = True
                pred, x_start = self.p_sample_repaint(pred, source, mask, t, last_resample, vec_cond, seq_conds, cond_scale)

        if self.normalize:
            return dac_latent_denormalize_heterogeneous(pred)
        else:
            return pred

    @torch.no_grad()
    def ddim_sample(self, shape, vec_cond = None, seq_conds = None, cond_scale = 3., return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.prediction_type

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        pred = torch.randn(shape, device = device)
        preds = [pred]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                x = pred, 
                t = time_cond,
                vec_cond=vec_cond,
                seq_conds=seq_conds,
                cond_scale = cond_scale
            )

            if time_next < 0:
                pred = x_start
                preds.append(pred)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(pred)

            pred = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            preds.append(pred)

            # print(f"at diffusion step {time}")
            # print("generated mean and var:", pred.mean().cpu().item(), pred.var().cpu().item())
            # print(" ")

        ret = pred if not return_all_timesteps else torch.stack(preds, dim = 1)
        if self.normalize:
            ret = dac_latent_denormalize_heterogeneous(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, vec_cond = None, seq_conds = None, cond_scale = 3., return_all_timesteps = False):
        if exists(vec_cond):
            text_batch_size = vec_cond.shape[0]
            if text_batch_size != batch_size:
                batch_size = text_batch_size
                print("Warning: the desired batch size is different from the input text batch size, using text batch size instead")
        signal_len = self.signal_len
        latent_dim = self.latent_dim
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(
            (batch_size, latent_dim, signal_len),
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = cond_scale,
            return_all_timesteps = return_all_timesteps
        )

    @torch.no_grad()
    def sample_repaint(self, source, mask, vec_cond = None, seq_conds = None, cond_scale = 3.):
        if exists(vec_cond):
            assert source.shape[0] == vec_cond.shape[0]
        if exists(seq_conds):
            for seq_cond in seq_conds:
                if exists(seq_cond):
                    assert source.shape[0] == seq_cond.shape[0]
        signal_len = self.signal_len
        latent_dim = self.latent_dim
        return self.p_sample_loop_repaint(
            source,
            mask,
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = cond_scale,
        )

    def p_losses(self, x_start, t, vec_cond = None, seq_conds = None, noise = None):
    # given x0, calculate ||x0 - f_\theta(a' x0 + b' eta)|| (for x0 predictive net)
        b, d_cated, l = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
        model_out = self.denoise_model(x, t, vec_cond = vec_cond, seq_conds = seq_conds)
        target = self.get_training_target(x_start, t, noise)

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, latent, *args, **kwargs):
        b, latent_dim, l, device = *latent.shape, latent.device, 
        assert l == self.signal_len, f'length of signal must be {self.signal_len}, but got {l} instead'
        assert latent_dim == self.latent_dim, f'dim of signal must be {self.latent_dim}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        if self.normalize:
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
        prediction_type = 'sample',
        schedule_fn_kwargs = dict(),
    ):
        super().__init__(
            denoise_model = denoise_model_primary,
            signal_len = signal_len,
            latent_dim = latent_dim,
            timesteps = timesteps,
            loss_type = loss_type,
            prediction_type = prediction_type,
            schedule_fn_kwargs = schedule_fn_kwargs
        )
        self.denoise_model_secondary = denoise_model_secondary
        assert latent_dim % denoise_model_primary.num_codebooks == 0
        self.latent_dim_primary = latent_dim // denoise_model_primary.num_codebooks
        self.obj_shape_primiary = (self.latent_dim_primary, signal_len)
        self.obj_shape_secondary = (latent_dim - self.latent_dim_primary, signal_len)

    def model_predictions_primary(self, x, t, vec_cond = None, seq_conds = None, cond_scale = 3.):
        model_output_primary = self.denoise_model.forward_with_cond_scale(
            latent =x[:, :self.latent_dim_primary],
            timestep = t,
            cond_scale=cond_scale,
            vec_cond = vec_cond,
            seq_conds = seq_conds
        )
        # returning a tuple (epsilon_0|t, x_0)
        return self.model_output_to_prediction(model_output_primary, x[:, :self.latent_dim_primary], t)

    def p_mean_variance_primary(self, x, t, vec_cond = None, seq_conds = None, cond_scale = 3.): # p_\theta(x_{t-1}|x_t)?
        preds = self.model_predictions_primary(x, t, vec_cond = vec_cond, seq_conds = seq_conds, cond_scale = cond_scale) # epsilon_0|t, x_0
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start = x_start, x_t =x[:, :self.latent_dim_primary], t = t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def model_predictions_secondary(self, x, t, vec_cond = None, seq_conds = None, cond_scale = 3.):
        model_output_secondary = self.denoise_model_secondary.forward_with_cond_scale(
            latent = x,
            timestep = t,
            cond_scale=cond_scale,
            vec_cond = vec_cond,
            seq_conds = seq_conds,
        )
        # returning a tuple (epsilon_0|t, x_0)
        return self.model_output_to_prediction(model_output_secondary, x[:, self.latent_dim_primary:], t)

    def p_mean_variance_secondary(self, x, t, vec_cond = None, seq_conds = None, cond_scale = 3.): # p_\theta(x_{t-1}|x_t)?
        preds = self.model_predictions_secondary(x, t, vec_cond = vec_cond, seq_conds = seq_conds, cond_scale = cond_scale) # epsilon_0|t, x_0
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start = x_start, x_t =x[:, self.latent_dim_primary:], t = t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample_primary(self, x, t: int, vec_cond = None, seq_conds = None, cond_scale = 3.):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean_primary, _, model_log_variance_primary, x_start_primary = self.p_mean_variance_primary(
            x = x,
            t = batched_times,
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = cond_scale,
        )
        noise = torch.randn_like(model_mean_primary) if t > 0 else 0. # no noise if t == 0
        pred_primary = model_mean_primary + (0.5 * model_log_variance_primary).exp() * noise

        return pred_primary, x_start_primary

    @torch.no_grad()
    def p_sample_secondary(self, x, t: int, vec_cond = None, seq_conds = None, cond_scale=3.):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean_secondary, _, model_log_variance_secondary, x_start_secondary = self.p_mean_variance_secondary(
            x = x,
            t = batched_times,
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = cond_scale,
        )
        noise = torch.randn_like(model_mean_secondary) if t > 0 else 0.  # no noise if t == 0
        pred_secondary = model_mean_secondary + (0.5 * model_log_variance_secondary).exp() * noise

        return pred_secondary, x_start_secondary

    @torch.no_grad()
    def p_sample_loop_primary_halfway(self, pred, start_diffusion_step, vec_cond = None, seq_conds = None, cond_scale=3.):
        shape, device = pred.shape, self.betas.device
        pred = pred.to(device)
        assert start_diffusion_step <= self.num_timesteps
        for t in tqdm(reversed(range(0, start_diffusion_step)), desc='sampling loop time step', total=self.num_timesteps):
            pred_primary, x_start_primary = self.p_sample_primary(pred, t, vec_cond, seq_conds, cond_scale)
            pred[:, :self.latent_dim_primary] = pred_primary
        return pred

    @torch.no_grad()
    def p_sample_loop_primary(self, shape, vec_cond = None, seq_conds = None, cond_scale=3.):
        batch, device = shape[0], self.betas.device
        pred = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            pred_primary, x_start_primary = self.p_sample_primary(pred, t, vec_cond, seq_conds, cond_scale)
            pred[:, :self.latent_dim_primary] = pred_primary
        return pred

    @torch.no_grad()
    def p_sample_loop_secondary(self, noise_given_primary, vec_cond = None, seq_conds = None, cond_scale=3.):
        pred = noise_given_primary
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            pred_secondary, x_start_secondary = self.p_sample_secondary(noise_given_primary, t, vec_cond, seq_conds, cond_scale)
            pred[:, self.latent_dim_primary:] = pred_secondary
        if self.normalize:
            pred = dac_latent_denormalize_heterogeneous(pred)
        return pred

    @torch.no_grad()
    def sample(self, batch_size = 16, vec_cond = None, seq_conds = None, cond_scale = 3.):
        if exists(vec_cond):
            text_batch_size = vec_cond.shape[0]
            if text_batch_size != batch_size:
                batch_size = text_batch_size
                print("Warning: the desired batch size is different from the input text batch size, using text batch size instead")

        signal_len = self.signal_len
        latent_dim = self.latent_dim
        pred = self.p_sample_loop_primary(
            (batch_size, latent_dim, signal_len),
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = cond_scale,
        )
        pred = self.p_sample_loop_secondary(
            pred,
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = cond_scale,
        )
        return pred

    @torch.no_grad()
    def sample_editing(self, input_latents_normalized, start_diffusion_step, vec_cond = None, seq_conds = None, cond_scale = 3.):
        batch_size = input_latents_normalized.shape[0]
        if exists(vec_cond):
            text_batch_size = vec_cond.shape[0]
            assert batch_size == text_batch_size
        if type(start_diffusion_step) == int:
            start_diffusion_step = torch.full(
                (batch_size,), start_diffusion_step,
                device=input_latents_normalized.device, dtype=torch.long
            )
        pred = self.q_sample(input_latents_normalized, start_diffusion_step)
        pred = self.p_sample_loop_primary_halfway(
            pred = pred,
            start_diffusion_step = start_diffusion_step,
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = cond_scale,
        )
        pred = self.p_sample_loop_secondary(
            pred,
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = cond_scale,
        )
        return pred

    @torch.no_grad()
    def sample_given_primary(self, noise_given_primary, vec_cond = None, seq_conds = None, cond_scale = 3.):
        if exists(vec_cond):
            assert noise_given_primary.shape[0] == vec_cond.shape[0]
        if exists(seq_conds):
            for seq_cond in seq_conds:
                if exists(seq_cond):
                    assert noise_given_primary.shape[0] == seq_cond.shape[0]
        pred = self.p_sample_loop_secondary(
            noise_given_primary,
            vec_cond = vec_cond,
            seq_conds = seq_conds,
            cond_scale = cond_scale,
        )
        return pred

    def p_losses(self, x_start, t, vec_cond = None, seq_conds = None, noise = None, only_secondary = False, only_primary = False):
    # given x0, calculate ||x0 - f_\theta(a' x0 + b' eta)|| (for x0 predictive net)
        b, d_cated, l = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
        if not only_secondary:
            model_out_primary = self.denoise_model(
                x[:, :self.latent_dim_primary], t, vec_cond = vec_cond, seq_conds = seq_conds,
            )
            target_primary = self.get_training_target(
                x_start[:, :self.latent_dim_primary], t, noise[:, :self.latent_dim_primary]
            )

            loss_primary = self.loss_fn(model_out_primary, target_primary, reduction='none')
            loss_primary = reduce(loss_primary, 'b ... -> b (...)', 'mean').mean()
        else:
            print("only secondary")
            loss_primary = 0

        if not only_primary:
            x_secondary = x[:]
            x_secondary[:, :self.latent_dim_primary] = x_start[:, :self.latent_dim_primary]

            model_out_secondary = self.denoise_model_secondary(
                x_secondary, t, vec_cond = vec_cond, seq_conds = seq_conds,
            )
            target_secondary = self.get_training_target(
                x_start[:, self.latent_dim_primary:], t, noise[:, self.latent_dim_primary:]
            )

            loss_secondary = self.loss_fn(model_out_secondary, target_secondary, reduction='none')
            loss_secondary = reduce(loss_secondary, 'b ... -> b (...)', 'mean').mean()
        else:
            print("only primary")
            loss_secondary = 0

        return loss_primary, loss_secondary

    def forward(self, latent, only_secondary = False, only_primary = False, *args, **kwargs):
        b, latent_dim, l, device = *latent.shape, latent.device,
        assert l == self.signal_len, f'length of signal must be {self.signal_len}'
        assert latent_dim == self.latent_dim, f'dim of signal must be {self.latent_dim}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        if self.normalize:
            latent = dac_latent_normalize_heterogeneous(latent)
        return self.p_losses(
            latent, t,
            only_secondary = only_secondary,
            only_primary = only_primary,
            *args, **kwargs
        )