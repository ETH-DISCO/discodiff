# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
import sys
sys.path.append("..")

from collections import namedtuple

import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from utils import exists, extract

import matplotlib.pyplot as plt

EPSILON = 1e-20

def sum_except_batch(x):
    return x.reshape(x.shape[0], -1).sum(-1)

def index_to_log_onehot(x, num_classes):
# x should have shape [B,...], return [B, num_classes, ...]
    x_onehot = F.one_hot(x, num_classes)
    x_onehot = torch.movedim(x_onehot, -1, 1)
    log_x = torch.log(x_onehot.float().clamp(min=EPSILON))
    return log_x

def log_onehot_to_index(log_x):
# log_x should have shape [B, num_classes, ...], return [B, ...]
    return log_x.argmax(1)

def log_1_min_a(loga):
    return torch.log(1 - loga.exp() + 1e-40)

def alpha_gamma_schedule(
    time_step, 
    codebook_size = 1024,
    cum_alpha_1 = 1.0, # 0.99999,
    cum_alpha_T = 0.0, # 0.000009,
    cum_gamma_1 = 0.0, # 0.000009,
    cum_gamma_T = 0.9, # 0.99999
):
    cum_alpha = np.arange(0, time_step)/(time_step-1)*(cum_alpha_T - cum_alpha_1) + cum_alpha_1
    cum_alpha = np.concatenate(([1], cum_alpha)) # in length time_step + 1
    alpha = cum_alpha[1:]/cum_alpha[:-1] # in length time_step
    
    cum_gamma = np.arange(0, time_step)/(time_step-1)*(cum_gamma_T - cum_gamma_1) + cum_gamma_1
    cum_gamma = np.concatenate(([0], cum_gamma))
    one_minus_cum_gamma = 1 - cum_gamma
    one_minus_gamma = one_minus_cum_gamma[1:] / one_minus_cum_gamma[:-1]
    gamma = 1 - one_minus_gamma # in length time_step
    
    beta = (1 - alpha - gamma)/codebook_size # in length time_step
    cum_alpha = np.concatenate((cum_alpha[1:], [1])) # in length time_step + 1, where -1 means t=0
    cum_gamma = np.concatenate((cum_gamma[1:], [0])) # in length time_step + 1, where -1 means t=0

    cum_beta = (1 - cum_alpha - cum_gamma)/codebook_size # in length time_step + 1, where -1 means t=0
    return alpha, beta, gamma, cum_alpha, cum_beta, cum_gamma

class VQDDPM(nn.Module):
    def __init__(
        self, 
        model, 
        obj_shape, 
        diffusion_steps = 100, 
        loss_type = 'vb_stochastic',
        loss_w_class = [1.0, 1.0], 
        schedule_fn_kwargs = dict()
    ):
        super().__init__()
        
        self.model = model
        self.codebook_size = self.model.codebook_size
        self.num_classes = self.codebook_size + 1
        self.num_timesteps = int(diffusion_steps)
        self.obj_shape = obj_shape

        # define VQ transition parameters
        alpha, beta, gamma, cum_alpha, cum_beta, cum_gamma = alpha_gamma_schedule(
            self.num_timesteps, codebook_size = self.codebook_size
        )
        [alpha, beta, gamma, cum_alpha, cum_beta, cum_gamma] = map(
            lambda a : torch.tensor(a).to(float), [alpha, beta, gamma, cum_alpha, cum_beta, cum_gamma]
        )
        log_alpha, log_beta, log_gamma = \
            torch.log(alpha.clamp(min=EPSILON)), torch.log(beta.clamp(min=EPSILON)), torch.log(gamma.clamp(min=EPSILON))
        log_1_minus_gamma = log_1_min_a(log_gamma)

        log_cum_alpha, log_cum_beta, log_cum_gamma,  = \
            torch.log(cum_alpha.clamp(min=EPSILON)), torch.log(cum_beta.clamp(min=EPSILON)), torch.log(cum_gamma.clamp(min=EPSILON))
        log_1_minus_cum_gamma = log_1_min_a(log_cum_gamma)

        assert torch.logaddexp(log_gamma, log_1_minus_gamma).abs().sum().item() < 1.e-5
        assert torch.logaddexp(log_cum_gamma, log_1_minus_cum_gamma).abs().sum().item() < 1.e-5
        

        # to register param for the model, this is the method of parent nn.Module
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('log_alpha', log_alpha)
        register_buffer('log_beta', log_beta)
        register_buffer('log_gamma', log_gamma)
        register_buffer('log_1_minus_gamma', log_1_minus_gamma)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        register_buffer('log_cum_alpha', log_cum_alpha)
        register_buffer('log_cum_beta', log_cum_beta)
        register_buffer('log_cum_gamma', log_cum_gamma)
        register_buffer('log_1_minus_cum_gamma', log_1_minus_cum_gamma)

        register_buffer('loss_t_history', torch.zeros(self.num_timesteps))
        register_buffer('train_t_count', torch.zeros(self.num_timesteps))

        # for t samping
        self.uniform_t_sample_max_count = 10

        # for losses
        self.loss_type = loss_type
        self.loss_w_class = loss_w_class
        self.diffusion_acc_t = torch.zeros(self.num_timesteps)
        self.diffusion_keep_t = torch.zeros(self.num_timesteps)

        self.auxiliary_loss_weight = 0.05

        # some constants
        self.epsilon = EPSILON
        self.log_epsilon = math.log(self.epsilon)

    '''
        For each diffusion step, define the number of samples per step. Not yet used in this code.
        For example, we are at t=2 (where self.num_timesteps == 100), then we should repeat sampling self.n_sample[t] = 11 times
    '''
    def update_n_sample(self):
        if self.num_timesteps == 100:
            if self.prior_ps <= 10:
                self.n_sample = [1, 6] + [11, 10, 10] * 32 + [11, 15] # 2 + 32*3 + 2 = 1100
            else:
                self.n_sample = [1, 10] + [11, 10, 10] * 32 + [11, 11]
        elif self.num_timesteps == 50:
            self.n_sample = [10] + [21, 20] * 24 + [30] # 1 + 24*2 + 1 = 50
        elif self.num_timesteps == 25:
            self.n_sample = [21] + [41] * 23 + [60] # 1 + 23 + 1 = 25
        elif self.num_timesteps == 10:
            self.n_sample = [69] + [102] * 8 + [139] # 1 + 8 + 1 = 10
        else:
            assert 0, "Num diffusion timesteps should be chosen within [100, 50, 25, 10]"

    '''
        Calculates sum_{x_t} q(x_{t+1} | x_t)p(x_t) = x^T_{t+1} Q_{t} x_{t} in log-space
        where: Q_{t} = [alpha_t I + beta_t 11^T, 0; gamma_t 1, 1]
        log_x_t shape: (B, num_classes, ...)
        If x_t is deterministic, it calculates q(x_{t+1} | x_t)
    '''
    def state_transition_one_step(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_beta_t = extract(self.log_beta, t, log_x_t.shape)
        log_1_minus_gamma_t = extract(self.log_1_minus_gamma, t, log_x_t.shape)
        log_gamma_t = extract(self.log_gamma, t, log_x_t.shape)
        return torch.cat(
            [
                torch.logaddexp(log_x_t[:,:-1] + log_alpha_t, log_beta_t), # alpha_t x[0:-1] + beta_t
                torch.logaddexp(log_x_t[:,-1:] + log_1_minus_gamma_t, log_gamma_t) # gamma_t (1-x[-1]) + x[-1] = (1-gamma_t)x[-1] + gamma_t
            ],
            dim=1
        )

    '''
        Calculates sum_{x_0} q(x_t | x_0) p(x_0) = x^T_t \bar{Q}_{t} x_0 in log-space
        where: \bar{Q}_{t} = [cum_alpha_t I + cum_beta_t 11^T, 0; cum_gamma_t 1, 1]
        log_x shape: (B, num_classes, ...)
        If x_0 is deterministic, it calculates q(x_t | x_0)
    '''
    def state_transition_skip(self, log_x_0, t):
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1) # get self.num_timesteps when t=-1
        log_cum_alpha_t = extract(self.log_cum_alpha, t, log_x_0.shape)
        log_cum_beta_t = extract(self.log_cum_beta, t, log_x_0.shape)
        log_1_minus_cum_gamma_t = extract(self.log_1_minus_cum_gamma, t, log_x_0.shape)
        log_cum_gamma_t = extract(self.log_cum_gamma, t, log_x_0.shape)
        return torch.cat(
            [
                torch.logaddexp(log_x_0[:,:-1] + log_cum_alpha_t, log_cum_beta_t), # alpha_t x[0:-1] + beta_t
                torch.logaddexp(log_x_0[:,-1:] + log_1_minus_cum_gamma_t, log_cum_gamma_t) # gamma_t (1-x[-1]) + x[-1] = (1-gamma_t)x[-1] + gamma_t
            ],
            dim=1
        )

    '''
        Calculates sum_{x_t} p(x_0 | x_t)p(x_t) = model(x_t, t)
        log_x_t: [B, num_classes, ...]
        t: [B, ]
    '''
    def predict_start(self, log_x_t, text_emb, t, cond_scale = 1.0):
        assert log_x_t.size(1) == self.num_classes
        x_t = log_onehot_to_index(log_x_t) # [B, ...]
        out = self.model.forward_with_cond_scale(
            x_t,
            timestep = t,
            cond_scale = cond_scale,
            encoder_hidden_states = text_emb
        ) # [B, num_classes-1, ...]
        assert out.size(0) == log_x_t.size(0)
        assert out.size(1) == self.num_classes - 1
        assert out.size()[2:] == log_x_t.size()[2:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        log0_vec = torch.zeros_like(log_x_t).sum(dim=1, keepdim=True) + self.log_epsilon # [B, 1, ...]
        log_pred = torch.cat((log_pred, log0_vec), dim=1) # [B, num_classes, ...]
        log_pred = torch.clamp(log_pred, self.log_epsilon, 0)
        return log_pred

    # debug
    # def predict_start(self, log_x_t, cond, t):
    #     assert log_x_t.size(1) == self.num_classes
    #     x_t = log_onehot_to_index(log_x_t) # [B, ...]
    #     out = self.model(x_t, timestep = t, class_labels = cond) # [B, num_classes-1, ...]
    #     assert out.size(0) == log_x_t.size(0)
    #     assert out.size(1) == self.num_classes - 1
    #     assert out.size()[2:] == log_x_t.size()[2:]
    #     log0_vec = torch.zeros_like(out).sum(dim=1, keepdim=True) + self.log_epsilon  # [B, 1, ...]
    #     out = torch.cat((out, log0_vec), dim=1) # [B, num_classes, ...]
    #     log_pred = F.log_softmax(out.double(), dim=1).float()
    #     log_pred = torch.clamp(log_pred, self.log_epsilon, 0)
    #     return log_pred
    '''
        Calculates q(x_{t-1}|x_t) = sum_{x_0} [q(x_{t-1}|x_t,x_0) p(x_0)], where p(x_0) and p(x_t) are given

        q(x_{t-1}|x_t,x_0) = q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0) / q(x_t|x_0)
        sum_{x_0} [q(x_{t-1}|x_t,x_0) p(x_0)] = q(x_t|x_{t-1}) * sum_{x_0} q(x_{t-1}|x_0) [p(x_0) / q(x_t|x_0)]
        q(x_t|x_{t-1},x_0) = q(x_t|x_{t-1})
        where p(x_0) and q(x_t|x_0) are consts because x_0, x_t are given

        therefore, sum_{x_0} q(x_{t-1}|x_0) [p(x_0) / q(x_t|x_0)] = [sum_{x_0} q(x_{t-1}|x_0) [p(x_0) / q(x_t|x_0) / W]] * W
        where W = sum_{x_0} p(x_0) / q(x_t|x_0)
            
    '''
    def q_posterior(self, log_x_0, log_x_t, t):
    # log_x_0 and log_x_t are in shape [B, num_classes, ...]
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        assert log_x_t.size(1) == self.num_classes
        batch_size = log_x_0.size()[0]
        id_x_t = log_onehot_to_index(log_x_t) # [B, ...]
        mask = (id_x_t == self.num_classes-1).unsqueeze(1) # [B, 1, ...], places where x_t is mask token
        log1 = torch.zeros(batch_size, 1, 1, 1).type_as(log_x_t) # [B, 1, 1]
        log0_vec = torch.zeros_like(log_x_t).sum(dim=1, keepdim=True) + self.log_epsilon # [B, 1, ...]

        # calculate q(x_t | x_0), with x_t given; this is the quantity [v_{t}^T @ \bar{Q}][:,:-1]
        log_q_t_given_x_0 = self.state_transition_skip(log_x_t, t) # [B, num_classes, ...]
        log_q_t_given_x_0 = log_q_t_given_x_0[:,:-1] # [B, num_classes-1, ...]
        log_cum_gamma_t = extract(self.log_cum_gamma, t, log_x_0.shape).repeat(1, self.num_classes-1, 1, 1)
        log_q_t_given_x_0 = (~mask) * log_q_t_given_x_0 + mask * log_cum_gamma_t # [B, num_classes-1, ...]

        # calculate q(x_t | x_{t-1}), with x_t given; this is the quantity v_{t}^T @ Q
        log_q_t_one_step = self.state_transition_one_step(log_x_t, t)
        log_q_t_one_step = torch.cat([log_q_t_one_step[:,:-1], log0_vec], dim=1)
        log_gamma_t = extract(self.log_gamma, t, log_x_0.shape).repeat(1, self.num_classes-1, 1, 1)
        log_gamma_t = torch.cat([log_gamma_t, log1], dim=1) # [B, num_classes, 1]
        log_q_t_one_step = (~mask) * log_q_t_one_step + mask * log_gamma_t # [B, num_classes, ...]

        # calculate q(x_0) / q(x_t | x_0), and W = sum_{x_0} q(x_0) / q(x_t | x_0), with x_0, x_t given
        logq = log_x_0[:,:-1] - log_q_t_given_x_0 # p(x_0) / q(x_t|x_0), in shape [B, num_classes-1, ...]
        logq = torch.cat([logq, log0_vec], dim=1) # [B, num_classes, ...]
        logq_norm = torch.logsumexp(logq, dim=1, keepdim=True) # W = sum_{x_0} p(x_0) / q(x_t|x_0)

        # calculate sum_{x_0} q(x_{t-1}|x_0) [p(x_0) / q(x_t|x_0)]
        logq = logq - logq_norm # p(x_0) / q(x_t|x_0) / W
        logq = self.state_transition_skip(logq, t - 1) # sum_{x_0} q(x_{t-1}|x_0) [p(x_0) / q(x_t|x_0) / W], in shape [B, num_classes, ...]
        
        # calculate logsum_{x_0} [q(x_{t-1}|x_t,x_0) p(x_0)] = \
        #     log q(x_t|x_{t-1}) + \
        #     logsum_{x_0} q(x_{t-1}|x_0) [p(x_0) / q(x_t|x_0) / W] + \
        #     logW
        log_x_tmin1_given_x_t = log_q_t_one_step + logq + logq_norm

        return torch.clamp(log_x_tmin1_given_x_t, self.log_epsilon, 0)

    '''
        Calculates p(x_{t-1} | x_t) through p(x_0 | x_t) and q(x_{t-1}|x_t, x_0), where p(x_t) is given but p(x_0) is not given
        Returns sum_{x_t} p(x_{t-1} | x_t)p(x_t) and sum_{x_t} p(x_0 | x_t)p(x_t)
    '''
    def p_pred(self, log_x_t, text_emb, t):
        log_x_recon = self.predict_start(log_x_t, text_emb, t)
        log_xmin1_pred = self.q_posterior(
            log_x_0 = log_x_recon, 
            log_x_t = log_x_t, 
            t = t
        )
        return log_xmin1_pred, log_x_recon
    '''
        Use gumbel to sample onehot vector from log probability
        logits: [B, num_classes, ...] (in probability)
        log_sample: [B, num_classes, ...] (one-hot, sampled)
    '''
    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    '''
        Sample from p(x_{t-1} | x_t), given p(x_t)
        Returns a log one-hot vector with the same size as log_x_t
    '''
    @torch.no_grad()
    def p_sample(self, log_x_t, text_emb, t):
        log_xmin1_pred, log_x_recon = self.p_pred(log_x_t, text_emb, t)
        return self.log_sample_categorical(log_xmin1_pred)
    '''
        Sample from p(x_t | x_0), given p(x_0)
        Returns a log one-hot vector with the same size as log_x_0
    '''
    def q_sample(self, log_x_0, t):
        log_x_t_given_x_0 = self.state_transition_skip(log_x_0, t)
        return self.log_sample_categorical(log_x_t_given_x_0)

    '''
        Sample time steps in shape [B, ]
        Two modes: 'importance': sample according to loss history of each step; the greater loss the more important
        'uniform': sample uniformly
    '''
    def sample_time(self, b, device, method = 'uniform'):
        if method == 'importance':
            if not (self.train_t_count > self.uniform_t_sample_max_count).all():
                return self.sample_time(b, device, method = 'uniform')

            loss_t_sqrt = torch.sqrt(self.loss_t_history + 1)
            loss_t_sqrt[0] = loss_t_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = loss_t_sqrt / loss_t_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples = b, replacement = True)
            pt = pt_all.gather(dim = 0, index = t)

            # fixed_pt = torch.zeros(self.num_timesteps)
            # fixed_pt[1:] = 2*self.num_timesteps - torch.arange(self.num_timesteps - 1)
            # fixed_pt[0] = 6*self.num_timesteps
            # fixed_pt = fixed_pt / fixed_pt.sum()
            # t = torch.multinomial(fixed_pt, num_samples = b, replacement = True)
            # pt = fixed_pt.gather(dim = 0, index = t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device = device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        #TODO: log log_prob1 etc., check weird numbers (e.g., is it a log-prob?)
        #DEBUG:
        # sum1 = torch.logsumexp(log_prob1, 1)
        # sum2 = torch.logsumexp(log_prob2, 1)
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    '''
        Run a training step given x_0, and calculate
        loss_t = KL[q(x_{t-1}|x_t, x_0) || p_theta(x_{t-1}|x_t)]
        loss_0 = -log p_theta(x_0|x_1)

        Input: x being a long tensor with size [B, ...] indicating an input VQ object
    '''

    # TODO 1: derive the loss function, is there other forms? Read the paper more and comprehend
    # TODO 2: run overfit test on a single piece

    def _train_loss(self, x, text_emb):
        b, device = x.size(0), x.device

        assert self.loss_type == 'vb_stochastic'
        x_0 = x.long()

        loss0_t = []
        losst_t = []
        # DEBUG
        # i_show = [0, 1, 2, 3, 5, 10, 20, 40, 60, 80, 98]
        # fig, axs = plt.subplots(5, len(i_show), figsize = (24, 16))
        # s = 0
        # for i in i_show: # DEBUG: plot the t - kl loss curve

        # 1. sample a timestep t
        # t, pt = self.sample_time(b, device, 'importance')
        t, pt = self.sample_time(b, device, 'uniform') # debug: not using importance sampling
        # t = t - t + i # DEBUG: setting t to i

        # 2. get x_t by q_prior (transition)
        log_x_0 = index_to_log_onehot(x_0, self.num_classes)
        # log_x_t_prob = self.state_transition_skip(log_x_0, t) # DEBUG
        log_x_t = self.q_sample(log_x_0 = log_x_0, t = t)
        if torch.isnan(log_x_t).any():
            print("warning: nan encountered in log_x_t.")
            assert 0
        x_t_idx = log_onehot_to_index(log_x_t)
        # 3. calculate p(x_0_recon) sending x_t, cond, t to model
        log_x_0_recon = self.predict_start(log_x_t, text_emb, t = t)

        if torch.isnan(log_x_0_recon).any():
            print("warning: nan encountered in model forward! Replaced with zero tensor.")
            log_x_0_recon = torch.nan_to_num(log_x_0_recon, nan=0.0)

        # 4. calculate p_theta(x_{t-1}|x_t) through q_posterior given x_0_recon
        log_x_tmin1_pred = self.q_posterior(log_x_0 = log_x_0_recon, log_x_t = log_x_t, t = t) # DEBUG: setting t to t+1 and get normal results?
        # 5. calculate q(x_{t-1}|x_t, x_0) through q_posterior given x_0
        log_x_tmin1 = self.q_posterior(log_x_0 = log_x_0, log_x_t = log_x_t, t = t)
        # 6. calculate loss_t = KL[q(x_{t-1}|x_t, x_0) || p_theta(x_{t-1}|x_t)]
        kl = self.multinomial_kl(log_x_tmin1, log_x_tmin1_pred)
        mask_region = (x_t_idx == self.num_classes-1).float()
        mask_weight = mask_region * self.loss_w_class[0] + (1. - mask_region) * self.loss_w_class[1]
        kl = kl * mask_weight
        kl = sum_except_batch(kl)
        # 7. calculate loss_0 = -log p_theta(x_0|x_1) for the places where t=0
        # decoder_nll = - (log_x_0.exp() * log_x_tmin1_pred).sum(dim = 1)
        decoder_nll = self.multinomial_kl(log_x_0, log_x_tmin1_pred) # DEBUG
        decoder_nll = sum_except_batch(decoder_nll)
        mask = (t == torch.zeros_like(t)).float() # if t=0, then it is the last transition (t=1 in paper)
        kl_loss = mask * decoder_nll + (1. - mask) * kl

        # DEBUG
        # loss0_t.append(decoder_nll.detach().cpu().numpy().item())
        # losst_t.append(kl.detach().cpu().numpy().item())
        #
        # axs[0,s].plot(log_x_tmin1[0, :, 1, 0].detach().cpu().numpy())
        # axs[1,s].plot(log_x_t[0, :, 1, 0].detach().cpu().numpy())
        # axs[2,s].plot(log_x_t_prob[0, :, 1, 0].detach().cpu().numpy())
        # axs[3,s].plot(log_x_tmin1_pred[0, :, 1, 0].detach().cpu().numpy())
        # axs[4,s].plot(log_x_0_recon[0, :, 1, 0].detach().cpu().numpy())
        # axs[0,s].set_title(str(i))
        # s += 1
        #
        # # DEBUG
        # axs[0,0].set_title("p(x_{t-1}|x_t, x_0)")
        # axs[1,0].set_title("x_t ~ p(x_t|x_0)")
        # axs[2,0].set_title("p(x_t|x_0)")
        # axs[3,0].set_title("p_{theta}(x_{t-1}}|x_t)")
        # axs[4,0].set_title("p_{theta}(x_0}|x_t)")
        # plt.show()
        #
        # fig, ax = plt.subplots(1, 1)
        # ax.scatter(i_show, loss0_t)
        # ax.scatter(i_show, losst_t)
        # ax.set_yscale('log')
        # ax.legend([
        #     "KL(p(x_0) || p_{theta}(x_{t-1}}|x_t))",
        #     "KL(p(x_{t-1}|x_t, x_0) || p_{theta}(x_{t-1}|x_t))"
        # ])
        # ax.set_title("losses")
        # plt.show()

        # 8. log down the losses for time step importance sampling
        kl_loss_squared = kl_loss.pow(2)
        kl_loss_squared_prev = self.loss_t_history.gather(dim=0, index=t)
        new_loss_t_history = (0.1 * kl_loss_squared + 0.9 * kl_loss_squared_prev).detach()

        self.loss_t_history.scatter_(dim=0, index=t, src=new_loss_t_history)
        self.train_t_count.scatter_add_(dim=0, index=t, src=torch.ones_like(kl_loss_squared))

        # 9. upweigh loss term of the kl
        vb_loss = kl_loss / pt # + kl_prior, as auxiliary loss, not used in this code

        # for loss curve
        default_loss = vb_loss.detach().cpu()

        # TODO: auxiliary loss
        if self.auxiliary_loss_weight != 0:
            kl_aux = self.multinomial_kl(log_x_0[:,:-1,:], log_x_0_recon[:,:-1,:])
            kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux

            addition_loss_weight = (1 - t/self.num_timesteps) + 0.1

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2

        # for loss curve
        aux_loss = loss2.detach().cpu()

        # bonus: calculate accracy predicting x_0 and keep rate from x_t to x_{t-1}
        x_0_recon = log_onehot_to_index(log_x_0_recon)
        x_tmin1_pred = log_onehot_to_index(log_x_tmin1_pred)
        x_t = log_onehot_to_index(log_x_t)
        total_size = x_0.shape[1] * x_0.shape[2]

        same_rates = sum_except_batch((x_0_recon == x_0)).cpu() / total_size
        keep_rates = sum_except_batch((x_tmin1_pred == x_t)).cpu() / total_size
        t_cpu = t.cpu()
        self.diffusion_acc_t[t_cpu] = same_rates * 0.1 + self.diffusion_acc_t[t_cpu] * 0.9
        self.diffusion_keep_t[t_cpu] = keep_rates * 0.1 + self.diffusion_keep_t[t_cpu] * 0.9

        return log_x_tmin1_pred, vb_loss, default_loss, aux_loss


    @property
    def device(self):
        return next(self.model.parameters()).device

    '''
        The sample_fast method of original code
    '''
    @torch.no_grad()
    def sample(
        self,
        batch_size,
        text_emb = None,
        skip_step = 1,
        cond_scale = 3.0,
        **kwargs
    ):
        device = self.device
        if exists(text_emb):
            text_batch_size = text_emb.shape[0]
            if text_batch_size != batch_size:
                batch_size = text_batch_size
                print("Warning: the desired batch size is different from the input text batch size, using text batch size instead")

        # zero_logits = torch.zeros((batch_size, self.num_classes-1, *self.obj_shape), device = device) + self.log_epsilon
        # one_logits = torch.zeros((batch_size, 1, *self.obj_shape), device = device)

        # DEBUG: paper implementation starts from the prior rather than all [mask]
        beta_cum_logits = torch.zeros((batch_size, self.num_classes-1, *self.obj_shape), device = device) + self.log_cum_beta[-2].item()
        gamma_cum_logits = torch.zeros((batch_size, 1, *self.obj_shape), device = device) + self.log_cum_gamma[-2].item()
        log_z = torch.cat((beta_cum_logits, gamma_cum_logits), dim = 1)
        # log_z = torch.cat((zero_logits, one_logits), dim=1)
        start_step = self.num_timesteps

        assert skip_step > 0

        diffusion_list = [t for t in range(start_step-1, -1, -1-skip_step)]
        # diffusion_list += [t for t in range(10, -1)] # DEBUG: fine diffusion sampling in the last 10 steps
        if diffusion_list[-1] != 0:
            diffusion_list.append(0)

        for diffusion_index in tqdm(diffusion_list, desc = 'sampling loop time step'):
            
            t = torch.full((batch_size,), diffusion_index, device = device, dtype = torch.long)
            log_x_0_recon = self.predict_start(log_z, text_emb, t, cond_scale = cond_scale)
            if diffusion_index > skip_step:
                model_log_prob = self.q_posterior(log_x_0 = log_x_0_recon, log_x_t = log_z, t = t-skip_step)
            else:
                model_log_prob = self.q_posterior(log_x_0 = log_x_0_recon, log_x_t = log_z, t = t)

            log_z = self.log_sample_categorical(model_log_prob)

        x0 = log_onehot_to_index(log_z)
        return x0

    # TODO
    @torch.no_grad()
    def sample_repaint(
        self,
        mask_batch,
        cond = None,
        repeat_steps = 10,
        skip_steps = 10,
        **kwargs
    ):
        pass

    '''
        Run a training step and get the loss
        vq_tokens: in shape [B, K, T], K being the num of codebooks, and T being the frame len
        TODO: cond, not considered yet
    '''
    def forward(self, vq_tokens, text_emb = None, **kwargs):
        total_size = vq_tokens.shape[1] * vq_tokens.shape[2]

        log_x_tmin1_pred, loss, default_loss, aux_loss = self._train_loss(vq_tokens, text_emb)
        loss = loss.sum() / total_size
        default_loss, aux_loss = default_loss.sum() / total_size, aux_loss.sum() / total_size
        return loss, default_loss, aux_loss


