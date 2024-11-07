from dataclasses import dataclass
from typing import Tuple

import torch
import tqdm
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .inner_model import InnerModel, InnerModelConfig
from .utils import ComputeLossOutput
import random
import math
def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(cfg.inner_model,self.cfg.use_zeta)
        self.sample_sigma_training = None
        sigmaConfig = SigmaDistributionConfig(
            loc=-0.4,
            scale=1.2,
            sigma_min=0.002,
            sigma_max=20
        )
        self.setup_training(sigmaConfig)
        self.last_loss = [1e5]
        if self.cfg.use_zeta:
            self.init_zeta = nn.Parameter(torch.randn(list([16,64,64])), requires_grad=True)

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    def forward(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor) -> Tuple[Tensor, Tensor]:
        c_in, c_out, c_skip, c_noise = self._compute_conditioners(sigma)
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * c_in
        model_output = self.inner_model(rescaled_noise, c_noise, rescaled_obs, act)
        denoised = model_output * c_out + noisy_next_obs * c_skip
        return model_output, denoised

    def model_prediction(self,noisy_next_obs, sigma, zeta, act):
        c_in, c_out, c_skip, c_noise = self._compute_conditioners(sigma)
        # rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * c_in
        model_output,zeta = self.inner_model(rescaled_noise, c_noise, zeta, act)
        denoised = model_output * c_out + noisy_next_obs * c_skip
        return model_output,zeta, denoised

    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        _, d = self(noisy_next_obs, sigma, obs, act)
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d

    @torch.no_grad()
    def denoise_with_zeta(self, noisy_next_obs: Tensor, sigma: Tensor, zeta: Tensor, act: Tensor) -> Tensor:
        _, zeta, d = self.model_prediction(noisy_next_obs, sigma, zeta, act)
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return zeta,d

    @torch.no_grad()
    def sample_next_obs(self, obs: Tensor, act: Tensor,num_steps_denoising: int, zeta=None):

        def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            l = torch.linspace(0, 1, num_steps, device=device)
            sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
            return torch.cat((sigmas, sigmas.new_zeros(1)))
        if self.cfg.use_zeta:
            zeta = zeta.detach()
        device = obs.device
        order = 1
        sigmas = build_sigmas(num_steps_denoising, 0.002, 20, 7, device)
        b, t, c, h, w = obs.size()
        obs = obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(0/ (len(sigmas) - 1), 2**0.5 - 1)
        x = torch.randn(b, c, h, w, device=device)
        for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
            gamma = gamma_ if 0 <= sigma <= float("inf") else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * 1
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
            if self.cfg.use_zeta:
                next_zeta,denoised = self.denoise_with_zeta(x,sigma,zeta,act)
            else:
                denoised = self.denoise(x, sigma, obs, act)
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                if self.cfg.use_zeta:
                    next_zeta, denoised_2 = self.denoise_with_zeta(x_2, next_sigma * s_in, zeta, act)
                else:
                    denoised_2 = self.denoise(x_2, next_sigma * s_in, obs, act)
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        if self.cfg.use_zeta:
            return x, next_zeta
        else:
            return x

    def _compute_conditioners(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        return *(add_dims(c, 4) for c in (c_in, c_out, c_skip)), add_dims(c_noise, 1)

    # def add_scalable_mask(self,mask,obs,act):
    #
    #     pass

    def compute_loss_zeta(self, batch) -> ComputeLossOutput:
        seq_length = batch['observations'].size(1)
        all_obs = batch['observations'].clone()
        loss = 0

        zeta = self.init_zeta[None].expand(batch['observations'].size(0),*self.init_zeta.shape)

        batch['actions'] = torch.cat([torch.zeros_like(batch['actions'][:,:1]),batch['actions'][:,0:-1]],dim=1)
        for i in range(seq_length):
            next_obs = all_obs[:,i]
            act = batch['actions'][:, i]

            mask = batch['mask_padding'][:, i]


            # b, t, c, h, w = obs.shape
            # obs = obs.reshape(b, t * c, h, w)
            b,c,h,w = next_obs.shape

            sigma = self.sample_sigma_training(b, self.device)
            _, c_out, c_skip, _ = self._compute_conditioners(sigma)

            offset_noise = self.cfg.sigma_offset_noise * torch.randn(b, c, 1, 1, device=next_obs.device)
            noisy_next_obs = next_obs + offset_noise + torch.randn_like(next_obs) * add_dims(sigma, next_obs.ndim)

            model_output,zeta, denoised = self.model_prediction(noisy_next_obs, sigma, zeta, act)

            target = (next_obs - c_skip * noisy_next_obs) / c_out
            one_step_loss = F.mse_loss(model_output[mask], target[mask])
            loss += one_step_loss

        loss /= seq_length
        self.last_loss.append(float(loss.detach()))
        if len(self.last_loss) > 20:
            self.last_loss = self.last_loss[1:]
        return loss, {"loss_denoising": loss.detach()}


    def warm_up_zeta(self,batch):
        seq_length = batch['observations'].size(1)
        all_obs = batch['observations'].clone()
        loss = 0

        zeta = self.init_zeta[None].expand(batch['observations'].size(0),*self.init_zeta.shape)

        batch['actions'] = torch.cat([torch.zeros_like(batch['actions'][:,:1]),batch['actions'][:,0:-1]],dim=1)
        for i in range(seq_length):
            next_obs = all_obs[:,i]
            act = batch['actions'][:, i]

            mask = batch['mask_padding'][:, i]


            # b, t, c, h, w = obs.shape
            # obs = obs.reshape(b, t * c, h, w)
            b,c,h,w = next_obs.shape

            sigma = self.sample_sigma_training(b, self.device)
            sigma.fill_(0.01)
            _, c_out, c_skip, _ = self._compute_conditioners(sigma)

            offset_noise = self.cfg.sigma_offset_noise * torch.randn(b, c, 1, 1, device=next_obs.device)
            noisy_next_obs = next_obs + offset_noise + torch.randn_like(next_obs) * add_dims(sigma, next_obs.ndim)

            model_output,zeta, denoised = self.model_prediction(noisy_next_obs, sigma, zeta, act)

            target = (next_obs - c_skip * noisy_next_obs) / c_out
            one_step_loss = F.mse_loss(model_output[mask], target[mask])
            loss += one_step_loss

        return zeta

    def compute_loss(self, batch) -> ComputeLossOutput:
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = batch['observations'].size(1) - n
        all_obs = batch['observations'].clone()
        loss = 0

        def loss_to_probability(loss):
            # 当 loss 小于 0.008 时，直接返回 1
            left_range = 0.008
            right_range = 0.01
            if loss < left_range:
                return 1.0
            # 当 loss 小于 0.02 时，激活概率
            if loss < right_range:
                # 使用逻辑斯蒂函数将 loss 映射到 [0, 1] 范围内
                # 这里的参数可以根据实际需求进行调整
                return 1 / (1 + math.exp((loss - (left_range + right_range)/2) * 1000))
            # 当 loss 大于等于 0.02 时，返回 0
            return 0.0

        # use_sampler_augmented = random.random() <= loss_to_probability(sum(self.last_loss)/len(self.last_loss))
        # denoised_step = 0
        # if use_sampler_augmented:
        #     print("augmented begin")
        #     denoised_step = random.randint(1,4)
        # else:
        #     seq_length = 1
        # seq_length = 10
        sigma_stat = []
        for i in range(seq_length):
            obs = all_obs[:, i : n + i]
            next_obs = all_obs[:, n + i]

            act = batch['actions'][:, i : n + i]
            mask = batch['mask_padding'][:, n + i]

            if self.cfg.use_mask:
                bs, t = obs.shape[0], obs.shape[1]
                next_obs = obs[torch.arange(bs), batch['scalable_mask'].squeeze(1)].clone()

                time_steps = torch.arange(t).unsqueeze(0).expand(bs, t)
                time_steps = time_steps.to(obs.device)
                # 创建一个 mask 矩阵，表示哪些时间步需要被置为特定值
                mask_matrix = time_steps >= batch['scalable_mask']

                # 将 obs 中 (bs, mask:, c, h, w) 的部分全部置为 2
                obs[mask_matrix.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(obs)] = 2

                # 将 act 中 (bs, mask:, e) 的部分全部置为 10
                act[mask_matrix.unsqueeze(-1).expand_as(act)] = 10


            b, t, c, h, w = obs.shape
            obs = obs.reshape(b, t * c, h, w)

            sigma = self.sample_sigma_training(b, self.device)
            _, c_out, c_skip, _ = self._compute_conditioners(sigma)

            offset_noise = self.cfg.sigma_offset_noise * torch.randn(b, c, 1, 1, device=next_obs.device)
            noisy_next_obs = next_obs + offset_noise + torch.randn_like(next_obs) * add_dims(sigma, next_obs.ndim)

            model_output, denoised = self(noisy_next_obs, sigma, obs, act)

            target = (next_obs - c_skip * noisy_next_obs) / c_out
            one_step_loss = F.mse_loss(model_output[mask], target[mask])
            loss += one_step_loss
            # step_length = (20-0.002)/100
            # for i in range(101):
            #     sigma = torch.tensor(step_length*i + 0.002).to(self.device)
            #     _, c_out, c_skip, _ = self._compute_conditioners(sigma)
            #
            #     offset_noise = self.cfg.sigma_offset_noise * torch.randn(b, c, 1, 1, device=next_obs.device)
            #     noisy_next_obs = next_obs + offset_noise + torch.randn_like(next_obs) * add_dims(sigma, next_obs.ndim)
            #
            #     model_output, denoised = self(noisy_next_obs, sigma, obs, act)
            #
            #     target = (next_obs - c_skip * noisy_next_obs) / c_out
            #     one_step_loss = F.mse_loss(model_output[mask], target[mask])
            #     loss += one_step_loss
            #     sigma_stat.append([float(sigma),float(one_step_loss.detach())])
            # if use_sampler_augmented:
            #     denoised = self.sample_next_obs(obs.reshape(b, t, c, h, w), act, denoised_step)
                # for j in range(seq_length-1):
                #     denoised = self.sample_next_obs(obs.reshape(b, t, c, h, w),act,denoised_step)
                #     all_obs[:, n + j] = denoised.detach().clamp(-1, 1)
                # i = seq_length-2
            # one_step_denoised_loss = F.mse_loss(all_obs[:, n + i],denoised).detach()
            all_obs[:, n + i] = denoised.detach().clamp(-1, 1)



        loss /= seq_length
        self.last_loss.append(float(loss.detach()))
        if len(self.last_loss) > 20:
            self.last_loss = self.last_loss[1:]
        return loss, {"loss_denoising": loss.detach()}

