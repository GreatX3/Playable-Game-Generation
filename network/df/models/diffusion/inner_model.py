from dataclasses import dataclass
from typing import List, Optional,Type

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv3x3, FourierFeatures, GroupNorm, UNet
from .dit_models import DiT_models

@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    attn_head_num: Optional[int] = None
    num_actions: Optional[int] = None
    use_dit: Optional[bool] = False
    dit_model_name: Optional[str] = ''

class ResBlock2d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation: Type[nn.Module] = nn.ReLU):
        super(ResBlock2d, self).__init__()
        self.activation = activation()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out
class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig,use_zeta=False) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_zeta = use_zeta
        if self.use_zeta:
            zeta_channels = 16
            in_channels = cfg.img_channels + zeta_channels
            out_channels = zeta_channels
        else:
            in_channels = (cfg.num_steps_conditioning+1) * cfg.img_channels
            out_channels = cfg.img_channels
        if cfg.use_dit:
            self.dit = DiT_models[cfg.dit_model_name](in_channels=in_channels)
            cond_channels = self.dit.y_embedder.embedding_table.weight.shape[-1]
            self.noise_emb = FourierFeatures(cond_channels)
            self.act_emb = nn.Sequential(
                nn.Embedding(cfg.num_actions, cond_channels // cfg.num_steps_conditioning),
                nn.Flatten(),  # b t e -> b (t e)
            )
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_channels, cond_channels),
                nn.SiLU(),
                nn.Linear(cond_channels, cond_channels),
            )
            # self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, 4)
            # self.conv_out = nn.ConvTranspose2d(self.dit.in_channels, 1, kernel_size=4, stride=2, padding=1)
            self.conv_out = Conv3x3(in_channels, out_channels)

            # nn.init.zeros_(self.conv_out.weight)
        else:
            self.noise_emb = FourierFeatures(cfg.cond_channels)
            self.act_emb = nn.Sequential(
                nn.Embedding(cfg.num_actions, cfg.cond_channels // cfg.num_steps_conditioning),
                nn.Flatten(),  # b t e -> b (t e)
            )
            self.cond_proj = nn.Sequential(
                nn.Linear(cfg.cond_channels, cfg.cond_channels),
                nn.SiLU(),
                nn.Linear(cfg.cond_channels, cfg.cond_channels),
            )
            self.conv_in = Conv3x3(in_channels, cfg.channels[0])
            self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths,cfg.attn_head_num)
            self.norm_out = GroupNorm(cfg.channels[0])
            self.conv_out = Conv3x3(cfg.channels[0], out_channels)
            nn.init.zeros_(self.conv_out.weight)

        if self.cfg.use_zeta:
            self.x_from_z = nn.Sequential(
                ResBlock2d(out_channels, cfg.img_channels),
                nn.Conv2d(cfg.img_channels, cfg.img_channels, 1, padding=0),
            )

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        cond = self.cond_proj(self.noise_emb(c_noise) + self.act_emb(act))
        if self.cfg.use_dit:
            x = torch.cat((obs, noisy_next_obs), dim=1)
            # x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
            x = self.dit(x,cond)
            if self.use_zeta:
                z = self.conv_out(x)
                x = self.x_from_z(z)
            else:
                x = self.conv_out(x)
        else:
            x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
            x, _, _ = self.unet(x, cond)
            if self.use_zeta:
                z = self.conv_out(F.silu(self.norm_out(x)))
                x = self.x_from_z(z)
            else:
                x = self.conv_out(F.silu(self.norm_out(x)))
        if self.use_zeta:
            return x, z
        else:
            return x

