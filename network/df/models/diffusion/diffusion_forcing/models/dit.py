from functools import partial
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from einops import rearrange, repeat


from .utils import default, exists, cast_tuple, divisible_by
from .gru import Conv2dGRUCell
from .resnet import ResBlock1d
from .sin_emb import SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb
from .attend import Attend
from .dit_models import DiT_models

# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, emb_dim=None, groups=8):
        """
        :param dim: input channel
        :param dim_out:  output channel
        :param emb_dim: extra embedding to fuse, such as time or control
        :param groups: group for conv2d
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim_out * 2)) if exists(emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(emb):
            emb = self.mlp(emb)
            emb = rearrange(emb, "b c -> b c 1 1")
            scale_shift = emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class DiT(nn.Module):
    def __init__(
        self,
        model_name=None,
        dim=64,  # base number of channels that controls network size
        out_dim=None,
        z_cond_dim=None,
        external_cond_dim=None,
        channels=3,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_cond=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        config=None,
    ):
        super().__init__()

        # determine dimensions
        self.config = config
        self.channels = channels
        self.self_condition = self_condition
        self.z_cond_dim = z_cond_dim
        self.external_cond_dim = external_cond_dim
        # input_channels = channels * (2 if self_condition else 1)
        # input_channels += z_cond_dim if z_cond_dim else 0

        # init_dim = default(init_dim, dim)
        # self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        self.dit = DiT_models[model_name](in_channels=channels + z_cond_dim)
        self.hidden_dim = self.dit.hidden_size

        # time embeddings

        time_emb_dim = self.hidden_dim // 2
        external_cond_emb_dim = self.hidden_dim // 2

        # self.emb_dim = time_emb_dim + external_cond_emb_dim if external_cond_dim else time_emb_dim
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_cond

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_cond)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(fourier_dim, time_emb_dim), nn.GELU(), nn.Linear(time_emb_dim, time_emb_dim)
        )

        # self.external_cond_mlp = (
        #     nn.Sequential(
        #         nn.Linear(external_cond_dim, external_cond_emb_dim),
        #         nn.GELU(),
        #         nn.Linear(external_cond_emb_dim, external_cond_emb_dim),
        #     )
        #     if self.external_cond_dim
        #     else None
        # )

        self.external_cond_emb = nn.Sequential(
            nn.Embedding(self.config.world_model_action_num, external_cond_emb_dim//self.external_cond_dim),
            nn.Flatten()
        )



        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_conv = nn.Conv2d(channels + z_cond_dim, self.out_dim, 1)
        # self.final_res_block = ResnetBlock(dim=(channels + z_cond_dim) * 2, dim_out=(channels + z_cond_dim), emb_dim=self.hidden_dim,groups=4)
        self.tanh_layer = nn.Sequential(
            nn.Conv2d(channels + z_cond_dim,channels + z_cond_dim,1),
            nn.Tanh(),
        )
        if self.config.use_lrelu:
            self.lrelu_layer = nn.Sequential(
            nn.Conv2d(channels + z_cond_dim,channels + z_cond_dim,1),
            nn.LeakyReLU(),
        )

    def forward(self, x, time, z_cond, external_cond=None, x_self_cond=None):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        if self.z_cond_dim:
            x = torch.cat((z_cond, x), dim=1)
        # r = x.clone()
        # # x = self.init_conv(x)


        emb = self.time_mlp(time)
        if self.external_cond_dim:
            if external_cond is None:
                external_cond_emb = torch.zeros((emb.shape[0], self.external_cond_dim)).to(emb)
            else:
                external_cond_emb = self.external_cond_emb(external_cond.long())
            emb = torch.cat([emb, external_cond_emb], -1)

        x = self.dit(x,emb)
        if self.config.use_tanh:
            x = self.tanh_layer(x)
        elif self.config.use_lrelu:
            x = self.lrelu_layer(x)

        # x = nn.LSTM
        # x = torch.cat((x, r), dim=1)
        # x = self.final_res_block(x,emb)

        # x = self.final_res_block(x, emb)
        return self.final_conv(x)