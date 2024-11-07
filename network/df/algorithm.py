import torch
import torch.nn as nn
from einops import rearrange
from .models.diffusion.diffusion_forcing import DiffusionForcingBase
from .config.Config_DF import ConfigDF
from .config.Config import Config
from .models.vae.autoencoder import AutoencoderKL

class ZetaEnv():
    def __init__(self,zeta):
        self.zeta = zeta

    def update(self,zeta):
        self.zeta = zeta

class Algorithm(nn.Module):
    def __init__(self, model_name,device='cpu'):
        super().__init__()
        self.device = device
        self.config = ConfigDF(model_name=model_name)
        self.df_model = DiffusionForcingBase(self.config,device)
        self.use_ldm = 'ldm' in self.config.model_name
        if self.use_ldm:
            self.vae = AutoencoderKL(image_key='observations')

    # observation (b t), c, h,w = (1,c,h,w),
    # obs[0:1] = (1,c,h,w)
    def init_wm(self,observation):
        observation = observation[0]
        if self.use_ldm:
            latent = self.vae.encode(observation.reshape(-1, 3, Config.resolution, Config.resolution))
            latent = latent.sample() * Config.scale_factor
            latent = latent.reshape(*Config.vae_latent_shape)
            observation = latent
        init_zeta = self.df_model.init_df_model(observation)
        # env = ZetaEnv(init_zeta)
        return init_zeta

    def real_time_infer(self, zeta, cur_act, sampling_timestep):
        cur_act = cur_act.to(self.device)
        zeta,obs = self.df_model.step(zeta,cur_act.float(), sampling_timestep)
        # wm_env.update(zeta)
        if self.use_ldm:
            obs = self.vae.decode(obs / Config.scale_factor)
        return obs, zeta

    def forward(self, batch, predict_len):
        batch['observations'] = batch['observations'][:, :predict_len, :, :, :]
        batch['cur_actions'] = batch['cur_actions'][:, :predict_len, :]
        mask_padding = torch.ones(size=(batch['cur_actions'].shape[0], batch['cur_actions'].shape[1]),
                                       dtype=torch.bool).to(self.device)
        df_batch = [batch['observations'], batch['cur_actions'].float(), mask_padding]
        self.df_model.validation_step(df_batch)
        pred_image = self.df_model.validation_step_outputs[0][0]
        pred_image = rearrange(pred_image, "t b c h w -> b t c h w")
        pred_image = torch.clamp(pred_image, -1, 1)
        pred_image = pred_image.to(self.device)
        return pred_image.reshape(-1, 3, Config.resolution, Config.resolution)