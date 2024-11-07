from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Union, Sequence, Optional
from einops import rearrange

from .models.diffusion_transition import DiffusionTransitionModel


class DiffusionForcingBase(nn.Module):
    def __init__(self,config=None,device='cuda:0'):
        super().__init__()
        from omegaconf import OmegaConf
        import os
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "df_video_dmlab.yaml")
        cfg = OmegaConf.load(config_path)
        self.config = config
        use_mc_config = self.config.use_mc
        # cfg.context_frames = 16
        cfg.z_shape = [self.config.zeta, cfg.z_shape[1], cfg.z_shape[2]]
        if use_mc_config:
            cfg.z_shape = [32,64,64]
            cfg.diffusion.cum_snr_decay = 0.96
            cfg.diffusion.network_size = 64
            cfg.frame_stack = 8
            cfg.context_frames = cfg.frame_stack
            cfg.external_cond_dim = cfg.frame_stack
        # cfg.z_shape = [32, 64, 64]
        # cfg.diffusion.network_size = 64
        cfg.context_frames = self.config.context_len
        self.cfg = cfg
        self.device = device
        self.x_shape = cfg.x_shape
        self.z_shape = cfg.z_shape
        self.frame_stack = cfg.frame_stack
        self.cfg.diffusion.cum_snr_decay = self.cfg.diffusion.cum_snr_decay**self.frame_stack
        self.x_stacked_shape = list(cfg.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.is_spatial = len(self.x_shape) == 3  # pixel
        self.gt_cond_prob = cfg.gt_cond_prob  # probability to condition one-step diffusion o_t+1 on ground truth o_t
        self.gt_first_frame = cfg.gt_first_frame
        self.context_frames = cfg.context_frames  # number of context frames at validation time
        self.chunk_size = cfg.chunk_size
        self.calc_crps_sum = cfg.calc_crps_sum
        self.external_cond_dim = cfg.external_cond_dim
        self.uncertainty_scale = cfg.uncertainty_scale
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.validation_step_outputs = []
        self.min_crps_sum = float("inf")
        self.learnable_init_z = cfg.learnable_init_z


        self._build_model()
    def register_data_mean_std(
        self, mean: Union[str, float, Sequence], std: Union[str, float, Sequence], namespace: str = "data"
    ):
        """
        Register mean and std of data as tensor buffer.

        Args:
            mean: the mean of data.
            std: the std of data.
            namespace: the namespace of the registered buffer.
        """
        for k, v in [("mean", mean), ("std", std)]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    v = torch.from_numpy(np.load(v))
                elif v.endswith(".pt"):
                    v = torch.load(v)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                v = torch.tensor(v)
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))

    def _build_model(self):
        self.transition_model = DiffusionTransitionModel(
            self.x_stacked_shape, self.z_shape, self.external_cond_dim, self.cfg.diffusion
        ,self.config)
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)
        if self.learnable_init_z:
            if self.config.use_km:
                # self.init_z = nn.Parameter(torch.randn(list(self.z_shape)), requires_grad=True)
                self.init_z = nn.Parameter(torch.empty(list(self.z_shape)), requires_grad=True)
                torch.nn.init.kaiming_normal_( self.init_z, mode='fan_in', nonlinearity='relu')
            else:
                self.init_z = nn.Parameter(torch.randn(list(self.z_shape)), requires_grad=True)
            # print(self.init_z)


    def configure_optimizers(self):
        transition_params = list(self.transition_model.parameters())
        if self.learnable_init_z:
            transition_params.append(self.init_z)
        optimizer_dynamics = torch.optim.AdamW(
            transition_params, lr=self.config.lr, weight_decay=self.config.wd, betas=self.cfg.optimizer_beta
        )

        return optimizer_dynamics

    def configure_optimizers_gpt(self):
        transition_params = list(self.transition_model.parameters())
        if self.learnable_init_z:
            transition_params.append(self.init_z)
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d)
        # blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        blacklist_module_names = ['transition_model.model.dit']
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                    no_decay.add(fpn)
                elif 'bias' in pn:
                    # all biases will not be decayed
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert len(
            param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.cfg.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.lr)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.cfg.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr

    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        nonterminals = batch[-1]
        nonterminals = nonterminals.bool().permute(1, 0)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()
        n_frames = n_frames // self.frame_stack
        # Todo: modify conditions to fit action, current condition drop the first action
        if self.external_cond_dim:
            conditions = batch[1]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1])+45, conditions[:, 0:-1]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
        else:
            conditions = [None for _ in range(n_frames)]

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        if self.learnable_init_z:
            init_z = self.init_z[None].expand(batch_size, *self.z_shape)
        else:
            init_z = torch.zeros(batch_size, *self.z_shape)
            init_z = init_z.to(xs.device)

        return xs, conditions, masks, init_z

    def reweigh_loss(self, loss, weight=None):
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(weight, "(t fs) b ... -> t b fs ..." + " 1" * expand_dim, fs=self.frame_stack)
            loss = loss * weight

        return loss.mean()

    def training_step(self, batch):
        # training step for dynamics
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch)

        n_frames, batch_size, _, *_ = xs.shape

        xs_pred = []
        loss = []
        original_loss = []
        zeta_min = []
        zeta_max = []
        z = init_z
        cum_snr = None
        for t in range(0, n_frames):
            deterministic_t = None
            if random() <= self.gt_cond_prob or (t == 0 and random() <= self.gt_first_frame):
                deterministic_t = 0

            z_next, x_next_pred, l, cum_snr, o_l = self.transition_model(
                z, xs[t], conditions[t], deterministic_t=deterministic_t, cum_snr=cum_snr
            )

            z = z_next
            zeta_min.append(z.min())
            zeta_max.append(z.max())
            xs_pred.append(x_next_pred)
            loss.append(l)
            original_loss.append(o_l)

        xs_pred = torch.stack(xs_pred)
        loss = torch.stack(loss)
        x_loss = self.reweigh_loss(loss, masks)
        loss = x_loss

        original_loss_mean = torch.stack(original_loss).mean()

        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

        output_dict = {
            "loss": loss,
            "original_loss":original_loss_mean,
            "original_loss_list": original_loss,
            "xs_pred": self._unnormalize_x(xs_pred),
            "xs": self._unnormalize_x(xs),
            "zeta_min": zeta_min,
            "zeta_max": zeta_max,
        }

        return output_dict

    @torch.no_grad()
    def init_df_model(self,observation):
        observation = self._normalize_x(xs=observation)
        init_z = self.init_z[None].expand(1, *self.z_shape)
        device = observation.device
        z, _, _, _, _ = self.transition_model(init_z, torch.stack([observation]), torch.tensor([[45]]).float().to(device), deterministic_t=0)
        return z

    @torch.no_grad()
    def step(self,zeta,cur_act, sampling_timesteps):
        z = zeta
        horizon = 1

        chunk = [
            torch.randn((1,) + tuple(self.x_stacked_shape), device=self.device) for _ in range(horizon)
        ]

        pyramid_height = sampling_timesteps + int(horizon * self.uncertainty_scale)
        pyramid = np.zeros((pyramid_height, horizon), dtype=int)
        for m in range(pyramid_height):
            for t in range(horizon):
                pyramid[m, t] = m - int(t * self.uncertainty_scale)
        pyramid = np.clip(pyramid, a_min=0, a_max=sampling_timesteps, dtype=int)

        for m in range(pyramid_height):

            z_chunk = z.detach()
            for t in range(horizon):
                i = min(pyramid[m, t], sampling_timesteps - 1)

                chunk[t], z_chunk = self.transition_model.ddim_sample_step(
                    chunk[t], z_chunk, cur_act, i, sampling_timesteps
                )

                # theoretically, one shall feed new chunk[t] with last z_chunk into transition model again
                # to get the posterior z_chunk, and optionaly, with small noise level k>0 for stablization.
                # However, since z_chunk in the above line already contains info about updated chunk[t] in
                # our simplied math model, we deem it suffice to directly take this z_chunk estimated from
                # last z_chunk and noiser chunk[t]. This saves half of the compute from posterior steps.
                # The effect of the above simplification already contains stablization: we always stablize
                # (ddim_sample_step is never called with noise level k=0 above)
        zeta = z_chunk
        chunk[0] = self._unnormalize_x(chunk[0])
        return zeta, chunk[0]

    @torch.no_grad()
    def validation_step(self, batch):
        if self.calc_crps_sum:
            # repeat batch for crps sum for time series prediction
            batch = [d[None].expand(self.calc_crps_sum, *([-1] * len(d.shape))).flatten(0, 1) for d in batch]

        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch)

        endless_mode = True
        if endless_mode:
            endless_len = 1000
            conditions = conditions.repeat((endless_len // conditions.shape[0] + 2, 1, 1))[: xs.shape[0]+endless_len]
            zeros_tensor = torch.zeros((endless_len, self.frame_stack, self.cfg.x_shape[0], self.cfg.x_shape[1], self.cfg.x_shape[-1]),dtype=xs.dtype).to(xs.device)
            xs = torch.cat((xs, zeros_tensor), dim=0)
            ones_tensor = torch.ones((endless_len * self.frame_stack, 1),dtype=masks.dtype).to(xs.device)
            masks = torch.cat((masks, ones_tensor), dim=0)

        n_frames, batch_size, *_ = xs.shape
        xs_pred = []
        xs_pred_all = []
        z = init_z

        # context
        for t in range(0, self.context_frames // self.frame_stack):
            z, x_next_pred, _, _, _ = self.transition_model(z, xs[t], conditions[t], deterministic_t=0)
            xs_pred.append(x_next_pred)

        # prediction

        from tqdm import tqdm
        with tqdm(total=n_frames) as pbar:
            while len(xs_pred) < n_frames:
                pbar.update(1)
                if self.chunk_size > 0:
                    horizon = min(n_frames - len(xs_pred), self.chunk_size)
                else:
                    horizon = n_frames - len(xs_pred)

                chunk = [
                    torch.randn((batch_size,) + tuple(self.x_stacked_shape), device=self.device) for _ in range(horizon)
                ]

                pyramid_height = self.sampling_timesteps + int(horizon * self.uncertainty_scale)
                pyramid = np.zeros((pyramid_height, horizon), dtype=int)
                for m in range(pyramid_height):
                    for t in range(horizon):
                        pyramid[m, t] = m - int(t * self.uncertainty_scale)
                pyramid = np.clip(pyramid, a_min=0, a_max=self.sampling_timesteps, dtype=int)

                for m in range(pyramid_height):
                    if self.transition_model.return_all_timesteps:
                        xs_pred_all.append(chunk)

                    z_chunk = z.detach()
                    for t in range(horizon):
                        i = min(pyramid[m, t], self.sampling_timesteps - 1)

                        chunk[t], z_chunk = self.transition_model.ddim_sample_step(
                            chunk[t], z_chunk, conditions[len(xs_pred) + t], i
                        )

                        # theoretically, one shall feed new chunk[t] with last z_chunk into transition model again
                        # to get the posterior z_chunk, and optionaly, with small noise level k>0 for stablization.
                        # However, since z_chunk in the above line already contains info about updated chunk[t] in
                        # our simplied math model, we deem it suffice to directly take this z_chunk estimated from
                        # last z_chunk and noiser chunk[t]. This saves half of the compute from posterior steps.
                        # The effect of the above simplification already contains stablization: we always stablize
                        # (ddim_sample_step is never called with noise level k=0 above)

                z = z_chunk
                xs_pred += chunk

        xs_pred = torch.stack(xs_pred)
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweigh_loss(loss, masks)

        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

        xs = self._unnormalize_x(xs)
        xs_pred = self._unnormalize_x(xs_pred)

        if not self.is_spatial:
            if self.transition_model.return_all_timesteps:
                xs_pred_all = [torch.stack(item) for item in xs_pred_all]
                limit = self.transition_model.sampling_timesteps
                for i in np.linspace(1, limit, 5, dtype=int):
                    xs_pred = xs_pred_all[i]
                    xs_pred = self._unnormalize_x(xs_pred)

        self.validation_step_outputs.append((xs_pred.detach().cpu(), xs.detach().cpu()))

        return loss

    def on_validation_epoch_end(self, namespace="validation"):
        if not self.validation_step_outputs:
            return

        self.validation_step_outputs.clear()


    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape).to(xs.device)
        std = self.data_std.reshape(shape).to(xs.device)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape).to(xs.device)
        std = self.data_std.reshape(shape).to(xs.device)
        return xs * std + mean
