import torch
import torch.nn.functional as F
from torch import nn
from .model import Encoder, Decoder
from .distributions import DiagonalGaussianDistribution
from .contperceptual import LPIPSWithDiscriminator

class AutoencoderKL(nn.Module):
    def __init__(self,
                 image_key="image",
                 ):
        super().__init__()
        # ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 64, 'in_channels': 3, 'out_ch': 3, 'ch': 64,
        #             'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
        ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 128, 'in_channels': 3, 'out_ch': 3, 'ch': 128,
                    'ch_mult': [1, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
        embed_dim = 4
        self.image_key = image_key
        self.learning_rate = 4.5e-6
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = LPIPSWithDiscriminator(disc_start=0,kl_weight=1e-6,disc_weight=0)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.global_step = 0

    def set_global_step(self,step):
        self.global_step = step

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def get_reconstructions(self,batch):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        return reconstructions
    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            return aeloss,log_dict_ae

        # if optimizer_idx == 1:
        #     # train the discriminator
        #     discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
        #                                         last_layer=self.get_last_layer(), split="train")
        #     return discloss


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        quant_conv_params = sum(p.numel() for p in self.quant_conv.parameters())
        post_quant_conv_params = sum(p.numel() for p in self.post_quant_conv.parameters())
        print('encoder params:',encoder_params)
        print('decoder params:',decoder_params)
        print('qc params:',quant_conv_params)
        print('pqc_params:',post_quant_conv_params)
        print("total params:", encoder_params + decoder_params + quant_conv_params + post_quant_conv_params)
        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x