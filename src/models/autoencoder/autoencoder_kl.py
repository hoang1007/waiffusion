from typing import Any, List, Optional

from src.models.attention import AttentionType

import torch
from torch import nn, Tensor

from src.models.base import BaseModel
from .vae_blocks import Encoder, Decoder, DiagonalGaussianDistribution
from src.losses.lpips import LPIPSWithDiscriminator


class AutoEncoderKL(BaseModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_channels: List[int] = [64],
        latent_channels: int = 4,
        attention_levels: List[int] = [],
        attn_type: AttentionType= "standard",
        num_attn_heads: int = 1,
        channels_per_head: int = -1,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        image_key: str = "image",
        loss_fn: Optional[LPIPSWithDiscriminator] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.image_key = image_key

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            hidden_channels=hidden_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            double_z=True,
            dropout=dropout,
            num_attn_heads=num_attn_heads,
            channels_per_head=channels_per_head,
            attn_type=attn_type,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            attn_type=attn_type,
            num_attn_heads=num_attn_heads,
            channels_per_head=channels_per_head,
            dropout=dropout,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

        # disable to training with multiple optimizers
        self.automatic_optimization = False
        self.loss_fn = loss_fn

    def encode(self, x: Tensor):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        return posterior
    
    def decode(self, z: Tensor):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        return dec

    def forward(self, sample: Tensor, sample_posterior: bool = True):
        posterior = self.encode(sample)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decode(z)

        return dec, posterior
    
    def get_input(self, batch, key):
        x = batch[key]
        return x
    
    # def get_loss(self, input, recon, posterior):
    #     rec_loss = nn.functional.mse_loss(input, recon)
    #     # rec_loss = nn.functional.binary_cross_entropy_with_logits(recon, input)    

    #     kl_loss = -0.5 * (1 + posterior.logvar - posterior.mean ** 2 - posterior.var)
    #     kl_loss = kl_loss.mean()

    #     return {
    #         "rec_loss": rec_loss,
    #         "kl_loss": kl_loss,
    #         "loss": rec_loss + 1e-4 * kl_loss,
    #     }

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        dec, posterior = self(x)

        opt_g, opt_d = self.optimizers()
        # train autoencoder
        # self.toggle_optimizer(opt_g)
        aeloss, log_dict_ae = self.loss_fn(x, dec, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        opt_g.zero_grad()
        self.manual_backward(aeloss)
        opt_g.step()
        # self.untoggle_optimizer(opt_g)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        # train discriminator
        # self.toggle_optimizer(opt_d)
        discloss, log_dict_disc = self.loss_fn(x, dec, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
        opt_d.zero_grad()
        self.manual_backward(discloss)
        opt_d.step()
        # self.untoggle_optimizer(opt_d)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        dec, posterior = self(x)
        
        aeloss, log_dict_ae = self.loss_fn(x, dec, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss_fn(x, dec, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def configure_optimizers(self):
        lr = 4.5e-6
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss_fn.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
