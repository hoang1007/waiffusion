from typing import Any, List

from src.models.attention import AttentionType

import torch
from torch import nn, Tensor

from src.models.base import BaseModel
from .vae_blocks import Encoder, Decoder, DiagonalGaussianDistribution


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
    
    def get_loss(self, input, recon, posterior):
        rec_loss = nn.functional.l1_loss(input, recon)
        # rec_loss = nn.functional.binary_cross_entropy_with_logits(recon, input)    

        kl_loss = -0.5 * (1 + posterior.logvar - posterior.mean ** 2 - posterior.var).sum(dim=1)
        kl_loss = kl_loss.mean()

        return {
            "rec_loss": rec_loss,
            "kl_loss": kl_loss,
            "loss": rec_loss + 1e-4 * kl_loss,
        }

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        dec, posterior = self(x)

        loss_dict = self.get_loss(x, dec, posterior)

        self.log_dict(
            {f"train/{k}": v.item() for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True
        )
        
        return loss_dict["loss"]
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        dec, posterior = self(x)
        loss_dict = self.get_loss(x, dec, posterior)

        self.log_dict(
            {f"val/{k}": v.item() for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True
        )

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
        return self.init_optimizers(self.parameters())
