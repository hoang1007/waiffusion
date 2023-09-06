from typing import List, Optional
from tqdm import tqdm

from src.models.autoencoder import AutoEncoderKL
from .ddpm import DDPM

import torch


class LDM(DDPM):
    def __init__(
        self,
        vae: AutoEncoderKL,
        vae_pretrained_path: str,
        vae_scale_factor: float = 0.18215,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.vae = vae.load_from_checkpoint(vae_pretrained_path)
        self.vae.freeze()
        self.vae.eval()

        self.vae_scale_factor = vae_scale_factor

    def encode_latent(self, x):
        posterior = self.vae.encode(x)
        z = posterior.sample()
        z = self.vae_scale_factor * z

        return z
    
    def decode_latent(self, z):
        z = 1 / self.vae_scale_factor * z
        reconstructed = self.vae.decode(z)

        return reconstructed

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        class_labels = batch.get(self.conditional_stage_key, None)
        z = self.encode_latent(z)
        loss, loss_dict = self(z, class_labels=class_labels)
        return loss, loss_dict

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        classes: Optional[List[int]] = None,
        return_intermediates: bool = False,
    ):
        if classes is not None:
            assert len(classes) == batch_size, "Number of classes must match batch size"
            class_labels = torch.tensor(classes, device=self.device).long()
        else:
            class_labels = None

        latents = torch.randn(
            (batch_size, self.unet.in_channels, *self.unet.sample_size),
            device=self.device,
        )
        intermediates = [self.decode_latent(latents)]

        for i, t in enumerate(tqdm(self.sampler.timesteps, desc="Sampling t")):
            t = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            model_output = self.unet(latents, t, class_labels=class_labels)
            latents = self.sampler.reverse_step(model_output, t, latents)

            if i % self.log_every_t == 0 or i == len(self.sampler.timesteps) - 1:
                intermediates.append(self.decode_latent(latents))

        if return_intermediates:
            return intermediates[-1], intermediates
        return intermediates[-1]
