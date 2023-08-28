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
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.vae = vae.load_from_checkpoint(vae_pretrained_path)
        self.vae.freeze()
        self.vae.eval()

    def get_input(self, batch, key):
        x = batch[key]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        # x = rearrange(x, "b h w c -> b c h w")
        x = x.contiguous().float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        class_labels = batch.get(self.conditional_stage_key, None)
        posterior = self.vae.encode(x)
        z = posterior.sample()
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
        intermediates = [self.vae.decode(latents)]

        for i, t in enumerate(tqdm(self.sampler.timesteps, desc="Sampling t")):
            t = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            model_output = self.unet(latents, t, class_labels=class_labels)
            latents = self.sampler.reverse_step(model_output, t, latents)

            if i % self.log_every_t == 0 or i == len(self.sampler.timesteps) - 1:
                intermediates.append(self.vae.decode(latents))

        if return_intermediates:
            return intermediates[-1], intermediates
        return intermediates[-1]
