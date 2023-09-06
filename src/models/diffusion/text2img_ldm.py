from typing import Optional, List, Union
from tqdm import tqdm

import torch
import numpy as np
from einops import repeat
from .ldm import LDM
from src.models.condition_encoders import CLIPEncoder


class Text2ImgLDM(LDM):
    def __init__(
        self,
        text_encoder: CLIPEncoder,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.text_encoder = text_encoder
        self.text_encoder.eval().freeze()

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        prompts = batch.get(self.conditional_stage_key, None)

        z = self.encode_latent(x)

        prompt_embeds = self.text_encoder(prompts)
        loss, loss_dict = self(z, context=prompt_embeds)
        return loss, loss_dict

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        prompts: Union[str, List[str]],
        return_intermediates: bool = False,
    ):
        if isinstance(prompts, list):
            assert len(prompts) == batch_size, "Number of prompts must match to batch size"
            prompt_embeds = self.text_encoder(prompts)
        elif isinstance(prompts, str):
            prompts = [prompts]
            prompt_embeds = self.text_encoder(prompts)
            prompt_embeds = torch.repeat_interleave(prompt_embeds, repeats=batch_size, dim=0)

        latents = torch.randn(
            (batch_size, self.unet.in_channels, *self.unet.sample_size),
            device=self.device,
        )
        intermediates = [self.decode_latent(latents)]

        for i, t in enumerate(tqdm(self.sampler.timesteps, desc="Sampling t")):
            t = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            model_output = self.unet(latents, t, context=prompt_embeds)
            latents = self.sampler.reverse_step(model_output, t, latents)

            if i % self.log_every_t == 0 or i == len(self.sampler.timesteps) - 1:
                intermediates.append(self.decode_latent(latents))

        if return_intermediates:
            return intermediates[-1], intermediates
        return intermediates[-1]

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        prompts = batch.get(self.conditional_stage_key, None)
        N = min(x.shape[0], N)
        if prompts is not None and isinstance(prompts, list):
            prompts = prompts[:N]
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for i, t in enumerate(reversed(self.sampler.timesteps)):
            if i % self.log_every_t == 0 or i == len(self.sampler.timesteps) - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.sampler.step(sample=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(
                    batch_size=N, prompts=prompts, return_intermediates=True
                )

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
