from typing import Optional, List, Union
from tqdm import tqdm

import torch
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

        posterior = self.vae.encode(x)
        z = posterior.sample()

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
        intermediates = [self.vae.decode(latents)]

        for i, t in enumerate(tqdm(self.sampler.timesteps, desc="Sampling t")):
            t = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            model_output = self.unet(latents, t, context=prompt_embeds)
            latents = self.sampler.reverse_step(model_output, t, latents)

            if i % self.log_every_t == 0 or i == len(self.sampler.timesteps) - 1:
                intermediates.append(self.vae.decode(latents))

        if return_intermediates:
            return intermediates[-1], intermediates
        return intermediates[-1]
