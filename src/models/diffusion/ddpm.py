from warnings import warn
from functools import partial
from typing import Dict, List, Optional, Literal

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.utils import make_grid
from tqdm import tqdm

from src.models.base import BaseModel
from src.models.samplers import BaseSampler
from src.models.unet import Unet
from torch_ema import ExponentialMovingAverage


class DDPM(BaseModel):
    def __init__(
        self,
        unet: Unet,
        sampler: BaseSampler,
        optimizer: Optional[partial] = None,
        scheduler_config: Optional[Dict] = None,
        num_train_timesteps: int = 1000,
        first_stage_key: str = "image",
        conditional_stage_key: str = "class",
        parameterization: Literal['eps', 'x0'] = "eps",  # all assuming fixed variance schedules
        validation_mode: Literal['forward', 'reverse', 'all'] = "forward",  # forward: validate noise prediction, reverse: validate image generation or both
        learn_logvar: bool = False,
        logvar_init: float = 0.0,
        use_ema: bool = True,
        compile_unet: bool = False
    ):
        super().__init__(optimizer, scheduler_config)

        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.first_stage_key = first_stage_key
        self.conditional_stage_key = conditional_stage_key
        self.num_train_timesteps = num_train_timesteps
        self.unet = unet
        self.sampler = sampler

        self.use_ema = use_ema
        if self.use_ema:
            self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.999)
        
        self.compile_unet = compile_unet
        if self.compile_unet:
            try:
                self.unet = torch.compile(self.unet, mode='reduce-overhead', fullgraph=True)
            except Exception as e:
                warn(f'Cannot compile {self.unet.__class__.__name__} model. Got error: {e}')

        self.validation_mode = validation_mode
        if self.validation_mode in ("reverse", "all"):
            self.inception_score = InceptionScore(normalize=True)
            self.fid = FrechetInceptionDistance(feature=192, normalize=True)

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_train_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        classes: Optional[List[int]] = None,
        return_intermediates: bool = False,
        n_intermediates: int = 5
    ):
        save_steps = len(self.sampler.timesteps) // n_intermediates
        if save_steps == 0:
            warn(f'Number of intermediate images is set too large!. Got {n_intermediates} / {(len(self.sampler.timesteps))}')
            save_steps = 1

        if classes is not None:
            assert len(classes) == batch_size, "Number of classes must match batch size"
            class_labels = torch.tensor(classes, device=self.device).long()
        else:
            class_labels = None

        imgs = torch.randn(
            (batch_size, self.unet.in_channels, *self.unet.sample_size),
            device=self.device,
        )
        intermediates = [imgs]

        for i, t in enumerate(tqdm(self.sampler.timesteps, desc="Sampling t")):
            t = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            model_output = self.unet(imgs, t, class_labels=class_labels)
            imgs = self.sampler.reverse_step(model_output, t, imgs)

            if i % save_steps == 0 or i == len(self.sampler.timesteps) - 1:
                intermediates.append(imgs)

        if return_intermediates:
            return imgs, intermediates
        return imgs

    def get_loss(self, pred, target, mean=True):
        if mean:
            loss = torch.nn.functional.mse_loss(target, pred)
        else:
            loss = torch.nn.functional.mse_loss(target, pred, reduction="none")

        return loss

    def forward(self, x, class_labels=None, context=None):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_train_timesteps, (x.size(0),), device=self.device).long()
        noise = torch.randn_like(x)
        x_noisy = self.sampler.step(sample=x, t=t, noise=noise)
        model_out = self.unet(x_noisy, t, class_labels=class_labels, context=context)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x
        else:
            raise NotImplementedError(
                f"Parameterization {self.parameterization} not yet supported"
            )

        loss = self.get_loss(model_out, target)
        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss": loss.item()})

        return loss, loss_dict

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        # x = rearrange(x, "b h w c -> b c h w")
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        class_labels = batch.get(self.conditional_stage_key, None)
        loss, loss_dict = self(x, class_labels=class_labels)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.validation_mode == "forward":
            self._forward_validate(batch, batch_idx)
        elif self.validation_mode == "reverse":
            self._reverse_validate(batch, batch_idx)
        elif self.validation_mode == "all":
            self._forward_validate(batch, batch_idx)
            self._reverse_validate(batch, batch_idx)
        else:
            raise NotImplementedError(f"Validation mode {self.validation_mode} not implemented")

    @torch.no_grad()
    def _forward_validate(self, batch, batch_idx):
        """Validate the diffusion forward process."""
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema.average_parameters():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    @torch.no_grad()
    def _reverse_validate(self, batch, batch_idx):
        """Validate the diffusion reverse process."""
        imgs = self.get_input(batch, self.first_stage_key)
        samples = self.sample(imgs.size(0))

        self.fid.update(imgs, real=True)
        self.fid.update(samples, real=False)

        self.inception_score.update(samples)

    def on_validation_epoch_end(self):
        if self.validation_mode in ("reverse", "both"):
            fid = self.fid.compute()
            iscore_mean, iscore_std = self.inception_score.compute()

            self.log_dict(
                {
                    "fid": fid,
                    "inception_score_mean": iscore_mean,
                    "inception_score_std": iscore_std,
                }
            )

    def on_train_batch_end(self, *args, **kwargs):
        self.ema.update()

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, n_intermediates=5, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        class_labels = batch.get(self.conditional_stage_key, None)
        N = min(x.shape[0], N)
        class_labels = class_labels[:N] if class_labels is not None else None
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        log_steps = max(self.sampler.num_inference_timesteps // n_intermediates, 1)

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for i, t in enumerate(reversed(self.sampler.timesteps)):
            if i % log_steps == 0 or i == len(self.sampler.timesteps) - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.sampler.step(sample=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema.average_parameters():
                samples, denoise_row = self.sample(
                    batch_size=N,
                    classes=class_labels,
                    return_intermediates=True,
                    n_intermediates=n_intermediates
                )

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        params = list(self.unet.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]

        return self.init_optimizers(params)
