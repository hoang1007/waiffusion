from contextlib import contextmanager
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.utils import make_grid
from tqdm import tqdm

from src.models.base import BaseModel
from src.models.ema import EMA
from src.models.samplers import BaseSampler
from src.models.unet import Unet
from src.utils.module_utils import load_ckpt


class DDPM(BaseModel):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet: Unet,
        sampler: BaseSampler,
        optimizer: Optional[partial] = None,
        scheduler_config: Optional[Dict] = None,
        num_train_timesteps=1000,
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        use_ema=True,
        first_stage_key="image",
        conditional_stage_key="class",
        channels=3,
        log_every_t=100,
        parameterization="eps",  # all assuming fixed variance schedules
        validation_mode="forward",  # forward: validate noise prediction, reverse: validate image generation or both
        learn_logvar=False,
        logvar_init=0.0,
    ):
        super().__init__(optimizer, scheduler_config)
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.conditional_stage_key = conditional_stage_key
        self.channels = channels
        self.num_train_timesteps = num_train_timesteps
        self.unet = unet
        self.sampler = sampler

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = EMA(self.unet)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.loss_type = loss_type
        self.validation_mode = validation_mode
        if self.validation_mode in ("reverse", "both"):
            self.inception_score = InceptionScore(normalize=True)
            self.fid = FrechetInceptionDistance(feature=192, normalize=True)

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_train_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.unet.parameters())
            self.model_ema.copy_to(self.unet)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.unet.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = load_ckpt(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False)
            if not only_model
            else self.unet.load_state_dict(sd, strict=False)
        )
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

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

        imgs = torch.randn(
            (batch_size, self.unet.in_channels, *self.unet.sample_size),
            device=self.device,
        )
        intermediates = [imgs]

        for i, t in enumerate(tqdm(self.sampler.timesteps, desc="Sampling t")):
            t = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            model_output = self.unet(imgs, t, class_labels=class_labels)
            imgs = self.sampler.reverse_step(model_output, t, imgs)

            if i % self.log_every_t == 0 or i == len(self.sampler.timesteps) - 1:
                intermediates.append(imgs)

        if return_intermediates:
            return imgs, intermediates
        return imgs

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def forward(self, x, class_labels=None, context=None):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_train_timesteps, (x.shape[0],), device=self.device).long()
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
        elif self.validation_mode == "both":
            self._forward_validate(batch, batch_idx)
            self._reverse_validate(batch, batch_idx)
        else:
            raise NotImplementedError(f"Validation mode {self.validation_mode} not implemented")

    @torch.no_grad()
    def _forward_validate(self, batch, batch_idx):
        """Validate the diffusion forward process."""
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
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
        if self.use_ema:
            self.model_ema(self.unet)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        class_labels = batch.get(self.conditional_stage_key, None)
        N = min(x.shape[0], N)
        class_labels = class_labels[:N] if class_labels is not None else None
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
                    batch_size=N, classes=class_labels, return_intermediates=True
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
