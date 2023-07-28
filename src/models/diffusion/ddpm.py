from contextlib import contextmanager

import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.utils import make_grid
from tqdm import tqdm

from src.models.ema import EMA
from src.models.unet import Unet
from src.utils.module_utils import default, load_ckpt

from .modules import DiffusionWrapper
from src.samplers import BaseSampler


class DDPM(LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        model: Unet,
        sampler: BaseSampler,
        num_timesteps=1000,
        loss_type="l2",
        learning_rate=1e-5,
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        image_size=256,
        channels=3,
        log_every_t=100,
        original_elbo_weight=0.0,
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        validation_mode="forward",  # forward: validate noise prediction, reverse: validate image generation or both
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.learning_rate = learning_rate
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.num_timesteps = num_timesteps
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(model, conditioning_key)
        self.sampler = sampler

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = EMA(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.loss_type = loss_type
        self.validation_mode = validation_mode
        if self.validation_mode in ("reverse", "both"):
            self.inception_score = InceptionScore(normalize=True)
            self.fid = FrechetInceptionDistance(feature=192, normalize=True)

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
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
            else self.model.load_state_dict(sd, strict=False)
        )
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    # def q_mean_variance(self, x_start, t):
    #     """Get the distribution q(x_t | x_0).

    #     :param x_start: the [N x C x ...] tensor of noiseless inputs.
    #     :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
    #     :return: A tuple (mean, variance, log_variance), all of x_start's shape.
    #     """
    #     mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    #     variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
    #     log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
    #     return mean, variance, log_variance

    def sample(self, batch_size: int, shape: tuple, return_intermediates: bool = False):
        imgs = torch.randn((batch_size, *shape), device=self.device)
        intermediates = [imgs]

        for i in tqdm(
            reversed(range(0, self.sampler.num_timesteps)),
            desc="Sampling t",
            total=self.sampler.num_timesteps,
        ):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            model_output = self.model(imgs, t)
            imgs = self.sampler.reverse_step(model_output, t, imgs)
            
            if i % self.log_every_t == 0 or i == self.sampler.num_timesteps - 1:
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

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.sampler.step(sample=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(
                f"Parameterization {self.parameterization} not yet supported"
            )

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        # TODO: add loss vlb
        loss = loss_simple

        # loss_vlb = (self.lvlb_weights[t] * loss).mean()
        # loss_dict.update({f"{log_prefix}/loss_vlb": loss_vlb})

        # loss = loss_simple + self.original_elbo_weight * loss_vlb

        # loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        # x = rearrange(x, "b h w c -> b c h w")
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
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

    def validation_epoch_end(self, outputs):
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
            self.model_ema(self.model)

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
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
