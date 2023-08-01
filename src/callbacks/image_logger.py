from typing import Any, Optional, Dict, Literal

import math
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from torchvision.utils import make_grid


class ImageLogger(Callback):
    def __init__(self, 
            frequency: int = 1,
            logging_interval: Literal['step', 'epoch'] = 'step',
            max_images: int = 16,
            clamp: bool = True,
            disabled: bool = False,
            log_images_kwargs: Optional[Dict] = None
        ):
        super().__init__()

        self.frequency = frequency
        self.logging_interval = logging_interval
        self.max_images = max_images
        self.clamp = clamp
        self.disabled = disabled
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}

    def __make_image_grid(self, images: torch.Tensor):
        nrow = math.ceil(math.sqrt(images.size(1)))
        return make_grid(images, nrow=nrow)

    def log_img(self, pl_module: pl.LightningModule, batch: Any, batch_idx: int, split="train"):
        check_idx = pl_module.current_epoch if self.logging_interval == 'epoch' else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                if images[k].dim() == 4:
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                    images[k] = self.__make_image_grid(images[k])

                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            # logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            # logger_log_images(pl_module, images, pl_module.global_step, split)
            pl_module.logger.log_image(key=f'{split}/images', images=list(images.values()), caption=list(images.keys()))

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.frequency == 0

    # def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
    #     if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
    #         self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.disabled:
            self.log_batch_idx = [
                torch.randint(0, num_val_batches, (1,)).item()
                for num_val_batches in trainer.num_val_batches
            ]

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not self.disabled and pl_module.global_step > 0 and batch_idx == self.log_batch_idx[dataloader_idx]:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)
