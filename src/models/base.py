from functools import partial
from typing import Dict, Optional

from pytorch_lightning import LightningModule


class BaseModel(LightningModule):
    def __init__(
        self, optimizer: Optional[partial] = None, scheduler_config: Optional[Dict] = None
    ):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config

        self.save_hyperparameters()

    def init_optimizers(self, params):
        if self.optimizer is None:
            raise ValueError("Optimizer not defined.")
        else:
            opt = self.optimizer(params=params)

            if self.scheduler_config is not None:
                scheduler = self.scheduler_config.pop("scheduler")(optimizer=opt)

                print(self.scheduler_config)

                return {
                    "optimizer": opt,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self.scheduler_config,
                    },
                }
            else:
                return opt
