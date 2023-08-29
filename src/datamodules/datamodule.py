from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Generator


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        generator: Optional[Generator] = None
    ):
        super().__init__()

        self.__dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.generator = generator
        self.train_size, self.val_size = self.__get_train_val_size(train_ratio, val_ratio)

        self.data_train = None
        self.data_val = None

    def __get_train_val_size(self, train_ratio, val_ratio):
        assert not (train_ratio is None and val_ratio is None), f'{train_ratio} and {val_ratio} cannot be same None'
        if train_ratio is None:
            train_ratio = 1 - val_ratio
        elif val_ratio is None:
            val_ratio = 1 - train_ratio
        else:
            assert train_ratio + val_ratio == 1, f'{train_ratio} + {val_ratio} must be equal to 1'
        
        train_size = int(len(self.__dataset) * train_ratio)
        val_size = len(self.__dataset) - train_size

        return train_size, val_size

    def setup(self, stage: Optional[str] = None):
        if self.data_train is None or self.data_val is None:
            self.data_train, self.data_val = random_split(self.__dataset, [self.train_size, self.val_size], generator=self.generator)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )