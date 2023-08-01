from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


class Cifar10Dataset(Dataset):
    CLASSES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]
    def __init__(self, train: bool, data_dir: str = "data") -> None:
        super().__init__()

        self.dataset = CIFAR10(
            data_dir, train=train, download=True, transform=transforms.ToTensor()
        )

        self.label2class = {label: i for i, label in enumerate(self.CLASSES)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = image * 2 - 1  # normalize to [-1, 1]
        return {"image": image, "label": label, "class": self.label2class[label]}


class Cifar10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None

    def setup(self, stage: Optional[str] = None):
        if self.data_train is None:
            self.data_train = Cifar10Dataset(train=True, data_dir=self.hparams.data_dir)

        if self.data_val is None:
            self.data_val = Cifar10Dataset(train=False, data_dir=self.hparams.data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
