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
        "truck",
    ]

    def __init__(self, train: bool, data_dir: str = "data") -> None:
        super().__init__()

        self.dataset = CIFAR10(
            data_dir, train=train, download=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label_idx = self.dataset[index]
        image = image * 2 - 1  # normalize to [-1, 1]
        return {"image": image, "label": Cifar10Dataset.CLASSES[label_idx], "class": label_idx}


class Cifar10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train = None
        self.data_val = None

    def setup(self, stage: Optional[str] = None):
        if self.data_train is None:
            self.data_train = Cifar10Dataset(train=True, data_dir=self.data_dir)

        if self.data_val is None:
            self.data_val = Cifar10Dataset(train=False, data_dir=self.data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
