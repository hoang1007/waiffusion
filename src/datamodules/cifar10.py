from typing import Callable, Optional

from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import CIFAR10


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

    def __init__(self, data_dir: str = "data", transform: Optional[Callable] = None):
        super().__init__()

        datasets = [
            CIFAR10(data_dir, train=train, download=True, transform=transform)
            for train in (True, False)
        ]

        self.dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label_idx = self.dataset[index]
        return {
            "image": image,
            "label": Cifar10Dataset.CLASSES[label_idx],
            "class": label_idx,
        }
