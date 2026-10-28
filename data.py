"""Official train/test partitions; validation is drawn only from training data."""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class SHDFrames(Dataset[tuple[Tensor, int]]):
    """Bin each SHD sample into 20 equal-duration frames over its own duration.

    Counts, not binary occupancy, are supplied to the digital input projection.
    This demo preprocessing is explicit; exact original SHD preprocessing was
    not supplied with the model and benchmark equivalence is not claimed.
    """

    def __init__(self, path: Path) -> None:
        import h5py

        self.samples: list[Tensor] = []
        with h5py.File(path, "r") as source:
            self.labels = [int(label) for label in source["labels"][:]]
            for times, units in zip(source["spikes/times"], source["spikes/units"]):
                frames = np.zeros((20, 700), dtype=np.float32)
                if len(times):
                    duration = max(float(np.max(times)), np.finfo(np.float32).eps)
                    bins = np.minimum((times / duration * 20).astype(np.int64), 19)
                    np.add.at(frames, (bins, units.astype(np.int64)), 1)
                self.samples.append(torch.from_numpy(frames))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.samples[index], self.labels[index]


def load_data(name: str, root: Path, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    match name:
        case "cifar10":
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            evaluation = transforms.Compose([transforms.ToTensor(), normalize])
            augmentation = transforms.Compose([
                transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize,
            ])
            train = datasets.CIFAR10(str(root), train=True, download=True, transform=augmentation)
            validation = datasets.CIFAR10(str(root), train=True, transform=evaluation)
            test = datasets.CIFAR10(str(root), train=False, download=True, transform=evaluation)
            val_size = 5000
        case "shd":
            train = SHDFrames(root / "shd_train.h5")
            validation = train
            test = SHDFrames(root / "shd_test.h5")
            val_size = max(1, len(train) // 10)
        case _:
            raise ValueError(f"unsupported dataset: {name}")
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed)).tolist()
    return Subset(train, indices[val_size:]), Subset(validation, indices[:val_size]), test
