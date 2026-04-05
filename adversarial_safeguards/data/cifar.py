from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from adversarial_safeguards.config import CIFAR_MEAN, CIFAR_STD


def cifar10_transforms(train: bool) -> transforms.Compose:
    t: list = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
    if train:
        t = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ] + t
    return transforms.Compose(t)


def get_cifar10_loader(
    batch_size: int,
    data_dir: str = "./data",
    num_workers: int = 0,
    train: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    ds = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=cifar10_transforms(train=train),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin_memory)
