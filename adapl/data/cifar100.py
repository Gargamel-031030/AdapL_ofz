"""CIFAR-100 dataset loading."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from adapl.constants import CIFAR100_MEAN, CIFAR100_STD


def build_cifar100_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return train_transform, test_transform


def maybe_limit_dataset(dataset: Dataset, limit: Optional[int], seed: int) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def load_cifar100(
    data_dir: str,
    seed: int,
    limit_train: Optional[int],
    limit_test: Optional[int],
) -> Tuple[Dataset, Dataset]:
    """Load CIFAR-100, downloading it into data_dir on the current machine."""
    os.makedirs(data_dir, exist_ok=True)
    train_transform, test_transform = build_cifar100_transforms()
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )
    train_dataset = maybe_limit_dataset(train_dataset, limit_train, seed)
    test_dataset = maybe_limit_dataset(test_dataset, limit_test, seed)
    return train_dataset, test_dataset
