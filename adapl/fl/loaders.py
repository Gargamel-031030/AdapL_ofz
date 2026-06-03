"""DataLoader builders."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


def make_train_loaders(
    client_datasets: Sequence[Dataset],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> List[DataLoader]:
    loaders = []
    for client_id, dataset in enumerate(client_datasets):
        generator = torch.Generator().manual_seed(seed + client_id)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                generator=generator,
            )
        )
    return loaders


def make_test_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
