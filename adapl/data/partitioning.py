"""Client data partitioning utilities."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def iid_partition(dataset: Dataset, num_clients: int, seed: int) -> List[Subset]:
    if num_clients <= 0:
        raise ValueError("--num_clients must be positive.")
    if num_clients > len(dataset):
        raise ValueError("--num_clients cannot exceed the number of train samples.")

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(dataset), generator=generator).tolist()

    base_size = len(dataset) // num_clients
    remainder = len(dataset) % num_clients
    partitions = []
    cursor = 0
    for client_id in range(num_clients):
        client_size = base_size + (1 if client_id < remainder else 0)
        client_indices = shuffled_indices[cursor : cursor + client_size]
        partitions.append(Subset(dataset, client_indices))
        cursor += client_size
    return partitions


def get_dataset_targets(dataset: Dataset) -> List[int]:
    if isinstance(dataset, Subset):
        parent_targets = get_dataset_targets(dataset.dataset)
        return [int(parent_targets[int(index)]) for index in dataset.indices]
    if hasattr(dataset, "targets"):
        return [int(target) for target in dataset.targets]
    if hasattr(dataset, "labels"):
        return [int(target) for target in dataset.labels]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def ensure_min_client_samples(
    client_indices: List[List[int]],
    min_samples_per_client: int,
    rng: np.random.Generator,
) -> None:
    for client_id in range(len(client_indices)):
        while len(client_indices[client_id]) < min_samples_per_client:
            donor_id = max(
                range(len(client_indices)),
                key=lambda idx: len(client_indices[idx]),
            )
            if len(client_indices[donor_id]) <= min_samples_per_client:
                raise RuntimeError(
                    "Unable to rebalance non-iid partition without creating an "
                    "empty client."
                )
            donor_position = int(rng.integers(len(client_indices[donor_id])))
            client_indices[client_id].append(
                client_indices[donor_id].pop(donor_position)
            )


def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 1,
) -> List[Subset]:
    if num_clients <= 0:
        raise ValueError("--num_clients must be positive.")
    if num_clients > len(dataset):
        raise ValueError("--num_clients cannot exceed the number of train samples.")
    if alpha <= 0:
        raise ValueError("--dirichlet_alpha must be positive.")

    rng = np.random.default_rng(seed)
    targets = np.asarray(get_dataset_targets(dataset), dtype=np.int64)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for class_id in sorted(np.unique(targets).tolist()):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        class_splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(int(index) for index in split.tolist())

    ensure_min_client_samples(client_indices, min_samples_per_client, rng)
    for indices in client_indices:
        rng.shuffle(indices)

    return [Subset(dataset, indices) for indices in client_indices]


def build_client_partitions(
    dataset: Dataset,
    num_clients: int,
    partition: str,
    dirichlet_alpha: float,
    seed: int,
) -> List[Subset]:
    if partition == "iid":
        return iid_partition(dataset, num_clients, seed)
    if partition in {"dirichlet", "non-iid"}:
        return dirichlet_partition(dataset, num_clients, dirichlet_alpha, seed)
    raise ValueError(f"Unsupported partition: {partition}")


def compute_client_label_distribution(
    client_datasets: Sequence[Dataset],
    num_classes: int,
) -> List[Dict[str, object]]:
    distribution = []
    for client_id, dataset in enumerate(client_datasets):
        targets = get_dataset_targets(dataset)
        class_counts = Counter(targets)
        label_counts = {
            str(class_id): int(class_counts.get(class_id, 0))
            for class_id in range(num_classes)
        }
        distribution.append(
            {
                "client_id": client_id,
                "total_samples": len(targets),
                "num_classes": sum(1 for count in label_counts.values() if count > 0),
                "label_counts": label_counts,
            }
        )
    return distribution


def log_client_label_distribution(
    distribution: Sequence[Dict[str, object]],
) -> None:
    print("Client label distribution summary:")
    for item in distribution:
        print(
            f"  client {int(item['client_id']):02d}: "
            f"samples={int(item['total_samples'])}, "
            f"classes={int(item['num_classes'])}"
        )
