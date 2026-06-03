"""Dataset loading and federated partition utilities."""

from adapl.data.cifar100 import load_cifar100
from adapl.data.partitioning import (
    build_client_partitions,
    compute_client_label_distribution,
    get_dataset_targets,
    log_client_label_distribution,
)

__all__ = [
    "build_client_partitions",
    "compute_client_label_distribution",
    "get_dataset_targets",
    "load_cifar100",
    "log_client_label_distribution",
]
