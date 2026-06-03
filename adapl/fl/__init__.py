"""Common federated learning utilities."""

from adapl.fl.aggregation import fedavg_aggregate
from adapl.fl.client import train_client_sgd
from adapl.fl.evaluation import evaluate
from adapl.fl.sampling import select_clients

__all__ = [
    "evaluate",
    "fedavg_aggregate",
    "select_clients",
    "train_client_sgd",
]
