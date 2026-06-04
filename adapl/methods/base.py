"""Base interfaces for federated learning methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import Namespace
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Sequence

from adapl.methods.metadata import MethodInfo

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader


@dataclass
class ClientUpdate:
    client_id: int
    state_dict: OrderedDict[str, torch.Tensor]
    train_loss: float
    num_examples: int
    metadata: Dict[str, object] = field(default_factory=dict)


class FederatedMethod(ABC):
    """Algorithm hook points used by the shared experiment runner."""

    info: MethodInfo

    def __init__(self, args: Namespace) -> None:
        self.args = args

    @property
    def display_name(self) -> str:
        return self.info.display_name

    def startup_lines(self) -> List[str]:
        return []

    def config_rows(self) -> List[tuple[str, str, object]]:
        return [
            ("method", "name", self.info.canonical_name),
            ("method", "display_name", self.info.display_name),
            ("method", "description", self.info.description),
        ]

    def config_payload(self) -> Dict[str, object]:
        return {
            "name": self.info.canonical_name,
            "display_name": self.info.display_name,
            "description": self.info.description,
        }

    def begin_round(self, round_idx: int, selected_clients: Sequence[int]) -> None:
        """Optional hook for methods with round-dependent state."""

    @abstractmethod
    def train_client(
        self,
        client_id: int,
        model_fn: Callable[[], nn.Module],
        global_state: OrderedDict[str, torch.Tensor],
        train_loader: DataLoader,
        device: torch.device,
    ) -> ClientUpdate:
        """Run one selected client's local update."""

    @abstractmethod
    def aggregate(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> OrderedDict[str, torch.Tensor]:
        """Aggregate selected client updates into the next global state."""
