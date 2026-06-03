"""Privacy-free FedAvg baseline."""

from __future__ import annotations

from argparse import Namespace
from collections import OrderedDict
from typing import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from adapl.fl.aggregation import fedavg_aggregate
from adapl.fl.client import train_client_sgd
from adapl.methods.base import ClientUpdate, FederatedMethod
from adapl.methods.metadata import PRIVACY_FREE_INFO


class PrivacyFreeFedAvg(FederatedMethod):
    info = PRIVACY_FREE_INFO

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        if not getattr(args, "no_dp", True):
            raise ValueError("PF / PrivacyFree requires --no_dp.")

    def startup_lines(self) -> list[str]:
        return ["DP, clipping, noise, epsilon, delta, and accountants are disabled."]

    def config_rows(self) -> list[tuple[str, str, object]]:
        rows = super().config_rows()
        rows.append(("privacy", "dp_enabled", False))
        return rows

    def config_payload(self) -> dict[str, object]:
        payload = super().config_payload()
        payload["dp_enabled"] = False
        return payload

    def train_client(
        self,
        client_id: int,
        model_fn: Callable[[], nn.Module],
        global_state: OrderedDict[str, torch.Tensor],
        train_loader: DataLoader,
        device: torch.device,
    ) -> ClientUpdate:
        client_state, client_loss, client_size = train_client_sgd(
            model_fn=model_fn,
            global_state=global_state,
            train_loader=train_loader,
            local_steps=self.args.local_steps,
            local_epochs=self.args.local_epochs,
            local_update_mode=self.args.local_update_mode,
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            device=device,
        )
        return ClientUpdate(
            client_id=client_id,
            state_dict=client_state,
            train_loss=client_loss,
            num_examples=client_size,
        )

    def aggregate(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> OrderedDict[str, torch.Tensor]:
        return fedavg_aggregate(
            [update.state_dict for update in client_updates],
            [update.num_examples for update in client_updates],
        )
