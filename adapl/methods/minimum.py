"""Minimum-budget DP-FedAvg baseline."""

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
from adapl.methods.metadata import MINIMUM_INFO
from adapl.privacy.config import PrivacyConfig, build_minimum_privacy_config
from adapl.privacy.mechanisms import privatize_client_update


class MinimumDPFedAvg(FederatedMethod):
    info = MINIMUM_INFO

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.privacy_config: PrivacyConfig = build_minimum_privacy_config(args)
        self.noise_generator = torch.Generator().manual_seed(args.seed + 10_000)

    def startup_lines(self) -> list[str]:
        epsilon = (
            f"{self.privacy_config.epsilon}"
            if self.privacy_config.epsilon is not None
            else "not reported"
        )
        return [
            "DP is enabled at the client update level.",
            (
                "Min privacy config: "
                f"epsilon_min={epsilon}, "
                f"delta={self.privacy_config.delta}, "
                f"clipping_norm={self.privacy_config.clipping_norm}, "
                f"noise_multiplier={self.privacy_config.noise_multiplier:.6f}, "
                f"noise_std={self.privacy_config.noise_std:.6f}"
            ),
        ]

    def config_rows(self) -> list[tuple[str, str, object]]:
        rows = super().config_rows()
        budget_count = (
            len(self.privacy_config.privacy_budgets)
            if self.privacy_config.privacy_budgets
            else 0
        )
        rows.extend(
            [
                ("privacy", "dp_enabled", True),
                ("privacy", "mechanism", self.privacy_config.mechanism),
                ("privacy", "epsilon_min", self.privacy_config.epsilon),
                ("privacy", "delta", self.privacy_config.delta),
                ("privacy", "clipping_norm", self.privacy_config.clipping_norm),
                ("privacy", "noise_multiplier", self.privacy_config.noise_multiplier),
                ("privacy", "noise_std", self.privacy_config.noise_std),
                ("privacy", "noise_source", self.privacy_config.noise_source),
                ("privacy", "privacy_budget_count", budget_count),
            ]
        )
        return rows

    def config_payload(self) -> dict[str, object]:
        payload = super().config_payload()
        payload["dp_enabled"] = True
        payload["privacy"] = {
            "mechanism": self.privacy_config.mechanism,
            "epsilon_min": self.privacy_config.epsilon,
            "delta": self.privacy_config.delta,
            "clipping_norm": self.privacy_config.clipping_norm,
            "noise_multiplier": self.privacy_config.noise_multiplier,
            "noise_std": self.privacy_config.noise_std,
            "noise_source": self.privacy_config.noise_source,
            "privacy_budget_count": (
                len(self.privacy_config.privacy_budgets)
                if self.privacy_config.privacy_budgets
                else 0
            ),
        }
        return payload

    def train_client(
        self,
        client_id: int,
        model_fn: Callable[[], nn.Module],
        global_state: OrderedDict[str, torch.Tensor],
        train_loader: DataLoader,
        device: torch.device,
    ) -> ClientUpdate:
        local_state, client_loss, client_size = train_client_sgd(
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
        privatized = privatize_client_update(
            global_state=global_state,
            local_state=local_state,
            clipping_norm=self.privacy_config.clipping_norm,
            noise_std=self.privacy_config.noise_std,
            generator=self.noise_generator,
        )
        return ClientUpdate(
            client_id=client_id,
            state_dict=privatized.state_dict,
            train_loss=client_loss,
            num_examples=client_size,
            metadata={
                "update_norm": privatized.update_norm,
                "clipped_norm": privatized.clipped_norm,
                "clip_factor": privatized.clip_factor,
                "noise_std": privatized.noise_std,
                "epsilon_min": self.privacy_config.epsilon,
                "delta": self.privacy_config.delta,
            },
        )

    def aggregate(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> OrderedDict[str, torch.Tensor]:
        return fedavg_aggregate(
            [update.state_dict for update in client_updates],
            [update.num_examples for update in client_updates],
        )
