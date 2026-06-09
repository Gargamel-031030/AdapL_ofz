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
from adapl.privacy.accounting import (
    ClientPrivacyBudgetManager,
    gaussian_noise_multiplier,
)
from adapl.privacy.config import PrivacyConfig, build_minimum_privacy_config
from adapl.privacy.mechanisms import client_update_l2_norm


class MinimumDPFedAvg(FederatedMethod):
    info = MINIMUM_INFO

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.privacy_config: PrivacyConfig = build_minimum_privacy_config(args)
        self.noise_generator = torch.Generator().manual_seed(args.seed + 10_000)
        self.current_epsilon = self.privacy_config.epsilon
        self.current_noise_multiplier = self.privacy_config.noise_multiplier
        self.current_selected_budgets: list[float] = []
        self.private_state_keys: tuple[str, ...] | None = None

    def _private_keys(self, model_fn: Callable[[], nn.Module]) -> tuple[str, ...]:
        if self.private_state_keys is None:
            model = model_fn()
            self.private_state_keys = tuple(
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        return self.private_state_keys

    def build_privacy_budget_manager(
        self,
    ) -> ClientPrivacyBudgetManager | None:
        if self.privacy_config.privacy_budgets is not None:
            client_epsilons = list(self.privacy_config.privacy_budgets)
        elif self.privacy_config.epsilon is not None:
            client_epsilons = [
                self.privacy_config.epsilon
                for _ in range(self.args.num_clients)
            ]
        elif self.args.epsilon_max is not None:
            client_epsilons = [
                self.args.epsilon_max
                for _ in range(self.args.num_clients)
            ]
        else:
            return None
        return ClientPrivacyBudgetManager.from_client_epsilons(
            client_epsilons=client_epsilons,
            delta=self.privacy_config.delta,
            noise_multiplier=self.args.noise_multiplier,
            epsilon_floor=self.args.epsilon_min,
        )

    def startup_lines(self) -> list[str]:
        epsilon = (
            f"{self.privacy_config.epsilon}"
            if self.privacy_config.epsilon is not None
            else "not reported"
        )
        lines = [
            "DP is enabled at the client update level.",
            (
                "Min privacy config: "
                f"epsilon_min={epsilon}, "
                f"delta={self.privacy_config.delta}, "
                f"clipping_norm={self.privacy_config.clipping_norm}, "
                f"noise_multiplier={self.privacy_config.noise_multiplier:.6f}, "
                f"noise_std={self.privacy_config.noise_std:.6f}, "
                "private_update_scope=trainable_parameters"
            ),
        ]
        if self.privacy_config.privacy_scenario is not None:
            scenario = self.privacy_config.privacy_scenario
            lines.append(
                "Paper privacy scenario: "
                f"scenario={scenario.name}, "
                f"level_budgets={list(scenario.level_budgets)}, "
                f"level_counts={list(scenario.level_counts)}"
            )
        if (
            self.privacy_config.privacy_budgets is not None
            and self.args.noise_multiplier is None
            and self.args.epsilon_min is None
        ):
            lines.append(
                "Min epsilon is recomputed each round from the selected clients "
                "K_t, matching epsilon_min = min_{k in K_t} epsilon_k."
            )
        return lines

    def begin_round(self, round_idx: int, selected_clients: Sequence[int]) -> None:
        del round_idx
        if (
            self.privacy_config.privacy_budgets is None
            or self.args.noise_multiplier is not None
            or self.args.epsilon_min is not None
        ):
            self.current_epsilon = self.privacy_config.epsilon
            self.current_noise_multiplier = self.privacy_config.noise_multiplier
            self.current_selected_budgets = []
            return

        self.current_selected_budgets = [
            self.privacy_config.privacy_budgets[client_id]
            for client_id in selected_clients
        ]
        self.current_epsilon = min(self.current_selected_budgets)
        self.current_noise_multiplier = gaussian_noise_multiplier(
            self.current_epsilon,
            self.privacy_config.delta,
        )

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
                ("privacy", "private_update_scope", "trainable_parameters"),
                ("privacy", "privacy_budget_count", budget_count),
            ]
        )
        if self.privacy_config.privacy_scenario is not None:
            scenario = self.privacy_config.privacy_scenario
            rows.extend(
                [
                    ("privacy", "scenario", scenario.name),
                    ("privacy", "level_budgets", list(scenario.level_budgets)),
                    ("privacy", "level_counts", list(scenario.level_counts)),
                    ("privacy", "client_budgets", list(scenario.client_budgets)),
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
            "private_update_scope": "trainable_parameters",
            "privacy_budget_count": (
                len(self.privacy_config.privacy_budgets)
                if self.privacy_config.privacy_budgets
                else 0
            ),
        }
        if self.privacy_config.privacy_scenario is not None:
            scenario = self.privacy_config.privacy_scenario
            payload["privacy"]["scenario"] = {
                "name": scenario.name,
                "level_budgets": list(scenario.level_budgets),
                "level_counts": list(scenario.level_counts),
                "client_budgets": list(scenario.client_budgets),
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
            clipping_norm=self.privacy_config.clipping_norm,
            noise_multiplier=self.current_noise_multiplier,
            noise_generator=self.noise_generator,
        )
        update_norm = client_update_l2_norm(
            global_state,
            local_state,
            self._private_keys(model_fn),
        )
        return ClientUpdate(
            client_id=client_id,
            state_dict=local_state,
            train_loss=client_loss,
            num_examples=client_size,
            metadata={
                "update_norm": update_norm,
                "clipped_norm": min(update_norm, self.privacy_config.clipping_norm),
                "clip_factor": min(1.0, self.privacy_config.clipping_norm / (update_norm + 1e-12)),
                "noise_std": self.privacy_config.clipping_norm * self.current_noise_multiplier,
                "epsilon_min": self.current_epsilon,
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
