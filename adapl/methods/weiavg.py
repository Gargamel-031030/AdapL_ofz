"""Budget-weighted heterogeneous DP-FedAvg baseline."""

from __future__ import annotations

from argparse import Namespace
from collections import OrderedDict
from typing import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from adapl.fl.aggregation import weighted_aggregate
from adapl.fl.client import train_client_sgd
from adapl.methods.base import ClientUpdate, FederatedMethod
from adapl.methods.metadata import WEIAVG_INFO
from adapl.privacy.accounting import gaussian_noise_multiplier
from adapl.privacy.config import PrivacyConfig, build_heterogeneous_privacy_config
from adapl.privacy.mechanisms import privatize_client_update


def normalize_budget_weights(privacy_budgets: Sequence[float]) -> list[float]:
    total_budget = float(sum(privacy_budgets))
    if total_budget <= 0:
        raise ValueError("Privacy budget weights must have positive total.")
    return [float(budget) / total_budget for budget in privacy_budgets]


class WeiAvgDPFedAvg(FederatedMethod):
    info = WEIAVG_INFO

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.privacy_config: PrivacyConfig = build_heterogeneous_privacy_config(
            args,
            self.info.display_name,
        )
        self.noise_generator = torch.Generator().manual_seed(args.seed + 20_000)
        self.private_state_keys: tuple[str, ...] | None = None
        self.current_selected_clients: list[int] = []
        self.current_selected_budgets: list[float] = []
        self.current_selected_weights: list[float] = []

    @property
    def privacy_budgets(self) -> list[float]:
        if self.privacy_config.privacy_budgets is None:
            raise RuntimeError("WeiAvg requires resolved privacy budgets.")
        return self.privacy_config.privacy_budgets

    def _private_keys(self, model_fn: Callable[[], nn.Module]) -> tuple[str, ...]:
        if self.private_state_keys is None:
            model = model_fn()
            self.private_state_keys = tuple(
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        return self.private_state_keys

    def _client_noise_multiplier(self, client_id: int) -> float:
        if self.args.noise_multiplier is not None:
            return float(self.args.noise_multiplier)
        return gaussian_noise_multiplier(
            self.privacy_budgets[client_id],
            self.privacy_config.delta,
        )

    def _client_noise_std(self, client_id: int) -> float:
        return self.privacy_config.clipping_norm * self._client_noise_multiplier(
            client_id
        )

    def _aggregation_weight(self, client_id: int) -> float:
        if not self.current_selected_clients:
            raise RuntimeError("begin_round must be called before train_client.")
        try:
            selected_index = self.current_selected_clients.index(client_id)
        except ValueError as exc:
            raise RuntimeError(
                f"Client {client_id} was not selected in the current round."
            ) from exc
        return self.current_selected_weights[selected_index]

    def startup_lines(self) -> list[str]:
        budgets = self.privacy_budgets
        noise_stds = [
            self._client_noise_std(client_id)
            for client_id in range(len(budgets))
        ]
        lines = [
            "DP is enabled at the client update level.",
            (
                "WeiAvg privacy config: "
                f"epsilon_min={min(budgets)}, "
                f"epsilon_max={max(budgets)}, "
                f"delta={self.privacy_config.delta}, "
                f"clipping_norm={self.privacy_config.clipping_norm}, "
                f"noise_std_min={min(noise_stds):.6f}, "
                f"noise_std_max={max(noise_stds):.6f}, "
                f"noise_source={self.privacy_config.noise_source}, "
                "aggregation_weight=epsilon_k/sum_selected_epsilon, "
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
        return lines

    def begin_round(self, round_idx: int, selected_clients: Sequence[int]) -> None:
        del round_idx
        self.current_selected_clients = list(selected_clients)
        self.current_selected_budgets = [
            self.privacy_budgets[client_id]
            for client_id in self.current_selected_clients
        ]
        self.current_selected_weights = normalize_budget_weights(
            self.current_selected_budgets
        )

    def config_rows(self) -> list[tuple[str, str, object]]:
        rows = super().config_rows()
        budgets = self.privacy_budgets
        noise_stds = [
            self._client_noise_std(client_id)
            for client_id in range(len(budgets))
        ]
        rows.extend(
            [
                ("privacy", "dp_enabled", True),
                ("privacy", "mechanism", self.privacy_config.mechanism),
                ("privacy", "epsilon_min", min(budgets)),
                ("privacy", "epsilon_max", max(budgets)),
                ("privacy", "delta", self.privacy_config.delta),
                ("privacy", "clipping_norm", self.privacy_config.clipping_norm),
                ("privacy", "noise_std_min", min(noise_stds)),
                ("privacy", "noise_std_max", max(noise_stds)),
                ("privacy", "noise_source", self.privacy_config.noise_source),
                ("privacy", "private_update_scope", "trainable_parameters"),
                ("privacy", "privacy_budget_count", len(budgets)),
                ("aggregation", "weight_rule", "epsilon_k/sum_selected_epsilon"),
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
        else:
            rows.append(("privacy", "client_budgets", list(budgets)))
        return rows

    def config_payload(self) -> dict[str, object]:
        payload = super().config_payload()
        budgets = self.privacy_budgets
        noise_stds = [
            self._client_noise_std(client_id)
            for client_id in range(len(budgets))
        ]
        payload["dp_enabled"] = True
        payload["privacy"] = {
            "mechanism": self.privacy_config.mechanism,
            "epsilon_min": min(budgets),
            "epsilon_max": max(budgets),
            "delta": self.privacy_config.delta,
            "clipping_norm": self.privacy_config.clipping_norm,
            "noise_std_min": min(noise_stds),
            "noise_std_max": max(noise_stds),
            "noise_source": self.privacy_config.noise_source,
            "private_update_scope": "trainable_parameters",
            "privacy_budget_count": len(budgets),
            "aggregation_weight_rule": "epsilon_k/sum_selected_epsilon",
        }
        if self.privacy_config.privacy_scenario is not None:
            scenario = self.privacy_config.privacy_scenario
            payload["privacy"]["scenario"] = {
                "name": scenario.name,
                "level_budgets": list(scenario.level_budgets),
                "level_counts": list(scenario.level_counts),
                "client_budgets": list(scenario.client_budgets),
            }
        else:
            payload["privacy"]["client_budgets"] = list(budgets)
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
        epsilon = self.privacy_budgets[client_id]
        noise_std = self._client_noise_std(client_id)
        aggregation_weight = self._aggregation_weight(client_id)
        privatized = privatize_client_update(
            global_state=global_state,
            local_state=local_state,
            clipping_norm=self.privacy_config.clipping_norm,
            noise_std=noise_std,
            generator=self.noise_generator,
            private_keys=self._private_keys(model_fn),
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
                "epsilon": epsilon,
                "epsilon_min": min(self.current_selected_budgets),
                "delta": self.privacy_config.delta,
                "aggregation_weight": aggregation_weight,
            },
        )

    def aggregate(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> OrderedDict[str, torch.Tensor]:
        # Since sum(weights) = 1, weighted states equal global + weighted deltas.
        return weighted_aggregate(
            [update.state_dict for update in client_updates],
            [
                float(update.metadata["aggregation_weight"])
                for update in client_updates
            ],
        )
