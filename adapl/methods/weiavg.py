"""Budget-weighted FedAvg baseline."""

from __future__ import annotations

from argparse import Namespace
from collections import OrderedDict
from typing import Callable, Sequence

from torch import nn
from torch.utils.data import DataLoader

from adapl.fl.aggregation import weighted_aggregate
from adapl.fl.client import train_client_sgd
from adapl.methods.base import ClientUpdate, FederatedMethod
from adapl.methods.metadata import WEIAVG_INFO
from adapl.privacy.config import (
    HeterogeneousBudgetConfig,
    build_heterogeneous_budget_config,
)
from adapl.privacy.mechanisms import client_update_l2_norm


def normalize_budget_weights(privacy_budgets: Sequence[float]) -> list[float]:
    total_budget = float(sum(privacy_budgets))
    if total_budget <= 0:
        raise ValueError("Privacy budget weights must have positive total.")
    return [float(budget) / total_budget for budget in privacy_budgets]


class WeiAvgFedAvg(FederatedMethod):
    info = WEIAVG_INFO

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.budget_config: HeterogeneousBudgetConfig = (
            build_heterogeneous_budget_config(
                args,
                self.info.display_name,
            )
        )
        self.private_state_keys: tuple[str, ...] | None = None
        self.current_selected_clients: list[int] = []
        self.current_selected_budgets: list[float] = []
        self.current_selected_weights: list[float] = []

    @property
    def privacy_budgets(self) -> list[float]:
        return self.budget_config.privacy_budgets

    def _private_keys(self, model_fn: Callable[[], nn.Module]) -> tuple[str, ...]:
        if self.private_state_keys is None:
            model = model_fn()
            self.private_state_keys = tuple(
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        return self.private_state_keys

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
        lines = [
            (
                "WeiAvg budget-weighted aggregation: "
                f"epsilon_min={min(budgets)}, "
                f"epsilon_max={max(budgets)}, "
                "aggregation_weight=epsilon_k/sum_selected_epsilon, "
                "dp_clipping_noise=disabled"
            ),
        ]
        if self.budget_config.privacy_scenario is not None:
            scenario = self.budget_config.privacy_scenario
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
        rows.extend(
            [
                ("privacy", "dp_enabled", False),
                ("privacy", "mechanism", "none"),
                ("privacy", "epsilon_min", min(budgets)),
                ("privacy", "epsilon_max", max(budgets)),
                ("privacy", "privacy_budget_count", len(budgets)),
                ("aggregation", "weight_rule", "epsilon_k/sum_selected_epsilon"),
            ]
        )
        if self.budget_config.privacy_scenario is not None:
            scenario = self.budget_config.privacy_scenario
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
        payload["dp_enabled"] = False
        payload["privacy"] = {
            "mechanism": "none",
            "epsilon_min": min(budgets),
            "epsilon_max": max(budgets),
            "privacy_budget_count": len(budgets),
            "aggregation_weight_rule": "epsilon_k/sum_selected_epsilon",
        }
        if self.budget_config.privacy_scenario is not None:
            scenario = self.budget_config.privacy_scenario
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
        aggregation_weight = self._aggregation_weight(client_id)
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
                "clipped_norm": update_norm,
                "clip_factor": 1.0,
                "noise_std": 0.0,
                "epsilon": epsilon,
                "epsilon_min": min(self.current_selected_budgets),
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
