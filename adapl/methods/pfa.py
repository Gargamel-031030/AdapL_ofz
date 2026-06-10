"""Projected Federated Averaging with heterogeneous privacy budgets."""

from __future__ import annotations

from argparse import Namespace
from collections import OrderedDict
from typing import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from adapl.fl.client import train_client_sgd
from adapl.fl.sampling import select_clients as default_select_clients
from adapl.methods.base import ClientUpdate, FederatedMethod
from adapl.methods.metadata import PFA_INFO
from adapl.privacy.accounting import (
    ClientPrivacyBudgetManager,
    PrivacyBudgetContext,
    gaussian_noise_multiplier,
)
from adapl.privacy.config import (
    HeterogeneousBudgetConfig,
    build_heterogeneous_budget_config,
)
from adapl.privacy.mechanisms import client_update_l2_norm
from adapl.utils import clone_state_dict


def _normalize_weights(values: Sequence[float]) -> list[float]:
    total = float(sum(values))
    if total <= 0:
        raise ValueError("PFA aggregation weights must have positive total.")
    return [float(value) / total for value in values]


class PFAFedAvg(FederatedMethod):
    """Projected FedAvg using high-budget clients as the public subspace source."""

    info = PFA_INFO

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.budget_config: HeterogeneousBudgetConfig = (
            build_heterogeneous_budget_config(args, self.info.display_name)
        )
        self.no_dp: bool = getattr(args, "no_dp", False)
        self.public_fraction: float = args.pfa_public_fraction
        self.projection_dim: int = args.pfa_projection_dim
        self.weighted_projection: bool = args.pfa_weighted_projection
        self.selection_attempts: int = args.pfa_selection_attempts
        if not 0 < self.public_fraction < 1:
            raise ValueError("--pfa_public_fraction must be in (0, 1).")
        if self.projection_dim <= 0:
            raise ValueError("--pfa_projection_dim must be positive.")
        if self.selection_attempts <= 0:
            raise ValueError("--pfa_selection_attempts must be positive.")
        if args.num_clients < 2:
            raise ValueError("PFA requires at least two clients.")

        self.public_clients = self._build_public_clients(args.num_clients)
        self.private_clients = [
            client_id
            for client_id in range(args.num_clients)
            if client_id not in self.public_clients
        ]
        if not self.private_clients:
            raise ValueError("PFA requires at least one private client.")

        self.current_selected_clients: list[int] = []
        self.current_selected_budgets: list[float] = []
        self.current_selected_weights: list[float] = []
        self.current_noise_multipliers: list[float] = []
        self.current_privacy_context: dict[int, PrivacyBudgetContext] = {}
        self.private_state_keys: tuple[str, ...] | None = None
        self.current_global_state: OrderedDict[str, torch.Tensor] | None = None
        self.noise_generator = (
            torch.Generator().manual_seed(args.seed + 30_000)
            if not self.no_dp
            else None
        )

        if not self.no_dp:
            self.clipping_norm: float = args.clipping_norm
            self.delta: float = args.delta
            self.user_noise_multiplier: float | None = args.noise_multiplier
            self.user_epsilon_min: float | None = args.epsilon_min

    @property
    def privacy_budgets(self) -> list[float]:
        return self.budget_config.privacy_budgets

    def _build_public_clients(self, num_clients: int) -> set[int]:
        public_count = max(1, int(num_clients * self.public_fraction))
        public_count = min(public_count, num_clients - 1)
        sorted_budgets = sorted(self.budget_config.privacy_budgets)
        threshold = sorted_budgets[-public_count]
        public_clients = {
            client_id
            for client_id, budget in enumerate(self.budget_config.privacy_budgets)
            if budget >= threshold
        }
        if len(public_clients) >= num_clients:
            ranked = sorted(
                range(num_clients),
                key=lambda client_id: (
                    self.budget_config.privacy_budgets[client_id],
                    -client_id,
                ),
                reverse=True,
            )
            public_clients = set(ranked[:public_count])
        return public_clients

    def _private_keys(self, model_fn: Callable[[], nn.Module]) -> tuple[str, ...]:
        if self.private_state_keys is None:
            model = model_fn()
            self.private_state_keys = tuple(
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        return self.private_state_keys

    def _noise_multiplier_for_budget(self, epsilon: float) -> float:
        if self.user_noise_multiplier is not None:
            return self.user_noise_multiplier
        noise_epsilon = float(epsilon)
        if self.user_epsilon_min is not None:
            noise_epsilon = max(noise_epsilon, self.user_epsilon_min)
        return gaussian_noise_multiplier(noise_epsilon, self.delta)

    def _has_public_and_private(self, client_ids: Sequence[int]) -> bool:
        has_public = any(client_id in self.public_clients for client_id in client_ids)
        has_private = any(client_id not in self.public_clients for client_id in client_ids)
        return has_public and has_private

    def select_clients(
        self,
        num_clients: int,
        client_fraction: float,
        round_idx: int,
        seed: int,
        candidate_client_ids: Sequence[int] | None = None,
    ) -> list[int]:
        candidates = (
            list(range(num_clients))
            if candidate_client_ids is None
            else sorted(set(int(client_id) for client_id in candidate_client_ids))
        )
        has_public_candidate = any(
            client_id in self.public_clients for client_id in candidates
        )
        has_private_candidate = any(
            client_id not in self.public_clients for client_id in candidates
        )
        if not has_public_candidate:
            return []
        if not has_private_candidate:
            return default_select_clients(
                num_clients,
                client_fraction,
                round_idx,
                seed,
                candidate_client_ids=candidates,
            )

        selected = []
        for attempt in range(self.selection_attempts):
            selected = default_select_clients(
                num_clients,
                client_fraction,
                round_idx + attempt * 10_000,
                seed,
                candidate_client_ids=candidates,
            )
            if self._has_public_and_private(selected):
                return selected
        return []

    def build_privacy_budget_manager(
        self,
    ) -> ClientPrivacyBudgetManager | None:
        if self.no_dp:
            return None
        return ClientPrivacyBudgetManager.from_client_epsilons(
            client_epsilons=self.privacy_budgets,
            delta=self.delta,
            noise_multiplier=self.user_noise_multiplier,
            epsilon_floor=self.user_epsilon_min,
        )

    def set_privacy_budget_context(
        self,
        context: Mapping[int, PrivacyBudgetContext],
    ) -> None:
        self.current_privacy_context = dict(context)

    def begin_round(self, round_idx: int, selected_clients: Sequence[int]) -> None:
        del round_idx
        self.current_selected_clients = list(selected_clients)
        self.current_selected_budgets = [
            self.privacy_budgets[client_id]
            for client_id in self.current_selected_clients
        ]
        self.current_selected_weights = _normalize_weights(
            self.current_selected_budgets
            if self.weighted_projection
            else [1.0 for _ in self.current_selected_clients]
        )
        self.current_global_state = None
        if not self.no_dp:
            if self.current_privacy_context:
                self.current_noise_multipliers = [
                    self.current_privacy_context[client_id].noise_multiplier
                    for client_id in self.current_selected_clients
                ]
            else:
                self.current_noise_multipliers = [
                    self._noise_multiplier_for_budget(epsilon)
                    for epsilon in self.current_selected_budgets
                ]
        else:
            self.current_noise_multipliers = [0.0] * len(self.current_selected_clients)

    def startup_lines(self) -> list[str]:
        budgets = self.privacy_budgets
        lines = [
            (
                "PFA projected aggregation: "
                f"public_fraction={self.public_fraction}, "
                f"public_clients={sorted(self.public_clients)}, "
                f"epsilon_min={min(budgets)}, "
                f"epsilon_max={max(budgets)}, "
                f"projection_dim={self.projection_dim}, "
                f"weighted_projection={self.weighted_projection}"
            ),
        ]
        if self.no_dp:
            lines.append("PFA DP local clipping/noise is disabled.")
        else:
            lines.append(
                "PFA DP local training: "
                f"delta={self.delta}, "
                f"clipping_norm={self.clipping_norm}, "
                "per_client_noise=epsilon_k_based"
            )
        if self.budget_config.privacy_scenario is not None:
            scenario = self.budget_config.privacy_scenario
            lines.append(
                "Paper privacy scenario: "
                f"scenario={scenario.name}, "
                f"level_budgets={list(scenario.level_budgets)}, "
                f"level_counts={list(scenario.level_counts)}"
            )
        return lines

    def config_rows(self) -> list[tuple[str, str, object]]:
        rows = super().config_rows()
        budgets = self.privacy_budgets
        rows.extend(
            [
                ("privacy", "dp_enabled", not self.no_dp),
                ("privacy", "epsilon_min", min(budgets)),
                ("privacy", "epsilon_max", max(budgets)),
                ("privacy", "privacy_budget_count", len(budgets)),
                ("privacy", "client_budgets", list(budgets)),
                ("pfa", "public_fraction", self.public_fraction),
                ("pfa", "public_clients", sorted(self.public_clients)),
                ("pfa", "projection_dim", self.projection_dim),
                ("pfa", "weighted_projection", self.weighted_projection),
                ("aggregation", "weight_rule", "epsilon_weighted_projected_updates"
                 if self.weighted_projection else "count_weighted_projected_updates"),
            ]
        )
        if self.budget_config.privacy_scenario is not None:
            scenario = self.budget_config.privacy_scenario
            rows.extend(
                [
                    ("privacy", "scenario", scenario.name),
                    ("privacy", "level_budgets", list(scenario.level_budgets)),
                    ("privacy", "level_counts", list(scenario.level_counts)),
                ]
            )
        if not self.no_dp:
            rows.extend(
                [
                    ("privacy", "mechanism", "per_client_gaussian"),
                    ("privacy", "delta", self.delta),
                    ("privacy", "clipping_norm", self.clipping_norm),
                    ("privacy", "noise_multiplier",
                     self.user_noise_multiplier if self.user_noise_multiplier is not None
                     else "per_client"),
                    ("privacy", "epsilon_min_override", self.user_epsilon_min),
                    ("privacy", "private_update_scope", "trainable_parameters"),
                ]
            )
        return rows

    def config_payload(self) -> dict[str, object]:
        payload = super().config_payload()
        budgets = self.privacy_budgets
        payload["dp_enabled"] = not self.no_dp
        payload["privacy"] = {
            "epsilon_min": min(budgets),
            "epsilon_max": max(budgets),
            "privacy_budget_count": len(budgets),
            "client_budgets": list(budgets),
        }
        if self.budget_config.privacy_scenario is not None:
            scenario = self.budget_config.privacy_scenario
            payload["privacy"]["scenario"] = {
                "name": scenario.name,
                "level_budgets": list(scenario.level_budgets),
                "level_counts": list(scenario.level_counts),
            }
        if not self.no_dp:
            payload["privacy"].update(
                {
                    "mechanism": "per_client_gaussian",
                    "delta": self.delta,
                    "clipping_norm": self.clipping_norm,
                    "noise_multiplier": (
                        self.user_noise_multiplier
                        if self.user_noise_multiplier is not None
                        else "per_client"
                    ),
                    "epsilon_min_override": self.user_epsilon_min,
                    "private_update_scope": "trainable_parameters",
                }
            )
        payload["pfa"] = {
            "public_fraction": self.public_fraction,
            "public_clients": sorted(self.public_clients),
            "projection_dim": self.projection_dim,
            "weighted_projection": self.weighted_projection,
            "selection_attempts": self.selection_attempts,
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
        if not self.current_selected_clients:
            raise RuntimeError("begin_round must be called before train_client.")
        selected_index = self.current_selected_clients.index(client_id)
        epsilon = self.current_selected_budgets[selected_index]
        aggregation_weight = self.current_selected_weights[selected_index]
        noise_multiplier = self.current_noise_multipliers[selected_index]
        if self.current_global_state is None:
            self.current_global_state = clone_state_dict(global_state)

        train_kwargs = {}
        if not self.no_dp:
            train_kwargs = {
                "clipping_norm": self.clipping_norm,
                "noise_multiplier": noise_multiplier,
                "noise_generator": self.noise_generator,
            }
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
            **train_kwargs,
        )
        private_keys = self._private_keys(model_fn)
        update_norm = client_update_l2_norm(global_state, local_state, private_keys)
        metadata = {
            "update_norm": update_norm,
            "epsilon": epsilon,
            "epsilon_min": min(self.current_selected_budgets),
            "aggregation_weight": aggregation_weight,
            "pfa_is_public": client_id in self.public_clients,
            "pfa_projection_dim": self.projection_dim,
            "pfa_weighted_projection": self.weighted_projection,
        }
        if not self.no_dp:
            metadata.update(
                {
                    "clipped_norm": min(update_norm, self.clipping_norm),
                    "clip_factor": min(1.0, self.clipping_norm / (update_norm + 1e-12)),
                    "noise_std": noise_multiplier * self.clipping_norm / self.args.batch_size,
                    "delta": self.delta,
                    "noise_multiplier": noise_multiplier,
                }
            )
        return ClientUpdate(
            client_id=client_id,
            state_dict=local_state,
            train_loss=client_loss,
            num_examples=client_size,
            metadata=metadata,
        )

    def _client_weight_values(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> list[float]:
        if self.weighted_projection:
            return [
                float(update.metadata.get("epsilon", self.privacy_budgets[update.client_id]))
                for update in client_updates
            ]
        return [1.0 for _ in client_updates]

    def _weighted_delta_mean(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        client_updates: Sequence[ClientUpdate],
        key: str,
        weights: Sequence[float],
    ) -> torch.Tensor:
        normalized = _normalize_weights(weights)
        value = torch.zeros_like(global_state[key], dtype=global_state[key].dtype)
        for update, weight in zip(client_updates, normalized):
            value += (update.state_dict[key] - global_state[key]) * weight
        return value

    def _weighted_state_mean(
        self,
        client_updates: Sequence[ClientUpdate],
        key: str,
        weights: Sequence[float],
    ) -> torch.Tensor:
        normalized = _normalize_weights(weights)
        value = torch.zeros_like(client_updates[0].state_dict[key])
        for update, weight in zip(client_updates, normalized):
            value += update.state_dict[key] * weight
        return value

    def _project_private_delta(
        self,
        public_deltas: Sequence[torch.Tensor],
        private_delta: torch.Tensor,
    ) -> torch.Tensor:
        if not public_deltas:
            return private_delta
        flat_public = torch.stack(
            [delta.detach().reshape(-1).float() for delta in public_deltas],
            dim=0,
        )
        flat_private = private_delta.detach().reshape(-1).float()
        if flat_public.size(0) == 1:
            mean = torch.zeros_like(flat_public[0])
            centered = flat_public
        else:
            mean = flat_public.mean(dim=0)
            centered = flat_public - mean
        rank = min(self.projection_dim, centered.size(0), centered.size(1))
        if rank <= 0 or float(torch.linalg.vector_norm(centered).item()) <= 1e-12:
            projected = mean
        else:
            try:
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                basis = vh[:rank].transpose(0, 1)
                centered_private = flat_private - mean
                projected = mean + basis @ (basis.transpose(0, 1) @ centered_private)
            except RuntimeError:
                projected = flat_private
        return projected.reshape_as(private_delta).to(dtype=private_delta.dtype)

    def aggregate(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> OrderedDict[str, torch.Tensor]:
        if not client_updates:
            raise ValueError("Cannot aggregate an empty client update list.")
        if self.current_global_state is None:
            raise RuntimeError("PFA aggregate called before any client trained.")

        global_state = self.current_global_state
        private_keys = set(self.private_state_keys or ())
        public_updates = [
            update for update in client_updates if update.client_id in self.public_clients
        ]
        private_updates = [
            update for update in client_updates if update.client_id not in self.public_clients
        ]
        all_weights = self._client_weight_values(client_updates)
        public_weights = self._client_weight_values(public_updates) if public_updates else []
        private_weights = self._client_weight_values(private_updates) if private_updates else []
        public_mass = sum(public_weights)
        private_mass = sum(private_weights)

        aggregated = OrderedDict()
        for name, global_value in global_state.items():
            if not torch.is_floating_point(global_value):
                aggregated[name] = client_updates[0].state_dict[name].clone()
                continue
            if name not in private_keys:
                aggregated[name] = self._weighted_state_mean(
                    client_updates,
                    name,
                    all_weights,
                )
                continue

            if public_updates and private_updates:
                public_delta_mean = self._weighted_delta_mean(
                    global_state,
                    public_updates,
                    name,
                    public_weights,
                )
                private_delta_mean = self._weighted_delta_mean(
                    global_state,
                    private_updates,
                    name,
                    private_weights,
                )
                public_deltas = [
                    update.state_dict[name] - global_state[name]
                    for update in public_updates
                ]
                projected_private_delta = self._project_private_delta(
                    public_deltas,
                    private_delta_mean,
                )
                delta = (
                    public_delta_mean * (public_mass / (public_mass + private_mass))
                    + projected_private_delta
                    * (private_mass / (public_mass + private_mass))
                )
            else:
                delta = self._weighted_delta_mean(
                    global_state,
                    client_updates,
                    name,
                    all_weights,
                )
            aggregated[name] = global_value + delta
        return aggregated
