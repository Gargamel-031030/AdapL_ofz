"""AdapL federated method implementation."""

from __future__ import annotations

from argparse import Namespace
from collections import OrderedDict
from dataclasses import dataclass, replace
import math
from typing import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from adapl.federated.aggregation import aggregate_state_dicts
from adapl.methods.base import ClientUpdate, FederatedMethod
from adapl.methods.metadata import ADAPL_INFO
from adapl.privacy.accountant import MomentsAccountant
from adapl.privacy.budgets import parse_privacy_budgets
from adapl.privacy.levels import PrivacyScenario, build_privacy_scenario
from adapl.privacy.mechanisms import client_update_l2_norm
from adapl.privacy.noise import initialize_noise_multiplier
from adapl.trainers.adapl_trainer import local_update_decay, local_update_first
from adapl.utils import clone_state_dict


@dataclass(frozen=True)
class _ClientPrivacyState:
    target_epsilon: float | None
    noise_multiplier: float
    noise_source: str
    q: float


class AdapL(FederatedMethod):
    """Adaptive local DP-FL with Fisher masks and per-minibatch noisy gradients."""

    info = ADAPL_INFO
    uses_internal_privacy_accountant = True

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        if getattr(args, "no_dp", False):
            raise ValueError("AdapL requires DP. Remove --no_dp.")
        if args.clipping_norm is None or args.clipping_norm <= 0:
            raise ValueError("--clipping_bound/--clipping_norm must be positive.")
        if args.delta is None or not 0 < args.delta < 1:
            raise ValueError("--target_delta/--delta must be in (0, 1).")
        if not 0 <= args.fisher_threshold <= 1:
            raise ValueError("--fisher_threshold must be in [0, 1].")
        if args.fisher_estimator not in {"sample", "batch"}:
            raise ValueError("--fisher_estimator must be sample or batch.")
        if args.fisher_batches < 0:
            raise ValueError("--fisher_batches must be non-negative.")
        if args.gamma < 0:
            raise ValueError("--gamma must be non-negative.")
        if not 0 <= args.adapl_alpha <= 1:
            raise ValueError("--adapl_alpha/--phi must be in [0, 1].")
        if args.adapl_noise_decay_factor <= 0:
            raise ValueError("--adapl_noise_decay_factor/--decay_factor must be positive.")
        if args.max_clip_norm is not None and args.max_clip_norm <= 0:
            raise ValueError("--max_clip_norm must be positive.")

        self.privacy_scenario: PrivacyScenario | None = None
        self.client_target_epsilons = self._resolve_client_epsilons()
        self.client_privacy: dict[int, _ClientPrivacyState] = {}
        self.accountants: dict[int, MomentsAccountant] = {}
        self.planned_steps_by_client: dict[int, int] = {}
        self.current_round_idx = 0
        self.current_selected_clients: list[int] = []
        self.latest_global_state: OrderedDict[str, torch.Tensor] | None = None
        self.test_accuracy_history: list[float] = []

    def _resolve_client_epsilons(self) -> list[float | None]:
        budgets = parse_privacy_budgets(self.args.privacy_budgets)
        if budgets is None and self.args.privacy_scenario is not None:
            self.privacy_scenario = build_privacy_scenario(
                scenario=self.args.privacy_scenario,
                num_clients=self.args.num_clients,
                seed=self.args.privacy_budget_seed,
            )
            budgets = list(self.privacy_scenario.client_budgets)

        if budgets is not None:
            if len(budgets) != self.args.num_clients:
                raise ValueError(
                    "The number of privacy budgets must match --num_clients "
                    f"({len(budgets)} != {self.args.num_clients})."
                )
            return [float(value) for value in budgets]

        if self.args.epsilon_max is not None:
            epsilon = float(self.args.epsilon_max)
        elif self.args.epsilon_min is not None:
            epsilon = float(self.args.epsilon_min)
        elif self.args.noise_multiplier is not None:
            return [None for _ in range(self.args.num_clients)]
        else:
            raise ValueError(
                "AdapL requires --epsilon_min, --epsilon_max, --privacy_scenario, "
                "--epsilon_file/--privacy_budgets, --noise_multiplier, or "
                "--noise_multiplier_override."
            )

        if epsilon <= 0:
            raise ValueError("epsilon values must be positive.")
        return [epsilon for _ in range(self.args.num_clients)]

    def _planned_local_steps(self, train_loader: DataLoader) -> int:
        if self.args.local_update_mode == "random-batch":
            return self.args.local_steps
        if self.args.local_update_mode == "full-epoch":
            local_epochs = (
                self.args.local_epochs
                if self.args.local_epochs is not None
                else self.args.local_steps
            )
            return local_epochs * len(train_loader)
        raise ValueError(f"Unsupported local update mode: {self.args.local_update_mode}")

    def prepare_privacy_accountants(self, train_loaders: Sequence[DataLoader]) -> None:
        if len(train_loaders) != self.args.num_clients:
            raise ValueError("train_loaders length must match --num_clients.")

        self.client_privacy.clear()
        self.accountants.clear()
        self.planned_steps_by_client.clear()
        for client_id, train_loader in enumerate(train_loaders):
            dataset_size = len(train_loader.dataset)
            if dataset_size <= 0:
                raise ValueError(f"Client {client_id} has no training samples.")
            planned_steps = self._planned_local_steps(train_loader)
            total_steps = planned_steps * self.args.global_rounds
            q = min(1.0, self.args.batch_size / float(dataset_size))
            target_epsilon = self.client_target_epsilons[client_id]
            noise_init = initialize_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=self.args.delta,
                q=q,
                total_steps=total_steps,
                manual_override=self.args.noise_multiplier_override
                if self.args.noise_multiplier_override is not None
                else self.args.noise_multiplier,
                use_decay_search=self.args.nm_decay,
                fallback_epsilon=self.args.epsilon_min,
            )
            self.client_privacy[client_id] = _ClientPrivacyState(
                target_epsilon=target_epsilon,
                noise_multiplier=noise_init.noise_multiplier,
                noise_source=noise_init.source,
                q=q,
            )
            self.accountants[client_id] = MomentsAccountant(
                q=q,
                noise_multiplier=noise_init.noise_multiplier,
                target_delta=self.args.delta,
                target_epsilon=target_epsilon,
            )
            self.planned_steps_by_client[client_id] = planned_steps

    def eligible_client_ids(self, client_ids: Sequence[int]) -> list[int]:
        if not self.accountants:
            raise RuntimeError("prepare_privacy_accountants must be called first.")
        eligible = []
        for client_id in client_ids:
            planned_steps = self.planned_steps_by_client[client_id]
            if self.accountants[client_id].can_train(planned_steps):
                eligible.append(client_id)
        return eligible

    @property
    def num_finished_accountants(self) -> int:
        return self.num_accountants - self.num_active_accountants

    @property
    def num_active_accountants(self) -> int:
        if not self.accountants:
            return 0
        return len(self.eligible_client_ids(list(self.accountants.keys())))

    @property
    def num_accountants(self) -> int:
        return len(self.accountants)

    def startup_lines(self) -> list[str]:
        lines = [
            (
                "AdapL DP local training: "
                "noise_timing=after_each_minibatch, "
                "gradient=per_sample_clipped_then_averaged, "
                "noise_scope=important_fisher_mask, "
                f"fisher_threshold={self.args.fisher_threshold}, "
                f"fisher_estimator={self.args.fisher_estimator}, "
                f"gamma={self.args.gamma}, "
                f"adapl_alpha={self.args.adapl_alpha}, "
                f"adapl_noise_decay_factor={self.args.adapl_noise_decay_factor}, "
                f"max_clip_norm={self.args.max_clip_norm}, "
                f"nm_decay={self.args.nm_decay}"
            )
        ]
        if self.privacy_scenario is not None:
            lines.append(
                "Paper privacy scenario: "
                f"scenario={self.privacy_scenario.name}, "
                f"level_budgets={list(self.privacy_scenario.level_budgets)}, "
                f"level_counts={list(self.privacy_scenario.level_counts)}"
            )
        return lines

    def config_rows(self) -> list[tuple[str, str, object]]:
        rows = super().config_rows()
        budget_count = sum(
            1 for epsilon in self.client_target_epsilons if epsilon is not None
        )
        rows.extend(
            [
                ("privacy", "dp_enabled", True),
                ("privacy", "mechanism", "minibatch_gradient_gaussian_moments"),
                ("privacy", "target_delta", self.args.delta),
                ("privacy", "clipping_bound", self.args.clipping_norm),
                ("privacy", "accountant", "moments"),
                ("privacy", "q", "batch_size/client_dataset_size"),
                ("privacy", "privacy_budget_count", budget_count),
                ("adapl", "fisher_threshold", self.args.fisher_threshold),
                ("adapl", "fisher_estimator", self.args.fisher_estimator),
                ("adapl", "fisher_batches", self.args.fisher_batches),
                ("adapl", "gamma", self.args.gamma),
                ("adapl", "adapl_alpha", self.args.adapl_alpha),
                (
                    "adapl",
                    "adapl_noise_decay_factor",
                    self.args.adapl_noise_decay_factor,
                ),
                ("adapl", "max_clip_norm", self.args.max_clip_norm),
                ("adapl", "nm_decay", self.args.nm_decay),
            ]
        )
        if self.privacy_scenario is not None:
            rows.extend(
                [
                    ("privacy", "scenario", self.privacy_scenario.name),
                    ("privacy", "level_budgets", list(self.privacy_scenario.level_budgets)),
                    ("privacy", "level_counts", list(self.privacy_scenario.level_counts)),
                    ("privacy", "client_budgets", list(self.privacy_scenario.client_budgets)),
                ]
            )
        elif budget_count:
            rows.append(("privacy", "client_budgets", list(self.client_target_epsilons)))
        return rows

    def config_payload(self) -> dict[str, object]:
        payload = super().config_payload()
        payload["dp_enabled"] = True
        payload["privacy"] = {
            "mechanism": "minibatch_gradient_gaussian_moments",
            "target_delta": self.args.delta,
            "clipping_bound": self.args.clipping_norm,
            "accountant": "moments",
            "q": "batch_size/client_dataset_size",
            "client_budgets": list(self.client_target_epsilons),
        }
        payload["adapl"] = {
            "fisher_threshold": self.args.fisher_threshold,
            "fisher_estimator": self.args.fisher_estimator,
            "fisher_batches": self.args.fisher_batches,
            "gamma": self.args.gamma,
            "adapl_alpha": self.args.adapl_alpha,
            "adapl_noise_decay_factor": self.args.adapl_noise_decay_factor,
            "max_clip_norm": self.args.max_clip_norm,
            "nm_decay": self.args.nm_decay,
        }
        if self.privacy_scenario is not None:
            payload["privacy"]["scenario"] = {
                "name": self.privacy_scenario.name,
                "level_budgets": list(self.privacy_scenario.level_budgets),
                "level_counts": list(self.privacy_scenario.level_counts),
                "client_budgets": list(self.privacy_scenario.client_budgets),
            }
        return payload

    def begin_round(self, round_idx: int, selected_clients: Sequence[int]) -> None:
        self.current_round_idx = round_idx
        self.current_selected_clients = list(selected_clients)

    def train_client(
        self,
        client_id: int,
        model_fn: Callable[[], nn.Module],
        global_state: OrderedDict[str, torch.Tensor],
        train_loader: DataLoader,
        device: torch.device,
    ) -> ClientUpdate:
        if client_id not in self.current_selected_clients:
            raise RuntimeError(f"Client {client_id} is not selected this round.")
        if client_id not in self.accountants:
            raise RuntimeError("prepare_privacy_accountants must be called first.")

        accountant = self.accountants[client_id]
        planned_steps = self.planned_steps_by_client[client_id]
        if not accountant.can_train(planned_steps):
            raise RuntimeError(
                f"Client {client_id} cannot train {planned_steps} more steps "
                "without exceeding its privacy budget."
            )

        privacy_state = self.client_privacy[client_id]
        if self.current_round_idx <= 1 or self.latest_global_state is None:
            result = local_update_first(
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
                clipping_bound=self.args.clipping_norm,
                base_noise_multiplier=privacy_state.noise_multiplier,
                gamma=self.args.gamma,
            )
            update_phase = "first"
        else:
            result = local_update_decay(
                model_fn=model_fn,
                global_state=global_state,
                latest_global_state=self.latest_global_state,
                train_loader=train_loader,
                local_steps=self.args.local_steps,
                local_epochs=self.args.local_epochs,
                local_update_mode=self.args.local_update_mode,
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                device=device,
                clipping_bound=self.args.clipping_norm,
                base_noise_multiplier=privacy_state.noise_multiplier,
                gamma=self.args.gamma,
                fisher_threshold=self.args.fisher_threshold,
                fisher_estimator=self.args.fisher_estimator,
                fisher_batches=self.args.fisher_batches,
                max_clip_norm=self.args.max_clip_norm,
            )
            update_phase = "decay"

        if not accountant.can_train(result.actual_minibatch_steps):
            raise RuntimeError(
                f"Client {client_id} actual steps exceed its remaining privacy budget."
            )
        before_steps = accountant.current_steps
        epsilon_after = accountant.commit_steps(result.actual_minibatch_steps)
        committed_steps = result.actual_minibatch_steps
        if accountant.current_steps - before_steps != committed_steps:
            raise RuntimeError(
                "Accountant committed steps do not match the local trainer result."
            )
        update_norm = client_update_l2_norm(global_state, result.state_dict)

        return ClientUpdate(
            client_id=client_id,
            state_dict=result.state_dict,
            train_loss=result.train_loss,
            num_examples=result.num_examples,
            metadata={
                "actual_minibatch_steps": result.actual_minibatch_steps,
                "accountant_committed_steps": committed_steps,
                "accountant_total_steps": accountant.current_steps,
                "epsilon": epsilon_after,
                "epsilon_target": privacy_state.target_epsilon,
                "epsilon_min": min(
                    epsilon
                    for epsilon in self.client_target_epsilons
                    if epsilon is not None
                )
                if any(epsilon is not None for epsilon in self.client_target_epsilons)
                else None,
                "delta": self.args.delta,
                "q": privacy_state.q,
                "noise_multiplier": result.noise_multiplier_mean,
                "noise_multiplier_min": result.noise_multiplier_min,
                "noise_multiplier_max": result.noise_multiplier_max,
                "base_noise_multiplier": privacy_state.noise_multiplier,
                "noise_source": privacy_state.noise_source,
                "noise_std": result.noise_std_mean,
                "update_norm": update_norm,
                "clipped_norm": result.sample_clipped_norm_mean,
                "clip_factor": result.sample_clip_factor_mean,
                "layer_clip_factor": result.layer_clip_factor_mean,
                "fisher_threshold": self.args.fisher_threshold,
                "fisher_estimator": self.args.fisher_estimator,
                "fisher_important_ratio": result.important_ratio,
                "fisher_important_params": result.important_params,
                "fisher_total_params": result.total_params,
                "privacy_scenario": (
                    self.privacy_scenario.name if self.privacy_scenario else ""
                ),
                "adapl_update_phase": update_phase,
            },
        )

    def aggregate(
        self,
        client_updates: Sequence[ClientUpdate],
    ) -> OrderedDict[str, torch.Tensor]:
        if not client_updates:
            raise ValueError("Cannot aggregate an empty client update list.")

        client_sizes = [float(update.num_examples) for update in client_updates]
        total_size = sum(client_sizes)
        if total_size <= 0:
            raise ValueError("Total selected client examples must be positive.")
        data_weights = [client_size / total_size for client_size in client_sizes]

        epsilons: list[float] = []
        for update in client_updates:
            epsilon = self.client_target_epsilons[update.client_id]
            if epsilon is None:
                raise ValueError(
                    "AdapL aggregation requires each selected client to have a "
                    "target epsilon privacy budget."
                )
            epsilons.append(float(epsilon))

        max_epsilon = max(epsilons)
        exp_values = [math.exp(epsilon - max_epsilon) for epsilon in epsilons]
        exp_total = sum(exp_values)
        if exp_total <= 0:
            raise ValueError("AdapL epsilon weights must sum to a positive value.")
        epsilon_weights = [value / exp_total for value in exp_values]

        alpha = self.args.adapl_alpha
        aggregation_weights = [
            (1.0 - alpha) * data_weight + alpha * epsilon_weight
            for data_weight, epsilon_weight in zip(data_weights, epsilon_weights)
        ]
        for update, weight in zip(client_updates, aggregation_weights):
            update.metadata["aggregation_weight"] = weight

        return aggregate_state_dicts(
            [update.state_dict for update in client_updates],
            aggregation_weights,
        )

    def observe_global_accuracy(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        test_accuracy: float,
    ) -> bool:
        if not math.isfinite(test_accuracy):
            return False

        previous_best = (
            max(self.test_accuracy_history)
            if self.test_accuracy_history
            else -math.inf
        )
        is_new_best = test_accuracy > previous_best
        should_decay = False
        if len(self.test_accuracy_history) >= 2:
            previous = self.test_accuracy_history[-1]
            previous_previous = self.test_accuracy_history[-2]
            should_decay = (
                test_accuracy >= previous
                and previous >= previous_previous
                and is_new_best
            )

        self.test_accuracy_history.append(float(test_accuracy))
        if is_new_best:
            self.latest_global_state = clone_state_dict(global_state)
        if not should_decay:
            return False

        for client_id, privacy_state in list(self.client_privacy.items()):
            new_sigma = (
                privacy_state.noise_multiplier
                * self.args.adapl_noise_decay_factor
            )
            self.client_privacy[client_id] = replace(
                privacy_state,
                noise_multiplier=new_sigma,
            )
            if client_id in self.accountants:
                self.accountants[client_id].noise_multiplier = new_sigma
        return True
