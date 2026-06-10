"""FedDPA personalized FL with adaptive DP."""

from __future__ import annotations

from argparse import Namespace
from collections import OrderedDict
from typing import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from adapl.fl.aggregation import fedavg_aggregate
from adapl.methods.base import ClientUpdate, FederatedMethod
from adapl.methods.metadata import FEDDPA_INFO
from adapl.privacy.accounting import (
    ClientPrivacyBudgetManager,
    PrivacyBudgetContext,
    gaussian_noise_multiplier,
)
from adapl.privacy.config import PrivacyConfig, build_minimum_privacy_config
from adapl.privacy.mechanisms import client_update_l2_norm
from adapl.utils import clone_state_dict


class FedDPA(FederatedMethod):
    """Dynamic personalized FL using Fisher masks and DP client updates."""

    info = FEDDPA_INFO

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.privacy_config: PrivacyConfig = build_minimum_privacy_config(
            args,
            self.info.display_name,
        )
        self.fisher_threshold: float = args.feddpa_fisher_threshold
        self.lambda1: float = args.feddpa_lambda1
        self.lambda2: float = args.feddpa_lambda2
        self.fisher_batches: int = args.feddpa_fisher_batches
        if not 0 <= self.fisher_threshold <= 1:
            raise ValueError("--feddpa_fisher_threshold must be in [0, 1].")
        if self.lambda1 < 0:
            raise ValueError("--feddpa_lambda1 must be non-negative.")
        if self.lambda2 < 0:
            raise ValueError("--feddpa_lambda2 must be non-negative.")
        if self.fisher_batches < 0:
            raise ValueError("--feddpa_fisher_batches must be non-negative.")

        self.current_epsilon = self.privacy_config.epsilon
        self.current_noise_multiplier = self.privacy_config.noise_multiplier
        self.current_selected_clients: list[int] = []
        self.current_selected_budgets: list[float] = []
        self.current_noise_multipliers: list[float] = []
        self.current_privacy_context: dict[int, PrivacyBudgetContext] = {}
        self.client_states: dict[int, OrderedDict[str, torch.Tensor]] = {}
        self.private_state_keys: tuple[str, ...] | None = None

    def _private_keys(self, model_fn: Callable[[], nn.Module]) -> tuple[str, ...]:
        if self.private_state_keys is None:
            model = model_fn()
            self.private_state_keys = tuple(
                name
                for name, parameter in model.named_parameters()
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
        elif self.args.noise_multiplier is not None:
            return None
        else:
            raise ValueError(
                "FedDPA requires --epsilon_min, --epsilon_max, "
                "--privacy_budgets, --privacy_scenario, or --noise_multiplier."
            )
        return ClientPrivacyBudgetManager.from_client_epsilons(
            client_epsilons=client_epsilons,
            delta=self.privacy_config.delta,
            noise_multiplier=self.args.noise_multiplier,
            epsilon_floor=self.args.epsilon_min,
        )

    def set_privacy_budget_context(
        self,
        context: Mapping[int, PrivacyBudgetContext],
    ) -> None:
        self.current_privacy_context = dict(context)

    def privacy_budget_local_steps(self, train_loader: DataLoader) -> int:
        return 2 * super().privacy_budget_local_steps(train_loader)

    def _estimate_fisher_masks(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        criterion = nn.CrossEntropyLoss()
        model.eval()
        fisher = {
            name: torch.zeros_like(parameter, device=device)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not fisher:
            raise ValueError("FedDPA requires at least one trainable parameter.")

        max_batches = self.fisher_batches or None
        seen_batches = 0
        seen_examples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            batch_size = targets.size(0)
            seen_examples += batch_size
            seen_batches += 1
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    fisher[name].add_(parameter.grad.detach().pow(2) * batch_size)
            if max_batches is not None and seen_batches >= max_batches:
                break

        if seen_examples <= 0:
            raise ValueError("FedDPA Fisher estimation saw no training examples.")

        masks: dict[str, torch.Tensor] = {}
        for name, score in fisher.items():
            score = score / float(seen_examples)
            max_score = torch.max(score)
            if float(max_score.item()) > 0:
                normalized = score / max_score
                masks[name] = (normalized >= self.fisher_threshold).detach().cpu()
            else:
                masks[name] = torch.zeros_like(score, dtype=torch.bool).cpu()
        model.zero_grad(set_to_none=True)
        return masks

    def _merge_personalized_state(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        client_state: OrderedDict[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        merged = OrderedDict()
        for name, global_value in global_state.items():
            if name in masks:
                mask = masks[name]
                client_value = client_state[name]
                merged[name] = torch.where(mask, client_value, global_value).clone()
            else:
                merged[name] = global_value.detach().clone()
        return merged

    def _regularizer_norm(
        self,
        model: nn.Module,
        reference_state: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        use_personal_mask: bool,
        device: torch.device,
    ) -> torch.Tensor:
        squared_norm = torch.zeros((), device=device)
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            mask = masks[name].to(device=device)
            if not use_personal_mask:
                mask = torch.logical_not(mask)
            diff = parameter - reference_state[name]
            diff = diff * mask.to(dtype=parameter.dtype)
            squared_norm = squared_norm + torch.sum(diff * diff)
        return torch.sqrt(squared_norm + 1e-12)

    def _clip_and_noise_masked_gradients(
        self,
        model: nn.Module,
        masks: Mapping[str, torch.Tensor],
        use_personal_mask: bool,
        batch_size: int,
        noise_multiplier: float,
    ) -> dict[str, float]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        masked_gradients: list[tuple[torch.Tensor, torch.Tensor]] = []
        squared_norm = 0.0
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            mask = masks[name].to(device=parameter.grad.device)
            if not use_personal_mask:
                mask = torch.logical_not(mask)
            mask = mask.to(dtype=parameter.grad.dtype)
            parameter.grad.mul_(mask)
            gradient = parameter.grad.detach().double()
            squared_norm += float(torch.sum(gradient * gradient).item())
            masked_gradients.append((parameter.grad, mask))

        grad_norm = squared_norm ** 0.5
        clip_factor = min(1.0, self.privacy_config.clipping_norm / (grad_norm + 1e-12))
        clipped_norm = min(grad_norm, self.privacy_config.clipping_norm)
        noise_std = (
            noise_multiplier * self.privacy_config.clipping_norm / float(batch_size)
        )

        for gradient, mask in masked_gradients:
            gradient.mul_(clip_factor)
            if noise_std > 0:
                gradient.add_(
                    torch.normal(
                        0,
                        noise_std,
                        size=gradient.shape,
                        device=gradient.device,
                        dtype=gradient.dtype,
                    )
                    * mask
                )

        return {
            "grad_norm": grad_norm,
            "clipped_norm": clipped_norm,
            "clip_factor": clip_factor,
            "noise_std": noise_std,
        }

    def _restore_inactive_parameters(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        masks: Mapping[str, torch.Tensor],
        use_personal_mask: bool,
        pre_step_state: Mapping[str, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                active_mask = masks[name].to(device=parameter.device)
                if not use_personal_mask:
                    active_mask = torch.logical_not(active_mask)
                parameter.copy_(
                    torch.where(
                        active_mask,
                        parameter,
                        pre_step_state[name].to(device=parameter.device),
                    )
                )
                state = optimizer.state.get(parameter, {})
                for value in state.values():
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        value.mul_(active_mask.to(dtype=value.dtype))

    def _train_masked_batch(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reference_state: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        use_personal_mask: bool,
        regularizer_weight: float,
        noise_multiplier: float,
        device: torch.device,
    ) -> tuple[float, int, dict[str, float]]:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        task_loss = criterion(logits, targets)
        regularizer = self._regularizer_norm(
            model,
            reference_state,
            masks,
            use_personal_mask,
            device,
        )
        if use_personal_mask:
            loss = task_loss + 0.5 * regularizer_weight * regularizer
        else:
            target_norm = torch.as_tensor(
                self.privacy_config.clipping_norm,
                device=device,
                dtype=regularizer.dtype,
            )
            loss = task_loss + 0.5 * regularizer_weight * torch.abs(
                regularizer - target_norm
            )
        loss.backward()
        gradient_stats = self._clip_and_noise_masked_gradients(
            model,
            masks,
            use_personal_mask,
            batch_size,
            noise_multiplier,
        )
        pre_step_state = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        optimizer.step()
        self._restore_inactive_parameters(
            model,
            optimizer,
            masks,
            use_personal_mask,
            pre_step_state,
        )

        return float(task_loss.item()) * batch_size, batch_size, gradient_stats

    def _run_masked_phase(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        reference_state: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        use_personal_mask: bool,
        regularizer_weight: float,
        noise_multiplier: float,
        device: torch.device,
    ) -> tuple[float, int, dict[str, float]]:
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_examples = 0
        gradient_stats = {
            "grad_norm_sum": 0.0,
            "clipped_norm_sum": 0.0,
            "clip_factor_sum": 0.0,
            "noise_std_sum": 0.0,
            "count": 0.0,
        }
        model.train()

        if self.args.local_update_mode == "random-batch":
            local_steps = (
                self.args.local_epochs
                if self.args.local_epochs is not None
                else self.args.local_steps
            )
            if local_steps <= 0:
                raise ValueError("--local_steps must be positive.")
            train_iter = iter(train_loader)
            for _ in range(local_steps):
                try:
                    inputs, targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    inputs, targets = next(train_iter)
                batch_loss, batch_examples, batch_stats = self._train_masked_batch(
                    model,
                    criterion,
                    optimizer,
                    inputs,
                    targets,
                    reference_state,
                    masks,
                    use_personal_mask,
                    regularizer_weight,
                    noise_multiplier,
                    device,
                )
                total_loss += batch_loss
                total_examples += batch_examples
                gradient_stats["grad_norm_sum"] += batch_stats["grad_norm"]
                gradient_stats["clipped_norm_sum"] += batch_stats["clipped_norm"]
                gradient_stats["clip_factor_sum"] += batch_stats["clip_factor"]
                gradient_stats["noise_std_sum"] += batch_stats["noise_std"]
                gradient_stats["count"] += 1.0
        elif self.args.local_update_mode == "full-epoch":
            local_epochs = (
                self.args.local_epochs
                if self.args.local_epochs is not None
                else self.args.local_steps
            )
            if local_epochs <= 0:
                raise ValueError("--local_epochs must be positive.")
            for _ in range(local_epochs):
                for inputs, targets in train_loader:
                    batch_loss, batch_examples, batch_stats = self._train_masked_batch(
                        model,
                        criterion,
                        optimizer,
                        inputs,
                        targets,
                        reference_state,
                        masks,
                        use_personal_mask,
                        regularizer_weight,
                        noise_multiplier,
                        device,
                    )
                    total_loss += batch_loss
                    total_examples += batch_examples
                    gradient_stats["grad_norm_sum"] += batch_stats["grad_norm"]
                    gradient_stats["clipped_norm_sum"] += batch_stats["clipped_norm"]
                    gradient_stats["clip_factor_sum"] += batch_stats["clip_factor"]
                    gradient_stats["noise_std_sum"] += batch_stats["noise_std"]
                    gradient_stats["count"] += 1.0
        else:
            raise ValueError(
                f"Unsupported local update mode: {self.args.local_update_mode}"
            )
        return total_loss, total_examples, gradient_stats

    def _device_reference_state(
        self,
        state_dict: OrderedDict[str, torch.Tensor],
        private_keys: Sequence[str],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return {
            name: state_dict[name].detach().to(device=device)
            for name in private_keys
        }

    def _mask_ratio(self, masks: Mapping[str, torch.Tensor]) -> tuple[float, int, int]:
        personal_params = 0
        total_params = 0
        for mask in masks.values():
            total_params += mask.numel()
            personal_params += int(mask.sum().item())
        ratio = personal_params / total_params if total_params else 0.0
        return ratio, personal_params, total_params

    def _uses_per_client_noise(self) -> bool:
        return (
            self.privacy_config.privacy_budgets is not None
            and self.args.noise_multiplier is None
        )

    def _noise_multiplier_for_budget(self, epsilon: float) -> float:
        if self.args.noise_multiplier is not None:
            return self.privacy_config.noise_multiplier
        noise_epsilon = float(epsilon)
        if self.args.epsilon_min is not None:
            noise_epsilon = max(noise_epsilon, self.args.epsilon_min)
        return gaussian_noise_multiplier(noise_epsilon, self.privacy_config.delta)

    def _noise_multiplier_config_value(self) -> object:
        if self._uses_per_client_noise():
            return "per_client"
        return self.privacy_config.noise_multiplier

    def _noise_source(self) -> str:
        if self.args.noise_multiplier is not None:
            return "user_noise_multiplier"
        if self._uses_per_client_noise():
            if self.args.epsilon_min is not None:
                return "per_client_eps_floor"
            return "per_client_epsilon_delta_bound"
        if self.args.epsilon_min is not None:
            return "epsilon_min_delta_bound"
        return self.privacy_config.noise_source

    def startup_lines(self) -> list[str]:
        epsilon = (
            f"{self.privacy_config.epsilon}"
            if self.privacy_config.epsilon is not None
            else "not reported"
        )
        fisher_scope = (
            "full_train_loader"
            if self.fisher_batches == 0
            else f"{self.fisher_batches}_batch"
        )
        lines = [
            (
                "FedDPA Fisher-personalized DP training: "
                f"fisher_threshold={self.fisher_threshold}, "
                f"fisher_scope={fisher_scope}, "
                f"lambda1={self.lambda1}, "
                f"lambda2={self.lambda2}, "
                f"epsilon_min={epsilon}, "
                f"delta={self.privacy_config.delta}, "
                f"clipping_norm={self.privacy_config.clipping_norm}, "
                f"noise_multiplier={self._noise_multiplier_config_value()}, "
                "noise_timing=per_minibatch_gradient, "
                "noise_std=noise_multiplier*clipping_norm/batch_size, "
                "private_update_scope=masked_trainable_gradients"
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
        ):
            if self.args.epsilon_min is not None:
                lines.append(
                    "FedDPA uses per-client minibatch gradient noise multipliers "
                    "from max(epsilon_k, epsilon_min), matching WeiAvg DP "
                    "training with an epsilon floor."
                )
            else:
                lines.append(
                    "FedDPA uses per-client minibatch gradient noise multipliers "
                    "from epsilon_k when heterogeneous budgets are provided, "
                    "matching WeiAvg DP training."
                )
        return lines

    def begin_round(self, round_idx: int, selected_clients: Sequence[int]) -> None:
        del round_idx
        self.current_selected_clients = list(selected_clients)

        if self.current_privacy_context:
            self.current_selected_budgets = [
                self.current_privacy_context[client_id].epsilon
                for client_id in self.current_selected_clients
            ]
            self.current_epsilon = min(self.current_selected_budgets)
            self.current_noise_multipliers = [
                self.current_privacy_context[client_id].noise_multiplier
                for client_id in self.current_selected_clients
            ]
            self.current_noise_multiplier = max(self.current_noise_multipliers)
        else:
            self.current_selected_budgets = (
                [
                    self.privacy_config.privacy_budgets[client_id]
                    for client_id in self.current_selected_clients
                ]
                if self.privacy_config.privacy_budgets is not None
                else []
            )
            self.current_epsilon = self.privacy_config.epsilon
            if self._uses_per_client_noise() and self.current_selected_budgets:
                self.current_noise_multipliers = [
                    self._noise_multiplier_for_budget(epsilon)
                    for epsilon in self.current_selected_budgets
                ]
                self.current_noise_multiplier = max(self.current_noise_multipliers)
            else:
                self.current_noise_multiplier = self.privacy_config.noise_multiplier
                self.current_noise_multipliers = [
                    self.current_noise_multiplier
                    for _ in self.current_selected_clients
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
                ("privacy", "mechanism", "minibatch_gradient_gaussian"),
                ("privacy", "epsilon_min", self.privacy_config.epsilon),
                ("privacy", "delta", self.privacy_config.delta),
                ("privacy", "clipping_norm", self.privacy_config.clipping_norm),
                ("privacy", "noise_multiplier", self._noise_multiplier_config_value()),
                (
                    "privacy",
                    "noise_std",
                    "noise_multiplier*clipping_norm/batch_size",
                ),
                ("privacy", "noise_timing", "per_minibatch_gradient"),
                ("privacy", "noise_source", self._noise_source()),
                ("privacy", "private_update_scope", "masked_trainable_gradients"),
                ("privacy", "privacy_budget_count", budget_count),
                ("feddpa", "fisher_threshold", self.fisher_threshold),
                ("feddpa", "fisher_batches", self.fisher_batches),
                ("feddpa", "lambda1", self.lambda1),
                ("feddpa", "lambda2", self.lambda2),
                ("feddpa", "personalized_state", "per_client_persistent"),
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
        elif self.privacy_config.privacy_budgets is not None:
            rows.append(
                (
                    "privacy",
                    "client_budgets",
                    list(self.privacy_config.privacy_budgets),
                )
            )
        return rows

    def config_payload(self) -> dict[str, object]:
        payload = super().config_payload()
        payload["dp_enabled"] = True
        payload["privacy"] = {
            "mechanism": "minibatch_gradient_gaussian",
            "epsilon_min": self.privacy_config.epsilon,
            "delta": self.privacy_config.delta,
            "clipping_norm": self.privacy_config.clipping_norm,
            "noise_multiplier": self._noise_multiplier_config_value(),
            "noise_std": "noise_multiplier*clipping_norm/batch_size",
            "noise_timing": "per_minibatch_gradient",
            "noise_source": self._noise_source(),
            "private_update_scope": "masked_trainable_gradients",
            "privacy_budget_count": (
                len(self.privacy_config.privacy_budgets)
                if self.privacy_config.privacy_budgets
                else 0
            ),
        }
        payload["feddpa"] = {
            "fisher_threshold": self.fisher_threshold,
            "fisher_batches": self.fisher_batches,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "personalized_state": "per_client_persistent",
        }
        if self.privacy_config.privacy_scenario is not None:
            scenario = self.privacy_config.privacy_scenario
            payload["privacy"]["scenario"] = {
                "name": scenario.name,
                "level_budgets": list(scenario.level_budgets),
                "level_counts": list(scenario.level_counts),
                "client_budgets": list(scenario.client_budgets),
            }
        elif self.privacy_config.privacy_budgets is not None:
            payload["privacy"]["client_budgets"] = list(
                self.privacy_config.privacy_budgets
            )
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
        try:
            selected_index = self.current_selected_clients.index(client_id)
        except ValueError as exc:
            raise RuntimeError(
                f"Client {client_id} was not selected in the current round."
            ) from exc
        epsilon = (
            self.current_selected_budgets[selected_index]
            if self.current_selected_budgets
            else self.current_epsilon
        )
        noise_multiplier = self.current_noise_multipliers[selected_index]

        private_keys = self._private_keys(model_fn)
        previous_client_state = self.client_states.get(client_id)
        if previous_client_state is None:
            previous_client_state = clone_state_dict(global_state)

        fisher_model = model_fn().to(device)
        fisher_model.load_state_dict(previous_client_state)
        masks = self._estimate_fisher_masks(fisher_model, train_loader, device)
        del fisher_model

        initial_state = self._merge_personalized_state(
            global_state,
            previous_client_state,
            masks,
        )
        model = model_fn().to(device)
        model.load_state_dict(initial_state)

        personal_reference = self._device_reference_state(
            previous_client_state,
            private_keys,
            device,
        )
        shared_reference = self._device_reference_state(
            global_state,
            private_keys,
            device,
        )
        personal_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        shared_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        personal_loss, personal_examples, personal_stats = self._run_masked_phase(
            model,
            train_loader,
            personal_optimizer,
            personal_reference,
            masks,
            use_personal_mask=True,
            regularizer_weight=self.lambda1,
            noise_multiplier=noise_multiplier,
            device=device,
        )
        shared_loss, shared_examples, shared_stats = self._run_masked_phase(
            model,
            train_loader,
            shared_optimizer,
            shared_reference,
            masks,
            use_personal_mask=False,
            regularizer_weight=self.lambda2,
            noise_multiplier=noise_multiplier,
            device=device,
        )
        loss_examples = personal_examples + shared_examples
        client_loss = (personal_loss + shared_loss) / max(1, loss_examples)
        local_state = clone_state_dict(model.state_dict())
        self.client_states[client_id] = local_state

        gradient_stats = {
            "grad_norm_sum": (
                personal_stats["grad_norm_sum"] + shared_stats["grad_norm_sum"]
            ),
            "clipped_norm_sum": (
                personal_stats["clipped_norm_sum"]
                + shared_stats["clipped_norm_sum"]
            ),
            "clip_factor_sum": (
                personal_stats["clip_factor_sum"] + shared_stats["clip_factor_sum"]
            ),
            "noise_std_sum": (
                personal_stats["noise_std_sum"] + shared_stats["noise_std_sum"]
            ),
            "count": personal_stats["count"] + shared_stats["count"],
        }
        stat_count = max(1.0, gradient_stats["count"])
        update_norm = client_update_l2_norm(
            global_state,
            local_state,
            private_keys,
        )
        mask_ratio, personal_params, total_params = self._mask_ratio(masks)
        return ClientUpdate(
            client_id=client_id,
            state_dict=local_state,
            train_loss=client_loss,
            num_examples=len(train_loader.dataset),
            metadata={
                "update_norm": update_norm,
                "clipped_norm": (
                    gradient_stats["clipped_norm_sum"] / stat_count
                ),
                "clip_factor": gradient_stats["clip_factor_sum"] / stat_count,
                "noise_std": gradient_stats["noise_std_sum"] / stat_count,
                "epsilon": epsilon,
                "epsilon_min": self.current_epsilon,
                "delta": self.privacy_config.delta,
                "noise_multiplier": noise_multiplier,
                "fisher_threshold": self.fisher_threshold,
                "fisher_personalized_ratio": mask_ratio,
                "fisher_personalized_params": personal_params,
                "fisher_total_params": total_params,
                "lambda1": self.lambda1,
                "lambda2": self.lambda2,
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
