"""AdapL local training with per-sample clipping and per-minibatch noise."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from adapl.fisher import (
    all_trainable_important_masks,
    compute_fisher_diag,
    fisher_important_means,
    make_important_masks,
)
from adapl.noise_strategy import (
    LayerNoiseStats,
    MAX_NOISE_RATIO,
    layerwise_noise_stats,
)
from adapl.utils import clone_state_dict


@dataclass(frozen=True)
class AdapLTrainResult:
    state_dict: OrderedDict[str, torch.Tensor]
    train_loss: float
    num_examples: int
    actual_minibatch_steps: int
    sample_grad_norm_mean: float
    sample_grad_norm_p50: float
    sample_grad_norm_p90: float
    sample_grad_norm_p99: float
    sample_clipped_norm_mean: float
    sample_clip_factor_mean: float
    sample_clip_fraction: float
    layer_clip_factor_mean: float
    coordinate_clip_fraction_mean: float
    coordinate_clip_radius_mean: float
    proximal_norm_mean: float
    noise_std_mean: float
    signal_l2_mean: float
    noise_l2_mean: float
    noise_to_signal_ratio_mean: float
    noise_multiplier_min: float
    noise_multiplier_max: float
    noise_multiplier_mean: float
    important_ratio: float
    important_params: int
    total_params: int
    min_fisher_mean: float = 0.0
    max_fisher_mean: float = 0.0
    max_noise_ratio: float = 0.0
    max_noise_ratio_configured: float = MAX_NOISE_RATIO
    fallback_layers: str = ""


def _trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def _batch_norm_modules(model: nn.Module) -> list[nn.Module]:
    return [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]


def _update_batch_norm_once_and_freeze(
    model: nn.Module,
    inputs: torch.Tensor,
    freeze_running_stats: bool,
) -> list[tuple[nn.Module, bool]]:
    batch_norm_modules = _batch_norm_modules(model)
    states = [(module, module.training) for module in batch_norm_modules]
    if not batch_norm_modules:
        return states

    if not freeze_running_stats:
        for module in batch_norm_modules:
            module.train(True)
        with torch.no_grad():
            model(inputs)
        model.zero_grad(set_to_none=True)
    for module in batch_norm_modules:
        module.train(False)
    return states


def _restore_batch_norm_training(states: list[tuple[nn.Module, bool]]) -> None:
    for module, training in states:
        module.train(training)


def _planned_minibatches(
    train_loader: DataLoader,
    local_steps: int,
    local_epochs: Optional[int],
    local_update_mode: str,
) -> int:
    if local_update_mode == "random-batch":
        steps = local_epochs if local_epochs is not None else local_steps
        if steps <= 0:
            raise ValueError("--local_steps must be positive.")
        return steps
    if local_update_mode == "full-epoch":
        epochs = local_epochs if local_epochs is not None else local_steps
        if epochs <= 0:
            raise ValueError("--local_epochs must be positive.")
        return epochs * len(train_loader)
    raise ValueError(f"Unsupported local update mode: {local_update_mode}")


def _minibatches(
    train_loader: DataLoader,
    local_steps: int,
    local_epochs: Optional[int],
    local_update_mode: str,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    planned = _planned_minibatches(
        train_loader,
        local_steps,
        local_epochs,
        local_update_mode,
    )
    if planned <= 0:
        return

    if local_update_mode == "random-batch":
        train_iter = iter(train_loader)
        for _ in range(planned):
            try:
                yield next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                try:
                    yield next(train_iter)
                except StopIteration as exc:
                    raise ValueError("Client train_loader produced no minibatches.") from exc
        return

    epochs = local_epochs if local_epochs is not None else local_steps
    for _ in range(epochs):
        for batch in train_loader:
            yield batch


def _per_sample_clipped_batch_gradient(
    model: nn.Module,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    clipping_bound: float,
    enable_clipping: bool,
    freeze_batch_norm: bool,
) -> tuple[dict[str, torch.Tensor], float, int, dict[str, object]]:
    if enable_clipping and clipping_bound <= 0:
        raise ValueError("clipping_bound must be positive.")

    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    batch_size = int(targets.size(0))
    if batch_size <= 0:
        raise ValueError("Empty minibatch is not supported.")

    named_parameters = _trainable_named_parameters(model)
    batch_grads = {
        name: torch.zeros_like(parameter, device=device)
        for name, parameter in named_parameters
    }
    total_loss = 0.0
    grad_norm_sum = 0.0
    clipped_norm_sum = 0.0
    clip_factor_sum = 0.0
    clipped_sample_count = 0
    grad_norm_values: list[float] = []

    batch_norm_states = _update_batch_norm_once_and_freeze(
        model,
        inputs,
        freeze_running_stats=freeze_batch_norm,
    )
    try:
        for sample_idx in range(batch_size):
            model.zero_grad(set_to_none=True)
            logits = model(inputs[sample_idx : sample_idx + 1])
            loss = criterion(logits, targets[sample_idx : sample_idx + 1])
            loss.backward()
            total_loss += float(loss.item())

            sample_grads: dict[str, torch.Tensor] = {}
            squared_norm = 0.0
            for name, parameter in named_parameters:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach().clone()
                sample_grads[name] = grad
                squared_norm += float(torch.sum(grad.double() * grad.double()).item())

            grad_norm = squared_norm ** 0.5
            clip_factor = (
                min(1.0, clipping_bound / (grad_norm + 1e-12))
                if enable_clipping
                else 1.0
            )
            grad_norm_sum += grad_norm
            clipped_norm_sum += grad_norm * clip_factor
            clip_factor_sum += clip_factor
            clipped_sample_count += int(clip_factor < 1.0)
            grad_norm_values.append(grad_norm)
            for name, grad in sample_grads.items():
                batch_grads[name].add_(grad, alpha=clip_factor)

        for grad in batch_grads.values():
            grad.div_(float(batch_size))
    finally:
        _restore_batch_norm_training(batch_norm_states)

    model.zero_grad(set_to_none=True)
    return (
        batch_grads,
        total_loss,
        batch_size,
        {
            "grad_norm_sum": grad_norm_sum,
            "clipped_norm_sum": clipped_norm_sum,
            "clip_factor_sum": clip_factor_sum,
            "clipped_sample_count": float(clipped_sample_count),
            "sample_count": float(batch_size),
            "grad_norm_values": grad_norm_values,
        },
    )


def _batch_gradient(
    model: nn.Module,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float, int, dict[str, object]]:
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    batch_size = int(targets.size(0))
    if batch_size <= 0:
        raise ValueError("Empty minibatch is not supported.")

    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()

    named_parameters = _trainable_named_parameters(model)
    batch_grads: dict[str, torch.Tensor] = {}
    squared_norm = 0.0
    for name, parameter in named_parameters:
        if parameter.grad is None:
            batch_grads[name] = torch.zeros_like(parameter, device=device)
            continue
        grad = parameter.grad.detach().clone()
        batch_grads[name] = grad
        squared_norm += float(torch.sum(grad.double() * grad.double()).item())

    grad_norm = squared_norm ** 0.5
    model.zero_grad(set_to_none=True)
    return (
        batch_grads,
        float(loss.item()) * batch_size,
        batch_size,
        {
            "grad_norm_sum": grad_norm * batch_size,
            "clipped_norm_sum": grad_norm * batch_size,
            "clip_factor_sum": float(batch_size),
            "clipped_sample_count": 0.0,
            "sample_count": float(batch_size),
            "grad_norm_values": [grad_norm],
        },
    )


def _apply_decay_layer_clipping(
    batch_grads: Mapping[str, torch.Tensor],
    max_clip_norm: float | None,
) -> float:
    if max_clip_norm is None:
        return 1.0
    if max_clip_norm <= 0:
        raise ValueError("max_clip_norm must be positive.")

    factor_sum = 0.0
    count = 0
    for grad in batch_grads.values():
        layer_norm = float(grad.detach().norm(2).item())
        factor = min(1.0, max_clip_norm / (layer_norm + 1e-12))
        grad.mul_(factor)
        factor_sum += factor
        count += 1
    return factor_sum / max(1, count)


def _apply_default_gradient_clipping(
    batch_grads: Mapping[str, torch.Tensor],
    clipping_bound: float,
) -> float:
    if clipping_bound <= 0:
        raise ValueError("clipping_bound must be positive.")
    grad_norm = _l2_norm(batch_grads)
    factor = min(1.0, clipping_bound / (grad_norm + 1e-12))
    for grad in batch_grads.values():
        grad.mul_(factor)
    return factor


def _apply_proximal_gradient(
    batch_grads: Mapping[str, torch.Tensor],
    model: nn.Module,
    reference_state: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    prox_mu: float,
) -> float:
    if prox_mu < 0:
        raise ValueError("prox_mu must be non-negative.")
    if prox_mu == 0:
        return 0.0

    squared_norm = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name not in batch_grads:
            continue
        reference = reference_state.get(name)
        if reference is None:
            continue
        reference = reference.to(device=parameter.device, dtype=parameter.dtype)
        proximal_grad = parameter.detach() - reference
        mask = masks.get(name)
        if mask is not None:
            proximal_grad = proximal_grad * mask.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
        batch_grads[name].add_(proximal_grad, alpha=2.0 * prox_mu)
        squared_norm += float(torch.sum(proximal_grad.double() * proximal_grad.double()).item())
    return squared_norm ** 0.5


def _apply_layerwise_gradient_clipping(
    batch_grads: Mapping[str, torch.Tensor],
    model: nn.Module,
    best_global_state: Mapping[str, torch.Tensor],
    learning_rate: float,
    privacy_level: float,
    max_clip_norm: float,
) -> tuple[float, float, float, dict[str, float]]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if max_clip_norm <= 0:
        raise ValueError("max_clip_norm must be positive.")
    if privacy_level <= 0:
        raise ValueError("privacy_level must be positive.")

    clipped_coordinates = 0
    total_coordinates = 0
    radius_sum = 0.0
    radius_count = 0
    layer_factor_sum = 0.0
    layer_count = 0
    layer_noise_bounds: dict[str, float] = {}

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name not in batch_grads:
                continue
            if parameter.numel() == 0:
                continue
            best_value = best_global_state.get(name)
            if best_value is None:
                continue
            best_value = best_value.to(device=parameter.device, dtype=parameter.dtype)
            best_detached = best_value.detach()
            min_val = torch.min(best_detached)
            max_val = torch.max(best_detached)
            center = (min_val + max_val) / 2.0
            radius = (max_val - min_val) / 2.0 * float(privacy_level)
            lower_parameter = center - radius
            upper_parameter = center + radius

            lower_grad = (parameter.detach() - upper_parameter) / learning_rate
            upper_grad = (parameter.detach() - lower_parameter) / learning_rate

            grad = batch_grads[name]
            original = grad.detach().clone()
            clipped = torch.maximum(torch.minimum(grad, upper_grad), lower_grad)
            grad.copy_(clipped)
            clipped_coordinates += int((clipped != original).sum().item())
            total_coordinates += grad.numel()
            radius_sum += float(radius.item())
            radius_count += 1

            layer_norm = float(grad.detach().norm(2).item())
            factor = min(1.0, float(max_clip_norm) / (layer_norm + 1e-12))
            grad.mul_(factor)
            layer_factor_sum += factor
            layer_count += 1
            noise_bound = max(float(radius.item()) / learning_rate, 1e-12)
            layer_noise_bounds[name] = noise_bound

    coordinate_clip_fraction = (
        clipped_coordinates / float(total_coordinates) if total_coordinates else 0.0
    )
    radius_mean = radius_sum / float(radius_count) if radius_count else 0.0
    layer_factor_mean = layer_factor_sum / float(layer_count) if layer_count else 1.0
    return (
        layer_factor_mean,
        coordinate_clip_fraction,
        radius_mean,
        layer_noise_bounds,
    )


def _manual_gradient_step(
    model: nn.Module,
    batch_grads: Mapping[str, torch.Tensor],
    learning_rate: float,
) -> None:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            grad = batch_grads.get(name)
            if grad is None:
                continue
            parameter.add_(grad, alpha=-learning_rate)


def _masked_fisher_means(
    fisher_mean_by_layer: Mapping[str, float],
    masks: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    means = {}
    for name, value in fisher_mean_by_layer.items():
        mask = masks.get(name)
        if mask is not None and bool(mask.any().item()):
            means[name] = float(value)
    return means


def _mask_summary(masks: Mapping[str, torch.Tensor]) -> tuple[float, int, int]:
    important = 0
    total = 0
    for mask in masks.values():
        important += int(mask.sum().item())
        total += mask.numel()
    ratio = important / total if total else 0.0
    return ratio, important, total


def _noised_layer_names(
    masks: Mapping[str, torch.Tensor],
    layer_stats: Mapping[str, LayerNoiseStats],
    noise_scope: str,
) -> list[str]:
    if noise_scope not in {"fisher", "all"}:
        raise ValueError("noise_scope must be 'fisher' or 'all'.")
    if noise_scope == "all":
        return list(layer_stats)
    names = []
    for name in layer_stats:
        mask = masks.get(name)
        if mask is None or bool(mask.any().item()):
            names.append(name)
    return names


def _apply_masked_noise(
    batch_grads: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    layer_stats: Mapping[str, LayerNoiseStats],
    noise_scope: str,
) -> tuple[float, float, float, float, float]:
    noised_names = _noised_layer_names(masks, layer_stats, noise_scope)
    if noised_names:
        sigma_values = [layer_stats[name].sigma for name in noised_names]
        std_values = [layer_stats[name].std for name in noised_names]
        sigma_min = min(sigma_values)
        sigma_max = max(sigma_values)
        sigma_mean = sum(sigma_values) / len(sigma_values)
        std_mean = sum(std_values) / len(std_values)
    else:
        sigma_min = sigma_max = sigma_mean = std_mean = 0.0

    noise_squared_norm: torch.Tensor | None = None
    for name, grad in batch_grads.items():
        stats = layer_stats.get(name)
        if stats is None or stats.std <= 0:
            continue
        mask = None if noise_scope == "all" else masks.get(name)
        if mask is None:
            mask_tensor = torch.ones_like(grad, dtype=grad.dtype, device=grad.device)
        else:
            mask_tensor = mask.to(device=grad.device, dtype=grad.dtype)
        if not bool(mask_tensor.bool().any().item()):
            continue
        noise = torch.normal(
            mean=0.0,
            std=stats.std,
            size=grad.shape,
            device=grad.device,
            dtype=grad.dtype,
        )
        applied_noise = noise * mask_tensor
        grad.add_(applied_noise)
        squared = torch.sum(applied_noise.detach().float().square())
        noise_squared_norm = (
            squared if noise_squared_norm is None else noise_squared_norm + squared
        )

    noise_l2 = (
        0.0
        if noise_squared_norm is None
        else float(torch.sqrt(noise_squared_norm).item())
    )
    return sigma_min, sigma_max, sigma_mean, std_mean, noise_l2


def _l2_norm(tensors: Mapping[str, torch.Tensor]) -> float:
    squared_norm: torch.Tensor | None = None
    for tensor in tensors.values():
        squared = torch.sum(tensor.detach().float().square())
        squared_norm = squared if squared_norm is None else squared_norm + squared
    if squared_norm is None:
        return 0.0
    return float(torch.sqrt(squared_norm).item())


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be in [0, 1].")
    ordered = sorted(values)
    position = percentile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _install_gradients_and_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_grads: Mapping[str, torch.Tensor],
) -> None:
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter.grad = batch_grads[name].detach().clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _run_adapl_update(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    local_steps: int,
    local_epochs: Optional[int],
    local_update_mode: str,
    lr: float,
    device: torch.device,
    clipping_bound: float,
    base_noise_multiplier: float,
    gamma: float,
    max_noise_ratio: float = MAX_NOISE_RATIO,
    masks: Mapping[str, torch.Tensor],
    fisher_mean_by_layer: Mapping[str, float],
    max_clip_norm: float | None,
    global_reference_state: Mapping[str, torch.Tensor],
    coordinate_clip_center_state: Mapping[str, torch.Tensor] | None,
    privacy_level: float,
    prox_mu: float,
    enable_clipping: bool,
    enable_noise: bool,
    noise_scope: str,
    freeze_batch_norm: bool,
) -> AdapLTrainResult:
    if noise_scope not in {"fisher", "all"}:
        raise ValueError("noise_scope must be 'fisher' or 'all'.")
    model.train()
    criterion = nn.CrossEntropyLoss()

    named_parameters = _trainable_named_parameters(model)
    if not fisher_mean_by_layer:
        fisher_mean_by_layer = {name: 0.0 for name, _ in named_parameters}

    total_loss = 0.0
    total_examples = 0
    actual_steps = 0
    sample_grad_norm_sum = 0.0
    sample_clipped_norm_sum = 0.0
    sample_clip_factor_sum = 0.0
    clipped_sample_count = 0.0
    sample_count = 0.0
    sample_grad_norm_values: list[float] = []
    layer_clip_factor_sum = 0.0
    coordinate_clip_fraction_sum = 0.0
    coordinate_clip_radius_sum = 0.0
    proximal_norm_sum = 0.0
    noise_std_sum = 0.0
    signal_l2_sum = 0.0
    noise_l2_sum = 0.0
    noise_to_signal_ratio_sum = 0.0
    noise_multiplier_min_values: list[float] = []
    noise_multiplier_max_values: list[float] = []
    noise_multiplier_mean_values: list[float] = []

    fisher_values = [float(v) for v in fisher_mean_by_layer.values()]
    min_fisher_mean = min(fisher_values) if fisher_values else 0.0
    max_fisher_mean = max(fisher_values) if fisher_values else 0.0
    max_noise_ratio_value = 0.0
    fallback_layer_names: set[str] = set()

    for inputs, targets in _minibatches(
        train_loader,
        local_steps,
        local_epochs,
        local_update_mode,
    ):
        batch_grads, batch_loss, batch_size, batch_stats = _batch_gradient(
            model=model,
            criterion=criterion,
            inputs=inputs,
            targets=targets,
            device=device,
        )
        proximal_norm_sum += _apply_proximal_gradient(
            batch_grads=batch_grads,
            model=model,
            reference_state=global_reference_state,
            masks=masks,
            prox_mu=prox_mu,
        )
        if enable_clipping and coordinate_clip_center_state is not None:
            if max_clip_norm is None:
                raise ValueError("max_clip_norm is required for layer-wise clipping.")
            (
                layer_clip_factor,
                coordinate_clip_fraction,
                coordinate_clip_radius,
                noise_bounds,
            ) = _apply_layerwise_gradient_clipping(
                batch_grads=batch_grads,
                model=model,
                best_global_state=coordinate_clip_center_state,
                learning_rate=lr,
                privacy_level=privacy_level,
                max_clip_norm=max_clip_norm,
            )
        elif enable_clipping:
            layer_clip_factor = _apply_default_gradient_clipping(
                batch_grads,
                clipping_bound,
            )
            coordinate_clip_fraction = 0.0
            coordinate_clip_radius = 0.0
            noise_bounds = {name: float(clipping_bound) for name in batch_grads}
        else:
            layer_clip_factor = 1.0
            coordinate_clip_fraction = 0.0
            coordinate_clip_radius = 0.0
            noise_bounds = {name: float(clipping_bound) for name in batch_grads}
        signal_l2 = _l2_norm(batch_grads)
        for name in list(noise_bounds.keys()):
            if noise_bounds[name] <= 0:
                noise_bounds[name] = max(float(clipping_bound), 1e-12)
                fallback_layer_names.add(name)
        for name in fisher_mean_by_layer:
            if name not in noise_bounds:
                noise_bounds[name] = max(float(clipping_bound), 1e-12)
        if enable_noise:
            stats_by_layer = layerwise_noise_stats(
                base_noise_multiplier=base_noise_multiplier,
                fisher_mean_by_layer=fisher_mean_by_layer,
                gamma=gamma,
                clipping_bound=noise_bounds,
                max_noise_ratio=max_noise_ratio,
            )
            for s in stats_by_layer.values():
                max_noise_ratio_value = max(
                    max_noise_ratio_value,
                    s.sigma / base_noise_multiplier,
                )
            sigma_min, sigma_max, sigma_mean, std_mean, noise_l2 = (
                _apply_masked_noise(
                    batch_grads,
                    masks,
                    stats_by_layer,
                    noise_scope,
                )
            )
        else:
            sigma_min = sigma_max = sigma_mean = std_mean = noise_l2 = 0.0
        _manual_gradient_step(model, batch_grads, lr)

        total_loss += batch_loss
        total_examples += batch_size
        actual_steps += 1
        sample_grad_norm_sum += float(batch_stats["grad_norm_sum"])
        sample_clipped_norm_sum += float(batch_stats["clipped_norm_sum"])
        sample_clip_factor_sum += float(batch_stats["clip_factor_sum"])
        clipped_sample_count += float(batch_stats["clipped_sample_count"])
        sample_count += float(batch_stats["sample_count"])
        sample_grad_norm_values.extend(batch_stats["grad_norm_values"])
        layer_clip_factor_sum += layer_clip_factor
        coordinate_clip_fraction_sum += coordinate_clip_fraction
        coordinate_clip_radius_sum += coordinate_clip_radius
        noise_std_sum += std_mean
        signal_l2_sum += signal_l2
        noise_l2_sum += noise_l2
        noise_to_signal_ratio_sum += noise_l2 / max(signal_l2, 1e-12)
        noise_multiplier_min_values.append(sigma_min)
        noise_multiplier_max_values.append(sigma_max)
        noise_multiplier_mean_values.append(sigma_mean)

    if actual_steps <= 0:
        raise ValueError("AdapL local update produced no minibatch steps.")

    important_ratio, important_params, total_params = _mask_summary(masks)
    return AdapLTrainResult(
        state_dict=clone_state_dict(model.state_dict()),
        train_loss=total_loss / max(1, total_examples),
        num_examples=len(train_loader.dataset),
        actual_minibatch_steps=actual_steps,
        sample_grad_norm_mean=sample_grad_norm_sum / max(1.0, sample_count),
        sample_grad_norm_p50=_percentile(sample_grad_norm_values, 0.50),
        sample_grad_norm_p90=_percentile(sample_grad_norm_values, 0.90),
        sample_grad_norm_p99=_percentile(sample_grad_norm_values, 0.99),
        sample_clipped_norm_mean=sample_clipped_norm_sum / max(1.0, sample_count),
        sample_clip_factor_mean=sample_clip_factor_sum / max(1.0, sample_count),
        sample_clip_fraction=clipped_sample_count / max(1.0, sample_count),
        layer_clip_factor_mean=layer_clip_factor_sum / max(1, actual_steps),
        coordinate_clip_fraction_mean=(
            coordinate_clip_fraction_sum / max(1, actual_steps)
        ),
        coordinate_clip_radius_mean=coordinate_clip_radius_sum / max(1, actual_steps),
        proximal_norm_mean=proximal_norm_sum / max(1, actual_steps),
        noise_std_mean=noise_std_sum / max(1, actual_steps),
        signal_l2_mean=signal_l2_sum / max(1, actual_steps),
        noise_l2_mean=noise_l2_sum / max(1, actual_steps),
        noise_to_signal_ratio_mean=(
            noise_to_signal_ratio_sum / max(1, actual_steps)
        ),
        noise_multiplier_min=min(noise_multiplier_min_values),
        noise_multiplier_max=max(noise_multiplier_max_values),
        noise_multiplier_mean=sum(noise_multiplier_mean_values)
        / len(noise_multiplier_mean_values),
        important_ratio=important_ratio,
        important_params=important_params,
        total_params=total_params,
        min_fisher_mean=min_fisher_mean,
        max_fisher_mean=max_fisher_mean,
        max_noise_ratio=max_noise_ratio_value,
        max_noise_ratio_configured=float(max_noise_ratio),
        fallback_layers=",".join(sorted(fallback_layer_names)),
    )


def local_update_first(
    *,
    model_fn: Callable[[], nn.Module],
    global_state: OrderedDict[str, torch.Tensor],
    train_loader: DataLoader,
    local_steps: int,
    local_epochs: Optional[int],
    local_update_mode: str,
    lr: float,
    device: torch.device,
    clipping_bound: float,
    base_noise_multiplier: float,
    gamma: float,
    max_noise_ratio: float = MAX_NOISE_RATIO,
    fisher_threshold: float,
    fisher_estimator: str,
    fisher_batches: int,
    prox_mu: float,
    enable_clipping: bool = True,
    enable_noise: bool = True,
    enable_fisher: bool = True,
    noise_scope: str = "fisher",
    freeze_batch_norm: bool = False,
) -> AdapLTrainResult:
    if enable_fisher:
        fisher_model = model_fn().to(device)
        fisher_model.load_state_dict(global_state)
        fisher_diag = compute_fisher_diag(
            fisher_model,
            train_loader,
            device,
            estimator=fisher_estimator,
            max_batches=None if fisher_batches == 0 else fisher_batches,
        )
        masks = make_important_masks(fisher_diag, fisher_threshold)
        fisher_mean_by_layer = fisher_important_means(fisher_diag, masks)
        del fisher_model
    else:
        mask_model = model_fn().to(device)
        masks = all_trainable_important_masks(mask_model)
        fisher_mean_by_layer = {
            name: 0.0
            for name, parameter in mask_model.named_parameters()
            if parameter.requires_grad
        }
        del mask_model

    model = model_fn().to(device)
    model.load_state_dict(global_state)
    return _run_adapl_update(
        model=model,
        train_loader=train_loader,
        local_steps=local_steps,
        local_epochs=local_epochs,
        local_update_mode=local_update_mode,
        lr=lr,
        device=device,
        clipping_bound=clipping_bound,
        base_noise_multiplier=base_noise_multiplier,
        gamma=gamma,
        max_noise_ratio=max_noise_ratio,
        masks=masks,
        fisher_mean_by_layer=fisher_mean_by_layer,
        max_clip_norm=None,
        global_reference_state=global_state,
        coordinate_clip_center_state=None,
        privacy_level=1.0,
        prox_mu=prox_mu,
        enable_clipping=enable_clipping,
        enable_noise=enable_noise,
        noise_scope=noise_scope,
        freeze_batch_norm=freeze_batch_norm,
    )


def local_update_decay(
    *,
    model_fn: Callable[[], nn.Module],
    global_state: OrderedDict[str, torch.Tensor],
    latest_global_state: OrderedDict[str, torch.Tensor],
    train_loader: DataLoader,
    local_steps: int,
    local_epochs: Optional[int],
    local_update_mode: str,
    lr: float,
    device: torch.device,
    clipping_bound: float,
    base_noise_multiplier: float,
    gamma: float,
    max_noise_ratio: float = MAX_NOISE_RATIO,
    fisher_threshold: float,
    fisher_estimator: str,
    fisher_batches: int,
    max_clip_norm: float | None,
    privacy_level: float,
    prox_mu: float,
    enable_clipping: bool = True,
    enable_noise: bool = True,
    enable_fisher: bool = True,
    noise_scope: str = "fisher",
    freeze_batch_norm: bool = False,
) -> AdapLTrainResult:
    if enable_fisher:
        fisher_model = model_fn().to(device)
        fisher_model.load_state_dict(global_state)
        fisher_diag = compute_fisher_diag(
            fisher_model,
            train_loader,
            device,
            estimator=fisher_estimator,
            max_batches=None if fisher_batches == 0 else fisher_batches,
        )
        masks = make_important_masks(fisher_diag, fisher_threshold)
        fisher_mean_by_layer = fisher_important_means(fisher_diag, masks)
        del fisher_model
    else:
        mask_model = model_fn().to(device)
        masks = all_trainable_important_masks(mask_model)
        fisher_mean_by_layer = {
            name: 0.0
            for name, parameter in mask_model.named_parameters()
            if parameter.requires_grad
        }
        del mask_model

    model = model_fn().to(device)
    model.load_state_dict(global_state)
    return _run_adapl_update(
        model=model,
        train_loader=train_loader,
        local_steps=local_steps,
        local_epochs=local_epochs,
        local_update_mode=local_update_mode,
        lr=lr,
        device=device,
        clipping_bound=clipping_bound,
        base_noise_multiplier=base_noise_multiplier,
        gamma=gamma,
        max_noise_ratio=max_noise_ratio,
        masks=masks,
        fisher_mean_by_layer=fisher_mean_by_layer,
        max_clip_norm=max_clip_norm if enable_clipping else None,
        global_reference_state=global_state,
        coordinate_clip_center_state=latest_global_state,
        privacy_level=privacy_level,
        prox_mu=prox_mu,
        enable_clipping=enable_clipping,
        enable_noise=enable_noise,
        noise_scope=noise_scope,
        freeze_batch_norm=freeze_batch_norm,
    )
