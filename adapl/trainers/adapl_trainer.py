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
    fisher_means,
    make_important_masks,
)
from adapl.noise_strategy import LayerNoiseStats, layerwise_noise_stats
from adapl.utils import clone_state_dict


@dataclass(frozen=True)
class AdapLTrainResult:
    state_dict: OrderedDict[str, torch.Tensor]
    train_loss: float
    num_examples: int
    actual_minibatch_steps: int
    sample_grad_norm_mean: float
    sample_clipped_norm_mean: float
    sample_clip_factor_mean: float
    layer_clip_factor_mean: float
    noise_std_mean: float
    noise_multiplier_min: float
    noise_multiplier_max: float
    noise_multiplier_mean: float
    important_ratio: float
    important_params: int
    total_params: int


def _trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


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
) -> tuple[dict[str, torch.Tensor], float, int, dict[str, float]]:
    if clipping_bound <= 0:
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
        clip_factor = min(1.0, clipping_bound / (grad_norm + 1e-12))
        grad_norm_sum += grad_norm
        clipped_norm_sum += min(grad_norm, clipping_bound)
        clip_factor_sum += clip_factor
        for name, grad in sample_grads.items():
            batch_grads[name].add_(grad, alpha=clip_factor)

    for grad in batch_grads.values():
        grad.div_(float(batch_size))

    model.zero_grad(set_to_none=True)
    return (
        batch_grads,
        total_loss,
        batch_size,
        {
            "grad_norm_sum": grad_norm_sum,
            "clipped_norm_sum": clipped_norm_sum,
            "clip_factor_sum": clip_factor_sum,
            "sample_count": float(batch_size),
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
) -> list[str]:
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
) -> tuple[float, float, float, float]:
    noised_names = _noised_layer_names(masks, layer_stats)
    if noised_names:
        sigma_values = [layer_stats[name].sigma for name in noised_names]
        std_values = [layer_stats[name].std for name in noised_names]
        sigma_min = min(sigma_values)
        sigma_max = max(sigma_values)
        sigma_mean = sum(sigma_values) / len(sigma_values)
        std_mean = sum(std_values) / len(std_values)
    else:
        sigma_min = sigma_max = sigma_mean = std_mean = 0.0

    for name, grad in batch_grads.items():
        stats = layer_stats.get(name)
        if stats is None or stats.std <= 0:
            continue
        mask = masks.get(name)
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
        grad.add_(noise * mask_tensor)

    return sigma_min, sigma_max, sigma_mean, std_mean


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
    momentum: float,
    weight_decay: float,
    device: torch.device,
    clipping_bound: float,
    base_noise_multiplier: float,
    gamma: float,
    masks: Mapping[str, torch.Tensor],
    fisher_mean_by_layer: Mapping[str, float],
    max_clip_norm: float | None,
) -> AdapLTrainResult:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    named_parameters = _trainable_named_parameters(model)
    if not fisher_mean_by_layer:
        fisher_mean_by_layer = {name: 0.0 for name, _ in named_parameters}

    total_loss = 0.0
    total_examples = 0
    actual_steps = 0
    sample_grad_norm_sum = 0.0
    sample_clipped_norm_sum = 0.0
    sample_clip_factor_sum = 0.0
    sample_count = 0.0
    layer_clip_factor_sum = 0.0
    noise_std_sum = 0.0
    noise_multiplier_min_values: list[float] = []
    noise_multiplier_max_values: list[float] = []
    noise_multiplier_mean_values: list[float] = []

    for inputs, targets in _minibatches(
        train_loader,
        local_steps,
        local_epochs,
        local_update_mode,
    ):
        batch_grads, batch_loss, batch_size, batch_stats = (
            _per_sample_clipped_batch_gradient(
                model=model,
                criterion=criterion,
                inputs=inputs,
                targets=targets,
                device=device,
                clipping_bound=clipping_bound,
            )
        )
        layer_clip_factor = _apply_decay_layer_clipping(batch_grads, max_clip_norm)
        stats_by_layer = layerwise_noise_stats(
            base_noise_multiplier=base_noise_multiplier,
            fisher_mean_by_layer=fisher_mean_by_layer,
            gamma=gamma,
            clipping_bound=clipping_bound,
            batch_size=batch_size,
        )
        sigma_min, sigma_max, sigma_mean, std_mean = _apply_masked_noise(
            batch_grads,
            masks,
            stats_by_layer,
        )
        _install_gradients_and_step(model, optimizer, batch_grads)

        total_loss += batch_loss
        total_examples += batch_size
        actual_steps += 1
        sample_grad_norm_sum += batch_stats["grad_norm_sum"]
        sample_clipped_norm_sum += batch_stats["clipped_norm_sum"]
        sample_clip_factor_sum += batch_stats["clip_factor_sum"]
        sample_count += batch_stats["sample_count"]
        layer_clip_factor_sum += layer_clip_factor
        noise_std_sum += std_mean
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
        sample_clipped_norm_mean=sample_clipped_norm_sum / max(1.0, sample_count),
        sample_clip_factor_mean=sample_clip_factor_sum / max(1.0, sample_count),
        layer_clip_factor_mean=layer_clip_factor_sum / max(1, actual_steps),
        noise_std_mean=noise_std_sum / max(1, actual_steps),
        noise_multiplier_min=min(noise_multiplier_min_values),
        noise_multiplier_max=max(noise_multiplier_max_values),
        noise_multiplier_mean=sum(noise_multiplier_mean_values)
        / len(noise_multiplier_mean_values),
        important_ratio=important_ratio,
        important_params=important_params,
        total_params=total_params,
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
    momentum: float,
    weight_decay: float,
    device: torch.device,
    clipping_bound: float,
    base_noise_multiplier: float,
    gamma: float,
) -> AdapLTrainResult:
    model = model_fn().to(device)
    model.load_state_dict(global_state)
    masks = all_trainable_important_masks(model)
    fisher_mean_by_layer = {
        name: 0.0
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return _run_adapl_update(
        model=model,
        train_loader=train_loader,
        local_steps=local_steps,
        local_epochs=local_epochs,
        local_update_mode=local_update_mode,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        device=device,
        clipping_bound=clipping_bound,
        base_noise_multiplier=base_noise_multiplier,
        gamma=gamma,
        masks=masks,
        fisher_mean_by_layer=fisher_mean_by_layer,
        max_clip_norm=None,
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
    momentum: float,
    weight_decay: float,
    device: torch.device,
    clipping_bound: float,
    base_noise_multiplier: float,
    gamma: float,
    fisher_threshold: float,
    fisher_estimator: str,
    fisher_batches: int,
    max_clip_norm: float | None,
) -> AdapLTrainResult:
    fisher_model = model_fn().to(device)
    fisher_model.load_state_dict(latest_global_state)
    fisher_diag = compute_fisher_diag(
        fisher_model,
        train_loader,
        device,
        estimator=fisher_estimator,
        max_batches=None if fisher_batches == 0 else fisher_batches,
    )
    masks = make_important_masks(fisher_diag, fisher_threshold)
    fisher_mean_by_layer = fisher_means(fisher_diag)
    del fisher_model

    model = model_fn().to(device)
    model.load_state_dict(global_state)
    return _run_adapl_update(
        model=model,
        train_loader=train_loader,
        local_steps=local_steps,
        local_epochs=local_epochs,
        local_update_mode=local_update_mode,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        device=device,
        clipping_bound=clipping_bound,
        base_noise_multiplier=base_noise_multiplier,
        gamma=gamma,
        masks=masks,
        fisher_mean_by_layer=fisher_mean_by_layer,
        max_clip_norm=max_clip_norm,
    )
