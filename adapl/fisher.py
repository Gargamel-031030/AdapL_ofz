"""Fisher diagonal estimation and important-parameter masks for AdapL."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader


def _trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def compute_fisher_diag(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    *,
    estimator: str = "batch",
    max_batches: int | None = 1,
) -> dict[str, torch.Tensor]:
    """Estimate the diagonal Fisher information for trainable parameters."""
    if estimator not in {"sample", "batch"}:
        raise ValueError("estimator must be 'sample' or 'batch'.")
    if max_batches is not None and max_batches < 0:
        raise ValueError("max_batches must be non-negative or None.")

    named_parameters = _trainable_named_parameters(model)
    if not named_parameters:
        raise ValueError("Fisher estimation requires trainable parameters.")

    criterion = nn.CrossEntropyLoss()
    fisher = {
        name: torch.zeros_like(parameter, device=device)
        for name, parameter in named_parameters
    }
    model.eval()
    seen_examples = 0
    seen_batches = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = int(targets.size(0))
        if batch_size <= 0:
            continue

        if estimator == "sample":
            for sample_idx in range(batch_size):
                model.zero_grad(set_to_none=True)
                logits = model(inputs[sample_idx : sample_idx + 1])
                loss = criterion(logits, targets[sample_idx : sample_idx + 1])
                loss.backward()
                for name, parameter in named_parameters:
                    if parameter.grad is not None:
                        fisher[name].add_(parameter.grad.detach().pow(2))
                seen_examples += 1
        else:
            model.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            for name, parameter in named_parameters:
                if parameter.grad is not None:
                    fisher[name].add_(parameter.grad.detach().pow(2) * batch_size)
            seen_examples += batch_size

        seen_batches += 1
        if max_batches is not None and seen_batches >= max_batches:
            break

    model.zero_grad(set_to_none=True)
    if seen_examples <= 0:
        raise ValueError("Fisher estimation saw no training examples.")
    return {
        name: score.detach() / float(seen_examples)
        for name, score in fisher.items()
    }


def fisher_means(
    fisher_diag: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Return the mean Fisher score per trainable tensor."""
    return {
        name: float(score.detach().float().mean().item())
        for name, score in fisher_diag.items()
    }


def make_important_masks(
    fisher_diag: Mapping[str, torch.Tensor],
    threshold: float,
) -> dict[str, torch.Tensor]:
    """Build boolean masks using Eq. (8): Im_i = 1 if F_i >= kappa."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    masks: Dict[str, torch.Tensor] = {}
    for name, score in fisher_diag.items():
        score = score.detach()
        masks[name] = (score >= threshold).detach().cpu()
    return masks


def fisher_important_means(
    fisher_diag: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Return mean Fisher values over important coordinates layer-wise."""
    means: dict[str, float] = {}
    for name, score in fisher_diag.items():
        mask = masks.get(name)
        if mask is None:
            means[name] = 0.0
            continue
        mask = mask.to(device=score.device, dtype=torch.bool)
        if not bool(mask.any().item()):
            means[name] = 0.0
        else:
            means[name] = float(score.detach().float()[mask].mean().item())
    return means


def all_trainable_important_masks(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return all-true masks for the first AdapL round."""
    return {
        name: torch.ones_like(parameter, dtype=torch.bool).detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
