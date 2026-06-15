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
    """Build boolean masks from per-layer max-normalized Fisher scores."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")

    masks: Dict[str, torch.Tensor] = {}
    for name, score in fisher_diag.items():
        score = score.detach()
        max_score = torch.max(score)
        if float(max_score.item()) > 0:
            normalized = score / max_score
            masks[name] = (normalized >= threshold).detach().cpu()
        else:
            masks[name] = torch.zeros_like(score, dtype=torch.bool).cpu()
    return masks


def all_trainable_important_masks(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return all-true masks for the first AdapL round."""
    return {
        name: torch.ones_like(parameter, dtype=torch.bool).detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
