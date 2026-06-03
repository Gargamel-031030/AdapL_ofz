"""Client-side training utilities."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from adapl.utils import clone_state_dict


def train_minibatch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> Tuple[float, int]:
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    batch_size = targets.size(0)
    return loss.item() * batch_size, batch_size


def train_client_sgd(
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
) -> Tuple[OrderedDict[str, torch.Tensor], float, int]:
    model = model_fn().to(device)
    model.load_state_dict(global_state)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    total_loss = 0.0
    total_examples = 0
    if local_update_mode == "random-batch":
        if local_epochs is not None:
            local_steps = local_epochs
        if local_steps <= 0:
            raise ValueError("--local_steps must be positive.")

        train_iter = iter(train_loader)
        for _ in range(local_steps):
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)

            batch_loss, batch_size = train_minibatch(
                model,
                criterion,
                optimizer,
                inputs,
                targets,
                device,
            )
            total_loss += batch_loss
            total_examples += batch_size
    elif local_update_mode == "full-epoch":
        if local_epochs is None:
            local_epochs = local_steps
        if local_epochs <= 0:
            raise ValueError("--local_epochs must be positive.")
        for _ in range(local_epochs):
            for inputs, targets in train_loader:
                batch_loss, batch_size = train_minibatch(
                    model,
                    criterion,
                    optimizer,
                    inputs,
                    targets,
                    device,
                )
                total_loss += batch_loss
                total_examples += batch_size
    else:
        raise ValueError(f"Unsupported local update mode: {local_update_mode}")

    avg_loss = total_loss / max(1, total_examples)
    return clone_state_dict(model.state_dict()), avg_loss, len(train_loader.dataset)
