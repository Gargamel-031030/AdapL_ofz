"""ResNet model builders for CIFAR experiments."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torchvision import models

from adapl.constants import CIFAR100_NUM_CLASSES


def build_resnet18_cifar100(
    num_classes: int = CIFAR100_NUM_CLASSES,
) -> nn.Module:
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model_fn(model_name: str) -> Callable[[], nn.Module]:
    if model_name == "resnet18":
        return build_resnet18_cifar100
    raise ValueError(f"Unsupported model: {model_name}")


def validate_model_output(
    model_fn: Callable[[], nn.Module],
    num_classes: int,
) -> None:
    model = model_fn()
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 32, 32))
    expected_shape = (2, num_classes)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Expected model output shape {expected_shape}, got {tuple(output.shape)}."
        )
