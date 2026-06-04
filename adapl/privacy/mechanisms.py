"""DP mechanisms for client model updates."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class PrivatizedUpdate:
    state_dict: OrderedDict[str, torch.Tensor]
    update_norm: float
    clipped_norm: float
    clip_factor: float
    noise_std: float


def _floating_keys(
    global_state: OrderedDict[str, torch.Tensor],
    local_state: OrderedDict[str, torch.Tensor],
    private_keys: Iterable[str] | None = None,
) -> list[str]:
    keys = []
    names = list(private_keys) if private_keys is not None else list(global_state.keys())
    for name in names:
        if name not in global_state:
            raise KeyError(f"Missing key in global state_dict: {name}")
        if name not in local_state:
            raise KeyError(f"Missing key in local state_dict: {name}")
        global_value = global_state[name]
        local_value = local_state[name]
        if not (
            torch.is_floating_point(global_value)
            and torch.is_floating_point(local_value)
        ):
            if private_keys is not None:
                raise TypeError(f"Private update key is not floating point: {name}")
            continue
        keys.append(name)
    return keys


def client_update_l2_norm(
    global_state: OrderedDict[str, torch.Tensor],
    local_state: OrderedDict[str, torch.Tensor],
    keys: Sequence[str] | None = None,
) -> float:
    if keys is None:
        keys = _floating_keys(global_state, local_state)

    squared_norm = 0.0
    for name in keys:
        diff = local_state[name].detach().double() - global_state[name].detach().double()
        squared_norm += float(torch.sum(diff * diff).item())
    return squared_norm ** 0.5


def _normal_like(
    tensor: torch.Tensor,
    std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.normal(
        mean=0.0,
        std=std,
        size=tensor.shape,
        generator=generator,
        device=tensor.device,
        dtype=tensor.dtype,
    )


def privatize_client_update(
    global_state: OrderedDict[str, torch.Tensor],
    local_state: OrderedDict[str, torch.Tensor],
    clipping_norm: float,
    noise_std: float,
    generator: torch.Generator,
    private_keys: Iterable[str] | None = None,
) -> PrivatizedUpdate:
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    keys = _floating_keys(global_state, local_state, private_keys)
    key_set = set(keys)
    update_norm = client_update_l2_norm(global_state, local_state, keys)
    clip_factor = min(1.0, clipping_norm / (update_norm + 1e-12))
    clipped_norm = min(update_norm, clipping_norm)

    privatized = OrderedDict()
    for name, local_value in local_state.items():
        global_value = global_state[name]
        if name in key_set:
            clipped_update = (local_value - global_value) * clip_factor
            if noise_std > 0:
                clipped_update = clipped_update + _normal_like(
                    clipped_update,
                    std=noise_std,
                    generator=generator,
                )
            privatized[name] = global_value + clipped_update
        else:
            privatized[name] = local_value.detach().clone()

    return PrivatizedUpdate(
        state_dict=privatized,
        update_norm=update_norm,
        clipped_norm=clipped_norm,
        clip_factor=clip_factor,
        noise_std=noise_std,
    )
