"""Unified state_dict aggregation routines."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch


StateDict = OrderedDict[str, torch.Tensor]


def _normalize_weights(weights: Sequence[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Aggregation weights must sum to a positive value.")
    return [float(weight) / total for weight in weights]


def aggregate_state_dicts(
    client_states: Sequence[StateDict],
    weights: Sequence[float],
) -> StateDict:
    if not client_states:
        raise ValueError("Cannot aggregate an empty client state list.")
    if len(client_states) != len(weights):
        raise ValueError("client_states and weights must have the same length.")

    normalized_weights = _normalize_weights(weights)
    aggregated: StateDict = OrderedDict()
    for name in client_states[0].keys():
        first_value = client_states[0][name]
        if torch.is_floating_point(first_value):
            value = torch.zeros_like(first_value)
            for state, weight in zip(client_states, normalized_weights):
                value.add_(state[name], alpha=weight)
            aggregated[name] = value
        else:
            aggregated[name] = first_value.detach().clone()
    return aggregated


def FedAvg(
    client_states: Sequence[StateDict],
    client_sizes: Sequence[int],
) -> StateDict:
    return aggregate_state_dicts(client_states, [float(size) for size in client_sizes])


def WeiAvg(
    client_states: Sequence[StateDict],
    weights: Sequence[float],
) -> StateDict:
    return aggregate_state_dicts(client_states, weights)


def DeAvg(
    client_states: Sequence[StateDict],
    client_sizes: Sequence[int],
    decay: float = 1.0,
) -> StateDict:
    if decay <= 0:
        raise ValueError("decay must be positive.")
    decayed_weights = [
        float(size) * (decay ** index)
        for index, size in enumerate(client_sizes)
    ]
    return aggregate_state_dicts(client_states, decayed_weights)
