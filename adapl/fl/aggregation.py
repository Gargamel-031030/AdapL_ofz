"""Aggregation utilities for federated optimization."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch


def fedavg_aggregate(
    client_states: Sequence[OrderedDict[str, torch.Tensor]],
    client_sizes: Sequence[int],
) -> OrderedDict[str, torch.Tensor]:
    if not client_states:
        raise ValueError("Cannot aggregate an empty client state list.")
    if len(client_states) != len(client_sizes):
        raise ValueError("client_states and client_sizes must have the same length.")

    total_size = float(sum(client_sizes))
    if total_size <= 0:
        raise ValueError("Total client size must be positive.")

    aggregated = OrderedDict()
    for name in client_states[0].keys():
        first_value = client_states[0][name]
        if torch.is_floating_point(first_value):
            value = torch.zeros_like(first_value)
            for state, size in zip(client_states, client_sizes):
                value += state[name] * (size / total_size)
            aggregated[name] = value
        else:
            aggregated[name] = first_value.clone()
    return aggregated


def weighted_aggregate(
    client_states: Sequence[OrderedDict[str, torch.Tensor]],
    weights: Sequence[float],
) -> OrderedDict[str, torch.Tensor]:
    if not client_states:
        raise ValueError("Cannot aggregate an empty client state list.")
    if len(client_states) != len(weights):
        raise ValueError("client_states and weights must have the same length.")

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("Total aggregation weight must be positive.")

    normalized = [float(weight) / total_weight for weight in weights]
    aggregated = OrderedDict()
    for name in client_states[0].keys():
        first_value = client_states[0][name]
        if torch.is_floating_point(first_value):
            value = torch.zeros_like(first_value)
            for state, weight in zip(client_states, normalized):
                value += state[name] * weight
            aggregated[name] = value
        else:
            aggregated[name] = first_value.clone()
    return aggregated
