"""Layer-wise AdapL noise multiplier and standard-deviation strategy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerNoiseStats:
    sigma: float
    std: float


def layerwise_sigmas(
    base_noise_multiplier: float,
    fisher_mean_by_layer: Mapping[str, float],
    gamma: float,
) -> dict[str, float]:
    """Scale per-layer sigma from Fisher means without going below the base sigma."""
    if base_noise_multiplier <= 0:
        raise ValueError("base_noise_multiplier must be positive.")
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if not fisher_mean_by_layer:
        return {}

    max_mean = max(max(0.0, float(value)) for value in fisher_mean_by_layer.values())
    sigmas: dict[str, float] = {}
    for name, mean_value in fisher_mean_by_layer.items():
        normalized = 0.0 if max_mean <= 0 else max(0.0, float(mean_value)) / max_mean
        sigmas[name] = float(base_noise_multiplier) * (1.0 + gamma * normalized)
    return sigmas


def layerwise_noise_stats(
    *,
    base_noise_multiplier: float,
    fisher_mean_by_layer: Mapping[str, float],
    gamma: float,
    clipping_bound: float,
    batch_size: int,
) -> dict[str, LayerNoiseStats]:
    if clipping_bound <= 0:
        raise ValueError("clipping_bound must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    sigmas = layerwise_sigmas(
        base_noise_multiplier=base_noise_multiplier,
        fisher_mean_by_layer=fisher_mean_by_layer,
        gamma=gamma,
    )
    return {
        name: LayerNoiseStats(
            sigma=sigma,
            std=sigma * clipping_bound / float(batch_size),
        )
        for name, sigma in sigmas.items()
    }
