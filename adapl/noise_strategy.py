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
    """Scale per-layer sigma from Eq. (15) in the AdapL paper."""
    if base_noise_multiplier <= 0:
        raise ValueError("base_noise_multiplier must be positive.")
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if not fisher_mean_by_layer:
        return {}

    means = {
        name: max(0.0, float(value))
        for name, value in fisher_mean_by_layer.items()
    }
    positive_means = [value for value in means.values() if value > 0.0]
    if not positive_means:
        return {name: float(base_noise_multiplier) for name in means}

    min_mean = min(positive_means)
    sigmas: dict[str, float] = {}
    for name, mean_value in means.items():
        adjusted_mean = max(mean_value, min_mean)
        sigmas[name] = float(base_noise_multiplier) * (
            1.0 + ((adjusted_mean - min_mean) / min_mean) * gamma
        )
    return sigmas


def layerwise_noise_stats(
    *,
    base_noise_multiplier: float,
    fisher_mean_by_layer: Mapping[str, float],
    gamma: float,
    clipping_bound: float | Mapping[str, float],
) -> dict[str, LayerNoiseStats]:
    sigmas = layerwise_sigmas(
        base_noise_multiplier=base_noise_multiplier,
        fisher_mean_by_layer=fisher_mean_by_layer,
        gamma=gamma,
    )
    stats: dict[str, LayerNoiseStats] = {}
    for name, sigma in sigmas.items():
        if isinstance(clipping_bound, Mapping):
            bound = float(clipping_bound.get(name, 0.0))
        else:
            bound = float(clipping_bound)
        if bound <= 0:
            raise ValueError("clipping_bound must be positive.")
        stats[name] = LayerNoiseStats(
            sigma=sigma,
            std=sigma * bound,
        )
    return stats
