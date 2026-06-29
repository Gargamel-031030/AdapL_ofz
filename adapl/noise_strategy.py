"""Layer-wise AdapL noise multiplier and standard-deviation strategy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

FISHER_EPS: float = 1e-12
MAX_NOISE_RATIO: float = 100.0


@dataclass(frozen=True)
class LayerNoiseStats:
    sigma: float
    std: float


def layerwise_sigmas(
    base_noise_multiplier: float,
    fisher_mean_by_layer: Mapping[str, float],
    gamma: float,
    max_noise_ratio: float = MAX_NOISE_RATIO,
) -> dict[str, float]:
    """Scale per-layer sigma from Eq. (15) in the AdapL paper."""
    if base_noise_multiplier <= 0:
        raise ValueError("base_noise_multiplier must be positive.")
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if not fisher_mean_by_layer:
        return {}

    means = {
        name: float(value)
        for name, value in fisher_mean_by_layer.items()
    }
    min_mean = min(max(v, FISHER_EPS) for v in means.values())
    sigmas: dict[str, float] = {}
    for name, mean_value in means.items():
        adjusted_mean = max(mean_value, min_mean)
        noise_ratio = 1.0 + ((adjusted_mean - min_mean) / min_mean) * gamma
        if noise_ratio > max_noise_ratio:
            noise_ratio = max_noise_ratio
        sigmas[name] = float(base_noise_multiplier) * noise_ratio
    return sigmas


def layerwise_noise_stats(
    *,
    base_noise_multiplier: float,
    fisher_mean_by_layer: Mapping[str, float],
    gamma: float,
    clipping_bound: float | Mapping[str, float],
    max_noise_ratio: float = MAX_NOISE_RATIO,
) -> dict[str, LayerNoiseStats]:
    sigmas = layerwise_sigmas(
        base_noise_multiplier=base_noise_multiplier,
        fisher_mean_by_layer=fisher_mean_by_layer,
        gamma=gamma,
        max_noise_ratio=max_noise_ratio,
    )
    stats: dict[str, LayerNoiseStats] = {}
    for name, sigma in sigmas.items():
        if isinstance(clipping_bound, Mapping):
            bound = max(float(clipping_bound.get(name, FISHER_EPS)), FISHER_EPS)
        else:
            bound = max(float(clipping_bound), FISHER_EPS)
        stats[name] = LayerNoiseStats(
            sigma=sigma,
            std=sigma * bound,
        )
    return stats
