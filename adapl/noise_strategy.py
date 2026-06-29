"""Layer-wise AdapL noise multiplier and standard-deviation strategy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

FISHER_EPS: float = 1e-12
MAX_NOISE_RATIO: float = 5.0


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
    """Scale per-layer sigma from Eq. (15) in the AdapL paper.

    Numerical-safety guards:
    - If the minimum Fisher mean is effectively zero (<= FISHER_EPS),
      all layers fall back to noise_ratio=1.0.
    - If a layer's Fisher mean is NaN, Inf, or <= FISHER_EPS,
      that layer falls back to noise_ratio=1.0.
    - If the computed noise_ratio is NaN, Inf, or <= 0,
      it is replaced with 1.0.
    - The final ratio is clamped to [1.0, max_noise_ratio].
    """
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
    min_mean = min(means.values())

    # Safety: invalid min_mean (NaN/Inf) or all Fisher means effectively zero
    if not math.isfinite(min_mean) or min_mean <= FISHER_EPS:
        return {
            name: float(base_noise_multiplier)
            for name in means
        }

    sigmas: dict[str, float] = {}
    for name, mean_value in means.items():
        if not math.isfinite(mean_value) or mean_value <= FISHER_EPS:
            noise_ratio = 1.0
        else:
            noise_ratio = 1.0 + ((mean_value - min_mean) / min_mean) * gamma

        if not math.isfinite(noise_ratio) or noise_ratio <= 0:
            noise_ratio = 1.0

        noise_ratio = max(1.0, min(noise_ratio, max_noise_ratio))
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
