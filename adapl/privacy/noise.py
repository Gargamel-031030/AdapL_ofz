"""Noise multiplier initialization for AdapL."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from adapl.privacy.accountant import DEFAULT_MOMENT_ORDERS, MomentsAccountant
from adapl.privacy.accounting import gaussian_noise_multiplier


@dataclass(frozen=True)
class NoiseInitialization:
    noise_multiplier: float
    source: str


def closed_form_noise_multiplier(epsilon: float, delta: float) -> float:
    return gaussian_noise_multiplier(epsilon, delta)


def search_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    q: float,
    total_steps: int,
    moment_orders: Sequence[int] = DEFAULT_MOMENT_ORDERS,
    tolerance: float = 1e-4,
    max_iterations: int = 80,
) -> float:
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive.")
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative.")
    if total_steps == 0:
        return closed_form_noise_multiplier(target_epsilon, target_delta)

    def projected_epsilon(sigma: float) -> float:
        accountant = MomentsAccountant(
            q=q,
            noise_multiplier=sigma,
            target_delta=target_delta,
            target_epsilon=target_epsilon,
            moment_orders=moment_orders,
        )
        return accountant.projected_epsilon(total_steps)

    low = 1e-6
    high = max(1.0, closed_form_noise_multiplier(target_epsilon, target_delta))
    while projected_epsilon(high) > target_epsilon:
        high *= 2.0
        if not math.isfinite(high) or high > 1e6:
            raise RuntimeError("Failed to bracket a valid noise multiplier.")

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        if projected_epsilon(mid) <= target_epsilon:
            high = mid
        else:
            low = mid
        if high - low <= tolerance * max(1.0, high):
            break
    return high


def initialize_noise_multiplier(
    *,
    target_epsilon: float | None,
    target_delta: float,
    q: float,
    total_steps: int,
    manual_override: float | None = None,
    use_decay_search: bool = False,
    fallback_epsilon: float | None = None,
) -> NoiseInitialization:
    if manual_override is not None:
        if manual_override <= 0:
            raise ValueError("manual_override must be positive.")
        return NoiseInitialization(float(manual_override), "manual_override")

    epsilon = target_epsilon if target_epsilon is not None else fallback_epsilon
    if epsilon is None:
        raise ValueError(
            "Noise initialization requires a target epsilon or manual override."
        )
    if epsilon <= 0:
        raise ValueError("target epsilon must be positive.")

    if use_decay_search:
        return NoiseInitialization(
            search_noise_multiplier(
                target_epsilon=epsilon,
                target_delta=target_delta,
                q=q,
                total_steps=total_steps,
            ),
            "decay_search",
        )
    return NoiseInitialization(
        closed_form_noise_multiplier(epsilon, target_delta),
        "closed_form",
    )
