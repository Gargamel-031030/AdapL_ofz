"""Basic DP calibration helpers.

The current helper uses the classical Gaussian mechanism bound
sigma >= sqrt(2 log(1.25 / delta)) / epsilon for one clipped vector release.
It is intentionally conservative and does not yet compose privacy loss over
rounds. A tighter accountant can be added here without changing method code.
"""

from __future__ import annotations

import math


def validate_epsilon_delta(epsilon: float, delta: float) -> None:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")


def gaussian_noise_multiplier(epsilon: float, delta: float) -> float:
    validate_epsilon_delta(epsilon, delta)
    return math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
