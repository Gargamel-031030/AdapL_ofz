"""Moments accountant for per-minibatch Gaussian mechanisms."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence


DEFAULT_MOMENT_ORDERS = tuple(range(1, 65))


def _validate_delta(delta: float) -> None:
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")


def _validate_noise_multiplier(noise_multiplier: float) -> None:
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive.")


def _subsampled_gaussian_log_moment(
    lambda_order: int,
    q: float,
    noise_multiplier: float,
) -> float:
    if lambda_order <= 0:
        raise ValueError("lambda_order must be positive.")
    if q <= 0:
        return 0.0
    q = min(1.0, float(q))
    sigma = float(noise_multiplier)
    _validate_noise_multiplier(sigma)
    base = lambda_order * (lambda_order + 1.0) / (2.0 * sigma * sigma)
    return math.log1p(q * q * base)


def epsilon_from_log_moments(
    log_moments: Sequence[float],
    delta: float,
    moment_orders: Sequence[int] = DEFAULT_MOMENT_ORDERS,
) -> float:
    """Return min_lambda (log_moment_lambda - log(delta)) / lambda."""
    _validate_delta(delta)
    if len(log_moments) != len(moment_orders):
        raise ValueError("log_moments and moment_orders must have the same length.")
    best = float("inf")
    log_delta = math.log(delta)
    for log_moment, lambda_order in zip(log_moments, moment_orders):
        if lambda_order <= 0:
            raise ValueError("moment orders must be positive.")
        epsilon = (float(log_moment) - log_delta) / float(lambda_order)
        if epsilon < best:
            best = epsilon
    return best


@dataclass
class MomentsAccountant:
    """Composable moments accountant with side-effect-free prechecks."""

    q: float
    noise_multiplier: float
    target_delta: float
    target_epsilon: float | None = None
    moment_orders: Sequence[int] = DEFAULT_MOMENT_ORDERS
    current_steps: int = 0
    log_moments: list[float] = field(default_factory=list)
    finished: bool = False

    def __post_init__(self) -> None:
        if self.q <= 0:
            raise ValueError("q must be positive.")
        self.q = min(1.0, float(self.q))
        _validate_noise_multiplier(self.noise_multiplier)
        _validate_delta(self.target_delta)
        if self.target_epsilon is not None and self.target_epsilon <= 0:
            raise ValueError("target_epsilon must be positive.")
        self.moment_orders = tuple(int(order) for order in self.moment_orders)
        if any(order <= 0 for order in self.moment_orders):
            raise ValueError("moment orders must be positive.")
        if self.current_steps < 0:
            raise ValueError("current_steps must be non-negative.")
        if not self.log_moments:
            self.log_moments = [0.0 for _ in self.moment_orders]
        if len(self.log_moments) != len(self.moment_orders):
            raise ValueError("log_moments and moment_orders must have the same length.")
        if self.current_steps:
            increments = self._increments(self.current_steps)
            self.log_moments = [
                current + increment
                for current, increment in zip(self.log_moments, increments)
            ]

    def _increments(self, steps: int) -> list[float]:
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        return [
            steps
            * _subsampled_gaussian_log_moment(
                lambda_order=order,
                q=self.q,
                noise_multiplier=self.noise_multiplier,
            )
            for order in self.moment_orders
        ]

    def projected_log_moments(self, next_steps: int) -> list[float]:
        increments = self._increments(next_steps)
        return [
            current + increment
            for current, increment in zip(self.log_moments, increments)
        ]

    def epsilon(self, log_moments: Iterable[float] | None = None) -> float:
        moments = list(self.log_moments if log_moments is None else log_moments)
        return epsilon_from_log_moments(
            moments,
            delta=self.target_delta,
            moment_orders=self.moment_orders,
        )

    def projected_epsilon(self, next_steps: int) -> float:
        return self.epsilon(self.projected_log_moments(next_steps))

    def can_train(self, next_steps: int) -> bool:
        """Return whether next_steps fit the target budget without mutating state."""
        if next_steps < 0:
            raise ValueError("next_steps must be non-negative.")
        if self.finished:
            return False
        if self.target_epsilon is None:
            return True
        return self.projected_epsilon(next_steps) <= self.target_epsilon

    def commit_steps(self, actual_steps: int) -> float:
        """Commit exactly the number of minibatch mechanisms that actually ran."""
        if actual_steps < 0:
            raise ValueError("actual_steps must be non-negative.")
        increments = self._increments(actual_steps)
        self.log_moments = [
            current + increment
            for current, increment in zip(self.log_moments, increments)
        ]
        self.current_steps += actual_steps
        epsilon = self.epsilon()
        if self.target_epsilon is not None and epsilon > self.target_epsilon:
            self.finished = True
        return epsilon
