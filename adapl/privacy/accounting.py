"""Basic DP calibration and budget accounting helpers.

PrivacyBudgetAccountant tracks per-client privacy loss using Rényi
Differential Privacy (RDP) composition for the subsampled Gaussian mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

DEFAULT_RDP_ORDERS = (
    [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 65))
)


def _gaussian_rdp(alpha: float, sigma: float) -> float:
    return alpha / (2.0 * sigma * sigma)


def _subsampled_gaussian_rdp(alpha: float, q: float, sigma: float) -> float:
    if q <= 0.0 or sigma <= 0.0:
        return 0.0
    coef = q * q * alpha * (alpha - 1.0) / (2.0 * sigma * sigma)
    if coef <= 1e-30:
        return 0.0
    return math.log1p(coef) / (alpha - 1.0)


def compute_epsilon_from_rdp(
    steps: int,
    q: float,
    sigma: float,
    delta: float,
    orders: Sequence[float] | None = None,
) -> float:
    if orders is None:
        orders = DEFAULT_RDP_ORDERS
    best_epsilon = float("inf")
    for alpha in orders:
        per_step_rdp = _subsampled_gaussian_rdp(alpha, q, sigma)
        total_rdp = steps * per_step_rdp
        epsilon = total_rdp + math.log(1.0 / delta) / (alpha - 1.0)
        if epsilon < best_epsilon:
            best_epsilon = epsilon
    return best_epsilon


@dataclass
class PrivacyBudgetAccountant:
    epsilon: float
    delta: float
    noise_multiplier: float
    accumulated_budget: float = 0.0
    current_steps: int = 0
    finished: bool = False
    _tmp_budget: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_epsilon_delta(self.epsilon, self.delta)
        if self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive.")
        if self.accumulated_budget < 0:
            raise ValueError("accumulated_budget must be non-negative.")
        if self.current_steps < 0:
            raise ValueError("current_steps must be non-negative.")

    def precheck(
        self,
        dataset_size: int,
        batch_size: int,
        local_steps: int,
    ) -> bool:
        if self.finished:
            return False
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if local_steps < 0:
            raise ValueError("local_steps must be non-negative.")

        q = batch_size / dataset_size
        tmp_steps = self.current_steps + local_steps
        tmp_budget = compute_epsilon_from_rdp(
            steps=tmp_steps,
            q=q,
            sigma=self.noise_multiplier,
            delta=self.delta,
        )
        if tmp_budget > self.epsilon:
            self.finished = True
            self._tmp_budget = None
            return False

        self._tmp_budget = tmp_budget
        return True

    def update(self, local_steps: int) -> float:
        if local_steps < 0:
            raise ValueError("local_steps must be non-negative.")
        if self._tmp_budget is None:
            raise RuntimeError("precheck must pass before update.")

        self.current_steps += local_steps
        self.accumulated_budget = self._tmp_budget
        self._tmp_budget = None
        return self.accumulated_budget


@dataclass(frozen=True)
class PrivacyBudgetContext:
    client_id: int
    epsilon: float
    delta: float
    noise_multiplier: float
    accumulated_budget: float
    current_steps: int
    finished: bool


class ClientPrivacyBudgetManager:
    """Shared per-client privacy budget state for local DP-SGD methods."""

    def __init__(self, accountants: Mapping[int, PrivacyBudgetAccountant]) -> None:
        if not accountants:
            raise ValueError("At least one client privacy accountant is required.")
        self.accountants = dict(accountants)

    @classmethod
    def from_client_epsilons(
        cls,
        client_epsilons: Sequence[float],
        delta: float,
        noise_multiplier: float | None = None,
        epsilon_floor: float | None = None,
    ) -> "ClientPrivacyBudgetManager":
        if not client_epsilons:
            raise ValueError("client_epsilons must not be empty.")
        if epsilon_floor is not None and epsilon_floor <= 0:
            raise ValueError("epsilon_floor must be positive.")

        accountants = {}
        for client_id, epsilon in enumerate(client_epsilons):
            if epsilon <= 0:
                raise ValueError("Client epsilon values must be positive.")
            if noise_multiplier is None:
                noise_epsilon = (
                    max(float(epsilon), epsilon_floor)
                    if epsilon_floor is not None
                    else float(epsilon)
                )
                client_noise_multiplier = gaussian_noise_multiplier(
                    noise_epsilon,
                    delta,
                )
            else:
                client_noise_multiplier = noise_multiplier
            accountants[client_id] = PrivacyBudgetAccountant(
                epsilon=float(epsilon),
                delta=delta,
                noise_multiplier=client_noise_multiplier,
            )
        return cls(accountants)

    def precheck_client(
        self,
        client_id: int,
        dataset_size: int,
        batch_size: int,
        local_steps: int,
    ) -> bool:
        return self.accountants[client_id].precheck(
            dataset_size=dataset_size,
            batch_size=batch_size,
            local_steps=local_steps,
        )

    def eligible_client_ids(
        self,
        client_ids: Sequence[int],
        dataset_sizes: Mapping[int, int],
        batch_size: int,
        local_steps_by_client: Mapping[int, int],
    ) -> list[int]:
        eligible = []
        for client_id in client_ids:
            if self.precheck_client(
                client_id=client_id,
                dataset_size=dataset_sizes[client_id],
                batch_size=batch_size,
                local_steps=local_steps_by_client[client_id],
            ):
                eligible.append(client_id)
        return eligible

    def context_for_client(self, client_id: int) -> PrivacyBudgetContext:
        accountant = self.accountants[client_id]
        return PrivacyBudgetContext(
            client_id=client_id,
            epsilon=accountant.epsilon,
            delta=accountant.delta,
            noise_multiplier=accountant.noise_multiplier,
            accumulated_budget=accountant.accumulated_budget,
            current_steps=accountant.current_steps,
            finished=accountant.finished,
        )

    def context_for_clients(
        self,
        client_ids: Sequence[int],
    ) -> dict[int, PrivacyBudgetContext]:
        return {
            client_id: self.context_for_client(client_id)
            for client_id in client_ids
        }

    def update_client(self, client_id: int, local_steps: int) -> float:
        return self.accountants[client_id].update(local_steps)

    def metadata_for_client(self, client_id: int) -> dict[str, object]:
        accountant = self.accountants[client_id]
        return {
            "privacy_budget_epsilon": accountant.epsilon,
            "privacy_budget_accumulated": accountant.accumulated_budget,
            "privacy_budget_current_steps": accountant.current_steps,
            "privacy_budget_finished": accountant.finished,
            "privacy_budget_noise_multiplier": accountant.noise_multiplier,
        }

    @property
    def num_finished(self) -> int:
        return sum(1 for accountant in self.accountants.values() if accountant.finished)

    @property
    def num_clients(self) -> int:
        return len(self.accountants)


def validate_epsilon_delta(epsilon: float, delta: float) -> None:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")


def gaussian_noise_multiplier(epsilon: float, delta: float) -> float:
    validate_epsilon_delta(epsilon, delta)
    return math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
