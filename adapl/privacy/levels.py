"""Paper privacy-level presets.

TNSE defines five client privacy levels with maximum budgets:
Level 1: 0.5, Level 2: 1.0, Level 3: 2.0, Level 4: 4.0, Level 5: 8.0.
The scenario proportions below follow the experiment setup in the paper.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


PAPER_LEVEL_BUDGETS = (0.5, 1.0, 2.0, 4.0, 8.0)
PAPER_PRIVACY_SCENARIOS = {
    "1": (0.10, 0.10, 0.40, 0.20, 0.20),
    "2": (0.20, 0.20, 0.40, 0.10, 0.10),
    "3": (0.90, 0.00, 0.00, 0.00, 0.10),
}


@dataclass(frozen=True)
class PrivacyScenario:
    name: str
    level_budgets: tuple[float, ...]
    level_counts: tuple[int, ...]
    client_budgets: tuple[float, ...]

    @property
    def epsilon_min(self) -> float:
        return min(self.client_budgets)


def normalize_privacy_scenario(scenario: str) -> str:
    value = str(scenario).strip().lower()
    if value.startswith("scenario"):
        value = value.replace("scenario", "", 1).strip("_- ")
    if value not in PAPER_PRIVACY_SCENARIOS:
        choices = ", ".join(sorted(PAPER_PRIVACY_SCENARIOS))
        raise ValueError(f"Unsupported privacy scenario '{scenario}'. Choices: {choices}")
    return value


def _counts_from_proportions(
    num_clients: int,
    proportions: Sequence[float],
) -> list[int]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if abs(sum(proportions) - 1.0) > 1e-8:
        raise ValueError("Privacy scenario proportions must sum to 1.")

    raw_counts = [num_clients * proportion for proportion in proportions]
    counts = [int(count) for count in raw_counts]
    remainder = num_clients - sum(counts)
    fractional_order = sorted(
        range(len(raw_counts)),
        key=lambda idx: raw_counts[idx] - counts[idx],
        reverse=True,
    )
    for idx in fractional_order[:remainder]:
        counts[idx] += 1
    return counts


def build_privacy_scenario(
    scenario: str,
    num_clients: int,
    seed: int,
    level_budgets: Sequence[float] = PAPER_LEVEL_BUDGETS,
) -> PrivacyScenario:
    scenario_name = normalize_privacy_scenario(scenario)
    proportions = PAPER_PRIVACY_SCENARIOS[scenario_name]
    if len(level_budgets) != len(proportions):
        raise ValueError("level_budgets must define exactly five privacy levels.")
    if any(float(budget) <= 0 for budget in level_budgets):
        raise ValueError("All privacy level budgets must be positive.")

    counts = _counts_from_proportions(num_clients, proportions)
    budgets = []
    for budget, count in zip(level_budgets, counts):
        budgets.extend([float(budget)] * count)

    rng = random.Random(seed)
    rng.shuffle(budgets)
    return PrivacyScenario(
        name=scenario_name,
        level_budgets=tuple(float(budget) for budget in level_budgets),
        level_counts=tuple(counts),
        client_budgets=tuple(budgets),
    )
