"""Privacy configuration builders."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Optional

from adapl.privacy.accounting import gaussian_noise_multiplier
from adapl.privacy.budgets import parse_privacy_budgets, resolve_epsilon_min
from adapl.privacy.levels import PrivacyScenario, build_privacy_scenario


@dataclass(frozen=True)
class PrivacyConfig:
    enabled: bool
    mechanism: str
    clipping_norm: float
    delta: float
    epsilon: Optional[float]
    noise_multiplier: float
    noise_source: str
    privacy_budgets: Optional[list[float]] = None
    privacy_scenario: Optional[PrivacyScenario] = None

    @property
    def noise_std(self) -> float:
        return self.clipping_norm * self.noise_multiplier


@dataclass(frozen=True)
class HeterogeneousBudgetConfig:
    privacy_budgets: list[float]
    privacy_scenario: Optional[PrivacyScenario] = None


def build_heterogeneous_budget_config(
    args: Namespace,
    method_label: str,
) -> HeterogeneousBudgetConfig:
    privacy_scenario = None
    privacy_budgets = parse_privacy_budgets(args.privacy_budgets)
    if privacy_budgets is None and args.privacy_scenario is not None:
        privacy_scenario = build_privacy_scenario(
            scenario=args.privacy_scenario,
            num_clients=args.num_clients,
            seed=args.privacy_budget_seed,
        )
        privacy_budgets = list(privacy_scenario.client_budgets)
    if privacy_budgets is None:
        raise ValueError(
            f"{method_label} requires heterogeneous client budgets. "
            "Pass --privacy_scenario or --privacy_budgets."
        )
    if len(privacy_budgets) != args.num_clients:
        raise ValueError(
            "The number of privacy budgets must match --num_clients "
            f"({len(privacy_budgets)} != {args.num_clients})."
        )
    return HeterogeneousBudgetConfig(
        privacy_budgets=list(privacy_budgets),
        privacy_scenario=privacy_scenario,
    )


def build_minimum_privacy_config(args: Namespace) -> PrivacyConfig:
    if getattr(args, "no_dp", False):
        raise ValueError("Min / Minimum requires DP. Remove --no_dp or use --method PF.")

    clipping_norm = args.clipping_norm
    if clipping_norm is None:
        clipping_norm = 1.0
    if clipping_norm <= 0:
        raise ValueError("--clipping_norm must be positive for DP methods.")

    delta = args.delta
    if delta is None:
        delta = 1e-5
    if not 0 < delta < 1:
        raise ValueError("--delta must be in (0, 1).")

    privacy_scenario = None
    privacy_budgets = parse_privacy_budgets(args.privacy_budgets)
    if privacy_budgets is None and args.privacy_scenario is not None:
        privacy_scenario = build_privacy_scenario(
            scenario=args.privacy_scenario,
            num_clients=args.num_clients,
            seed=args.privacy_budget_seed,
        )
        privacy_budgets = list(privacy_scenario.client_budgets)
    if privacy_budgets is not None and len(privacy_budgets) != args.num_clients:
        raise ValueError(
            "The number of privacy budgets must match --num_clients "
            f"({len(privacy_budgets)} != {args.num_clients})."
        )
    epsilon = resolve_epsilon_min(args.epsilon_min, privacy_budgets)

    if args.noise_multiplier is not None:
        if args.noise_multiplier <= 0:
            raise ValueError("--noise_multiplier must be positive.")
        noise_multiplier = args.noise_multiplier
        noise_source = "user_noise_multiplier"
    else:
        if epsilon is None:
            raise ValueError(
                "Min / Minimum requires --epsilon_min, --privacy_budgets, "
                "or --noise_multiplier."
            )
        noise_multiplier = gaussian_noise_multiplier(epsilon, delta)
        noise_source = "gaussian_epsilon_delta_bound"

    return PrivacyConfig(
        enabled=True,
        mechanism="client_update_gaussian",
        clipping_norm=clipping_norm,
        delta=delta,
        epsilon=epsilon,
        noise_multiplier=noise_multiplier,
        noise_source=noise_source,
        privacy_budgets=list(privacy_budgets) if privacy_budgets else None,
        privacy_scenario=privacy_scenario,
    )


def build_heterogeneous_privacy_config(
    args: Namespace,
    method_label: str,
) -> PrivacyConfig:
    if getattr(args, "no_dp", False):
        raise ValueError(
            f"{method_label} requires DP. Remove --no_dp or use --method PF."
        )

    clipping_norm = args.clipping_norm
    if clipping_norm is None:
        clipping_norm = 1.0
    if clipping_norm <= 0:
        raise ValueError("--clipping_norm must be positive for DP methods.")

    delta = args.delta
    if delta is None:
        delta = 1e-5
    if not 0 < delta < 1:
        raise ValueError("--delta must be in (0, 1).")

    privacy_scenario = None
    privacy_budgets = parse_privacy_budgets(args.privacy_budgets)
    if privacy_budgets is None and args.privacy_scenario is not None:
        privacy_scenario = build_privacy_scenario(
            scenario=args.privacy_scenario,
            num_clients=args.num_clients,
            seed=args.privacy_budget_seed,
        )
        privacy_budgets = list(privacy_scenario.client_budgets)
    if privacy_budgets is None:
        raise ValueError(
            f"{method_label} requires heterogeneous client budgets. "
            "Pass --privacy_scenario or --privacy_budgets."
        )
    if len(privacy_budgets) != args.num_clients:
        raise ValueError(
            "The number of privacy budgets must match --num_clients "
            f"({len(privacy_budgets)} != {args.num_clients})."
        )

    epsilon_min = min(privacy_budgets)
    if args.noise_multiplier is not None:
        if args.noise_multiplier <= 0:
            raise ValueError("--noise_multiplier must be positive.")
        noise_multiplier = args.noise_multiplier
        noise_source = "user_noise_multiplier"
    else:
        noise_multiplier = gaussian_noise_multiplier(epsilon_min, delta)
        noise_source = "per_client_gaussian_epsilon_delta_bound"

    return PrivacyConfig(
        enabled=True,
        mechanism="client_update_gaussian",
        clipping_norm=clipping_norm,
        delta=delta,
        epsilon=epsilon_min,
        noise_multiplier=noise_multiplier,
        noise_source=noise_source,
        privacy_budgets=list(privacy_budgets),
        privacy_scenario=privacy_scenario,
    )
