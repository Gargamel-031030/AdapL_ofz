"""Method metadata that is safe to import without ML dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MethodInfo:
    canonical_name: str
    aliases: Sequence[str]
    display_name: str
    description: str
    implemented: bool


PRIVACY_FREE_INFO = MethodInfo(
    canonical_name="pf",
    aliases=("pf", "privacyfree", "fedavg"),
    display_name="PF / PrivacyFree FedAvg",
    description="Vanilla FedAvg without DP, clipping, noise, or privacy accounting.",
    implemented=True,
)


MINIMUM_INFO = MethodInfo(
    canonical_name="min",
    aliases=("min", "minimum"),
    display_name="Min / Minimum DP-FedAvg",
    description=(
        "DP-FedAvg where all clients use the strictest privacy budget epsilon_min."
    ),
    implemented=True,
)


WEIAVG_INFO = MethodInfo(
    canonical_name="weiavg",
    aliases=("weiavg", "weightedavg", "weighted-avg"),
    display_name="WeiAvg",
    description=(
        "Heterogeneous DP-FedAvg where aggregation weights are determined "
        "solely by client privacy budgets."
    ),
    implemented=True,
)


PLANNED_METHODS = [
    MethodInfo(
        canonical_name="feddpa",
        aliases=("feddpa",),
        display_name="FedDPA",
        description=(
            "Dynamic personalized FL with adaptive DP using Fisher information "
            "to separate parameter importance."
        ),
        implemented=False,
    ),
    MethodInfo(
        canonical_name="ppfed",
        aliases=("ppfed",),
        display_name="PPFed",
        description=(
            "Privacy-preserving personalized FL that separates global and local "
            "model parts and aggregates only global parameters."
        ),
        implemented=False,
    ),
    MethodInfo(
        canonical_name="pfa",
        aliases=("pfa",),
        display_name="PFA / Projected Federated Averaging",
        description=(
            "Projected FedAvg that separates public and private clients and "
            "maps private updates using public updates."
        ),
        implemented=False,
    ),
    MethodInfo(
        canonical_name="efl",
        aliases=("efl",),
        display_name="EFL",
        description=(
            "Efficient FL privacy preservation with heterogeneous DP and "
            "noise-aware aggregation weights."
        ),
        implemented=False,
    ),
    MethodInfo(
        canonical_name="adapl",
        aliases=("adapl", "ours"),
        display_name="AdapL / Ours",
        description="Project method placeholder for the final adaptive algorithm.",
        implemented=False,
    ),
]


METHOD_INFOS = [PRIVACY_FREE_INFO, MINIMUM_INFO, WEIAVG_INFO, *PLANNED_METHODS]
