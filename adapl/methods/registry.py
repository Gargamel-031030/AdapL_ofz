"""Method aliases and factory functions."""

from __future__ import annotations

from argparse import Namespace
from typing import Dict, List

from adapl.methods.base import FederatedMethod
from adapl.methods.metadata import METHOD_INFOS, MethodInfo

_ALIAS_TO_INFO: Dict[str, MethodInfo] = {
    alias: info for info in METHOD_INFOS for alias in info.aliases
}


def method_choices() -> List[str]:
    return sorted(_ALIAS_TO_INFO.keys())


def canonicalize_method(method_name: str) -> str:
    key = method_name.lower()
    if key not in _ALIAS_TO_INFO:
        choices = ", ".join(method_choices())
        raise ValueError(f"Unsupported method '{method_name}'. Choices: {choices}")
    return _ALIAS_TO_INFO[key].canonical_name


def get_method_info(method_name: str) -> MethodInfo:
    key = method_name.lower()
    if key not in _ALIAS_TO_INFO:
        choices = ", ".join(method_choices())
        raise ValueError(f"Unsupported method '{method_name}'. Choices: {choices}")
    return _ALIAS_TO_INFO[key]


def build_method(method_name: str, args: Namespace) -> FederatedMethod:
    key = method_name.lower()
    if key not in _ALIAS_TO_INFO:
        choices = ", ".join(method_choices())
        raise ValueError(f"Unsupported method '{method_name}'. Choices: {choices}")

    info = _ALIAS_TO_INFO[key]
    if info.canonical_name == "pf":
        from adapl.methods.privacy_free import PrivacyFreeFedAvg

        return PrivacyFreeFedAvg(args)
    if info.canonical_name == "min":
        from adapl.methods.minimum import MinimumDPFedAvg

        return MinimumDPFedAvg(args)
    if info.canonical_name == "weiavg":
        from adapl.methods.weiavg import WeiAvgFedAvg

        return WeiAvgFedAvg(args)
    if info.canonical_name == "feddpa":
        from adapl.methods.feddpa import FedDPA

        return FedDPA(args)
    if info.canonical_name == "pfa":
        from adapl.methods.pfa import PFAFedAvg

        return PFAFedAvg(args)
    if info.canonical_name == "adapl":
        from adapl.methods.adapl import AdapL

        return AdapL(args)

    raise NotImplementedError(
        f"{info.display_name} is registered but not implemented yet. "
        "Add its algorithm hooks under adapl/methods/ and register the factory "
        "in adapl/methods/registry.py."
    )
