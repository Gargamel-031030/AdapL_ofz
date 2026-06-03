"""Federated method registry."""

from adapl.methods.registry import (
    build_method,
    canonicalize_method,
    get_method_info,
    method_choices,
)

__all__ = [
    "build_method",
    "canonicalize_method",
    "get_method_info",
    "method_choices",
]
