"""Command-line entry point for AdapL experiments."""

from __future__ import annotations

from typing import Optional, Sequence

from adapl.config import parse_args
from adapl.methods import get_method_info


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    method_info = get_method_info(args.method)
    if not method_info.implemented:
        raise SystemExit(
            f"{method_info.display_name} is registered but not implemented yet. "
            "Implement it under adapl/methods/ before running this method."
        )

    try:
        from adapl.experiment import run_experiment
    except ModuleNotFoundError as exc:
        if exc.name in {"numpy", "torch", "torchvision"}:
            raise SystemExit(
                f"Missing dependency: {exc.name}. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc
        raise

    run_experiment(args)
