#!/usr/bin/env python3
"""Summarize Min grid result CSV files."""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path


def parse_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed


def summarize_metrics(path: Path) -> dict[str, object] | None:
    with path.open(newline="") as csvfile:
        rows = list(csv.DictReader(csvfile))
    if not rows:
        return None
    params = parse_run_params(path.stem)

    evaluated = [
        row for row in rows
        if not math.isnan(parse_float(row.get("test_accuracy", "")))
    ]
    if not evaluated:
        return {
            "run": path.stem,
            **params,
            "rounds": len(rows),
            "final_round": "",
            "final_test_accuracy": "",
            "final_test_loss": "",
            "best_round": "",
            "best_test_accuracy": "",
            "csv_path": str(path),
        }

    final = evaluated[-1]
    best = max(evaluated, key=lambda row: parse_float(row["test_accuracy"]))
    return {
        "run": path.stem,
        **params,
        "rounds": len(rows),
        "final_round": final["round"],
        "final_test_accuracy": final["test_accuracy"],
        "final_test_loss": final["test_loss"],
        "best_round": best["round"],
        "best_test_accuracy": best["test_accuracy"],
        "csv_path": str(path),
    }


def unsanitize_number(value: str) -> str:
    return value.replace("p", ".")


def parse_run_params(run: str) -> dict[str, str]:
    patterns = {
        "epsilon": r"_eps([0-9p]+)_",
        "lr": r"_lr([0-9p]+)_",
        "local_steps": r"_steps([0-9]+)_",
        "clipping_norm": r"_clip([0-9p]+)_",
        "rounds_config": r"_r([0-9]+)$",
    }
    params = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, run)
        params[key] = unsanitize_number(match.group(1)) if match else ""
    return params


def main() -> None:
    result_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/root/autodl-tmp/results/min_grid"
    )
    files = sorted(result_root.glob("**/*.csv"))
    files = [
        path for path in files
        if not path.name.endswith("_client_dist.csv")
        and path.name != "summary.csv"
    ]

    summaries = [
        summary for path in files
        if (summary := summarize_metrics(path)) is not None
    ]

    fieldnames = [
        "run",
        "epsilon",
        "lr",
        "local_steps",
        "clipping_norm",
        "rounds_config",
        "rounds",
        "final_round",
        "final_test_accuracy",
        "final_test_loss",
        "best_round",
        "best_test_accuracy",
        "csv_path",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summaries)


if __name__ == "__main__":
    main()
