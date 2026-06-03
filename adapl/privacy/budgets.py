"""Privacy budget parsing utilities."""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Iterable, List, Optional, Sequence


def _parse_float_values(values: Iterable[object]) -> List[float]:
    budgets: List[float] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            budget = float(text)
        except ValueError:
            continue
        if budget <= 0:
            raise ValueError("Privacy budgets must be positive.")
        budgets.append(budget)
    return budgets


def _parse_budget_text(text: str) -> List[float]:
    return _parse_float_values(re.split(r"[\s,;]+", text.strip()))


def _parse_budget_json(path: str) -> List[float]:
    with open(path) as jsonfile:
        payload = json.load(jsonfile)
    if isinstance(payload, dict):
        if "budgets" in payload:
            payload = payload["budgets"]
        elif "epsilons" in payload:
            payload = payload["epsilons"]
        else:
            payload = payload.values()
    if not isinstance(payload, list):
        raise ValueError("Privacy budget JSON must contain a list of values.")
    return _parse_float_values(payload)


def _parse_budget_csv(path: str) -> List[float]:
    budgets: List[float] = []
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            budgets.extend(_parse_float_values(row))
    return budgets


def parse_privacy_budgets(spec: Optional[str]) -> Optional[List[float]]:
    """Parse comma-separated budgets or a JSON/CSV/text file path."""
    if spec is None:
        return None
    if os.path.exists(spec):
        _, ext = os.path.splitext(spec)
        if ext.lower() == ".json":
            budgets = _parse_budget_json(spec)
        elif ext.lower() in {".csv", ".tsv"}:
            budgets = _parse_budget_csv(spec)
        else:
            with open(spec) as textfile:
                budgets = _parse_budget_text(textfile.read())
    else:
        budgets = _parse_budget_text(spec)

    if not budgets:
        raise ValueError("No valid positive privacy budgets were found.")
    return budgets


def resolve_epsilon_min(
    epsilon_min: Optional[float],
    privacy_budgets: Optional[Sequence[float]],
) -> Optional[float]:
    if epsilon_min is not None:
        if epsilon_min <= 0:
            raise ValueError("--epsilon_min must be positive.")
        return epsilon_min
    if privacy_budgets:
        return min(privacy_budgets)
    return None
