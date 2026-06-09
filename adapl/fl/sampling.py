"""Client sampling utilities."""

from __future__ import annotations

import math
from typing import List, Sequence

import torch


def select_clients(
    num_clients: int,
    client_fraction: float,
    round_idx: int,
    seed: int,
    candidate_client_ids: Sequence[int] | None = None,
) -> List[int]:
    if not 0 < client_fraction <= 1:
        raise ValueError("--client_fraction must be in (0, 1].")
    candidates = (
        list(range(num_clients))
        if candidate_client_ids is None
        else sorted(set(int(client_id) for client_id in candidate_client_ids))
    )
    if not candidates:
        return []

    num_selected = min(
        len(candidates),
        max(1, int(math.ceil(num_clients * client_fraction))),
    )
    generator = torch.Generator().manual_seed(seed + round_idx)
    selected_indices = torch.randperm(len(candidates), generator=generator)[:num_selected]
    return sorted(candidates[index] for index in selected_indices.tolist())
