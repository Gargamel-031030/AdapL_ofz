"""Client sampling utilities."""

from __future__ import annotations

import math
from typing import List

import torch


def select_clients(
    num_clients: int,
    client_fraction: float,
    round_idx: int,
    seed: int,
) -> List[int]:
    if not 0 < client_fraction <= 1:
        raise ValueError("--client_fraction must be in (0, 1].")
    num_selected = max(1, int(math.ceil(num_clients * client_fraction)))
    generator = torch.Generator().manual_seed(seed + round_idx)
    selected = torch.randperm(num_clients, generator=generator)[:num_selected]
    return sorted(selected.tolist())
