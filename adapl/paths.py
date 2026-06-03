"""Project path helpers.

Paths passed on the command line are resolved relative to the repository root
unless they are already absolute. This keeps server runs independent of the
current shell working directory.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTODL_DATA_DIR = Path("/root/autodl-tmp/data")
AUTODL_RESULTS_DIR = Path("/root/autodl-tmp/results")

DEFAULT_DATA_ROOT = Path(os.environ.get("ADAPL_DATA_DIR", str(AUTODL_DATA_DIR)))
DEFAULT_CIFAR100_DIR = DEFAULT_DATA_ROOT
DEFAULT_RESULTS_DIR = Path(os.environ.get("ADAPL_RESULTS_DIR", str(AUTODL_RESULTS_DIR)))


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved
