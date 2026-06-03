"""Project path helpers.

Paths passed on the command line are resolved relative to the repository root
unless they are already absolute. This keeps server runs independent of the
current shell working directory.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_CIFAR100_DIR = DEFAULT_DATA_ROOT / "cifar100"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved
