"""Compatibility entry point for the old baseline script name.

Prefer running experiments through:

    python main.py --method pf
"""

from adapl.cli import main


if __name__ == "__main__":
    main()
