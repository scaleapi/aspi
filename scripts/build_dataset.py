#!/usr/bin/env python3
"""ASPI dataset construction CLI.

Thin wrapper around ``aspi.dataset.pipeline.main``.

Usage:
    python scripts/build_dataset.py --model gemini/gemini-2.5-pro --suite all --limit 25 --workers 4

See ``python scripts/build_dataset.py --help`` for all flags.
"""

from aspi.dataset.pipeline import main

if __name__ == "__main__":
    main()
