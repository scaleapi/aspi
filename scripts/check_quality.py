#!/usr/bin/env python3
"""ASPI quality checker CLI.

Thin wrapper around ``aspi.dataset.quality.main``. Runs all 10 QC
checks over a saved dataset JSONL and prints a per-check pass-rate
summary.

Usage:
    python scripts/check_quality.py --input data/aspi_banking_25pairs.jsonl
"""

from aspi.dataset.quality import main

if __name__ == "__main__":
    main()
