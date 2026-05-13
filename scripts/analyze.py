#!/usr/bin/env python3
"""ASPI analysis CLI (tables + plots from saved JSONL).

Thin wrapper around ``aspi.eval.analyze.main``. No API calls — pure
offline analysis of result files.

Usage:
    python scripts/analyze.py \\
        --add "GPT-4o" results/judged/aspi_eval_..._judged.jsonl \\
        --summary --plot all
"""

from aspi.eval.analyze import main

if __name__ == "__main__":
    main()
