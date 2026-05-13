#!/usr/bin/env python3
"""ASPI LLM-as-judge CLI.

Thin wrapper around the judge CLI defined in ``aspi.eval.judge``. Reads
an eval results JSONL, annotates each group with judge classifications,
writes a new JSONL with the ``_judged.jsonl`` suffix.

Usage:
    python scripts/run_judge.py results/eval/aspi_eval_*.jsonl \\
        --model openai/gpt-4o-mini-2024-07-18 --workers 4
"""

import argparse
import sys
from pathlib import Path

from aspi.eval.judge import ASPIJudge


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge on saved ASPI evaluation results",
    )
    parser.add_argument("input", help="Path to evaluation results JSONL")
    parser.add_argument(
        "-o", "--output",
        help="Output path (default: input with _judged suffix, in results/judged/)",
    )
    parser.add_argument(
        "--model", default="openai/gpt-4o-mini-2024-07-18",
        help="Judge model (default: openai/gpt-4o-mini-2024-07-18)",
    )
    parser.add_argument(
        "--backend", default="litellm", choices=["litellm", "openai"],
        help="Backend: litellm (proxy) or openai (direct)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max groups to judge (0 = all)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel judge threads (default: 4)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "If output already exists, skip groups that already have a "
            "'judge' field (matched by suite/user_task_id/injection_task_id). "
            "Rows with truncation/UNCLEAR judges still count as judged — "
            "delete the corrupted output before resuming if you need to "
            "re-judge."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        # Default: results/judged/{stem}_judged.jsonl
        from aspi.common.paths import JUDGED_RESULTS_DIR, ensure_dirs
        ensure_dirs(JUDGED_RESULTS_DIR)
        stem = input_path.stem
        if stem.endswith("_judged"):
            output_path = str(JUDGED_RESULTS_DIR / f"{stem}.jsonl")
        else:
            output_path = str(JUDGED_RESULTS_DIR / f"{stem}_judged.jsonl")

    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Model:   {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Workers: {args.workers}")
    if args.limit:
        print(f"Limit:   {args.limit}")
    if args.resume:
        print(f"Resume:  enabled")
    print()

    judge = ASPIJudge(model=args.model, backend=args.backend)
    judge.run_judge_pass(
        str(input_path), output_path,
        limit=args.limit, workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
