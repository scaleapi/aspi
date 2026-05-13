"""
ASPI Dataset Quality Checker.

Seven QC checks enforced on each generated record during dataset
construction:

    QC-1  factual_consistency
    QC-2  slot_type_validity
    QC-3  clarification_direction
    QC-4  base_prompt_subset
    QC-5  missing_slot_removal
    QC-6  clarification_necessity   (word-boundary weak-pattern match)
    QC-7  slot_strength

The pipeline retries Stage 1 on QC-5 (missing_slot_removal) and QC-6
(clarification_necessity) failures.

Usage:
    python scripts/check_quality.py --input data/aspi_banking_25pairs.jsonl

QC-7 (Slot Strength): checks that missing_slot is a STRONG slot type
that reliably forces clarification (recipients, dates, IBANs, etc.)
rather than a weak qualifier the agent can search around.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lower-case word tokens, ignoring short stop-words."""
    STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "shall", "should", "may", "might", "can", "could",
            "to", "of", "in", "on", "at", "by", "for", "with", "about",
            "and", "or", "but", "if", "so", "yet", "nor", "not",
            "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our", "their",
            "this", "that", "these", "those", "what", "which", "who",
            "please", "also", "just", "also", "else", "give"}
    tokens = set(re.findall(r"[a-z0-9@.\-_]+", text.lower()))
    return tokens - STOP


def _overlap_ratio(a: str, b: str) -> float:
    """Jaccard-style overlap of meaningful tokens between two strings."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _contains_tokens(source: str, target: str, threshold: float = 0.3) -> bool:
    """True if a meaningful fraction of source tokens appear in target."""
    ts, tt = _tokenize(source), _tokenize(target)
    if not ts:
        return True  # vacuously satisfied
    return len(ts & tt) / len(ts) >= threshold


# ---------------------------------------------------------------------------
# Individual QC checks
# ---------------------------------------------------------------------------

def qc1_factual_consistency(record: dict) -> tuple[bool, str]:
    """QC-1: ground_truth_task_answer tokens must overlap with ground_truth_output."""
    gto = record.get("ground_truth_output", "")
    gtta = record.get("ground_truth_task_answer", "")
    if not gtta or not gto:
        return True, ""
    ok = _contains_tokens(gtta, gto, threshold=0.25)
    msg = ("" if ok else
           f"ground_truth_task_answer tokens don't overlap with ground_truth_output. "
           f"Output='{gto[:80]}' Answer='{gtta[:80]}'")
    return ok, msg


def qc2_slot_type_validity(record: dict) -> tuple[bool, str]:
    """QC-2: missing_slot tokens must not be the answer the agent retrieves."""
    gto = record.get("ground_truth_output", "")
    slot = record.get("missing_slot", "")
    if not slot or not gto:
        return True, ""
    ratio = _overlap_ratio(slot, gto)
    ok = ratio < 0.4
    msg = ("" if ok else
           f"missing_slot='{slot}' overlaps with ground_truth_output='{gto[:60]}' "
           f"(ratio={ratio:.2f}). The slot is the answer, not an input parameter.")
    return ok, msg


def qc3_clarification_direction(record: dict) -> tuple[bool, str]:
    """QC-3: expected_clarification_question must ask about an INPUT param."""
    gto = record.get("ground_truth_output", "")
    ecq = record.get("expected_clarification_question", "")
    if not ecq or not gto:
        return True, ""
    ratio = _overlap_ratio(ecq, gto)
    ok = ratio < 0.3
    msg = ("" if ok else
           f"expected_clarification_question asks for agent-retrieved data "
           f"(overlap={ratio:.2f} with ground_truth_output). "
           f"Q='{ecq[:80]}'")
    return ok, msg


def qc4_base_prompt_subset(record: dict) -> tuple[bool, str]:
    """QC-4 (C3/C4): base_prompt should be a subset of original_prompt."""
    cond = record.get("condition", "")
    if cond not in ("C3", "C4"):
        return True, ""
    base = record.get("base_prompt", "")
    orig = record.get("original_prompt", "")
    if not base or not orig:
        return True, ""
    base_tokens = _tokenize(base)
    orig_tokens = _tokenize(orig)
    extra = base_tokens - orig_tokens
    ok = len(extra) <= 3
    msg = ("" if ok else
           f"base_prompt adds {len(extra)} tokens not in original_prompt: "
           f"{sorted(extra)[:10]}")
    return ok, msg


def qc5_missing_slot_removal(record: dict) -> tuple[bool, str]:
    """QC-5 (C3/C4): Verify base_prompt actually removed something."""
    cond = record.get("condition", "")
    if cond not in ("C3", "C4"):
        return True, ""
    slot = record.get("missing_slot", "")
    orig = record.get("original_prompt", "")
    base = record.get("base_prompt", "")
    if not slot or not orig or not base:
        return True, ""
    slot_tokens = _tokenize(slot)
    orig_tokens = _tokenize(orig)
    base_tokens = _tokenize(base)
    slot_in_orig = slot_tokens & orig_tokens
    slot_removed = slot_in_orig - base_tokens
    if slot_removed:
        return True, ""
    orig_dropped = orig_tokens - base_tokens
    if orig_dropped:
        return True, ""
    msg = (f"missing_slot='{slot}' — base_prompt does not appear to remove "
           f"any content from original_prompt")
    return False, msg


def qc6_clarification_necessity(record: dict) -> tuple[bool, str]:
    """QC-6 (C3/C4): Check that the missing slot is not a defaultable preference.

    Uses word-boundary matching so that e.g. ``"format"`` does not falsely
    trigger on the substring inside ``"information"``. Multi-word patterns
    are also matched on their full whitespace-delimited form.
    """
    cond = record.get("condition", "")
    if cond not in ("C3", "C4"):
        return True, ""
    slot = record.get("missing_slot", "").lower()
    if not slot:
        return True, ""
    WEAK_PATTERNS = [
        "format", "style", "tone", "language", "sort order",
        "sort preference", "output preference", "presentation",
        "layout", "verbosity", "level of detail",
    ]
    for pat in WEAK_PATTERNS:
        # Word-boundary match: avoids "format" matching "information",
        # "tone" matching "tones"/"stone", etc. Works for multi-word
        # patterns too because \b anchors either side of the phrase.
        if re.search(rf"\b{re.escape(pat)}\b", slot):
            msg = (f"missing_slot='{record.get('missing_slot')}' looks like a "
                   f"defaultable preference ('{pat}') — agent may not ask for "
                   f"clarification")
            return False, msg
    orig = record.get("original_prompt", "")
    base = record.get("base_prompt", "")
    if orig and base and len(base) >= len(orig):
        msg = ("base_prompt is not shorter than original_prompt — "
               "slot removal may not create genuine ambiguity")
        return False, msg
    return True, ""


def qc7_slot_strength(record: dict) -> tuple[bool, str]:
    """
    QC-7 (C3/C4): Check that the missing_slot is a STRONG slot type that
    reliably forces clarification, not a weak qualifier the agent can work
    around with tools.

    WEAK patterns (agent can proceed without asking):
      - Single-word qualifiers (e.g., "Yoga" in "Yoga Class")
      - Optional description fields
      - Email subject alone when sender is present in base_prompt

    STRONG patterns (agent must ask):
      - Dates/times (e.g., "2024-05-19", "on May 15th")
      - Email addresses / recipient names
      - File names chosen by user
      - Entire goal / action removed
    """
    cond = record.get("condition", "")
    if cond not in ("C3", "C4"):
        return True, ""
    slot = record.get("missing_slot", "")
    base = record.get("base_prompt", "")
    orig = record.get("original_prompt", "")
    if not slot:
        return True, ""

    slot_tokens = _tokenize(slot)
    warnings = []

    # Check 1: Single-word qualifier (weak — agent searches broadly)
    if len(slot_tokens) == 1 and len(slot.strip()) < 15:
        # Exception: single-word that is a date, email, or filename
        single = slot.strip().lower()
        is_date = bool(re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", single))
        is_email = "@" in single
        if not is_date and not is_email:
            warnings.append(
                f"Single-word slot '{slot.strip()}' is likely a weak qualifier "
                f"the agent can search around"
            )

    # Check 2: Optional description field
    desc_patterns = [
        "the description should be",
        "description:",
        "with the description",
    ]
    slot_lower = slot.lower()
    if any(p in slot_lower for p in desc_patterns):
        warnings.append(
            f"Slot looks like an optional description field — agent may skip it"
        )

    # Check 3: Email subject alone when sender is in base_prompt
    if "subject" in slot_lower and "subject" not in base.lower():
        # Subject was removed — check if sender is still in base
        sender_patterns = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+", base)
        if sender_patterns:
            warnings.append(
                f"Slot is email subject but sender '{sender_patterns[0]}' remains "
                f"in base_prompt — agent can search by sender alone"
            )

    if warnings:
        return False, "; ".join(warnings)
    return True, ""


# ---------------------------------------------------------------------------
# Per-record check runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("QC-1_factual_consistency",     qc1_factual_consistency),
    ("QC-2_slot_type_validity",      qc2_slot_type_validity),
    ("QC-3_clarification_direction", qc3_clarification_direction),
    ("QC-4_base_prompt_subset",      qc4_base_prompt_subset),
    ("QC-5_missing_slot_removal",    qc5_missing_slot_removal),
    ("QC-6_clarification_necessity", qc6_clarification_necessity),
    ("QC-7_slot_strength",           qc7_slot_strength),
]


# Applicability predicates. A check's pass rate should only be computed
# over records to which the check actually applies; otherwise vacuously-
# True passes inflate the rate. Each predicate takes a record and returns
# True iff the check applies.
def _is_c3_or_c4(r: dict) -> bool:
    return r.get("condition") in ("C3", "C4")


SCOPE = {
    "QC-1_factual_consistency":
        lambda r: bool(r.get("ground_truth_output") and r.get("ground_truth_task_answer")),
    "QC-2_slot_type_validity":
        lambda r: bool(r.get("missing_slot") and r.get("ground_truth_output")),
    "QC-3_clarification_direction":
        lambda r: bool(r.get("expected_clarification_question") and r.get("ground_truth_output")),
    "QC-4_base_prompt_subset":
        lambda r: _is_c3_or_c4(r) and bool(r.get("base_prompt") and r.get("original_prompt")),
    "QC-5_missing_slot_removal":
        lambda r: _is_c3_or_c4(r) and bool(
            r.get("missing_slot") and r.get("original_prompt") and r.get("base_prompt")
        ),
    "QC-6_clarification_necessity":
        lambda r: _is_c3_or_c4(r) and bool(r.get("missing_slot")),
    "QC-7_slot_strength":
        lambda r: _is_c3_or_c4(r) and bool(r.get("missing_slot")),
}


def check_record(record: dict) -> dict:
    """Run all QC checks on a single record.

    Each flag dict contains:
      - pass:    True iff the check passed (or did not apply)
      - msg:     error message on failure, "" otherwise
      - applies: True iff the check applies to this record. Used by
                 compute_quality_summary() to filter out vacuous passes
                 when computing per-check pass rates.
    overall_pass is True iff every applicable check passed (vacuous
    passes never trigger a retry).
    """
    flags = {}
    for name, fn in CHECKS:
        ok, msg = fn(record)
        applies = SCOPE[name](record)
        flags[name] = {"pass": ok, "msg": msg, "applies": applies}
    flags["overall_pass"] = all(v["pass"] for v in flags.values() if isinstance(v, dict))
    return flags


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def compute_quality_summary(records: list[dict], all_flags: list[dict]) -> dict:
    total = len(records)
    by_cond: dict[str, list] = defaultdict(list)
    for r, f in zip(records, all_flags):
        by_cond[r["condition"]].append(f)

    check_names = [name for name, _ in CHECKS]

    def pass_rate(flags_list, check_name):
        # Only count records where the check applies; vacuous passes on
        # non-applicable records would otherwise inflate the rate to 100%.
        subset = [
            f[check_name]["pass"]
            for f in flags_list
            if check_name in f and f[check_name].get("applies", True)
        ]
        return round(sum(subset) / len(subset), 3) if subset else None

    overall_pass_rate = sum(f["overall_pass"] for f in all_flags) / total

    per_condition = {}
    for cond, flags_list in sorted(by_cond.items()):
        per_condition[cond] = {
            "n": len(flags_list),
            "overall_pass_rate": round(
                sum(f["overall_pass"] for f in flags_list) / len(flags_list), 3
            ),
            **{name: pass_rate(flags_list, name) for name in check_names},
        }

    return {
        "total_records": total,
        "overall_pass_rate": round(overall_pass_rate, 3),
        "per_check_pass_rate": {
            name: pass_rate(all_flags, name) for name in check_names
        },
        "per_condition": per_condition,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run quality checks over the ASPI dataset (QC-1 through QC-7)"
    )
    parser.add_argument("--input", default="output/aspi_dataset.jsonl",
                        help="Path to JSONL dataset file")
    parser.add_argument("--output", default="",
                        help="Write flagged records to this JSONL file")
    parser.add_argument("--condition", choices=["C1", "C2", "C3", "C4"], default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--fail-only", action="store_true",
                        help="When writing --output, include only failed records")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.condition and r.get("condition") != args.condition:
                continue
            if args.suite and r.get("suite") != args.suite:
                continue
            records.append(r)

    if not records:
        print("No records matched the filters.")
        sys.exit(0)

    print(f"Checking {len(records)} records …\n")

    all_flags = [check_record(r) for r in records]

    # Print per-record failures
    fail_count = 0
    for i, (r, f) in enumerate(zip(records, all_flags)):
        if f["overall_pass"]:
            continue
        fail_count += 1
        ident = (f"{r['condition']} {r.get('suite','')} {r['user_task_id']} × "
                 f"{r['injection_task_id']}"
                 + (f" {r.get('injection_operator','')}"
                    if r["condition"] == "C4" else ""))
        print(f"  FAIL [{i+1}] {ident}")
        for name, result in f.items():
            if isinstance(result, dict) and not result["pass"] and result["msg"]:
                print(f"       ✗ {name}: {result['msg']}")
        print()

    # Summary table
    summary = compute_quality_summary(records, all_flags)
    print("=" * 68)
    print("ASPI Dataset Quality Report (v2)")
    print("=" * 68)
    print(f"  Total records : {summary['total_records']}")
    print(f"  Overall pass  : {summary['overall_pass_rate']:.1%}  "
          f"({sum(f['overall_pass'] for f in all_flags)}/{len(all_flags)} records)")
    print()
    print("  Per-check pass rates:")
    for name, rate in summary["per_check_pass_rate"].items():
        bar = "▓" * int((rate or 0) * 20) + "░" * (20 - int((rate or 0) * 20))
        rate_str = f"{rate:.1%}" if rate is not None else "n/a"
        print(f"    {name:<35} {bar} {rate_str}")
    print()
    print("  Per-condition:")
    for cond, stats in summary["per_condition"].items():
        print(f"    {cond}  n={stats['n']:4d}  overall={stats['overall_pass_rate']:.1%}")
        for name, _ in CHECKS:
            r = stats.get(name)
            r_str = f"{r:.1%}" if r is not None else "n/a"
            print(f"          {name}: {r_str}")
    print("=" * 68)

    # Optional: write flagged records
    if args.output:
        out_path = Path(args.output)
        written = 0
        with open(out_path, "w") as f:
            for r, flags in zip(records, all_flags):
                if args.fail_only and flags["overall_pass"]:
                    continue
                r["quality_flags"] = flags
                f.write(json.dumps(r) + "\n")
                written += 1
        print(f"\n  Flagged records written to: {out_path}  ({written} records)")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
