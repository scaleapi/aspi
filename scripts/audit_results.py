#!/usr/bin/env python3
"""
Audit an ASPI results JSONL file for schema completeness and health.

Usage:
    python scripts/audit_results.py results/eval/pilot_p2/*.jsonl

Checks:
    1. Every record has all 8 condition keys (exec_benign, exec_tool, ...,
       clarif_user) plus clarification block.
    2. Every attacked condition has utility AND security.
    3. clarif_user has all 3 operator variants (HR, GS, CP) or a valid
       skip_reason.
    4. Skipped clarif conditions have skip_reason set.
    5. Health metrics: ask_user_rate, enforcement rate, task_completed rate.
    6. No obviously-malformed rows (missing suite, etc).

Prints summary + a compact pass/fail table.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

EXPECTED_CONDITIONS = [
    "exec_benign", "exec_tool", "exec_user", "exec_next_turn",
    "clarif_benign", "clarif_tool", "clarif_ask_user", "clarif_user",
]
ATTACKED_CONDITIONS = [
    "exec_tool", "exec_user", "exec_next_turn",
    "clarif_tool", "clarif_ask_user",
]
CLARIF_CONDITIONS = [
    "clarif_benign", "clarif_tool", "clarif_ask_user", "clarif_user",
]
OPERATORS = ("HR", "GS", "CP")


def _iter_assistant_turns(record):
    """Yield every assistant message across every condition's
    full_conversation. Handles the clarif_user variants structure."""
    for cond in EXPECTED_CONDITIONS:
        entry = record.get(cond) or {}
        if cond == "clarif_user":
            # variants: {"HR": {...}, "GS": {...}, "CP": {...}}
            for op, variant in (entry.get("variants") or {}).items():
                conv = variant.get("full_conversation") or []
                for m in conv:
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        yield cond, op, m
        else:
            conv = entry.get("full_conversation") or []
            for m in conv:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    yield cond, None, m


def _load_summary_sidecar(results_path: Path) -> dict | None:
    """Try to find the matching _summary.json next to a results file.

    Accepts either ``aspi_eval_<tag>.jsonl`` or
    ``aspi_eval_<tag>_judged.jsonl`` as input; strips ``_judged`` before
    looking up the summary.
    """
    stem = results_path.stem
    if stem.endswith("_judged"):
        stem = stem[: -len("_judged")]
    candidate = results_path.parent / f"{stem}_summary.json"
    if candidate.exists():
        try:
            return json.loads(candidate.read_text())
        except Exception:
            return None
    # Also try the conventional eval/ sibling if someone ran judge output from results/judged/
    alt = results_path.parent.parent / "eval" / f"{stem}_summary.json"
    if alt.exists():
        try:
            return json.loads(alt.read_text())
        except Exception:
            return None
    return None


def audit_record(r):
    """Return list of (severity, msg) issues found in one record."""
    issues = []

    # Required top-level fields
    for f in ("suite", "user_task_id", "injection_task_id", "clarification"):
        if f not in r:
            issues.append(("error", f"missing required field: {f}"))

    # Clarification block
    clar = r.get("clarification", {})
    for f in ("ask_user_called", "task_completed", "enforced"):
        if f not in clar:
            issues.append(("warn", f"clarification.{f} missing"))

    # All 8 conditions present
    for cond in EXPECTED_CONDITIONS:
        if cond not in r:
            issues.append(("error", f"missing condition: {cond}"))
            continue
        c = r[cond]

        # Skipped clarif conditions need skip_reason
        if cond in CLARIF_CONDITIONS and c.get("skipped"):
            if not c.get("skip_reason"):
                issues.append(("warn", f"{cond} skipped but no skip_reason"))

        # Attacked conditions need utility + security when not skipped
        if cond in ATTACKED_CONDITIONS and not c.get("skipped"):
            if "utility" not in c:
                issues.append(("error", f"{cond} missing utility"))
            if "security" not in c:
                issues.append(("error", f"{cond} missing security"))

    # clarif_user special: 3 operator variants
    cu = r.get("clarif_user", {})
    if not cu.get("skipped"):
        variants = cu.get("variants", {})
        for op in OPERATORS:
            if op not in variants:
                issues.append(("error", f"clarif_user.variants.{op} missing"))
            else:
                v = variants[op]
                if not v.get("skipped"):
                    if "utility" not in v or "security" not in v:
                        issues.append(
                            ("error", f"clarif_user.variants.{op} missing util/sec"),
                        )
        # Aggregate fields
        for f in ("utility_max", "utility_mean", "security_max", "security_mean"):
            if f not in cu:
                issues.append(("warn", f"clarif_user.{f} missing"))

    return issues


def summarize(records):
    """Print run-level health summary."""
    n = len(records)
    if n == 0:
        print("  (no records)"); return

    # Clarification health
    n_ask = sum(1 for r in records if r.get("clarification", {}).get("ask_user_called"))
    n_done = sum(1 for r in records if r.get("clarification", {}).get("task_completed"))
    n_enf = sum(1 for r in records if r.get("clarification", {}).get("enforced"))
    n_enf_ok = sum(
        1 for r in records
        if r.get("clarification", {}).get("enforced")
        and r.get("clarification", {}).get("ask_user_called")
    )

    # Per-condition counts (not-skipped)
    cond_ran = Counter()
    cond_skip_reasons = {c: Counter() for c in CLARIF_CONDITIONS}
    for r in records:
        for c in EXPECTED_CONDITIONS:
            entry = r.get(c, {})
            if not entry.get("skipped"):
                cond_ran[c] += 1
            elif c in CLARIF_CONDITIONS:
                cond_skip_reasons[c][entry.get("skip_reason", "unknown")] += 1

    # Per-condition utility + security (on ran subset)
    util = Counter()
    sec = Counter()
    for r in records:
        for c in EXPECTED_CONDITIONS:
            entry = r.get(c, {})
            if entry.get("skipped"):
                continue
            if c == "clarif_user":
                if entry.get("utility_max"):
                    util[c] += 1
                if entry.get("security_max"):
                    sec[c] += 1
            else:
                if entry.get("utility"):
                    util[c] += 1
                if c in ATTACKED_CONDITIONS and entry.get("security"):
                    sec[c] += 1

    print(f"\n  Health:")
    print(f"    n_groups:              {n}")
    print(f"    ask_user_rate:         {n_ask}/{n} ({n_ask/n:.0%})")
    print(f"    task_completed:        {n_done}/{n} ({n_done/n:.0%})")
    print(f"    enforced:              {n_enf}/{n} ({n_enf/n:.0%})")
    print(f"    enforced_success:      {n_enf_ok}/{n_enf if n_enf else 1} "
          f"({n_enf_ok/n_enf if n_enf else 0:.0%})")

    print(f"\n  Per-condition run counts (non-skipped):")
    for c in EXPECTED_CONDITIONS:
        ran = cond_ran[c]
        u = util[c]
        s_cnt = sec[c] if c in ATTACKED_CONDITIONS else None
        utl = f"u={u}/{ran}" if ran else "u=0/0"
        asr = f"ASR={s_cnt}/{ran}" if s_cnt is not None and ran else ""
        print(f"    {c:<20} ran={ran:>3}  {utl:>14}  {asr:>12}")

    print(f"\n  Clarif skip reasons:")
    for c in CLARIF_CONDITIONS:
        if cond_skip_reasons[c]:
            print(f"    {c}: {dict(cond_skip_reasons[c])}")


def reasoning_audit(records, summary: dict | None):
    """Audit reasoning-content capture across conditions.

    Reports: per-condition fraction of assistant turns that carry
    non-empty reasoning_content, global counts, and source-field
    distribution. Cross-checks against the run's _summary.json so we
    can flag "reasoning was supposed to be on but the trace is empty".
    """
    # Aggregate over every assistant turn in every condition
    total_turns = 0
    turns_with_rc = 0
    turns_with_ts = 0  # thought_signature (Gemini replay token)
    source_counts: Counter = Counter()
    per_cond_total: Counter = Counter()
    per_cond_rc: Counter = Counter()
    # Group-level: at least one turn somewhere in this record had RC
    records_with_any_rc = 0

    for r in records:
        any_rc = False
        for cond, _op, msg in _iter_assistant_turns(r):
            total_turns += 1
            per_cond_total[cond] += 1
            rc = msg.get("reasoning_content") or ""
            if isinstance(rc, str) and rc.strip():
                turns_with_rc += 1
                per_cond_rc[cond] += 1
                any_rc = True
                src = msg.get("reasoning_source")
                if isinstance(src, list):
                    for s in src:
                        source_counts[s] += 1
                elif src:
                    source_counts[src] += 1
                else:
                    source_counts["(unlabeled)"] += 1
            if msg.get("thought_signature"):
                turns_with_ts += 1
        if any_rc:
            records_with_any_rc += 1

    print(f"\n  Reasoning capture:")
    if summary:
        rm = summary.get("reasoning_mode") or "?"
        rs = summary.get("reasoning_source") or "?"
        rk = summary.get("reasoning_kwargs")
        print(f"    reasoning_mode:        {rm}")
        print(f"    reasoning_source:      {rs}")
        print(f"    reasoning_kwargs:      "
              f"{json.dumps(rk) if rk else '{}'}")
    else:
        print(f"    (no _summary.json found — cannot cross-check intended config)")

    n_records = len(records)
    if n_records:
        print(f"    records with any RC:   {records_with_any_rc}/{n_records} "
              f"({records_with_any_rc/n_records:.0%})")
    if total_turns:
        print(f"    assistant turns w/ RC: {turns_with_rc}/{total_turns} "
              f"({turns_with_rc/total_turns:.0%})")
        print(f"    thought_signature:     {turns_with_ts}/{total_turns} "
              f"({turns_with_ts/total_turns:.0%} — Gemini+tools replay token)")
    else:
        print(f"    (no assistant turns found)")

    if source_counts:
        print(f"    reasoning_source distribution:")
        for src, n in source_counts.most_common():
            print(f"      {src:<32} {n}")

    if total_turns:
        print(f"\n    Per-condition RC populated rate:")
        for c in EXPECTED_CONDITIONS:
            t = per_cond_total[c]
            r_rc = per_cond_rc[c]
            if t:
                print(f"      {c:<20} {r_rc:>4}/{t:<4} "
                      f"({r_rc/t:>4.0%})")

    # Sanity flag: if summary says reasoning should be on but capture
    # rate is suspiciously low, warn the user.
    if summary and total_turns:
        rs = summary.get("reasoning_source") or ""
        on_expected = rs.startswith("auto:") and rs != "auto:unknown" or rs == "env"
        # auto:unknown means we didn't find a reasoning-native family,
        # so zero RC is EXPECTED and not a warning.
        if on_expected:
            rate = turns_with_rc / total_turns
            if rate < 0.10:
                print(f"\n    [WARN] reasoning was enabled (source={rs}) "
                      f"but only {rate:.0%} of assistant turns carried "
                      f"reasoning_content. Possible causes:")
                print(f"           - LiteLLM route doesn't surface reasoning "
                      f"for this model (e.g. Bedrock-Qwen3 extra_body path)")
                print(f"           - Model ID is the non-thinking variant "
                      f"(e.g. Kimi K2 chat instead of K2-thinking)")
                print(f"           - Provider stripped thought parts at the "
                      f"proxy layer")


def main():
    parser = argparse.ArgumentParser(description="Audit ASPI results JSONL")
    parser.add_argument("paths", nargs="+", help="Path(s) to results JSONL")
    args = parser.parse_args()

    total_issues = Counter()
    for path in args.paths:
        p = Path(path)
        if not p.exists():
            print(f"SKIP (not found): {p}")
            continue
        print(f"\n{'='*70}")
        print(f"File: {p}")
        print(f"{'='*70}")

        records = [json.loads(l) for l in open(p) if l.strip()]
        print(f"  Loaded {len(records)} records")

        # Schema audit
        issue_count = Counter()
        example_issues_by_severity = {"error": [], "warn": []}
        for i, r in enumerate(records):
            issues = audit_record(r)
            for sev, msg in issues:
                issue_count[sev] += 1
                total_issues[sev] += 1
                if len(example_issues_by_severity[sev]) < 3:
                    example_issues_by_severity[sev].append(
                        (i, r.get("suite", "?"),
                         r.get("user_task_id", "?"),
                         r.get("injection_task_id", "?"),
                         msg),
                    )

        # Print issue summary
        if not issue_count:
            print(f"  [OK] schema: all {len(records)} records complete")
        else:
            print(f"  [!!] schema: {dict(issue_count)}")
            for sev in ("error", "warn"):
                for i, suite, ut, it, msg in example_issues_by_severity[sev]:
                    print(f"    {sev.upper()}  row={i}  {suite}/{ut}x{it}: {msg}")

        # Health summary
        summarize(records)

        # Reasoning-capture audit (cross-checks with _summary.json)
        summary = _load_summary_sidecar(p)
        reasoning_audit(records, summary)

    print(f"\n{'='*70}")
    if total_issues.get("error", 0):
        print(f"FAIL — {total_issues['error']} schema errors across all files")
        sys.exit(1)
    elif total_issues.get("warn", 0):
        print(f"OK — {total_issues['warn']} warnings (non-blocking)")
        sys.exit(0)
    else:
        print("OK — all files pass schema audit")
        sys.exit(0)


if __name__ == "__main__":
    main()
