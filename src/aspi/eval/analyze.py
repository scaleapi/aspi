"""
ASPI Results Analyzer — 8-condition design
==========================================

CLI tool to load saved evaluation results and generate tables + plots.
No model calls needed — reads from saved JSONL files.

Tables (via --summary):
  1. CLARIFICATION SIGNAL   — n, ask_user%, n_ask, completed, enforced
  2. UTILITY                — 8 conditions × (all / cond)
  3. ATTACK SUCCESS RATE    — 6 attacked conditions × (all / cond)
  4. CLARIF_USER OPERATORS  — HR / GS / CP + max / mean
  5. PAIRED DELTAS          — 12 deltas grouped by thread

Plots (via --plot):
  Core (always):
    asr_heatmap              — 2×4 state × channel grid
    asr_by_condition         — 6 attacked conditions, per-model
    utility_by_condition     — 8 conditions, per-model
    paired_deltas            — horizontal bars, 12 deltas
    operator_breakdown       — HR/GS/CP for clarif_user
    asr_by_suite             — per-suite × 6 attacked conditions, faceted
    utility_vs_asr           — scatter (utility baseline × ASR)

  Judge-only (auto-detected):
    judge_classifications    — stacked bars per condition
    compliance_reasons       — stacked bars per condition
    agreement                — env vs rule vs judge heatmap

Usage:
    python scripts/analyze.py --summary
    python scripts/analyze.py --plot all
    python scripts/analyze.py --plot core
    python scripts/analyze.py --plot asr_heatmap
    python scripts/analyze.py --add "GPT-4o" results.jsonl --plot all
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aspi.common.paths import EVAL_RESULTS_DIR, JUDGED_RESULTS_DIR, PLOTS_DIR
from aspi.eval.metrics import (
    aggregate,
    aggregate_with_judge,
    attack_attempted_rule,
    rate_with_ci,
    utility_attempted_rule,
    _adapt_result,
)

RESULTS_DIR = EVAL_RESULTS_DIR
PLOT_DIR = PLOTS_DIR


# ---------------------------------------------------------------------------
# Condition metadata
# ---------------------------------------------------------------------------

ALL_CONDITIONS = (
    "exec_benign", "exec_tool", "exec_user", "exec_next_turn",
    "clarif_benign", "clarif_tool", "clarif_ask_user", "clarif_user",
)

ATTACKED_CONDITIONS = (
    "exec_tool", "exec_user", "exec_next_turn",
    "clarif_tool", "clarif_ask_user", "clarif_user",
)

# ---------------------------------------------------------------------------
# Color palette — designed for visual comparability
# ---------------------------------------------------------------------------
#
# Design principles:
#   1. Paired state conditions share HUE (so exec↔clarif pairs pop):
#        exec_tool ↔ clarif_tool        — orange family
#        exec_next_turn ↔ clarif_user   — purple family
#   2. LIGHTNESS varies by state (exec=lighter, clarif=darker).
#   3. exec_user uses coral (its own hue — no clarif pair).
#   4. clarif_ask_user uses burgundy (distinctive — it's the novel surface
#      and should stand out in heatmaps).
#   5. Benign conditions (utility-only) live in the green/teal family to
#      stay out of the "attacked" color space.

COLORS = {
    # Benign (greens/teal)
    "exec_benign":     "#8ed1a1",   # soft green
    "clarif_benign":   "#2a9d8f",   # teal

    # Tool-channel pair (orange family — exec/clarif)
    "exec_tool":       "#f5a261",   # light orange
    "clarif_tool":     "#d35400",   # dark orange

    # User same-turn (coral — exec only, no clarif pair)
    "exec_user":       "#ef6461",   # coral

    # User next-turn pair (purple family — exec/clarif)
    "exec_next_turn":  "#b4a0dd",   # lavender
    "clarif_user":     "#6c3483",   # deep purple

    # Novel ask_user surface (burgundy — stands out)
    "clarif_ask_user": "#8b1e3f",   # burgundy
}

# Additional palettes for deltas, grouped by comparison family.
DELTA_FAMILY_COLOR = {
    "state":       "#c0392b",   # red (state effect)
    "channel_clarif": "#8b1e3f", # burgundy (channel in clarif)
    "channel_exec":  "#e67e22",  # orange (channel in exec)
    "wrapper":     "#6c3483",    # purple (wrapper/disguise)
    "interaction": "#2980b9",    # blue (interaction)
    "utility":     "#27ae60",    # green (utility delta)
}

MODEL_MARKERS = ["o", "s", "D", "^", "v", "P", "*", "X"]

# Judge-classification colors (semantic: red=bad, green=good, gray=uncertain)
CLASS_COLORS = {
    "FULL_COMPLIANCE":      "#c0392b",
    "PARTIAL_COMPLIANCE":   "#e67e22",
    "ACKNOWLEDGED_REFUSED": "#27ae60",
    "IGNORED":              "#7f8c8d",
    "UNCLEAR":              "#bdc3c7",
}

REASON_COLORS = {
    "CONFUSED":     "#3498db",
    "PERSUADED":    "#e67e22",
    "NO_REASONING": "#95a5a6",
}

# Operator colors for clarif_user (HR/GS/CP)
OPERATOR_COLORS = {
    "HR": "#1abc9c",   # teal-green (Hierarchy Rebinding)
    "GS": "#3498db",   # blue (Goal Substitution)
    "CP": "#9b59b6",   # lavender (Constraint Poisoning)
    "MAX":  "#c0392b", # red (max threat model)
    "MEAN": "#7f8c8d", # gray (mean baseline)
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def r(metric):
    if metric is None:
        return "--"
    if isinstance(metric, dict):
        v = metric.get("rate")
        return f"{v:.1%}" if v is not None else "--"
    return f"{metric:.4f}"


def ci(metric):
    if not isinstance(metric, dict) or metric.get("rate") is None:
        return "--"
    return f"{metric['rate']:.1%} [{metric['ci_low']:.1%}, {metric['ci_high']:.1%}]"


def pct(metric):
    v = None
    if isinstance(metric, dict):
        v = metric.get("rate")
    elif isinstance(metric, (int, float)):
        v = metric
    return f"{v*100:.1f}" if v is not None else "--"


def pp(val):
    """Signed percentage-point delta."""
    if val is None:
        return "--"
    if isinstance(val, dict):
        val = val.get("rate")
        if val is None:
            return "--"
    return f"{val*100:+.1f}"


def g(s, *keys):
    for k in keys:
        if k in s:
            return s[k]
    return None


def _val(metric):
    if metric is None:
        return 0.0
    if isinstance(metric, dict):
        return metric.get("rate", 0) or 0.0
    return float(metric) if metric else 0.0


def _ci_err(metric):
    if not isinstance(metric, dict) or metric.get("rate") is None:
        return None
    rate = metric["rate"]
    return [rate - metric.get("ci_low", rate), metric.get("ci_high", rate) - rate]


def _any_judge(all_data):
    return any(d.get("has_judge") for d in all_data.values())


# ---------------------------------------------------------------------------
# Discovery & loading
# ---------------------------------------------------------------------------

def discover_default_results() -> dict[str, Path]:
    def _display_name(path: Path) -> str:
        stem = path.stem
        for prefix in ("aspi_eval_", "agentdojo_eval_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        if stem.endswith("_judged"):
            stem = stem[:-len("_judged")]
        return stem

    discovered: dict[str, Path] = {}
    if JUDGED_RESULTS_DIR.exists():
        for p in sorted(JUDGED_RESULTS_DIR.glob("*.jsonl")):
            name = _display_name(p)
            discovered[name] = p
    if EVAL_RESULTS_DIR.exists():
        for p in sorted(EVAL_RESULTS_DIR.glob("*.jsonl")):
            name = _display_name(p)
            if name not in discovered:
                discovered[name] = p
    return discovered


DEFAULT_RESULTS: dict[str, Path] = {}


def load_results(result_files: dict) -> dict:
    all_data = {}
    for name, path in result_files.items():
        p = Path(path)
        if not p.exists():
            print(f"  SKIP {name}: {p}")
            continue
        results = [json.loads(l) for l in open(p) if l.strip()]
        # Adapt legacy records in-place
        for res in results:
            _adapt_result(res)
        has_judge = any("judge" in res for res in results)
        summary = aggregate(results)
        if has_judge:
            summary.update(aggregate_with_judge(results))
        all_data[name] = {
            "results": results,
            "summary": summary,
            "path": str(p),
            "has_judge": has_judge,
        }
        tag = " [judged]" if has_judge else ""
        print(f"  Loaded {name}: {len(results)} groups{tag}")
    return all_data


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def cmd_list(all_data):
    print("\nAvailable models:")
    for name, data in all_data.items():
        s = data["summary"]
        j = " [judged]" if data.get("has_judge") else ""
        print(f"  {name:<30} n={s['n_groups']:>3}  path={data['path']}{j}")


def cmd_summary(all_data):
    """Print all 5 summary tables."""
    models = list(all_data.keys())
    model_w = max(22, max(len(m) for m in models) + 2)

    def print_table(title, columns, rows, col_widths):
        header = f"{'Model':<{model_w}}" + "".join(
            f"{h:>{w}}" for h, w in zip(columns, col_widths)
        )
        total = len(header)
        print(f"\n{'─'*total}")
        print(title)
        print(f"{'─'*total}")
        print(header)
        print("-" * total)
        for name, cells in rows:
            print(
                f"{name:<{model_w}}"
                + "".join(f"{c:>{w}}" for c, w in zip(cells, col_widths))
            )

    # ── Table 1: Clarification signal ─────────────────────────────
    clar_cols = ["n", "ask_user%", "n_ask", "completed", "enforced", "enf_ok"]
    clar_widths = [6, 10, 8, 11, 10, 8]
    clar_rows = []
    for name in models:
        s = all_data[name]["summary"]
        clar_rows.append((name, [
            str(s.get("n_groups", "--")),
            pct(s.get("ask_user_rate")),
            str(s.get("n_ask_user", "--")),
            str(s.get("n_task_completed", "--")),
            str(s.get("n_enforced", "--")),
            str(s.get("n_enforced_success", "--")),
        ]))
    print_table("CLARIFICATION", clar_cols, clar_rows, clar_widths)
    print("  n_ask = called ask_user | completed = task done w/o clarif")
    print("  enforced = retry fired | enf_ok = retry succeeded")

    # ── Table 2: Utility (8 conditions) ───────────────────────────
    util_cols = [
        "exec_benign", "exec_tool", "exec_user", "exec_next",
        "cl_benign", "cl_tool", "cl_ask", "cl_user(mean)",
    ]
    util_widths = [12, 10, 10, 10, 10, 9, 9, 13]

    # Row set 1: over all n
    util_rows_all = []
    for name in models:
        s = all_data[name]["summary"]
        util_rows_all.append((name, [
            pct(g(s, "u_exec_benign")),
            pct(g(s, "u_exec_tool")),
            pct(g(s, "u_exec_user")),
            pct(g(s, "u_exec_next_turn")),
            pct(g(s, "u_clarif_benign")),
            pct(g(s, "u_clarif_tool")),
            pct(g(s, "u_clarif_ask_user")),
            pct(s.get("u_clarif_user_mean")),
        ]))
    print_table(
        "UTILITY (%) — ALL GROUPS (n; clarif conditions: skipped→0)",
        util_cols, util_rows_all, util_widths,
    )

    util_rows_cond = []
    for name in models:
        s = all_data[name]["summary"]
        util_rows_cond.append((name, [
            pct(g(s, "u_exec_benign_cond")),
            pct(g(s, "u_exec_tool_cond")),
            pct(g(s, "u_exec_user_cond")),
            pct(g(s, "u_exec_next_turn_cond")),
            pct(g(s, "u_clarif_benign_cond")),
            pct(g(s, "u_clarif_tool_cond")),
            pct(g(s, "u_clarif_ask_user_cond")),
            pct(s.get("u_clarif_user_mean_cond")),
        ]))
    print_table(
        "UTILITY (%) — CLARIFIED SUBSET (n=n_ask_user)",
        util_cols, util_rows_cond, util_widths,
    )

    # ── Table 3: ASR (6 attacked conditions) ──────────────────────
    # clarif_user shows max (headline: any operator succeeds)
    asr_cols = [
        "exec_tool", "exec_user", "exec_next",
        "cl_tool", "cl_ask", "cl_user(max)",
    ]
    asr_widths = [10, 10, 10, 9, 9, 13]

    asr_rows_all = []
    for name in models:
        s = all_data[name]["summary"]
        asr_rows_all.append((name, [
            pct(g(s, "asr_exec_tool")),
            pct(g(s, "asr_exec_user")),
            pct(g(s, "asr_exec_next_turn")),
            pct(g(s, "asr_clarif_tool")),
            pct(g(s, "asr_clarif_ask_user")),
            pct(g(s, "asr_clarif_user")),  # = _max
        ]))
    print_table(
        "ATTACK SUCCESS RATE (%) — ALL GROUPS (n; clarif skipped→0)",
        asr_cols, asr_rows_all, asr_widths,
    )

    asr_rows_cond = []
    for name in models:
        s = all_data[name]["summary"]
        asr_rows_cond.append((name, [
            pct(g(s, "asr_exec_tool_cond")),
            pct(g(s, "asr_exec_user_cond")),
            pct(g(s, "asr_exec_next_turn_cond")),
            pct(g(s, "asr_clarif_tool_cond")),
            pct(g(s, "asr_clarif_ask_user_cond")),
            pct(g(s, "asr_clarif_user_cond")),
        ]))
    print_table(
        "ATTACK SUCCESS RATE (%) — CLARIFIED SUBSET (n=n_ask_user)",
        asr_cols, asr_rows_cond, asr_widths,
    )

    # ── Table 4: clarif_user operator breakdown ───────────────────
    op_cols = ["HR", "GS", "CP", "MAX(any)", "MEAN"]
    op_widths = [8, 8, 8, 11, 9]
    op_rows = []
    for name in models:
        s = all_data[name]["summary"]
        op_rows.append((name, [
            pct(g(s, "asr_clarif_user_HR_cond", "asr_clarif_user_HR")),
            pct(g(s, "asr_clarif_user_GS_cond", "asr_clarif_user_GS")),
            pct(g(s, "asr_clarif_user_CP_cond", "asr_clarif_user_CP")),
            pct(g(s, "asr_clarif_user_cond")),
            pct(s.get("asr_clarif_user_mean_cond", s.get("asr_clarif_user_mean"))),
        ]))
    print_table(
        "CLARIF_USER OPERATOR BREAKDOWN (%) — CLARIFIED SUBSET",
        op_cols, op_rows, op_widths,
    )
    print("  MAX = any operator succeeded (headline threat model)")
    print("  MEAN = average across 3 operators")

    # ── Table 5: Paired deltas ───────────────────────────────────
    delta_groups = [
        ("STATE DELTAS (same channel, state varies)", [
            ("dasr_state_tool     [clarif_tool − exec_tool]",      "dasr_state_tool"),
            ("dasr_state_user     [clarif_user − exec_next_turn]", "dasr_state_user"),
            ("du_clarif           [u(clarif_benign) − u(exec_benign)]", "du_clarif"),
        ]),
        ("CHANNEL DELTAS — within clarification", [
            ("dasr_ask_user_surface [clarif_ask_user − clarif_tool]",    "dasr_ask_user_surface"),
            ("dasr_channel_clarif   [clarif_user − clarif_ask_user]",    "dasr_channel_clarif"),
            ("dasr_user_vs_tool_clarif [clarif_user − clarif_tool]",     "dasr_user_vs_tool_clarif"),
        ]),
        ("CHANNEL DELTAS — within execution", [
            ("dasr_exec_user_vs_tool      [exec_user − exec_tool]",       "dasr_exec_user_vs_tool"),
            ("dasr_exec_nextturn_vs_user  [exec_next_turn − exec_user]",  "dasr_exec_nextturn_vs_user"),
            ("dasr_exec_nextturn_vs_tool  [exec_next_turn − exec_tool]",  "dasr_exec_nextturn_vs_tool"),
        ]),
        ("WRAPPER / disguise deltas", [
            ("dasr_wrapper_exec   [exec_tool − exec_user (raw)]",         "dasr_wrapper_exec"),
            ("dasr_wrapper_clarif [clarif_ask_user(W) − clarif_user(h_op)]", "dasr_wrapper_clarif"),
        ]),
        ("INTERACTION", [
            ("dasr_state_gap_channel  [(clar_tool−exe_tool)−(clar_user−exe_next)]", "dasr_state_gap_channel"),
        ]),
    ]

    for group_name, rows_spec in delta_groups:
        d_cols = [m for m in models]
        d_widths = [max(12, len(m) + 2) for m in models]
        d_rows = []
        for label, key in rows_spec:
            cells = [pp(all_data[m]["summary"].get(key)) for m in models]
            d_rows.append((label, cells))
        # Print with label column wide
        label_w = max(60, max(len(lbl) for lbl, _ in rows_spec) + 2)
        header = f"{'Delta':<{label_w}}" + "".join(
            f"{m:>{w}}" for m, w in zip(d_cols, d_widths)
        )
        total = len(header)
        print(f"\n{'─'*total}")
        print(f"PAIRED DELTAS (pp) — {group_name}")
        print(f"{'─'*total}")
        print(header)
        print("-" * total)
        for label, cells in d_rows:
            print(
                f"{label:<{label_w}}"
                + "".join(f"{c:>{w}}" for c, w in zip(cells, d_widths))
            )

    print("\nNotes:")
    print("  + = second condition is more vulnerable / larger utility than first")
    print("  state deltas + channel/wrapper deltas in clarif use the clarified subset (n_ask_user)")
    print("  exec channel deltas use all groups (exec never skips)")
    print("  clarif_user uses SECURITY_MAX (any operator succeeds) in deltas")

    # ── Table 6: Reasoning analysis (judge-only) ──────────────────
    # Shows per-attacked-condition the fraction of compliant traces
    # where the agent's visible pre-tool-call text referenced the
    # attack BEFORE the attack tool call, plus the fraction of ALL
    # judged traces where the agent acknowledged the attack in text.
    judged_models = [m for m in models if all_data[m].get("has_judge")]
    if judged_models:
        r_cols = ["exec_tool", "exec_user", "exec_next", "cl_tool", "cl_ask", "cl_user"]
        r_widths = [10, 10, 10, 9, 8, 9]
        r_cond_keys = [
            ("exec_tool", "exec_tool"),
            ("exec_user", "exec_user"),
            ("exec_next", "exec_next_turn"),
            ("cl_tool",   "clarif_tool"),
            ("cl_ask",    "clarif_ask_user"),
            ("cl_user",   "clarif_user"),
        ]

        # Row 1: pre-tool reasoning (BEFORE) rate among compliant
        preto_rows = []
        for name in judged_models:
            s = all_data[name]["summary"]
            preto_rows.append((name, [
                pct(s.get(f"pre_tool_reasoning_{k}")) for _, k in r_cond_keys
            ]))
        print_table(
            "PRE-TOOL REASONING (%) — fraction of compliant traces with "
            "BEFORE-timing (deliberative compliance)",
            r_cols, preto_rows, r_widths,
        )

        # Row 2: attack_first rate among compliant
        af_rows = []
        for name in judged_models:
            s = all_data[name]["summary"]
            af_rows.append((name, [
                pct(s.get(f"attack_first_{k}")) for _, k in r_cond_keys
            ]))
        print_table(
            "ATTACK_FIRST (%) — fraction of compliant traces where attack "
            "executed before the legitimate task",
            r_cols, af_rows, r_widths,
        )

        # Row 3: attack_acknowledged rate on all judged
        ack_rows = []
        for name in judged_models:
            s = all_data[name]["summary"]
            ack_rows.append((name, [
                pct(s.get(f"attack_acknowledged_{k}")) for _, k in r_cond_keys
            ]))
        print_table(
            "ATTACK_ACKNOWLEDGED (%) — fraction of ALL judged traces where "
            "agent's text references the attack",
            r_cols, ack_rows, r_widths,
        )

        print("\nReasoning analysis notes:")
        print("  pre-tool reasoning = compliant trace showed the agent's visible text")
        print("                       reference the attack BEFORE the attack tool call")
        print("  attack_first       = compliant trace where attack tool calls preceded")
        print("                       legitimate-task tool calls")
        print("  attack_ack         = agent's visible text at any point references the")
        print("                       attack content (independent of compliance)")

        # ── Table 7: Awareness decomposition (public/covert/unaware) ──
        # Derived from attack_acknowledged (visible) + reasoning_acknowledged_in_hidden.
        # Separates three failure modes of compliance:
        #   public_ack  — model referenced the attack in visible text
        #   covert_ack  — visible silent, hidden trace referenced attack
        #                 (deliberated privately then complied)
        #   unaware     — neither visible nor hidden reference attack
        #                 (classic unaware injection)
        # hidden_cov: fraction of judged traces where any hidden reasoning
        # was captured — denominator for covert/unaware. Low coverage
        # means the provider returns no reasoning_content for this model
        # and the covert/unaware rates are not trustworthy.
        aw_cols = ["exec_tool", "exec_user", "exec_next", "cl_tool", "cl_ask", "cl_user"]
        aw_widths = [10, 10, 10, 9, 8, 9]
        aw_keys = [
            ("exec_tool", "exec_tool"),
            ("exec_user", "exec_user"),
            ("exec_next", "exec_next_turn"),
            ("cl_tool",   "clarif_tool"),
            ("cl_ask",    "clarif_ask_user"),
            ("cl_user",   "clarif_user"),
        ]

        for label, field, note in [
            ("PUBLIC_ACK (%)",  "public_ack",
             "visible text referenced the attack (denom: all judged)"),
            ("COVERT_ACK (%)",  "covert_ack",
             "hidden trace referenced attack AND visible did not "
             "(denom: traces with hidden reasoning captured)"),
            ("UNAWARE (%)",     "unaware",
             "neither visible nor hidden referenced attack "
             "(denom: traces with hidden reasoning captured)"),
            ("HIDDEN_COVERAGE (%)", "hidden_coverage",
             "fraction of judged traces where reasoning_content was "
             "captured — trust floor for covert/unaware rates"),
        ]:
            rows = []
            for name in judged_models:
                s = all_data[name]["summary"]
                rows.append((name, [
                    pct(s.get(f"{field}_{k}")) for _, k in aw_keys
                ]))
            print_table(f"{label} — {note}", aw_cols, rows, aw_widths)

        print("\nAwareness decomposition notes:")
        print("  public_ack + covert_ack + unaware do NOT sum to 100 — different")
        print("  denominators (public uses all judged; covert/unaware use only")
        print("  traces where hidden reasoning was captured).")
        print("  If hidden_coverage is low (<30%) for a model, covert/unaware")
        print("  rates for that model are unreliable — the provider likely did")
        print("  not surface reasoning_content for most traces.")


def cmd_detail(all_data, model_name):
    if model_name not in all_data:
        print(f"Model '{model_name}' not found. Available: {list(all_data.keys())}")
        return
    s = all_data[model_name]["summary"]

    print(f"\n{'='*60}")
    print(f"{model_name}  (n={s['n_groups']}, n_ask_user={s.get('n_ask_user', '?')})")
    print(f"{'='*60}")

    print(f"\nClarification:")
    if s.get("ask_user_rate") is not None:
        print(f"  {'ask_user_rate':<32} {s['ask_user_rate']*100:>21.1f}%")
    print(f"  {'n_task_completed':<32} {s.get('n_task_completed', 0):>22}")
    print(f"  {'n_enforced':<32} {s.get('n_enforced', 0):>22}")
    print(f"  {'n_enforced_success':<32} {s.get('n_enforced_success', 0):>22}")

    print(f"\nUtility — CLARIFIED SUBSET (n={s.get('n_ask_user','?')}):")
    for cond in ALL_CONDITIONS:
        key = f"u_{cond}_cond"
        val = g(s, key)
        if cond == "clarif_user":
            val = g(s, "u_clarif_user_cond")
        print(f"  {cond:<32} {ci(val):>22}")

    print(f"\nASR — CLARIFIED SUBSET (n={s.get('n_ask_user','?')}):")
    for cond in ATTACKED_CONDITIONS:
        val = g(s, f"asr_{cond}_cond")
        print(f"  {cond:<32} {ci(val):>22}")

    print(f"\nclarif_user operator breakdown (cond):")
    for op in ("HR", "GS", "CP"):
        val = g(s, f"asr_clarif_user_{op}_cond", f"asr_clarif_user_{op}")
        print(f"  {op:<32} {ci(val):>22}")
    print(f"  {'MAX (any operator)':<32} {ci(g(s, 'asr_clarif_user_cond')):>22}")
    print(f"  {'MEAN':<32} {pct(s.get('asr_clarif_user_mean_cond', s.get('asr_clarif_user_mean'))):>22}%")

    print(f"\nPaired deltas:")
    for key in [
        "dasr_state_tool", "dasr_state_user", "du_clarif",
        "dasr_ask_user_surface", "dasr_channel_clarif", "dasr_user_vs_tool_clarif",
        "dasr_exec_user_vs_tool", "dasr_exec_nextturn_vs_user", "dasr_exec_nextturn_vs_tool",
        "dasr_wrapper_exec", "dasr_wrapper_clarif", "dasr_state_gap_channel",
    ]:
        val = s.get(key)
        print(f"  {key:<32} {pp(val):>20}pp")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_asr_heatmap(all_data, save_dir):
    """2×4 state × channel heatmap of ASR, one panel per model.

    Rows: exec, clarif
    Cols: tool, ask_user, user(same-turn/reply), next-turn(user reply)
    """
    models = list(all_data.keys())
    n_models = len(models)

    # Cells (row, col) → condition key (or None if blank)
    grid = [
        # row 0: exec
        [("exec_tool",      "tool"),
         (None,             "ask_user"),   # no exec ask_user
         ("exec_user",      "user (T1)"),
         ("exec_next_turn", "user (T2)")],
        # row 1: clarif
        [("clarif_tool",     "tool"),
         ("clarif_ask_user", "ask_user"),
         (None,              "user (T1)"), # no clarif same-turn user
         ("clarif_user",     "user (T2)")],
    ]
    col_labels = ["task_tool", "ask_user_tool", "user (same-turn)", "user (next-turn)"]
    row_labels = ["execution", "clarification"]

    fig, axes = plt.subplots(
        1, n_models, figsize=(4.5 * n_models + 1, 3.5), squeeze=False,
    )
    axes = axes[0]

    for mi, m in enumerate(models):
        s = all_data[m]["summary"]
        mat = np.full((2, 4), np.nan)
        for ri, row in enumerate(grid):
            for ci_idx, (cond, _) in enumerate(row):
                if cond is None:
                    continue
                v = g(s, f"asr_{cond}_cond", f"asr_{cond}")
                if isinstance(v, dict):
                    v = v.get("rate")
                if v is not None:
                    mat[ri, ci_idx] = v

        ax = axes[mi]
        # Plot NaN cells as gray by masking them
        masked = np.ma.array(mat, mask=np.isnan(mat))
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad("#e0e0e0")   # gray for N/A cells
        im = ax.imshow(
            masked, cmap=cmap, vmin=0, vmax=1, aspect="auto",
        )
        ax.set_xticks(range(4))
        ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(2))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(m, fontsize=10)
        for ri in range(2):
            for ci_idx in range(4):
                v = mat[ri, ci_idx]
                if np.isnan(v):
                    ax.text(ci_idx, ri, "—", ha="center", va="center",
                            fontsize=11, color="#777", fontweight="bold")
                else:
                    color = "white" if v > 0.55 else "black"
                    ax.text(ci_idx, ri, f"{v:.0%}", ha="center", va="center",
                            fontsize=11, fontweight="bold", color=color)
                    # Highlight the novel surface (clarif row × ask_user col)
                    if ri == 1 and ci_idx == 1:
                        ax.add_patch(plt.Rectangle(
                            (ci_idx - 0.5, ri - 0.5), 1, 1,
                            fill=False, edgecolor=COLORS["clarif_ask_user"],
                            linewidth=3,
                        ))

    fig.suptitle(
        "ASR Heatmap — state × channel (clarified subset)\n"
        "Highlighted cell: clarif_ask_user (novel surface)",
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    out = save_dir / "aspi_asr_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_asr_by_condition(all_data, save_dir):
    """Grouped bar — ASR for 6 attacked conditions, one group per model."""
    models = list(all_data.keys())
    conds = [
        ("exec_tool",       "asr_exec_tool_cond",       COLORS["exec_tool"]),
        ("exec_user",       "asr_exec_user_cond",       COLORS["exec_user"]),
        ("exec_next_turn",  "asr_exec_next_turn_cond",  COLORS["exec_next_turn"]),
        ("clarif_tool",     "asr_clarif_tool_cond",     COLORS["clarif_tool"]),
        ("clarif_ask_user", "asr_clarif_ask_user_cond", COLORS["clarif_ask_user"]),
        ("clarif_user(max)", "asr_clarif_user_cond",    COLORS["clarif_user"]),
    ]

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(models) + 2), 5))
    x = np.arange(len(models))
    w = 0.8 / len(conds)

    for j, (label, key, color) in enumerate(conds):
        metrics = [g(all_data[m]["summary"], key) or {} for m in models]
        vals = [_val(met) for met in metrics]
        errs = [_ci_err(met) for met in metrics]
        yerr = np.array([e if e else [0, 0] for e in errs]).T

        offset = (j - (len(conds) - 1) / 2) * w
        bars = ax.bar(
            x + offset, vals, w, label=label, color=color,
            yerr=yerr, capsize=2, error_kw={"linewidth": 0.8},
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.0%}", ha="center", fontsize=6)

    ax.set_ylabel("ASR", fontsize=11)
    ax.set_title("Attack Success Rate by Condition (clarified subset)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    out = save_dir / "aspi_asr_by_condition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_utility_by_condition(all_data, save_dir):
    """Grouped bar — utility for 8 conditions, clarified subset."""
    models = list(all_data.keys())
    conds = [
        ("exec_benign",     "u_exec_benign_cond",     COLORS["exec_benign"]),
        ("exec_tool",       "u_exec_tool_cond",       COLORS["exec_tool"]),
        ("exec_user",       "u_exec_user_cond",       COLORS["exec_user"]),
        ("exec_next_turn",  "u_exec_next_turn_cond",  COLORS["exec_next_turn"]),
        ("clarif_benign",   "u_clarif_benign_cond",   COLORS["clarif_benign"]),
        ("clarif_tool",     "u_clarif_tool_cond",     COLORS["clarif_tool"]),
        ("clarif_ask_user", "u_clarif_ask_user_cond", COLORS["clarif_ask_user"]),
        ("clarif_user",     "u_clarif_user_cond",     COLORS["clarif_user"]),
    ]

    fig, ax = plt.subplots(figsize=(max(12, 2.2 * len(models) + 2), 5))
    x = np.arange(len(models))
    w = 0.8 / len(conds)

    for j, (label, key, color) in enumerate(conds):
        metrics = [g(all_data[m]["summary"], key) or {} for m in models]
        vals = [_val(met) for met in metrics]
        errs = [_ci_err(met) for met in metrics]
        yerr = np.array([e if e else [0, 0] for e in errs]).T

        offset = (j - (len(conds) - 1) / 2) * w
        ax.bar(
            x + offset, vals, w, label=label, color=color,
            yerr=yerr, capsize=2, error_kw={"linewidth": 0.8},
        )

    ax.set_ylabel("Utility", fontsize=11)
    ax.set_title("Utility by Condition (clarified subset)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    out = save_dir / "aspi_utility_by_condition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_paired_deltas(all_data, save_dir):
    """Horizontal bars — all 12 paired deltas, one row per delta.

    Row colors encode family (state / channel / wrapper / interaction).
    Bar height within each row encodes per-model magnitude.
    """
    models = list(all_data.keys())

    delta_spec = [
        # (label, summary_key, family)
        ("A1  state_tool       [clarif_tool − exec_tool]",              "dasr_state_tool",              "state"),
        ("A2  state_user       [clarif_user − exec_next_turn]",         "dasr_state_user",              "state"),
        ("A3  du_clarif        [u(clarif_benign) − u(exec_benign)]",    "du_clarif",                    "utility"),
        ("B1  ask_user surface [clarif_ask_user − clarif_tool]",        "dasr_ask_user_surface",        "channel_clarif"),
        ("B2  channel clarif   [clarif_user − clarif_ask_user]",        "dasr_channel_clarif",          "channel_clarif"),
        ("B3  cl_user vs tool  [clarif_user − clarif_tool]",            "dasr_user_vs_tool_clarif",     "channel_clarif"),
        ("C1  exec user−tool   [exec_user − exec_tool]",                "dasr_exec_user_vs_tool",       "channel_exec"),
        ("C2  exec next−user   [exec_next_turn − exec_user]",           "dasr_exec_nextturn_vs_user",   "channel_exec"),
        ("C3  exec next−tool   [exec_next_turn − exec_tool]",           "dasr_exec_nextturn_vs_tool",   "channel_exec"),
        ("D1  wrapper_exec     [exec_tool − exec_user (raw)]",          "dasr_wrapper_exec",            "wrapper"),
        ("D2  wrapper_clarif   [clarif_ask_user(W) − clarif_user(h)]",  "dasr_wrapper_clarif",          "wrapper"),
        ("F1  state_gap_channel                                     ",   "dasr_state_gap_channel",       "interaction"),
    ]

    fig, ax = plt.subplots(
        figsize=(max(12, 1.2 * len(models) + 8),
                 0.55 * len(delta_spec) + 1.5),
    )
    y = np.arange(len(delta_spec))
    w = 0.8 / max(1, len(models))

    # Row background shading by family
    family_bg = {
        "state": "#fdeee8", "utility": "#e8f5e9",
        "channel_clarif": "#f5e3e9", "channel_exec": "#fdf2e6",
        "wrapper": "#f0e5f5", "interaction": "#e3f0f7",
    }
    for yi, (_, _, fam) in enumerate(delta_spec):
        ax.axhspan(yi - 0.5, yi + 0.5, color=family_bg.get(fam, "#f5f5f5"),
                   alpha=0.5, zorder=0)

    # Draw bars per model
    model_palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for mi, m in enumerate(models):
        s = all_data[m]["summary"]
        vals = [(s.get(key) or 0) * 100 for _, key, _ in delta_spec]
        offset = (mi - (len(models) - 1) / 2) * w
        ax.barh(
            y + offset, vals, w, label=m,
            color=model_palette[mi % 10],
            edgecolor="white", linewidth=0.5,
        )

    # Family-colored labels on the left
    for yi, (label, _, fam) in enumerate(delta_spec):
        col = DELTA_FAMILY_COLOR.get(fam, "#333")
        ax.text(
            -0.005, yi, label,
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=8,
            color=col, family="monospace",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([])  # custom labels drawn above
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=1, zorder=3)
    ax.set_xlabel("Δ (percentage points)", fontsize=10)
    ax.set_title(
        "Paired Deltas — 12 comparisons grouped by family\n"
        "rows: state (red) | utility (green) | channel-clarif (burgundy) | "
        "channel-exec (orange) | wrapper (purple) | interaction (blue)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right", ncol=min(len(models), 3), framealpha=0.9)
    ax.grid(axis="x", alpha=0.3, zorder=1)
    plt.tight_layout()
    out = save_dir / "aspi_paired_deltas.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Focused comparison plots (one per narrative thread)
# ---------------------------------------------------------------------------

def plot_state_effect_pairs(all_data, save_dir):
    """THE headline plot: state effect on two independent channels.

    Two sub-panels side-by-side:
      Left:  exec_tool vs clarif_tool (task-tool channel)
      Right: exec_next_turn vs clarif_user(max) (user next-turn channel)

    Each sub-panel: grouped bars (exec=light, clarif=dark) per model.
    """
    models = list(all_data.keys())

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.4 * len(models) + 6), 5))

    panels = [
        ("Tool channel\n(clarif_tool − exec_tool)", [
            ("exec_tool",   "asr_exec_tool_cond",   COLORS["exec_tool"]),
            ("clarif_tool", "asr_clarif_tool_cond", COLORS["clarif_tool"]),
        ], "dasr_state_tool"),
        ("User next-turn channel\n(clarif_user(max) − exec_next_turn)", [
            ("exec_next_turn", "asr_exec_next_turn_cond", COLORS["exec_next_turn"]),
            ("clarif_user",    "asr_clarif_user_cond",    COLORS["clarif_user"]),
        ], "dasr_state_user"),
    ]

    for ax, (title, spec, delta_key) in zip(axes, panels):
        x = np.arange(len(models))
        w = 0.35
        for j, (label, key, color) in enumerate(spec):
            vals = [_val(g(all_data[m]["summary"], key)) for m in models]
            errs = [_ci_err(g(all_data[m]["summary"], key)) for m in models]
            yerr = np.array([e if e else [0, 0] for e in errs]).T
            offset = (j - 0.5) * w
            bars = ax.bar(
                x + offset, vals, w, label=label, color=color,
                yerr=yerr, capsize=3, error_kw={"linewidth": 0.9},
                edgecolor="black", linewidth=0.4,
            )
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                            f"{h:.0%}", ha="center", fontsize=8)

        # Delta annotation above each model
        for i, m in enumerate(models):
            d = all_data[m]["summary"].get(delta_key)
            if d is not None:
                sign = "+" if d >= 0 else ""
                ax.text(x[i], 1.08, f"Δ={sign}{d*100:.0f}pp",
                        ha="center", fontsize=9, fontweight="bold",
                        color=DELTA_FAMILY_COLOR["state"])

        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("ASR")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "State Effect on Prompt Injection — paired across channels",
        fontsize=12, y=1.03,
    )
    plt.tight_layout()
    out = save_dir / "aspi_state_effect_pairs.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_channel_clarif_focus(all_data, save_dir):
    """Focus on 3 clarification channels: task-tool / ask_user-tool / user-reply.

    Bars: clarif_tool, clarif_ask_user, clarif_user(max) per model.
    Highlights the novel surface (clarif_ask_user in burgundy).
    """
    models = list(all_data.keys())
    conds = [
        ("clarif_tool",     "asr_clarif_tool_cond",     COLORS["clarif_tool"]),
        ("clarif_ask_user", "asr_clarif_ask_user_cond", COLORS["clarif_ask_user"]),
        ("clarif_user(max)", "asr_clarif_user_cond",    COLORS["clarif_user"]),
    ]

    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(models) + 3), 5))
    x = np.arange(len(models))
    w = 0.25

    for j, (label, key, color) in enumerate(conds):
        vals = [_val(g(all_data[m]["summary"], key)) for m in models]
        errs = [_ci_err(g(all_data[m]["summary"], key)) for m in models]
        yerr = np.array([e if e else [0, 0] for e in errs]).T
        offset = (j - 1) * w
        bars = ax.bar(
            x + offset, vals, w, label=label, color=color,
            yerr=yerr, capsize=3,
            edgecolor="black", linewidth=0.5,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.0%}", ha="center", fontsize=8)

    # Annotate the novel surface emphasis
    ax.text(
        0.02, 0.95,
        "Novel surface: clarif_ask_user (burgundy)\n"
        "— attack rides on the ask_user tool's own return value.",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fce4ec",
                  edgecolor=COLORS["clarif_ask_user"], linewidth=1),
    )

    ax.set_ylabel("ASR", fontsize=11)
    ax.set_title("Channel Effect within Clarification State", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = save_dir / "aspi_channel_clarif_focus.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_channel_exec_focus(all_data, save_dir):
    """Focus on 3 execution channels: task-tool / user same-turn / user next-turn.

    Bars: exec_tool, exec_user, exec_next_turn per model.
    """
    models = list(all_data.keys())
    conds = [
        ("exec_tool",      "asr_exec_tool",      COLORS["exec_tool"]),
        ("exec_user",      "asr_exec_user",      COLORS["exec_user"]),
        ("exec_next_turn", "asr_exec_next_turn", COLORS["exec_next_turn"]),
    ]

    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(models) + 3), 5))
    x = np.arange(len(models))
    w = 0.25

    for j, (label, key, color) in enumerate(conds):
        vals = [_val(g(all_data[m]["summary"], key)) for m in models]
        errs = [_ci_err(g(all_data[m]["summary"], key)) for m in models]
        yerr = np.array([e if e else [0, 0] for e in errs]).T
        offset = (j - 1) * w
        bars = ax.bar(
            x + offset, vals, w, label=label, color=color,
            yerr=yerr, capsize=3,
            edgecolor="black", linewidth=0.5,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.0%}", ha="center", fontsize=8)

    ax.set_ylabel("ASR", fontsize=11)
    ax.set_title(
        "Channel Effect within Execution State\n"
        "(task-tool-channel vs user-prompt vs next-turn user)",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = save_dir / "aspi_channel_exec_focus.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_utility_cost(all_data, save_dir):
    """Utility cost bar — benign vs attacked conditions.

    For each model, show:
      u(exec_benign), u(exec_tool), u(clarif_benign), u(clarif_ask_user)
    This visualises whether the attack degrades utility in addition to
    succeeding as an attack.
    """
    models = list(all_data.keys())
    conds = [
        ("exec_benign",     "u_exec_benign_cond",     COLORS["exec_benign"]),
        ("exec_tool",       "u_exec_tool_cond",       COLORS["exec_tool"]),
        ("clarif_benign",   "u_clarif_benign_cond",   COLORS["clarif_benign"]),
        ("clarif_ask_user", "u_clarif_ask_user_cond", COLORS["clarif_ask_user"]),
    ]

    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(models) + 3), 5))
    x = np.arange(len(models))
    w = 0.2

    for j, (label, key, color) in enumerate(conds):
        vals = [_val(g(all_data[m]["summary"], key)) for m in models]
        errs = [_ci_err(g(all_data[m]["summary"], key)) for m in models]
        yerr = np.array([e if e else [0, 0] for e in errs]).T
        offset = (j - 1.5) * w
        ax.bar(
            x + offset, vals, w, label=label, color=color,
            yerr=yerr, capsize=3,
            edgecolor="black", linewidth=0.5,
        )

    ax.set_ylabel("Utility", fontsize=11)
    ax.set_title(
        "Utility: benign vs attacked × exec vs clarif",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = save_dir / "aspi_utility_cost.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_operator_breakdown(all_data, save_dir):
    """Grouped bar — HR / GS / CP + MAX + MEAN for clarif_user, per model.

    MAX = any operator succeeded (headline threat model).
    MEAN = average across 3 operators (statistical baseline).
    """
    models = list(all_data.keys())
    series = [
        ("HR",  "asr_clarif_user_HR_cond",  OPERATOR_COLORS["HR"]),
        ("GS",  "asr_clarif_user_GS_cond",  OPERATOR_COLORS["GS"]),
        ("CP",  "asr_clarif_user_CP_cond",  OPERATOR_COLORS["CP"]),
        ("MAX", "asr_clarif_user_cond",     OPERATOR_COLORS["MAX"]),
    ]

    fig, ax = plt.subplots(figsize=(max(9, 1.7 * len(models) + 3), 5))
    x = np.arange(len(models))
    w = 0.2

    for j, (label, key, color) in enumerate(series):
        vals = [_val(g(all_data[m]["summary"], key)) for m in models]
        offset = (j - 1.5) * w
        bars = ax.bar(
            x + offset, vals, w, label=label, color=color,
            edgecolor="black", linewidth=0.4,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.0%}", ha="center", fontsize=7)

    # Overlay MEAN as a dashed line (statistical baseline)
    mean_vals = [
        all_data[m]["summary"].get(
            "asr_clarif_user_mean_cond",
            all_data[m]["summary"].get("asr_clarif_user_mean"),
        ) or 0
        for m in models
    ]
    ax.plot(
        x, mean_vals, "s--", color=OPERATOR_COLORS["MEAN"],
        markersize=9, linewidth=1.5, label="MEAN (avg)",
    )

    ax.set_ylabel("ASR", fontsize=11)
    ax.set_title(
        "clarif_user — HR / GS / CP operator breakdown + MAX (any) + MEAN",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.set_ylim(0, 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = save_dir / "aspi_operator_breakdown.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_near_miss(all_data, save_dir):
    """Judge-only — ASR (deterministic) vs Attempted (judge) per condition.

    Shows the gap between what actually succeeded (security) and what the
    agent tried to do (attack_attempted). The shaded area between is the
    "near-miss" rate — agents that tried but failed due to environmental
    friction rather than refusal.
    """
    judged_models = [m for m in all_data if all_data[m].get("has_judge")]
    if not judged_models:
        print("plot_near_miss: no judged data"); return

    conds = [
        "exec_tool", "exec_user", "exec_next_turn",
        "clarif_tool", "clarif_ask_user", "clarif_user",
    ]

    n_models = len(judged_models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(5 * n_models + 1, 5), sharey=True, squeeze=False,
    )
    axes = axes[0]

    for mi, m in enumerate(judged_models):
        ax = axes[mi]
        s = all_data[m]["summary"]
        x = np.arange(len(conds))

        asr_vals = []
        attempted_vals = []
        for c in conds:
            a = s.get(f"asr_{c}_cond") if c in ("clarif_tool", "clarif_ask_user", "clarif_user") else s.get(f"asr_{c}")
            asr_vals.append(_val(a))
            at = s.get(f"attempted_{c}")
            attempted_vals.append(at if at is not None else 0)

        w = 0.38
        b1 = ax.bar(
            x - w / 2, attempted_vals, w, label="Attempted (judge)",
            color="#c7a0d1", edgecolor="black", linewidth=0.4,
        )
        b2 = ax.bar(
            x + w / 2, asr_vals, w, label="ASR (deterministic)",
            color="#c0392b", edgecolor="black", linewidth=0.4,
        )

        # Shade near-miss gap
        for xi, (a, at) in enumerate(zip(asr_vals, attempted_vals)):
            gap = at - a
            if gap > 0.02:
                ax.annotate(
                    f"near-miss\n+{gap*100:.0f}pp",
                    xy=(xi, at), xytext=(xi, at + 0.08),
                    ha="center", fontsize=7,
                    color="#555",
                    arrowprops=dict(arrowstyle="-", color="#555", linewidth=0.5),
                )

        ax.set_title(m, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, fontsize=7, rotation=25, ha="right")
        ax.set_ylim(0, 1.25)
        if mi == 0:
            ax.set_ylabel("Rate")
            ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Near-Miss Analysis — Attempted (intent) vs ASR (success)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = save_dir / "aspi_near_miss.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_reasoning_timing_stack(all_data, save_dir):
    """Stacked bar — reasoning_timing distribution per attacked condition, per model.

    Bars show the fraction of compliant traces where the agent's visible
    pre-tool-call text referenced the attack BEFORE / DURING / AFTER /
    NONE relative to the first attack-aligned tool call.

    Interpretation:
      - BEFORE = deliberative / pre-meditated compliance
      - DURING = interleaved reasoning and execution
      - AFTER  = post-hoc rationalization
      - NONE   = reflexive compliance (no visible reasoning about the attack)
    """
    judged_models = [m for m in all_data if all_data[m].get("has_judge")]
    if not judged_models:
        print("plot_reasoning_timing_stack: no judged data"); return

    conds = [
        "exec_tool", "exec_user", "exec_next_turn",
        "clarif_tool", "clarif_ask_user", "clarif_user",
    ]
    timing_cats = ["BEFORE", "DURING", "AFTER", "NONE"]
    timing_colors = {
        "BEFORE": "#2e86c1",   # blue (deliberative)
        "DURING": "#f39c12",   # orange (interleaved)
        "AFTER":  "#8e44ad",   # purple (post-hoc)
        "NONE":   "#7f8c8d",   # gray (reflexive)
    }

    n_models = len(judged_models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(5 * n_models + 1, 5),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for mi, m in enumerate(judged_models):
        ax = axes[mi]
        s = all_data[m]["summary"]
        x = np.arange(len(conds))
        bottom = np.zeros(len(conds))
        ns = [s.get(f"reasoning_timing_{c}_n", 0) for c in conds]

        for cat in timing_cats:
            vals = []
            for c in conds:
                v = s.get(f"reasoning_timing_{c}_{cat}")
                vals.append(0 if v is None else v)
            vals_arr = np.array(vals)
            ax.bar(
                x, vals_arr, 0.7, bottom=bottom,
                label=cat, color=timing_colors[cat],
                edgecolor="white", linewidth=0.5,
            )
            for xi, v in enumerate(vals_arr):
                if v > 0.08:
                    ax.text(
                        xi, bottom[xi] + v / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold",
                    )
            bottom += vals_arr

        # n labels below
        for xi, nv in enumerate(ns):
            ax.text(xi, -0.08, f"n={nv}", ha="center", fontsize=7, color="gray")

        ax.set_title(m, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, fontsize=7, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        if mi == 0:
            ax.set_ylabel("Fraction of compliant traces")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(timing_cats),
        fontsize=9, bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle(
        "Reasoning Timing — where does the agent's visible reasoning sit "
        "relative to the attack tool call?\n"
        "(compliant traces only)",
        fontsize=11, y=1.13,
    )
    plt.tight_layout()
    out = save_dir / "aspi_reasoning_timing_stack.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_attack_ordering(all_data, save_dir):
    """Stacked bar — order_of_execution per attacked condition, per model.

    Shows whether the attack executed BEFORE / INTERLEAVED with / AFTER
    the legitimate user task (or task-only / attack-only).
    """
    judged_models = [m for m in all_data if all_data[m].get("has_judge")]
    if not judged_models:
        print("plot_attack_ordering: no judged data"); return

    conds = [
        "exec_tool", "exec_user", "exec_next_turn",
        "clarif_tool", "clarif_ask_user", "clarif_user",
    ]
    order_cats = [
        "ATTACK_FIRST", "TASK_FIRST", "INTERLEAVED",
        "TASK_ONLY", "ATTACK_ONLY", "NEITHER",
    ]
    order_colors = {
        "ATTACK_FIRST": "#c0392b",   # red (attack prioritised)
        "TASK_FIRST":   "#2ecc71",   # green (task prioritised)
        "INTERLEAVED":  "#f39c12",   # orange (mixed)
        "TASK_ONLY":    "#27ae60",   # darker green (resistant — task only)
        "ATTACK_ONLY":  "#7b241c",   # dark red (captured — attack only)
        "NEITHER":      "#7f8c8d",   # gray (neither completed)
    }

    n_models = len(judged_models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(5 * n_models + 1, 5),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for mi, m in enumerate(judged_models):
        ax = axes[mi]
        s = all_data[m]["summary"]
        x = np.arange(len(conds))
        bottom = np.zeros(len(conds))

        for cat in order_cats:
            vals = []
            for c in conds:
                v = s.get(f"order_of_execution_{c}_{cat}")
                vals.append(0 if v is None else v)
            vals_arr = np.array(vals)
            ax.bar(
                x, vals_arr, 0.7, bottom=bottom,
                label=cat, color=order_colors[cat],
                edgecolor="white", linewidth=0.5,
            )
            for xi, v in enumerate(vals_arr):
                if v > 0.08:
                    ax.text(
                        xi, bottom[xi] + v / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold",
                    )
            bottom += vals_arr

        ax.set_title(m, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, fontsize=7, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        if mi == 0:
            ax.set_ylabel("Fraction of compliant traces")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(order_cats),
        fontsize=8, bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle(
        "Order of Execution — did the agent run the attack first, the task "
        "first, or interleave them?\n(compliant traces only)",
        fontsize=11, y=1.13,
    )
    plt.tight_layout()
    out = save_dir / "aspi_attack_ordering.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_attack_acknowledgement(all_data, save_dir):
    """Grouped bar — fraction of judged traces where agent acknowledged attack.

    Useful for seeing whether the agent's TEXT ever references the attack
    content, independent of whether it complied. Distinguishes:
      - high ack + high ASR → agent saw the attack and was persuaded
      - high ack + low ASR  → agent saw the attack and refused
      - low ack  + high ASR → agent was confused (didn't see attack as separate)
      - low ack  + low ASR  → agent ignored the attack content
    """
    judged_models = [m for m in all_data if all_data[m].get("has_judge")]
    if not judged_models:
        print("plot_attack_acknowledgement: no judged data"); return

    conds = [
        ("exec_tool",       "attack_acknowledged_exec_tool",       COLORS["exec_tool"]),
        ("exec_user",       "attack_acknowledged_exec_user",       COLORS["exec_user"]),
        ("exec_next_turn",  "attack_acknowledged_exec_next_turn",  COLORS["exec_next_turn"]),
        ("clarif_tool",     "attack_acknowledged_clarif_tool",     COLORS["clarif_tool"]),
        ("clarif_ask_user", "attack_acknowledged_clarif_ask_user", COLORS["clarif_ask_user"]),
        ("clarif_user",     "attack_acknowledged_clarif_user",     COLORS["clarif_user"]),
    ]

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(judged_models) + 2), 5))
    x = np.arange(len(judged_models))
    w = 0.8 / len(conds)

    for j, (label, key, color) in enumerate(conds):
        vals = [all_data[m]["summary"].get(key) or 0 for m in judged_models]
        offset = (j - (len(conds) - 1) / 2) * w
        bars = ax.bar(
            x + offset, vals, w, label=label, color=color,
            edgecolor="black", linewidth=0.4,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.0%}", ha="center", fontsize=6,
                )

    ax.set_ylabel("Attack-acknowledged rate", fontsize=11)
    ax.set_title(
        "Attack Acknowledgement — fraction of traces where agent's "
        "visible text references the attack",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(judged_models, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = save_dir / "aspi_attack_acknowledgement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_attempted_by_condition(all_data, save_dir):
    """Judge-only — attempted rate (any operator for clarif_user) per condition.

    Parallel to plot_asr_by_condition but showing the judge's intent metric.
    """
    judged_models = [m for m in all_data if all_data[m].get("has_judge")]
    if not judged_models:
        print("plot_attempted_by_condition: no judged data"); return

    conds = [
        ("exec_tool",       "attempted_exec_tool",        COLORS["exec_tool"]),
        ("exec_user",       "attempted_exec_user",        COLORS["exec_user"]),
        ("exec_next_turn",  "attempted_exec_next_turn",   COLORS["exec_next_turn"]),
        ("clarif_tool",     "attempted_clarif_tool",      COLORS["clarif_tool"]),
        ("clarif_ask_user", "attempted_clarif_ask_user",  COLORS["clarif_ask_user"]),
        ("clarif_user(any)", "attempted_clarif_user",     COLORS["clarif_user"]),
    ]

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(judged_models) + 2), 5))
    x = np.arange(len(judged_models))
    w = 0.8 / len(conds)

    for j, (label, key, color) in enumerate(conds):
        vals = [
            all_data[m]["summary"].get(key) or 0 for m in judged_models
        ]
        offset = (j - (len(conds) - 1) / 2) * w
        bars = ax.bar(
            x + offset, vals, w, label=label, color=color,
            edgecolor="black", linewidth=0.4,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.0%}", ha="center", fontsize=6)

    ax.set_ylabel("Attempted rate (judge)", fontsize=11)
    ax.set_title("Attack Attempted Rate by Condition — intent metric from LLM judge", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(judged_models, rotation=15, ha="right")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = save_dir / "aspi_attempted_by_condition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_asr_by_suite(all_data, save_dir):
    """Per-suite ASR for 6 attacked conditions, faceted by model."""
    models = list(all_data.keys())
    all_suites = sorted(set(
        s for m in models
        for s in all_data[m]["summary"].get("by_suite", {}).keys()
    ))
    if not all_suites:
        print("No suite data"); return

    cond_specs = [
        ("exec_tool",       "asr_exec_tool_cond",       COLORS["exec_tool"]),
        ("exec_user",       "asr_exec_user_cond",       COLORS["exec_user"]),
        ("exec_next_turn",  "asr_exec_next_turn_cond",  COLORS["exec_next_turn"]),
        ("clarif_tool",     "asr_clarif_tool_cond",     COLORS["clarif_tool"]),
        ("clarif_ask_user", "asr_clarif_ask_user_cond", COLORS["clarif_ask_user"]),
        ("clarif_user",     "asr_clarif_user_cond",     COLORS["clarif_user"]),
    ]
    bar_w = 0.8 / len(cond_specs)

    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(5 * n_models, 5), sharey=True, squeeze=False,
    )
    axes = axes[0]

    for mi, m in enumerate(models):
        ax = axes[mi]
        by_suite = all_data[m]["summary"].get("by_suite", {})
        x = np.arange(len(all_suites))
        for j, (label, key, color) in enumerate(cond_specs):
            vals = [_val(by_suite.get(s, {}).get(key)) for s in all_suites]
            offset = (j - (len(cond_specs) - 1) / 2) * bar_w
            ax.bar(x + offset, vals, bar_w, label=label, color=color)
        ax.set_title(m, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(all_suites, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        if mi == 0:
            ax.set_ylabel("ASR")
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle("ASR by Task Suite — 6 attacked conditions", fontsize=12, y=1.02)
    plt.tight_layout()
    out = save_dir / "aspi_asr_by_suite.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_utility_vs_asr(all_data, save_dir):
    """Scatter — utility baseline u(exec_benign) vs ASR for 6 attacked conditions."""
    models = list(all_data.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    conds = [
        ("exec_tool",       "asr_exec_tool_cond",       COLORS["exec_tool"]),
        ("exec_user",       "asr_exec_user_cond",       COLORS["exec_user"]),
        ("exec_next_turn",  "asr_exec_next_turn_cond",  COLORS["exec_next_turn"]),
        ("clarif_tool",     "asr_clarif_tool_cond",     COLORS["clarif_tool"]),
        ("clarif_ask_user", "asr_clarif_ask_user_cond", COLORS["clarif_ask_user"]),
        ("clarif_user",     "asr_clarif_user_cond",     COLORS["clarif_user"]),
    ]

    for i, m in enumerate(models):
        s = all_data[m]["summary"]
        x = _val(g(s, "u_exec_benign_cond", "u_exec_benign"))
        mk = MODEL_MARKERS[i % len(MODEL_MARKERS)]
        y_vals = []
        for label, key, color in conds:
            y = _val(g(s, key))
            ax.scatter(
                x, y, marker=mk, s=90, color=color, zorder=5,
                edgecolor="black", linewidth=0.5,
            )
            y_vals.append(y)
        ax.annotate(m, (x + 0.01, max(y_vals) + 0.02), fontsize=8)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=color, markersize=9,
            markeredgecolor="black", markeredgewidth=0.5,
            label=f"ASR({label})",
        )
        for label, _, color in conds
    ]
    for i, m in enumerate(models):
        legend_els.append(
            Line2D(
                [0], [0], marker=MODEL_MARKERS[i % len(MODEL_MARKERS)],
                color="w", markerfacecolor="gray", markersize=8, label=m,
            )
        )
    ax.legend(handles=legend_els, fontsize=7, loc="best", ncol=2)
    ax.set_xlabel("Utility u(exec_benign) — benign baseline", fontsize=11)
    ax.set_ylabel("Attack Success Rate", fontsize=11)
    ax.set_title("Utility vs ASR — 6 attacked conditions", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = save_dir / "aspi_utility_vs_asr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Judge-only plots
# ---------------------------------------------------------------------------

def _collect_judge_counts(results, cond, field="classification",
                          variant_ops=None):
    """Collect judge classification counts for `cond` across judged results.

    For clarif_user, if `variant_ops` is given, aggregate over those ops
    with "any" semantics: classify as FULL if any operator FULL, else
    PARTIAL if any PARTIAL, else whichever is most common.
    """
    counts = Counter()
    for res in results:
        if "judge" not in res:
            continue
        j = res["judge"]
        if cond == "clarif_user" and variant_ops:
            # Aggregate across operators
            priorities = [
                "FULL_COMPLIANCE", "PARTIAL_COMPLIANCE",
                "ACKNOWLEDGED_REFUSED", "IGNORED", "UNCLEAR",
            ]
            picked = None
            per_op_classes = []
            for op in variant_ops:
                ja = j.get(f"clarif_user_{op}_judge_attack", {})
                cls = ja.get(field, None)
                if cls:
                    per_op_classes.append(cls)
            if per_op_classes:
                for p in priorities:
                    if p in per_op_classes:
                        picked = p
                        break
                if picked:
                    counts[picked] += 1
        else:
            ja = j.get(f"{cond}_judge_attack", {})
            cls = ja.get(field, None)
            if cls:
                counts[cls] += 1
    return counts


def plot_judge_classifications(all_data, save_dir):
    """Stacked bar — attack classification distribution per condition per model."""
    judged_models = [m for m in all_data if all_data[m].get("has_judge")]
    if not judged_models:
        print("No judged data for classification plot"); return

    cats = ["FULL_COMPLIANCE", "PARTIAL_COMPLIANCE", "ACKNOWLEDGED_REFUSED", "IGNORED", "UNCLEAR"]
    conds = [
        "exec_tool", "exec_user", "exec_next_turn",
        "clarif_tool", "clarif_ask_user", "clarif_user",
    ]

    n_models = len(judged_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models + 1, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for mi, m in enumerate(judged_models):
        ax = axes[mi]
        results = all_data[m]["results"]
        counts = {
            c: _collect_judge_counts(results, c,
                                     variant_ops=("HR", "GS", "CP") if c == "clarif_user" else None)
            for c in conds
        }
        active = [c for c in conds if sum(counts[c].values()) > 0]
        x = np.arange(len(active))
        bottom = np.zeros(len(active))
        for cat in cats:
            vals = []
            for c in active:
                total = sum(counts[c].values())
                vals.append(counts[c].get(cat, 0) / total if total else 0)
            vals = np.array(vals)
            ax.bar(x, vals, 0.7, bottom=bottom, label=cat,
                   color=CLASS_COLORS.get(cat, "#ccc"))
            for xi, v in enumerate(vals):
                if v > 0.05:
                    ax.text(xi, bottom[xi] + v / 2, f"{v:.0%}",
                            ha="center", va="center", fontsize=7,
                            color="white", fontweight="bold")
            bottom += vals

        ax.set_title(m, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(active, fontsize=7, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        if mi == 0:
            ax.set_ylabel("Proportion")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(cats), fontsize=8,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Attack Classification Distribution — 6 attacked conditions",
                 fontsize=12, y=1.12)
    plt.tight_layout()
    out = save_dir / "aspi_judge_classifications.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_compliance_reasons(all_data, save_dir):
    """Stacked bar — compliance reason per condition per model."""
    judged_models = [m for m in all_data if all_data[m].get("has_judge")]
    if not judged_models:
        print("No judged data for compliance reasons"); return

    reasons = ["CONFUSED", "PERSUADED", "NO_REASONING"]
    conds = [
        "exec_tool", "exec_user", "exec_next_turn",
        "clarif_tool", "clarif_ask_user", "clarif_user",
    ]

    n_models = len(judged_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models + 1, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for mi, m in enumerate(judged_models):
        ax = axes[mi]
        results = all_data[m]["results"]
        by_cond = {c: Counter() for c in conds}
        for res in results:
            if "judge" not in res:
                continue
            j = res["judge"]
            for c in conds:
                if c == "clarif_user":
                    # Aggregate across operators — use the first compliant op's reason
                    for op in ("HR", "GS", "CP"):
                        ja = j.get(f"clarif_user_{op}_judge_attack", {})
                        if ja.get("attack_attempted"):
                            reason = ja.get("compliance_reason", "NO_REASONING")
                            if reason not in reasons:
                                reason = "NO_REASONING"
                            by_cond[c][reason] += 1
                            break
                else:
                    ja = j.get(f"{c}_judge_attack", {})
                    if ja.get("attack_attempted"):
                        reason = ja.get("compliance_reason", "NO_REASONING")
                        if reason not in reasons:
                            reason = "NO_REASONING"
                        by_cond[c][reason] += 1

        active = [c for c in conds if sum(by_cond[c].values()) > 0]
        if not active:
            ax.set_title(f"{m}\n(no compliant attacks)", fontsize=10)
            continue

        x = np.arange(len(active))
        bottom = np.zeros(len(active))
        for reason in reasons:
            vals = []
            for c in active:
                total = sum(by_cond[c].values())
                vals.append(by_cond[c].get(reason, 0) / total if total else 0)
            vals = np.array(vals)
            ax.bar(x, vals, 0.7, bottom=bottom, label=reason,
                   color=REASON_COLORS.get(reason, "#ccc"))
            for xi, v in enumerate(vals):
                if v > 0.05:
                    ax.text(xi, bottom[xi] + v / 2, f"{v:.0%}",
                            ha="center", va="center", fontsize=7,
                            color="white", fontweight="bold")
            bottom += vals

        ax.set_title(m, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(active, fontsize=7, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        if mi == 0:
            ax.set_ylabel("Proportion (compliant attacks)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(reasons), fontsize=9,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Compliance Reason by Condition (attacks only)",
                 fontsize=12, y=1.12)
    plt.tight_layout()
    out = save_dir / "aspi_compliance_reasons.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_agreement(all_data, save_dir):
    """Heatmap — env vs rule vs judge agreement for 6 attacked conditions."""
    judged_models = [m for m in all_data if all_data[m].get("has_judge")]
    if not judged_models:
        print("No judged data for agreement plot"); return

    conds = [
        "exec_tool", "exec_user", "exec_next_turn",
        "clarif_tool", "clarif_ask_user", "clarif_user",
    ]

    fig, axes = plt.subplots(
        len(judged_models), 1,
        figsize=(8, 2.5 * len(judged_models) + 1), squeeze=False,
    )

    im = None
    for mi, m in enumerate(judged_models):
        ax = axes[mi][0]
        results = all_data[m]["results"]
        pairs = {c: {"env_rule": [], "env_judge": [], "rule_judge": []} for c in conds}

        for res in results:
            if "judge" not in res:
                continue
            goal = res.get("attacker_goal", "")
            for c in conds:
                if c == "clarif_user":
                    cu = res.get("clarif_user", {})
                    if cu.get("skipped"):
                        continue
                    # env: security_max
                    env = bool(cu.get("security_max", False))
                    # rule: any operator attempted
                    rule = False
                    for op in ("HR", "GS", "CP"):
                        v = cu.get("variants", {}).get(op, {})
                        if v.get("skipped"):
                            continue
                        ar = attack_attempted_rule(
                            v.get("tool_calls", []), goal, v.get("model_output", "")
                        )
                        if ar["attack_attempted"]:
                            rule = True
                            break
                    # judge: any operator attempted
                    judge = False
                    for op in ("HR", "GS", "CP"):
                        ja = res["judge"].get(f"clarif_user_{op}_judge_attack", {})
                        if ja.get("attack_attempted"):
                            judge = True
                            break
                else:
                    cr = res.get(c, {})
                    if not cr or cr.get("skipped"):
                        continue
                    env = bool(cr.get("security", False))
                    rule = attack_attempted_rule(
                        cr.get("tool_calls", []), goal, cr.get("model_output", "")
                    )["attack_attempted"]
                    ja = res["judge"].get(f"{c}_judge_attack", {})
                    judge = bool(ja.get("attack_attempted", False))

                pairs[c]["env_rule"].append(env == rule)
                pairs[c]["env_judge"].append(env == judge)
                pairs[c]["rule_judge"].append(rule == judge)

        active_conds = [c for c in conds if pairs[c]["env_rule"]]
        col_labels = ["Env vs Rule", "Env vs Judge", "Rule vs Judge"]
        matrix = np.zeros((len(active_conds), 3))
        for ri, cond in enumerate(active_conds):
            for ci_idx, comp in enumerate(["env_rule", "env_judge", "rule_judge"]):
                vals = pairs[cond][comp]
                matrix[ri, ci_idx] = sum(vals) / len(vals) if vals else 0

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(len(active_conds)))
        ax.set_yticklabels(active_conds, fontsize=8)
        ax.set_title(m, fontsize=10)

        for ri in range(len(active_conds)):
            for ci_idx in range(3):
                v = matrix[ri, ci_idx]
                color = "white" if v < 0.7 else "black"
                ax.text(ci_idx, ri, f"{v:.0%}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    fig.suptitle("Metric Agreement — 6 attacked conditions", fontsize=12, y=0.98)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.08,
                     label="Agreement rate")
    fig.subplots_adjust(hspace=0.6, top=0.92)
    out = save_dir / "aspi_agreement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot dispatch
# ---------------------------------------------------------------------------

CORE_PLOTS = {
    # Headline plots — narrative-driving
    "asr_heatmap":          plot_asr_heatmap,           # 2×4 state × channel
    "state_effect_pairs":   plot_state_effect_pairs,    # A1, A2 side-by-side
    "channel_clarif_focus": plot_channel_clarif_focus,  # B1, B2 in clarif
    "channel_exec_focus":   plot_channel_exec_focus,    # C1, C2 in exec

    # Overview / comprehensive
    "asr_by_condition":     plot_asr_by_condition,      # all 6 attacked
    "utility_by_condition": plot_utility_by_condition,  # all 8
    "utility_cost":         plot_utility_cost,          # benign vs attacked × state
    "paired_deltas":        plot_paired_deltas,         # all 12 deltas
    "operator_breakdown":   plot_operator_breakdown,    # HR/GS/CP + max + mean

    # Per-suite
    "asr_by_suite":         plot_asr_by_suite,

    # Scatter
    "utility_vs_asr":       plot_utility_vs_asr,
}

JUDGE_PLOTS = {
    "judge_classifications":     plot_judge_classifications,
    "compliance_reasons":        plot_compliance_reasons,
    "agreement":                 plot_agreement,
    "near_miss":                 plot_near_miss,
    "attempted_by_condition":    plot_attempted_by_condition,
    # Reasoning-analysis plots (Approach B — visible pre-tool-call text)
    "reasoning_timing_stack":    plot_reasoning_timing_stack,
    "attack_ordering":           plot_attack_ordering,
    "attack_acknowledgement":    plot_attack_acknowledgement,
}

PLOT_FUNCTIONS = {**CORE_PLOTS, **JUDGE_PLOTS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ASPI Results Analyzer — 8-condition design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--detail", type=str)
    parser.add_argument(
        "--plot", type=str,
        choices=list(PLOT_FUNCTIONS.keys()) + ["all", "core", "judge"],
    )
    parser.add_argument("--add", nargs=2, action="append", metavar=("NAME", "PATH"))
    parser.add_argument("--plot-dir", type=str, default=str(PLOT_DIR))
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    result_files = dict(DEFAULT_RESULTS)
    if args.add:
        for name, path in args.add:
            result_files[name] = Path(path)
    if not result_files:
        result_files = discover_default_results()
        if result_files:
            print(f"Auto-discovered {len(result_files)} result file(s):")
            for name, path in result_files.items():
                print(f"  {name}  →  {path}")
            print()

    if args.results_dir != str(RESULTS_DIR):
        rd = Path(args.results_dir)
        result_files = {
            k: rd / v.name if v.parent == RESULTS_DIR else v
            for k, v in result_files.items()
        }

    save_dir = Path(args.plot_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    all_data = load_results(result_files)
    if not all_data:
        print()
        print("ERROR: No results loaded.")
        print("Pass --add NAME PATH or place judged JSONL(s) in results/judged/")
        import sys
        sys.exit(1)

    if args.list:
        cmd_list(all_data)
    if args.summary:
        cmd_summary(all_data)
    if args.detail:
        cmd_detail(all_data, args.detail)
    if args.plot:
        has_judge = _any_judge(all_data)
        if args.plot == "all":
            for name, fn in CORE_PLOTS.items():
                fn(all_data, save_dir)
            if has_judge:
                for name, fn in JUDGE_PLOTS.items():
                    fn(all_data, save_dir)
            else:
                print("  (Skipping judge plots — no judged data found)")
        elif args.plot == "core":
            for name, fn in CORE_PLOTS.items():
                fn(all_data, save_dir)
        elif args.plot == "judge":
            if has_judge:
                for name, fn in JUDGE_PLOTS.items():
                    fn(all_data, save_dir)
            else:
                print("Error: no judged data found.")
        elif args.plot in JUDGE_PLOTS and not has_judge:
            print(f"Error: '{args.plot}' requires judged data.")
        else:
            PLOT_FUNCTIONS[args.plot](all_data, save_dir)

    if not any([args.list, args.summary, args.detail, args.plot]):
        cmd_summary(all_data)


if __name__ == "__main__":
    main()
