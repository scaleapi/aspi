"""ASPI evaluation: agent runner, metrics, judge, simulator, analyzer.

Submodules:
    runner    — full evaluation driver (replays C4_tool, parallel groups)
    metrics   — env-state and rule-based aggregation; aggregate() main entry
    judge     — LLM-as-judge for richer attack/utility classification
    simulator — LLM user simulator for C3/C4_user Turn 3
    analyze   — tables + plots from saved JSONL results
"""
