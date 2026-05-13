"""ASPI — Ambiguous State Prompt Injection benchmark.

A 6-condition benchmark extending AgentDojo v1.2.2 to test whether
agents in a clarification-seeking state are more vulnerable to prompt
injection than agents in an execution state, and whether the user-message
channel constitutes a distinct attack surface.

Conditions:
    C1        — execution state, clean env (baseline utility)
    C2_tool   — execution state, tool-output injection
    C2_user   — execution state, user states attacker goal (control)
    C3        — clarification state, benign user answer
    C4_tool   — clarification state, tool-output injection
    C4_user   — clarification state, injected user answer

Entry points live in `scripts/` at the repo root. Import paths:
    from aspi.eval.metrics import aggregate
    from aspi.common.litellm import LiteLlmClient
    from aspi.prompts.dataset import AMBIGUITY_SYSTEM
"""

__version__ = "0.1.0"
