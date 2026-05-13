"""LiteLLM-backed pipeline element for AgentDojo.

Drop-in replacement for AgentDojo's ``OpenAILLM``. Calls
``litellm.completion()`` directly so reasoning kwargs
(``reasoning_effort``, ``thinking``, ``extra_body.thinkingConfig``,
``extra_body.enable_thinking``) pass through as documented, without
the monkey-patching gymnastics that would be needed when going
through an ``openai.OpenAI`` client pointed at a LiteLLM proxy.

Everything else in the AgentDojo pipeline (system prompt, tool
registration, runtime, scoring) is untouched — this element emits
the same ``ChatAssistantMessage`` shape that ``OpenAILLM`` emits,
plus a top-level ``reasoning_content`` field (plus
``provider_specific_fields`` / ``reasoning_items`` / ``thinking_blocks``
when present) so the eval's ``_serialize_messages`` captures them
uniformly.

Design parity with ``agentdojo.agent_pipeline.llms.openai_llm``:
  - message-format conversion mirrors ``_message_to_openai``
  - tool-format conversion mirrors ``_function_to_openai``
  - retry policy matches ``chat_completion_request`` (3 attempts,
    exponential backoff, do not retry BadRequest / Unprocessable)
  - the returned assistant message is a ``ChatAssistantMessage`` with
    ``role``, ``content`` (list of MessageContentBlock or None), and
    ``tool_calls`` (list of FunctionCall or None)
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

import litellm
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import (
    EmptyEnv,
    Env,
    Function,
    FunctionCall,
    FunctionsRuntime,
)
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    MessageContentBlock,
    text_content_block_from_string,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model-string coercion for LiteLLM proxy routing
# ---------------------------------------------------------------------------
#
# LiteLLM SDK uses the model's provider prefix to decide which backend
# to route through. For native calls (no api_base), ``anthropic/claude-
# sonnet-4-6`` correctly routes via Anthropic's SDK. But when we set
# ``api_base`` to a LiteLLM proxy, the SDK still honours the prefix and
# tries to hit api.anthropic.com with the proxy URL overriding its
# base path — which returns 403 because the proxy isn't anthropic.com.
#
# To route through the proxy's OpenAI-compat endpoint, the effective
# model string sent by the SDK must carry the ``openai/`` prefix, and
# the native-provider prefix must be stripped so the proxy's router
# matches its config.
#
# We apply this coercion ONLY at call time, inside the element. The
# model string the element is *constructed* with, and the string
# persisted to logs / summary.json / the dispatch table, remains the
# original LiteLLM-compatible name (e.g. ``anthropic/claude-sonnet-
# 4-6``). This is a proxy-routing detail, not a provider identity.

_NATIVE_PROVIDER_PREFIXES = (
    "anthropic/", "gemini/", "bedrock/", "fireworks_ai/",
    "vertex_ai/", "moonshot/", "deepseek/", "groq/",
    "mistral/", "xai/", "openrouter/",
)


def _coerce_model_for_proxy(model: str) -> str:
    """Rewrite a model string for LiteLLM-proxy routing.

    Treats ``openai/`` as a *transport* prefix (LiteLLM SDK telling
    itself "talk OpenAI-compatible to the proxy"), and otherwise
    PRESERVES the upstream provider group as the proxy expects it.
    The proxy's catalog uses the full provider-prefixed handles
    (``anthropic/claude-opus-4-7``, ``fireworks_ai/kimi-k2p5``,
    ``bedrock/eu.deepseek.v3.2``, ...), so stripping the prefix
    produced names the proxy did not serve — that path returned 200
    with empty / malformed content for some routes (notably
    ``openai/o4-mini`` after we'd stripped a different prefix).

    Rules:
    - Pass through ``openai/responses/...`` unchanged (Responses API
      bridge already has both transport and route in the right shape).
    - Pass through ``openai/...`` unchanged (already OpenAI-routed).
    - For other provider prefixes that the proxy serves verbatim
      (``anthropic/``, ``fireworks_ai/``, ``bedrock/``,
      ``mistral/``, ``xai/``, ``openrouter/``), prepend ``openai/`` as
      the SDK transport prefix WITHOUT stripping the provider group.
    - For ``gemini/``, keep the historical strip behavior — the
      proxy was confirmed to serve gemini handles under the
      stripped form.
    - Bare names get ``openai/`` prepended.
    """
    if not model:
        return model

    # Already routed via OpenAI Responses bridge — leave alone.
    if model.startswith("openai/responses/"):
        return model
    # Already openai/<...> — leave alone.
    if model.startswith("openai/"):
        return model

    # Providers the proxy serves with their full prefix preserved.
    PRESERVE = ("anthropic/", "fireworks_ai/", "bedrock/",
                "mistral/", "xai/", "openrouter/")
    for prefix in PRESERVE:
        if model.startswith(prefix):
            return f"openai/{model}"

    # Gemini: historical strip-and-prepend works on the proxy.
    if model.startswith("gemini/"):
        return f"openai/{model.split('/', 1)[1]}"

    # Bare name → prepend openai transport prefix.
    if "/" not in model:
        return f"openai/{model}"

    # Unrecognised slashed handle: leave as-is.
    return model


# ---------------------------------------------------------------------------
# Message / tool conversion — AgentDojo ChatMessage -> OpenAI-format dict
# ---------------------------------------------------------------------------

def _content_blocks_to_openai_content(
    message: ChatMessage,
) -> list[dict] | None:
    """Mirror of agentdojo.openai_llm._content_blocks_to_openai_content_blocks
    but emitting plain dicts (litellm accepts dicts, no need for the
    openai pydantic types).
    """
    content = message.get("content")
    if content is None:
        return None
    return [
        {"type": "text", "text": (el.get("content") or "")}
        for el in content
    ]


def _tool_call_to_openai(tc: FunctionCall) -> dict:
    if tc.id is None:
        raise ValueError("tool_call.id is required for OpenAI-format tool calls")
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function,
            "arguments": json.dumps(tc.args),
        },
    }


def _message_to_openai(message: ChatMessage) -> dict:
    """Convert a single AgentDojo ChatMessage to a LiteLLM/OpenAI
    message dict. Handles the four roles agentdojo emits: system, user,
    assistant (with or without tool_calls), and tool (a tool result).
    """
    role = message["role"]

    if role == "system":
        return {
            "role": "system",
            "content": _content_blocks_to_openai_content(message),
        }

    if role == "user":
        return {
            "role": "user",
            "content": _content_blocks_to_openai_content(message),
        }

    if role == "assistant":
        out: dict[str, Any] = {
            "role": "assistant",
            "content": _content_blocks_to_openai_content(message),
        }
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            out["tool_calls"] = [_tool_call_to_openai(tc) for tc in tool_calls]
        return out

    if role == "tool":
        tool_call_id = message.get("tool_call_id")
        if tool_call_id is None:
            raise ValueError(
                "tool_call_id is required on tool-role messages for OpenAI format",
            )
        tc = message.get("tool_call")
        # Tool-result content: prefer explicit error string, else join
        # text content blocks. OpenAI expects a string here, not a list.
        if message.get("error"):
            content = str(message["error"])
        else:
            parts = _content_blocks_to_openai_content(message) or []
            content = "\n".join(p.get("text", "") for p in parts)
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tc.function if tc is not None else "unknown",
            "content": content,
        }

    raise ValueError(f"Unknown message role: {role!r}")


def _function_to_openai(f: Function) -> dict:
    """Mirror of agentdojo.openai_llm._function_to_openai."""
    return {
        "type": "function",
        "function": {
            "name": f.name,
            "description": f.description,
            "parameters": f.parameters.model_json_schema(),
        },
    }


# ---------------------------------------------------------------------------
# Response parsing — litellm.completion response -> ChatAssistantMessage
# ---------------------------------------------------------------------------

def _psf_to_dict(psf: Any) -> dict:
    """Coerce a provider_specific_fields value to a plain dict.

    Different LiteLLM versions surface this as either a dict, a
    pydantic model, or an object with __dict__ — handle all three.
    """
    if psf is None:
        return {}
    if isinstance(psf, dict):
        return psf
    if hasattr(psf, "model_dump"):
        try:
            return psf.model_dump()
        except Exception:
            pass
    try:
        return dict(vars(psf))
    except Exception:
        return {}


def _response_to_assistant_message(response: Any) -> ChatAssistantMessage:
    """Parse a litellm.completion() response into AgentDojo's
    ChatAssistantMessage shape, with reasoning fields attached as
    top-level keys so the eval's ``_serialize_messages`` captures them.

    ChatAssistantMessage is a TypedDict at import time but a plain
    ``dict`` at runtime — extra keys are preserved through the
    pipeline and serialized to the output JSONL.
    """
    choice = response.choices[0]
    msg = choice.message

    # ── Content: list[MessageContentBlock] | None ────────────────────
    content_text = getattr(msg, "content", None) or ""
    content_blocks: list[MessageContentBlock] | None
    if content_text.strip():
        content_blocks = [text_content_block_from_string(content_text)]
    else:
        content_blocks = None

    # ── Tool calls: list[FunctionCall] | None ────────────────────────
    tool_calls_out: list[FunctionCall] | None
    raw_tool_calls = getattr(msg, "tool_calls", None)
    if raw_tool_calls:
        tool_calls_out = []
        for tc in raw_tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except Exception as e:
                logger.debug(f"tool_call arg parse failed: {e}")
                args = {}
            tool_calls_out.append(FunctionCall(
                function=tc.function.name,
                args=args,
                id=tc.id,
            ))
    else:
        tool_calls_out = None

    out: dict[str, Any] = {
        "role": "assistant",
        "content": content_blocks,
        "tool_calls": tool_calls_out,
    }

    # ── Reasoning capture — multiple provider shapes ────────────────
    # All reasoning-related fields are attached as TOP-LEVEL keys on the
    # assistant message dict. Downstream (runner.py::_serialize_messages,
    # eval/judge.py::_format_conversation, scripts/audit_results.py) all
    # read from top-level, so in-memory and serialised shapes agree.
    # sources tracks which extraction path(s) fired — ends up in
    # reasoning_source at the end of this function.
    sources: list[str] = []

    # (1) Top-level reasoning_content — Anthropic, Gemini, DeepSeek,
    #     Qwen3, Groq, Mistral Magistral, Vertex, OpenAI Responses.
    rc = getattr(msg, "reasoning_content", None) or ""
    if isinstance(rc, str) and rc.strip():
        out["reasoning_content"] = rc
        sources.append("message_reasoning_content")

    # (2) Anthropic-style thinking_blocks — redundant with reasoning_content
    #     but carries the signature token required for multi-turn thinking.
    thinking_blocks = getattr(msg, "thinking_blocks", None)
    if thinking_blocks:
        try:
            out["thinking_blocks"] = [
                dict(tb) if isinstance(tb, dict) else
                (tb.model_dump() if hasattr(tb, "model_dump") else dict(vars(tb)))
                for tb in thinking_blocks
            ]
            if "reasoning_content" not in out:
                texts = [
                    b.get("thinking", "") for b in out["thinking_blocks"]
                    if isinstance(b, dict) and b.get("thinking")
                ]
                if texts:
                    out["reasoning_content"] = "\n".join(texts)
            sources.append("content_block_thinking")
        except Exception as e:
            logger.debug(f"thinking_blocks parse failed: {e}")

    # (3) OpenAI Responses-bridge reasoning_items[].summary[].text.
    #     Only populated when the call requested summary: detailed/auto
    #     (see _DEFAULT_REASONING_KWARGS_BY_FAMILY in runner.py).
    items = getattr(msg, "reasoning_items", None)
    if items:
        try:
            out["reasoning_items"] = [
                dict(it) if isinstance(it, dict) else
                (it.model_dump() if hasattr(it, "model_dump") else dict(vars(it)))
                for it in items
            ]
        except Exception as e:
            logger.debug(f"reasoning_items parse failed: {e}")
            out["reasoning_items"] = []

        summary_texts: list[str] = []
        for item in out.get("reasoning_items", []):
            summary = item.get("summary") or []
            if not isinstance(summary, list):
                continue
            for s in summary:
                if isinstance(s, dict):
                    t = s.get("text") or s.get("content") or ""
                    if isinstance(t, str) and t.strip():
                        summary_texts.append(t)
        if summary_texts:
            prev = out.get("reasoning_content", "")
            extra = "\n\n".join(summary_texts)
            out["reasoning_content"] = (
                f"{prev}\n\n{extra}".strip() if prev else extra
            )
            sources.append("reasoning_items_summary")

    # (4) provider_specific_fields — e.g. Gemini thought_signature (opaque
    #     replay token, required for multi-turn tool use with thinking on).
    #     We keep the full psf dict on the message (for future forwarding
    #     to subsequent calls) AND hoist thought_signature to top-level
    #     for easy audit and symmetry with other reasoning fields.
    psf = getattr(msg, "provider_specific_fields", None)
    if psf is not None:
        psf_dict = _psf_to_dict(psf)
        if psf_dict:
            out["provider_specific_fields"] = psf_dict
            ts = psf_dict.get("thought_signature")
            if ts:
                # Top-level mirror — preserved in full. Signatures are
                # opaque replay tokens required by Gemini for multi-turn
                # tool use with thinking on; truncating them would make
                # the JSONL unusable for replay. No length cap.
                out["thought_signature"] = (
                    [str(t) for t in ts] if isinstance(ts, list)
                    else str(ts)
                )
                sources.append("thought_signature")

    # Final reasoning_source field — records which extraction path(s)
    # produced content. Single string if one source, list if multiple.
    # Downstream audit script reports per-source counts from this field.
    if sources:
        out["reasoning_source"] = sources[0] if len(sources) == 1 else sources

    return out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Pipeline element
# ---------------------------------------------------------------------------

_NON_RETRIABLE = (
    litellm.BadRequestError,
    litellm.UnprocessableEntityError,
)


class LiteLLMPipelineElement(BasePipelineElement):
    """AgentDojo pipeline element that calls ``litellm.completion()``.

    Parameters
    ----------
    model : str
        Model string passed to ``litellm.completion``. Provider prefix
        required (e.g. ``anthropic/claude-sonnet-4-6``, ``gemini/gemini-
        3-flash-preview``, ``openai/responses/o3``). The ``openai/
        responses/`` prefix forces LiteLLM onto the Responses API
        bridge (required for o-series to actually reason).
    reasoning_kwargs : dict | None
        Additional kwargs merged into every ``litellm.completion``
        call. Typical shapes:
          - OpenAI Responses: ``{"extra_body": {"reasoning":
            {"effort": "medium", "summary": "detailed"}}}``
          - Anthropic thinking: ``{"thinking": {"type": "enabled",
            "budget_tokens": 10000}}``
          - Gemini thinking: ``{"extra_body": {"thinkingConfig":
            {"includeThoughts": True, "thinkingBudget": 10000}}}``
          - Qwen3 thinking: ``{"extra_body": {"enable_thinking": True}}``
        Resolved by ``runner._resolve_reasoning_kwargs`` based on the
        ``--reasoning`` flag.
    temperature : float
        Forwarded to every call. Default 0.0.
    api_base : str | None
        If set, overrides the default endpoint for the provider —
        useful when routing through a LiteLLM proxy.
    api_key : str | None
        If set, overrides the default key for the provider.
    extra_static_kwargs : dict | None
        Additional kwargs that should land on every call regardless of
        reasoning mode (e.g. Gemini safety_settings). Merged before
        reasoning_kwargs so reasoning_kwargs can override.
    """

    def __init__(
        self,
        model: str,
        reasoning_kwargs: dict[str, Any] | None = None,
        temperature: float = 0.0,
        api_base: str | None = None,
        api_key: str | None = None,
        extra_static_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.reasoning_kwargs = dict(reasoning_kwargs or {})
        self.temperature = temperature
        self.api_base = api_base
        self.api_key = api_key
        self.extra_static_kwargs = dict(extra_static_kwargs or {})

    # Retry wrapper matching AgentDojo's chat_completion_request policy.
    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        reraise=True,
        retry=retry_if_not_exception_type(_NON_RETRIABLE),
    )
    def _call(self, **kwargs: Any) -> Any:
        return litellm.completion(**kwargs)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        openai_messages = [_message_to_openai(m) for m in messages]
        openai_tools = [_function_to_openai(f) for f in runtime.functions.values()]

        # Coerce only at call time — self.model (and anything logged or
        # persisted) stays as the original LiteLLM-compatible name.
        effective_model = (
            _coerce_model_for_proxy(self.model) if self.api_base else self.model
        )

        # Temperature override when reasoning is enabled.
        # Anthropic's extended-thinking API rejects the call with:
        #     "temperature may only be set to 1 when thinking is enabled"
        # if temp != 1.0. OpenAI reasoning models ignore temperature; Gemini
        # / Qwen3 / DeepSeek accept temp=1.0 with reasoning on. So: force 1.0
        # whenever reasoning_kwargs are present — safe universally, required
        # for Anthropic.
        effective_temperature = self.temperature
        if self.reasoning_kwargs:
            effective_temperature = 1.0

        call_kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": openai_messages,
            "temperature": effective_temperature,
        }
        if openai_tools:
            call_kwargs["tools"] = openai_tools
            call_kwargs["tool_choice"] = "auto"
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.api_key:
            call_kwargs["api_key"] = self.api_key

        # Merge static kwargs (e.g. Gemini safety_settings), then
        # reasoning_kwargs on top so reasoning can override static if
        # needed. extra_body is deep-merged rather than overwritten.
        def _merge(dst: dict, src: dict) -> None:
            for k, v in src.items():
                if (k == "extra_body" and isinstance(v, dict)
                        and isinstance(dst.get("extra_body"), dict)):
                    dst["extra_body"] = {**dst["extra_body"], **v}
                else:
                    dst[k] = v

        _merge(call_kwargs, self.extra_static_kwargs)
        _merge(call_kwargs, self.reasoning_kwargs)

        response = self._call(**call_kwargs)
        output = _response_to_assistant_message(response)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args
