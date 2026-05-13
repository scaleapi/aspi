"""LiteLLM proxy client and env-var helpers.

This module is the single source of truth for LiteLLM configuration
across ASPI. All modules that need to hit the proxy (dataset construction,
evaluation, judge, simulator, validation) import from here.

Environment variables (loaded from .env or the shell):
    LITE_LLM_KEY  — API key for the LiteLLM proxy
    LITE_LLM_URL  — Base URL of the proxy; /v1 is appended if missing
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Pick up .env from the repo root (or any ancestor of the current dir).
load_dotenv()


def get_litellm_base_url(url: str | None = None) -> str:
    """Normalize a LiteLLM proxy base URL.

    Reads LITE_LLM_URL from environment if no URL is passed. Ensures
    the URL has the /v1 suffix so callers can use it directly as the
    OpenAI ``base_url`` argument. Raises ValueError if neither
    argument nor environment variable is set.
    """
    if url is None:
        url = os.environ.get("LITE_LLM_URL", "")
    if not url:
        raise ValueError(
            "LITE_LLM_URL is not set. Add it to your .env file or pass "
            "it explicitly."
        )
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def get_litellm_api_key() -> str:
    """Read LITE_LLM_KEY from environment and raise if missing."""
    key = os.environ.get("LITE_LLM_KEY", "")
    if not key:
        raise ValueError(
            "LITE_LLM_KEY is not set. Add it to your .env file or "
            "export it in your shell."
        )
    return key


class LiteLlmClient:
    """Thin wrapper around LiteLLM's OpenAI-compatible endpoint.

    Routes chat completions through the configured LiteLLM proxy. Used
    by dataset construction (Stage 1 / Stage 2 LLM calls) and by the
    clarification validation utility.
    """

    def __init__(self, model: str):
        self.client = OpenAI(
            api_key=get_litellm_api_key(),
            base_url=get_litellm_base_url(),
        )
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(10))
    def completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def get_completion(self, messages: list[dict], **kwargs) -> object:
        """Send messages to the LLM and return the raw response object.

        Args:
            messages: list of {"role": ..., "content": ...} dicts.
            **kwargs: passed through to the API (model, max_tokens, etc.)

        Returns:
            Raw response object from the API.

        Native-reasoning passthrough:
            If the ASPI_REASONING_KWARGS env var is set to a JSON dict,
            those kwargs are merged into the request. Enables reasoning
            per provider (OpenAI: reasoning={"effort": ...}; Anthropic:
            thinking={...}; Gemini: extra_body.thinkingConfig; Qwen3:
            extra_body.enable_thinking). When unset or "{}", no
            reasoning is requested (default, current behaviour).
        """
        params = dict(kwargs)

        # Native-reasoning passthrough (opt-in via env var)
        reasoning_kwargs_env = os.environ.get("ASPI_REASONING_KWARGS", "")
        if reasoning_kwargs_env.strip() and reasoning_kwargs_env.strip() != "{}":
            try:
                import json as _json
                reasoning_kwargs = _json.loads(reasoning_kwargs_env)
            except Exception:
                reasoning_kwargs = {}
            # Deep-merge extra_body; shallow-set others
            for k, v in reasoning_kwargs.items():
                if k == "extra_body" and isinstance(v, dict):
                    params["extra_body"] = {**params.get("extra_body", {}), **v}
                else:
                    params[k] = v

        # Gemini-specific: disable safety filters for security research
        if "gemini" in self.model.lower():
            params["extra_body"] = params.get("extra_body", {})
            params["extra_body"]["safety_settings"] = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

        # Bedrock-specific: allow OpenAI-style tool args
        if "bedrock" in self.model.lower():
            params["extra_body"] = params.get("extra_body", {})
            params["extra_body"]["allowed_openai_params"] = ["tools", "tool_choice"]

        return self.completion_with_backoff(messages=messages, **params)

    def get_text(self, messages: list[dict], **kwargs) -> str:
        """Convenience method: call get_completion and extract text content.

        Handles response format differences between OpenAI and Gemini.
        """
        response = self.get_completion(messages, **kwargs)

        # Standard OpenAI format
        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError):
            pass

        # Gemini/other format: response.output[...].content[...].text
        try:
            return response.output[1].content[0].text
        except (AttributeError, IndexError):
            pass

        try:
            return response.output[0].content[0].text
        except (AttributeError, IndexError):
            pass

        raise ValueError(f"Could not extract text from response: {response}")
