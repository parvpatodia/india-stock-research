"""Provider-agnostic LLM access.

The analysts depend on this interface, never on a specific vendor SDK. The model string
picks the provider (LiteLLM routes it), so switching from NVIDIA NIM to a local Ollama
model to anything else is a config line, not a code change. No vendor lock-in, no mandatory
paid key.

Examples of LLM_MODEL:
  nvidia_nim/deepseek-ai/deepseek-v3.2   free hosted (build.nvidia.com, US cloud)
  ollama/llama3.1                        fully local, data never leaves the machine
  anthropic/claude-sonnet-4-6            if a paid key is ever preferred
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


def _clean(value: str | None) -> str | None:
    """Trim a config string; a blank/whitespace-only value becomes None (treated as absent).

    WHY (real money, Ask-tab reliability): env vars / .env / Streamlit TOML secrets very commonly
    carry a trailing space or newline (a quoted-with-space value, or a copy-paste). Left unstripped,
    a whitespace-only LLM_MODEL made `available` read True (then complete() errored on an empty
    model), and a trailing-space model string was passed straight to litellm, failing every lookup --
    silently breaking the Ask tab even though a valid model was intended. None-ing a blank value also
    lets a padded/empty API key fall back to the provider's own env instead of sending blank creds.
    """
    v = (value or "").strip()
    return v or None


class LLMClient(ABC):
    @property
    @abstractmethod
    def available(self) -> bool:
        """True if a model is configured and a call can be attempted."""

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1000,
                 json_mode: bool = False, json_schema: dict | None = None) -> str:
        """Return the model's text reply. json_mode forces valid JSON; json_schema forces a
        specific JSON structure (needed so small models fill the exact keys, not their own)."""

    @property
    def model_name(self) -> str:
        return "none"


class LiteLLMClient(LLMClient):
    """LiteLLM-backed client. Reads config from the constructor or these env vars:
    LLM_MODEL (litellm model string), LLM_API_KEY (optional), LLM_API_BASE (optional).
    Provider-specific env vars LiteLLM already understands (e.g. NVIDIA_NIM_API_KEY) also
    work without LLM_API_KEY. temperature=0 for grounded, repeatable extraction.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 api_base: str | None = None, temperature: float = 0.0):
        self.model = _clean(model or os.environ.get("LLM_MODEL"))
        self.api_key = _clean(api_key or os.environ.get("LLM_API_KEY"))
        self.api_base = _clean(api_base or os.environ.get("LLM_API_BASE"))
        self.temperature = temperature

    @property
    def available(self) -> bool:
        return bool(self.model)

    @property
    def model_name(self) -> str:
        return self.model or "none"

    def complete(self, system: str, user: str, max_tokens: int = 1000,
                 json_mode: bool = False, json_schema: dict | None = None) -> str:
        if not self.available:
            raise RuntimeError("no LLM configured (set LLM_MODEL)")
        # WHY: import lazily so the package loads (and unit tests with a fake client run)
        # without paying litellm's heavy import cost or requiring it to be installed.
        from litellm import completion
        kwargs: dict = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        if json_schema is not None:
            # WHY: force the EXACT key structure. With plain json_mode small models return valid
            # JSON but invent their own keys; a schema makes them fill the fields we asked for.
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "extraction", "schema": json_schema}}
        elif json_mode:
            # WHY: force valid-JSON output (prose otherwise breaks the parse). LiteLLM maps this
            # to Ollama's format=json / OpenAI json_object.
            kwargs["response_format"] = {"type": "json_object"}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        response = completion(**kwargs)
        return (response.choices[0].message.content or "").strip()
