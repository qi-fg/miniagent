"""LLM providers.

The framework is provider-agnostic. A provider only needs to implement
``complete(system, history) -> str``. We ship four:

* :class:`MockProvider` — a deterministic offline stub (no API key, no network).
  Perfect for tests and for the README demo.
* :class:`OpenAIProvider` / :class:`AnthropicProvider` / :class:`OllamaProvider`
  — real backends that talk to the respective HTTP APIs via ``requests``
  (imported lazily, so the core stays stdlib-only).
"""

from __future__ import annotations

import ast
import operator
import os
import re
from typing import Any, Protocol


class Provider(Protocol):
    def complete(self, system: str, history: list[dict[str, Any]]) -> str: ...


# --------------------------------------------------------------------------- #
# Offline stub
# --------------------------------------------------------------------------- #
class MockProvider:
    """A fake LLM that drives the ReAct loop without any network call.

    It pattern-matches the latest user turn and emits either a tool action or a
    final answer. After a tool observation appears in the history it returns a
    Final Answer. Used so the framework can be demonstrated and tested fully
    offline. Swap in a real provider for production use.
    """

    def complete(self, system: str, history: list[dict[str, Any]]) -> str:
        has_obs = any(
            m["role"] == "system" and m["content"].startswith("Observation:")
            for m in history
        )
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"), ""
        )
        low = last_user.lower()

        if has_obs:
            obs = [
                m["content"]
                for m in history
                if m["role"] == "system" and m["content"].startswith("Observation:")
            ][-1]
            val = obs.replace("Observation:", "", 1).strip()
            return f"Final Answer: The result is {val}."

        if ("calculate" in low or "compute" in low or "^" in low) and re.search(
            r"\d", low
        ):
            expr = re.search(r"[\d\.\s\+\-\*/\^\(\)%]+", last_user)
            e = expr.group(0).strip() if expr else "1+1"
            return (
                "Thought: I need to evaluate the arithmetic expression.\n"
                f"Action: calculator\nAction Input: {e}"
            )

        if "search" in low or "look up" in low:
            return (
                "Thought: I should search the web for this.\n"
                f"Action: search\nAction Input: {last_user}"
            )

        return f"Final Answer: {last_user}"


# --------------------------------------------------------------------------- #
# Real HTTP providers (lazy import of requests)
# --------------------------------------------------------------------------- #
def _post(url: str, headers: dict, payload: dict, timeout: int = 60) -> dict:
    try:
        import requests  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'requests' package is required for HTTP providers. "
            "Install it with: pip install requests"
        ) from exc
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


class OpenAIProvider:
    """OpenAI Chat Completions API (gpt-4o-mini, gpt-4o, o3, ...)."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY or pass api_key=.")
        self.model = model

    def complete(self, system: str, history: list[dict[str, Any]]) -> str:
        messages = [{"role": "system", "content": system}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history]
        data = _post(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            {"model": self.model, "messages": messages, "temperature": 0},
        )
        return data["choices"][0]["message"]["content"]


class AnthropicProvider:
    """Anthropic Messages API (claude-3-5-sonnet, claude-opus-4, ...)."""

    def __init__(
        self, api_key: str | None = None, model: str = "claude-3-5-sonnet-latest"
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Set ANTHROPIC_API_KEY or pass api_key=.")
        self.model = model

    def complete(self, system: str, history: list[dict[str, Any]]) -> str:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in history
            if m["role"] in ("user", "assistant")
        ]
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            {"model": self.model, "system": system, "messages": messages, "max_tokens": 1024},
        )
        # Anthropic returns content blocks; join text blocks.
        return "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )


class OllamaProvider:
    """Local models via Ollama (http://localhost:11434). No API key needed."""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, system: str, history: list[dict[str, Any]]) -> str:
        messages = [{"role": "system", "content": system}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history]
        data = _post(
            f"{self.base_url}/api/chat",
            {"Content-Type": "application/json"},
            {"model": self.model, "messages": messages, "stream": False},
        )
        return data.get("message", {}).get("content", "")
