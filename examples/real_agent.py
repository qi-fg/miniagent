"""Talk to a real LLM with miniagent.

Run it and chat. Pick a provider, set the right env var, and go:

    export OPENAI_API_KEY=sk-...
    python examples/real_agent.py --provider openai

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/real_agent.py --provider anthropic --model claude-3-5-sonnet-latest

    # local model, no key needed
    python examples/real_agent.py --provider ollama --model llama3.1

    # fully offline, no network, no key
    python examples/real_agent.py --provider mock

NOTE: the OpenAI / Anthropic / Ollama providers talk to real HTTP APIs, so the
optional `requests` package is required for them:  pip install requests
(The Mock provider needs nothing and runs fully offline.)
"""

from __future__ import annotations

import argparse
import ast
import datetime
import operator
import sys
from pathlib import Path

# Allow running directly from a fresh clone:  python examples/real_agent.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniagent import (
    Agent,
    ToolRegistry,
    tool,
    MockProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
)

# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
# Map AST operator nodes to safe Python operators. `^` is treated as exponent
# (users intuitively type 2 ^ 10 for "two to the tenth").
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.BitXor: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str):
    """Evaluate a tiny arithmetic expression via the `ast` module (no eval())."""

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _BINOPS[type(node.op)](_ev(node.operand))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported expression: {expr!r}")

    return _ev(ast.parse(expr, mode="eval"))


@tool
def calculator(expr: str) -> float:
    """Evaluate a basic arithmetic expression, e.g. '2 ** 10 + 1' or '7 ^ 3'."""
    return _safe_eval(expr)


@tool
def current_time(_: str) -> str:
    """Return the current local date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def count_words(text: str) -> int:
    """Count the number of whitespace-separated words in the given text."""
    return len(text.split())


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "ollama": "llama3.1",
    "mock": "mock",
}


def build_provider(name: str, model: str):
    if name == "mock":
        return MockProvider()
    if name == "openai":
        return OpenAIProvider(model=model)
    if name == "anthropic":
        return AnthropicProvider(model=model)
    if name == "ollama":
        return OllamaProvider(model=model)
    raise SystemExit(f"unknown provider: {name!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat with a real LLM via miniagent.")
    p.add_argument(
        "--provider",
        default="openai",
        choices=list(_DEFAULT_MODELS),
    )
    p.add_argument("--model", default=None, help="model name (provider default if omitted)")
    p.add_argument("--max-steps", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model or _DEFAULT_MODELS[args.provider]
    provider = build_provider(args.provider, model)
    registry = ToolRegistry([calculator, current_time, count_words])
    agent = Agent(provider, tools=registry, max_steps=args.max_steps)

    print(f"miniagent ready · provider={args.provider} model={model}")
    print("Tools: calculator, current_time, count_words  (type 'exit' to quit)\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        try:
            print(f"agent> {agent.run(q)}\n")
        except Exception as exc:  # surface API errors without killing the loop
            print(f"[error] {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
