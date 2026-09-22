"""Command-line interface.

Usage::

    python -m miniagent.cli "Calculate 123 * 456" --provider mock
    python -m miniagent.cli --provider ollama --model llama3.1 --interactive
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .agent import Agent  # noqa: E402
from .memory import Memory  # noqa: E402
from .providers import (  # noqa: E402
    AnthropicProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
)
from .tools import ToolRegistry, tool  # noqa: E402


@tool
def calculator(expr: str) -> float:
    """Evaluate a basic arithmetic expression (+, -, *, /, ^, %, parentheses)."""
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.pow,  # treat "^" as exponentiation (user intent)
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("only numbers are allowed")
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    return _eval(ast.parse(expr, mode="eval"))


def build_provider(name: str, model: str | None) -> object:
    if name == "mock":
        return MockProvider()
    if name == "openai":
        return OpenAIProvider(model=model or "gpt-4o-mini")
    if name == "anthropic":
        return AnthropicProvider(model=model or "claude-3-5-sonnet-latest")
    if name == "ollama":
        return OllamaProvider(model=model or "llama3.1")
    raise SystemExit(f"unknown provider: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="miniagent CLI")
    ap.add_argument("prompt", nargs="*", help="user prompt (omit with --interactive)")
    ap.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic", "ollama"])
    ap.add_argument("--model", default=None, help="model name for the chosen provider")
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--interactive", action="store_true", help="chat loop")
    args = ap.parse_args()

    provider = build_provider(args.provider, args.model)
    registry = ToolRegistry([calculator])
    agent = Agent(provider, tools=registry, max_steps=args.max_steps)

    if args.interactive:
        print("miniagent interactive (type 'exit' to quit)")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                print("agent>", agent.run(q))
        return

    prompt = " ".join(args.prompt)
    if not prompt:
        ap.error("provide a prompt or use --interactive")
    print(agent.run(prompt))


if __name__ == "__main__":
    main()
