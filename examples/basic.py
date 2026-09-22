"""Basic demo — runs fully offline using the MockProvider.

Run with:  python examples/basic.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniagent import Agent, Memory, MockProvider, ToolRegistry, tool


@tool
def calculator(expr: str) -> float:
    """Evaluate a basic arithmetic expression (+, -, *, /, ^, %, parentheses)."""
    import ast
    import operator

    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.BitXor: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("only numbers allowed")
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    return _eval(ast.parse(expr, mode="eval"))


@tool
def search(query: str) -> str:
    """Look up information on the web (stubbed in this demo)."""
    return f"(mock search results for: {query})"


def main() -> None:
    agent = Agent(
        provider=MockProvider(),
        tools=ToolRegistry([calculator, search]),
        memory=Memory(),
        max_steps=4,
    )

    question = "Calculate 123 * 456 for me"
    print(f"\n🗣  User: {question}\n")
    answer = agent.run(question)

    print("── transcript ──")
    for m in agent.memory.get_context():
        tag = {"user": "👤", "assistant": "🤖", "system": "🔧"}.get(m["role"], "·")
        print(f"{tag} {m['content']}")
    print("\n✅ Final answer:", answer)


if __name__ == "__main__":
    main()
