"""Show how to add your own tool to the agent.

A tool is any function decorated with ``@tool``. Its name and docstring are
automatically exposed to the model, so write a clear one-line docstring.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniagent import Agent, MockProvider, ToolRegistry, tool


@tool
def weather(city: str) -> str:
    """Return the current weather for a city, e.g. 'Zhengzhou'."""
    # Replace with a real API call (OpenWeather, etc.) in production.
    return f"{city}: 26C, partly cloudy"


@tool
def calculator(expr: str) -> float:
    """Evaluate a basic arithmetic expression."""
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
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    return _eval(ast.parse(expr, mode="eval"))


def main() -> None:
    agent = Agent(
        provider=MockProvider(),
        tools=ToolRegistry([weather, calculator]),
        max_steps=4,
    )
    print("Weather demo:", agent.run("What is the weather in Zhengzhou?"))
    print("Math demo:   ", agent.run("Calculate 2 ^ 10"))


if __name__ == "__main__":
    main()
