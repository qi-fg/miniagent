"""Smoke tests for the offline ReAct loop (no network, no API key)."""

import os
import sys
import unittest

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
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    return _eval(ast.parse(expr, mode="eval"))


class TestAgentLoop(unittest.TestCase):
    def _agent(self) -> Agent:
        return Agent(
            provider=MockProvider(),
            tools=ToolRegistry([calculator]),
            memory=Memory(),
            max_steps=4,
        )

    def test_calculator_tool_is_called(self):
        out = self._agent().run("Calculate 123 * 456")
        self.assertIn("56088", out)

    def test_unknown_tool_is_handled(self):
        # MockProvider only emits calculator/search; with no tools it must not crash.
        agent = Agent(MockProvider(), tools=ToolRegistry(), memory=Memory(), max_steps=3)
        out = agent.run("Calculate 9 * 9")
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) > 0)

    def test_memory_records_observation(self):
        agent = self._agent()
        agent.run("Calculate 2 + 3")
        roles = [m["role"] for m in agent.memory.get_context()]
        self.assertIn("system", roles)  # Observation line was appended


if __name__ == "__main__":
    unittest.main()
