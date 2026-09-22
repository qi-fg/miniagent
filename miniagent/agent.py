"""The agent: a minimal ReAct loop.

Given a user question the agent repeatedly:

1. Sends ``(system_prompt + transcript)`` to the provider.
2. Parses the response. If it contains ``Final Answer:`` the loop ends.
3. Otherwise it extracts ``Action`` / ``Action Input``, executes the tool, and
   appends the ``Observation`` back to memory.
4. Repeats until a final answer or ``max_steps`` is hit.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from .memory import Memory
from .tools import ToolRegistry

DEFAULT_SYSTEM = """\
You are a reasoning agent that can call tools to solve problems.

Available tools:
{tools}

To respond, use exactly one of the following formats:

1) When you can answer directly:
Final Answer: <your answer>

2) When you need a tool:
Thought: <one sentence about what to do next>
Action: <tool_name>
Action Input: <input for the tool>

After an Action you will receive an Observation line. Reason over it and either
take another Action or return a Final Answer. Do not invent tool outputs."""


class Agent:
    def __init__(
        self,
        provider: Protocol,
        tools: ToolRegistry | None = None,
        memory: Memory | None = None,
        max_steps: int = 5,
        system_prompt: str | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.memory = memory or Memory()
        self.max_steps = max_steps
        self.system_prompt = system_prompt or DEFAULT_SYSTEM

    # -- public API -------------------------------------------------------- #
    def run(self, user_input: str) -> str:
        """Run the agent on a single user turn and return the final answer."""
        self.memory.add("user", user_input)
        system = self.system_prompt.format(tools=self.tools.format_for_prompt())

        for _ in range(self.max_steps):
            history = self.memory.get_context()
            text = self.provider.complete(system, history).strip()

            if "Final Answer:" in text:
                answer = text.split("Final Answer:", 1)[1].strip()
                self.memory.add("assistant", answer)
                return answer

            action, action_input = self._parse_action(text)
            if action is None:
                # No actionable tool call — treat the whole reply as the answer.
                self.memory.add("assistant", text)
                return text

            observation = self._execute(action, action_input)
            self.memory.add("assistant", text)
            self.memory.add("system", f"Observation: {observation}")

        return "Reached max_steps without a final answer."

    # -- internals --------------------------------------------------------- #
    def _parse_action(self, text: str) -> tuple[str | None, str | None]:
        m = re.search(
            r"Action:\s*([A-Za-z0-9_]+)\s*\n?Action Input:\s*(.+?)(?:\n|$)",
            text,
            re.DOTALL,
        )
        if not m:
            return None, None
        return m.group(1).strip(), m.group(2).strip()

    def _execute(self, action: str, action_input: str) -> str:
        tool = self.tools.get(action)
        if tool is None:
            return f"Error: tool '{action}' is not available."
        try:
            return tool.run(action_input)
        except Exception as exc:  # surface tool errors back to the model
            return f"Error: {type(exc).__name__}: {exc}"
