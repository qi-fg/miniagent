"""Tool definition and registry.

A *tool* is just a Python function. Decorate it with ``@tool`` to turn it into a
:class:`Tool` that the agent can discover and call. The function name becomes the
tool name and its docstring becomes the tool description shown to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    """A callable tool the agent can use."""

    name: str
    description: str
    func: Callable[..., Any]

    def run(self, arg: str) -> str:
        """Execute the tool. ``arg`` is the raw "Action Input" string."""
        return str(self.func(arg))

    def to_prompt(self) -> str:
        return f"- {self.name}: {self.description}"


def tool(func: Callable[..., Any]) -> Tool:
    """Decorator: turn a function into a :class:`Tool`.

    The function name is used as the tool name and its docstring (first line)
    as the description. The decorated object is a :class:`Tool`, so register it
    with ``ToolRegistry([my_tool])``.
    """
    return Tool(name=func.__name__, description=(func.__doc__ or "").strip(), func=func)


class ToolRegistry:
    """Holds the set of tools an agent is allowed to call."""

    def __init__(self, tools: list[Tool] | None = None):
        self.tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> None:
        self.tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def format_for_prompt(self) -> str:
        if not self.tools:
            return "(no tools available)"
        return "\n".join(t.to_prompt() for t in self.tools.values())
