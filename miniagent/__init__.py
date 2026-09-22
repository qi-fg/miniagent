"""miniagent — a tiny, dependency-free LLM agent framework.

Build reasoning agents with tools + memory in a few hundred lines of pure
Python. No LangChain, no AutoGen, no bloat — just the ReAct loop and the
pieces you actually need.
"""

from .tools import Tool, ToolRegistry, tool
from .memory import Memory
from .providers import (
    Provider,
    MockProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
)
from .agent import Agent

__version__ = "0.1.0"
__all__ = [
    "Tool",
    "ToolRegistry",
    "tool",
    "Memory",
    "Provider",
    "MockProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "Agent",
]
