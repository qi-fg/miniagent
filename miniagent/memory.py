"""Conversation memory.

Stores the running transcript as a list of ``(role, content)`` turns. Roles are
``user``, ``assistant`` (model thoughts/actions) and ``system`` (tool
observations). The whole transcript is fed back to the model each step so it can
reason over what it has already done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


class Memory:
    def __init__(self) -> None:
        self.store: list[Message] = []

    def add(self, role: str, content: str) -> None:
        self.store.append(Message(role=role, content=content))

    def get_context(self) -> list[dict[str, Any]]:
        """Return the transcript as plain dicts (provider-friendly)."""
        return [{"role": m.role, "content": m.content} for m in self.store]

    def clear(self) -> None:
        self.store.clear()

    def __len__(self) -> int:
        return len(self.store)
