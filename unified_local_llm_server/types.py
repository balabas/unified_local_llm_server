from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None


@dataclass(slots=True)
class ChatChoice:
    index: int
    message: ChatMessage
    finish_reason: str | None = None


@dataclass(slots=True)
class ChatUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True)
class LocalChatCompletion:
    id: str | None = None
    model: str | None = None
    choices: list[ChatChoice] = field(default_factory=list)
    usage: ChatUsage | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        if not self.choices:
            return ""
        return self.choices[0].message.content

    @property
    def reasoning(self) -> str | None:
        if not self.choices:
            return None
        return self.choices[0].message.reasoning

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content,
                        "tool_calls": c.message.tool_calls,
                    },
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": None if self.usage is None else {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "raw": self.raw,
        }
