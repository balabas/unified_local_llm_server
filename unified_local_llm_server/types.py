"""Shared response data-classes for the OpenAI-compatible client.

Responsibility
--------------
Defines the typed in-memory representation of a chat-completion response.
These types mirror the OpenAI API object model closely enough that callers
can treat them interchangeably, but they are stdlib-only dataclasses with
no external dependency.

Layer position
--------------
Lowest layer — imported by ``client.py`` and surfaced through ``__init__.py``.
Has no imports from the rest of this package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChatMessage:
    """Content of a single chat message returned by the model.

    Attributes
    ----------
    role:
        Always ``"assistant"`` for model responses.
    content:
        The final text output.  Empty string when the model only returned
        tool calls or a reasoning block without a text answer.
    tool_calls:
        Zero or more tool-call requests emitted by the model.
        Each entry follows the OpenAI ``tool_calls`` item format.
    reasoning:
        Chain-of-thought / thinking text when the model exposes it
        (e.g. via a dedicated ``reasoning_content`` field or
        ``<think>…</think>`` blocks).  ``None`` when not present.
    """

    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None


@dataclass(slots=True)
class ChatChoice:
    """One completion alternative returned by the model.

    Most calls produce a single choice (``index=0``).  The ``n`` parameter
    for multiple completions is not used by this library.
    """

    index: int
    message: ChatMessage
    finish_reason: str | None = None


@dataclass(slots=True)
class ChatUsage:
    """Token-count metadata attached to a completion response.

    All fields are optional because not all providers include usage data
    and streaming responses may omit it entirely.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True)
class LocalChatCompletion:
    """Full chat-completion response, mirroring the OpenAI ``ChatCompletion`` shape.

    Attributes
    ----------
    id:
        Provider-assigned request ID (may be ``None`` for local servers that
        don't generate one).
    model:
        Model identifier echoed from the response; falls back to the
        requested model name.
    choices:
        Completion alternatives.  For this library always a single-element
        list.
    usage:
        Token counts; ``None`` when the provider omits usage data.
    raw:
        The original parsed JSON response dict, for callers that need
        provider-specific fields not captured above.
    """

    id: str | None = None
    model: str | None = None
    choices: list[ChatChoice] = field(default_factory=list)
    usage: ChatUsage | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """Shortcut: text content of the first choice."""
        if not self.choices:
            return ""
        return self.choices[0].message.content

    @property
    def reasoning(self) -> str | None:
        """Shortcut: reasoning text of the first choice, or ``None``."""
        if not self.choices:
            return None
        return self.choices[0].message.reasoning

    def model_dump(self) -> dict[str, Any]:
        """Serialize to an OpenAI-compatible dict, suitable for JSON encoding."""
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
