"""Stdlib-only OpenAI-compatible async client.

Responsibility
--------------
Provides a drop-in replacement for ``openai.AsyncOpenAI`` that works without
the ``openai`` package installed.  Used by ``pool.py`` to offer a
``pool.async_client`` property that external code can use like the real SDK.

The client is *not* used for the primary inference path — ``pool._call()``
talks to providers directly via ``OpenAICompatibleTransport``.  This module
exists so callers that already know the OpenAI SDK interface can work with
local providers without any code changes.

Layer position
--------------
Sits between ``transport.py`` (raw HTTP) and ``pool.py`` (provider logic).
Imports: ``transport``, ``types``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .transport import OpenAICompatibleTransport
from .types import ChatChoice, ChatMessage, ChatUsage, LocalChatCompletion


@dataclass(slots=True)
class _ChatCompletionsNamespace:
    """Mirrors ``openai.AsyncOpenAI().chat.completions`` attribute structure."""

    completions: "LocalChatCompletions"


class LocalChatCompletions:
    """Implements the ``chat.completions.create`` method of the OpenAI client interface.

    Non-streaming only — ``stream=True`` raises ``NotImplementedError`` because
    the primary streaming path goes through ``pool._stream_chat_sync`` instead.
    """

    def __init__(self, transport: OpenAICompatibleTransport):
        self._transport = transport

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> LocalChatCompletion:
        """Send a chat-completion request and return a typed response object.

        Parameters
        ----------
        model:
            Model identifier forwarded to the provider.
        messages:
            OpenAI-format message list.
        stream:
            Not supported; raises ``NotImplementedError`` if ``True``.
        **kwargs:
            Any extra OpenAI-compatible fields (``temperature``, ``tools``, …)
            are forwarded in the request payload; ``None`` values are dropped.
        """
        if stream:
            raise NotImplementedError("stream=True is not implemented in the local fallback client")

        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        result = await self._transport.post_json("/chat/completions", payload)
        data = result.json()
        if data is None:
            raise RuntimeError(
                f"Chat completions request failed (HTTP {result.status}): {result.text!r}"
            )

        choices = []
        for idx, choice in enumerate(data.get("choices", [])):
            message = choice.get("message") or {}
            content = message.get("content")
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            choices.append(
                ChatChoice(
                    index=choice.get("index", idx),
                    message=ChatMessage(
                        role=message.get("role", "assistant"),
                        content=content if isinstance(content, str) else "",
                        tool_calls=message.get("tool_calls") or [],
                        reasoning=reasoning if isinstance(reasoning, str) else None,
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            )

        usage = data.get("usage")
        usage_obj = None
        if isinstance(usage, dict):
            usage_obj = ChatUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )

        return LocalChatCompletion(
            id=data.get("id"),
            model=data.get("model", model),
            choices=choices,
            usage=usage_obj,
            raw=data,
        )


class _ModelsNamespace:
    """Mirrors the ``openai.AsyncOpenAI().models`` namespace."""

    def __init__(self, transport: OpenAICompatibleTransport):
        self._transport = transport

    async def list(self) -> Any:
        """List available models from the provider's ``/models`` endpoint."""
        return await self._transport.get_json("/models")


class LocalAsyncOpenAI:
    """Async OpenAI-compatible client backed by a local provider.

    Mirrors the ``openai.AsyncOpenAI`` attribute structure (``client.chat``,
    ``client.models``) so it can substitute for the real SDK in codebases that
    already use it.

    Parameters
    ----------
    base_url:
        Provider base URL including the ``/v1`` prefix, e.g.
        ``"http://127.0.0.1:11434/v1"``.
    api_key:
        Optional Bearer token; passed through to the transport layer.
    timeout:
        Request timeout in seconds (default 300).
    transport:
        Pre-constructed transport to reuse; created from ``base_url`` when
        omitted.  Injected by ``LLMProviderPool`` to share a single socket
        pool.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        transport: OpenAICompatibleTransport | None = None,
    ):
        self._transport = transport or OpenAICompatibleTransport(base_url, timeout=timeout, api_key=api_key)
        self.chat = _ChatCompletionsNamespace(completions=LocalChatCompletions(self._transport))
        self.models = _ModelsNamespace(self._transport)
