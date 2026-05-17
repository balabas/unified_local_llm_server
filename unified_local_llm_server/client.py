from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .transport import OpenAICompatibleTransport
from .types import ChatChoice, ChatMessage, ChatUsage, LocalChatCompletion




@dataclass(slots=True)
class _ChatCompletionsNamespace:
    completions: "LocalChatCompletions"


class LocalChatCompletions:
    def __init__(self, transport: OpenAICompatibleTransport):
        self._transport = transport

    async def create(self, *, model: str, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> LocalChatCompletion:
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
    def __init__(self, transport: OpenAICompatibleTransport):
        self._transport = transport

    async def list(self) -> Any:
        return await self._transport.get_json("/models")


class LocalAsyncOpenAI:
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
