from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


ToolHandler = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass(slots=True)
class ToolExecution:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: str


class ToolPipeline:
    """Owns tool-call normalization, execution, and tool-message construction."""

    def normalize_tool_calls(self, tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for idx, call in enumerate(tool_calls or []):
            function = call.get("function") or {}
            name = function.get("name") or call.get("name")
            if not name:
                continue
            arguments = function.get("arguments", call.get("arguments", {}))
            normalized.append(
                {
                    "id": str(call.get("id") or f"tool_call_{idx}"),
                    "type": call.get("type", "function"),
                    "function": {
                        "name": str(name),
                        "arguments": self._parse_arguments(arguments),
                    },
                }
            )
        return normalized

    async def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        tool_registry: dict[str, ToolHandler],
    ) -> list[ToolExecution]:
        executions: list[ToolExecution] = []
        for call in self.normalize_tool_calls(tool_calls):
            function = call["function"]
            name = function["name"]
            arguments = function["arguments"]
            handler = tool_registry.get(name)
            if handler is None:
                result = json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)
            else:
                try:
                    value = handler(arguments)
                    if inspect.isawaitable(value):
                        value = await value
                    result = self._stringify_result(value)
                except Exception as exc:
                    result = json.dumps({"error": f"Tool execution failed: {exc}"}, ensure_ascii=False)

            executions.append(
                ToolExecution(
                    tool_call_id=call["id"],
                    name=name,
                    arguments=arguments,
                    result=result,
                )
            )
        return executions

    def assistant_message(self, content: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        serialized = []
        for tc in self.normalize_tool_calls(tool_calls):
            tc = dict(tc)
            fn = dict(tc.get("function") or {})
            args = fn.get("arguments")
            fn["arguments"] = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else (args or "{}")
            tc["function"] = fn
            serialized.append(tc)
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": serialized,
        }

    def tool_messages(self, executions: list[ToolExecution]) -> list[dict[str, Any]]:
        return [
            {
                "role": "tool",
                "tool_call_id": execution.tool_call_id,
                "name": execution.name,
                "content": execution.result,
            }
            for execution in executions
        ]

    def openai_tools(self, tool_schemas: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tool_schemas:
            return None
        return [dict(schema) for schema in tool_schemas]

    def _parse_arguments(self, arguments: Any) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {"raw": arguments}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        return {"value": arguments}

    def _stringify_result(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)
