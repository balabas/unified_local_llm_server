from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..helpers.pydantic_helper import dict_to_pydantic_schema


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_json_comments(s: str) -> str:
    out: list[str] = []
    i = 0
    in_string = False
    while i < len(s):
        c = s[i]
        if in_string:
            out.append(c)
            if c == "\\":
                i += 1
                if i < len(s):
                    out.append(s[i])
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
                out.append(c)
            elif c == "/" and i + 1 < len(s) and s[i + 1] == "/":
                while i < len(s) and s[i] != "\n":
                    i += 1
                continue
            elif c == "/" and i + 1 < len(s) and s[i + 1] == "*":
                i += 2
                while i + 1 < len(s) and not (s[i] == "*" and s[i + 1] == "/"):
                    i += 1
                i += 2
                continue
            else:
                out.append(c)
        i += 1
    return "".join(out)


def _extract_balanced_json(text: str) -> str:
    start = None
    opening = ""
    closing = ""
    for idx, ch in enumerate(text):
        if ch == "{":
            start = idx
            opening = "{"
            closing = "}"
            break
        if ch == "[":
            start = idx
            opening = "["
            closing = "]"
            break
    if start is None:
        return text.strip()

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:].strip()


def _preview(text: str, limit: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _normalize_json_like(text: str) -> str:
    text = _THINK_RE.sub("", text)
    match = _CODE_FENCE_RE.search(text)
    if match:
        text = match.group(1)
    text = _strip_json_comments(text)
    text = text.strip()
    text = _extract_balanced_json(text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text.strip()


def _loads_best_effort(text: str) -> Any:
    if not isinstance(text, str):
        raise ValueError(f"Expected JSON output as text, got {type(text).__name__}")

    candidates = []
    raw = text.strip()
    if raw:
        candidates.append(raw)
    cleaned = _normalize_json_like(text)
    if cleaned and cleaned not in candidates:
        candidates.append(cleaned)

    if not candidates:
        raise ValueError("No JSON found in model output: response was empty")

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    has_json_start = any(ch in cleaned for ch in ("{", "["))
    if not has_json_start:
        raise ValueError(f"No JSON object or array found in model output: {_preview(text)!r}")

    pythonish = cleaned
    pythonish = re.sub(r"\btrue\b", "True", pythonish)
    pythonish = re.sub(r"\bfalse\b", "False", pythonish)
    pythonish = re.sub(r"\bnull\b", "None", pythonish)
    try:
        return ast.literal_eval(pythonish)
    except Exception as exc:
        raise ValueError(
            "Unable to repair JSON output. "
            f"Extracted candidate: {_preview(cleaned)!r}. "
            f"Raw output preview: {_preview(text)!r}"
        ) from exc


def _flat_dict_value_type(model_cls: type[BaseModel]) -> type | None:
    fields = model_cls.model_fields
    if not fields:
        return None
    types = {info.annotation for info in fields.values()}
    if len(types) == 1:
        t = next(iter(types))
        if t in (int, float, str, bool):
            return t
    return None


@dataclass(slots=True)
class StructuredOutputRequest:
    messages: list[dict[str, Any]]
    model_cls: type[BaseModel]
    schema_json: str


class JsonFixPipeline:
    def build_request(
        self,
        *,
        messages: list[dict[str, Any]],
        schema_dict: dict[str, Any],
    ) -> StructuredOutputRequest:
        model_cls = dict_to_pydantic_schema(schema_dict)
        schema_json = json.dumps(model_cls.model_json_schema(), ensure_ascii=False, indent=2)
        prepared_messages = [dict(m) for m in messages]

        replaced = False
        for message in prepared_messages:
            content = message.get("content")
            if isinstance(content, str) and "<SCHEMA_DICT>" in content:
                message["content"] = content.replace("<SCHEMA_DICT>", schema_json)
                replaced = True

        if not replaced:
            prepared_messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "Return valid JSON matching this schema:\n"
                        f"{schema_json}"
                    ),
                },
            )

        return StructuredOutputRequest(
            messages=prepared_messages,
            model_cls=model_cls,
            schema_json=schema_json,
        )

    def parse(self, text: str, model_cls: type[BaseModel]) -> dict[str, Any]:
        parsed = _loads_best_effort(text)
        if isinstance(parsed, dict):
            val_type = _flat_dict_value_type(model_cls)
            if val_type is not None:
                bad = [k for k, v in parsed.items() if not isinstance(v, val_type)]
                if bad:
                    raise ValueError(
                        f"Expected all values to be {val_type.__name__}, got wrong type for: {bad[:3]}"
                    )
                return parsed
        return model_cls.model_validate(parsed, strict=False).model_dump()

    def build_retry_messages(self, previous_text: str, error: Exception, schema_json: str | None) -> list[dict[str, str]]:
        schema_hint = f"\nSchema:\n{schema_json}" if schema_json else ""
        return [
            {
                "role": "system",
                "content": (
                    "The previous response was invalid JSON. "
                    "Return only valid JSON and keep the schema unchanged."
                    f"{schema_hint}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Fix this output:\n{previous_text[-4000:]}\n\n"
                    f"Parser error: {error}"
                ),
            },
        ]
