"""Structured-output pipeline: schema injection and JSON repair.

Responsibility
--------------
Handles the full lifecycle of structured (JSON) output requests:

1. **Build** — converts a ``schema_dict`` to a Pydantic model, injects a
   system prompt that tells the model to return JSON matching the schema.
2. **Parse** — extracts and validates JSON from the model's raw text output,
   tolerating common model quirks (code fences, ``<think>`` blocks,
   trailing commas, JS-style comments, Python literals).
3. **Retry** — produces corrective messages when parsing fails so the caller
   can re-submit with an error hint.

The parsing strategy is deliberately lenient: try raw JSON first, then
progressively more aggressive normalisation, finally fall back to
``ast.literal_eval`` for Python-formatted output.

Layer position
--------------
Pipeline layer.  Instantiated once by ``LLMProviderPool.__init__`` and
called from ``pool._call()`` when ``schema_dict`` is provided.
Imports: ``helpers.pydantic_helper``.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..helpers.pydantic_helper import dict_to_pydantic_schema


# Pre-compiled patterns used during JSON normalization
_THINK_RE      = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_json_comments(s: str) -> str:
    """Remove ``//`` and ``/* */`` comments from a JSON-like string.

    Operates character-by-character to avoid stripping comment-like sequences
    inside string values.
    """
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
    """Extract the first balanced ``{…}`` or ``[…]`` block from ``text``.

    Returns the full input stripped if no opening bracket is found, so the
    caller can still attempt ``json.loads`` on the raw text.
    """
    start = None
    opening = ""
    closing = ""
    for idx, ch in enumerate(text):
        if ch == "{":
            start, opening, closing = idx, "{", "}"
            break
        if ch == "[":
            start, opening, closing = idx, "[", "]"
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
    """Return a compact single-line preview of ``text`` for error messages."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _normalize_json_like(text: str) -> str:
    """Apply all cleanup steps in sequence to produce the best parse candidate."""
    text = _THINK_RE.sub("", text)           # remove reasoning blocks
    match = _CODE_FENCE_RE.search(text)
    if match:
        text = match.group(1)                # unwrap ```json … ```
    text = _strip_json_comments(text)
    text = text.strip()
    text = _extract_balanced_json(text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)  # trailing commas
    return text.strip()


def _loads_best_effort(text: str) -> Any:
    """Parse JSON from model output with progressive fallback strategies.

    Tries in order:
    1. Raw ``json.loads`` on the original text.
    2. ``json.loads`` after full normalisation (comments, fences, trailing commas …).
    3. ``ast.literal_eval`` after Python-to-JSON keyword substitution
       (``True``/``False``/``None``), for models that emit Python-formatted dicts.

    Raises ``ValueError`` with a descriptive message if all strategies fail.
    """
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

    # Last resort: treat Python-literal syntax as JSON
    pythonish = cleaned
    pythonish = re.sub(r"\btrue\b",  "True",  pythonish)
    pythonish = re.sub(r"\bfalse\b", "False", pythonish)
    pythonish = re.sub(r"\bnull\b",  "None",  pythonish)
    try:
        return ast.literal_eval(pythonish)
    except Exception as exc:
        raise ValueError(
            "Unable to repair JSON output. "
            f"Extracted candidate: {_preview(cleaned)!r}. "
            f"Raw output preview: {_preview(text)!r}"
        ) from exc


def _flat_dict_value_type(model_cls: type[BaseModel]) -> type | None:
    """Return the common value type if all fields share the same primitive type.

    Used to validate flat key→value dicts where the schema says every value
    must be e.g. ``int`` — avoids running full Pydantic validation when a
    simple isinstance check is sufficient and faster.
    """
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
    """Prepared state for a structured-output inference round.

    Produced by :meth:`JsonFixPipeline.build_request` and consumed by
    :meth:`JsonFixPipeline.parse` and :meth:`JsonFixPipeline.build_retry_messages`.
    Carries the Pydantic model class so validation can run without re-deriving
    it from the schema on every retry.
    """

    messages: list[dict[str, Any]]   # messages with schema system prompt injected
    model_cls: type[BaseModel]        # Pydantic model for response validation
    schema_json: str                  # JSON Schema string included in the prompt


class JsonFixPipeline:
    """Stateless pipeline for structured JSON output with automatic retry.

    The same instance is reused across all calls; all state is passed explicitly.
    """

    def build_request(
        self,
        *,
        messages: list[dict[str, Any]],
        schema_dict: dict[str, Any],
    ) -> StructuredOutputRequest:
        """Inject the JSON schema into the message list and build a Pydantic validator.

        If a message contains the placeholder ``<SCHEMA_DICT>``, it is replaced
        with the schema JSON.  Otherwise a system prompt is prepended.
        """
        model_cls  = dict_to_pydantic_schema(schema_dict)
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
                    "content": f"Return valid JSON matching this schema:\n{schema_json}",
                },
            )

        return StructuredOutputRequest(
            messages=prepared_messages,
            model_cls=model_cls,
            schema_json=schema_json,
        )

    def parse(self, text: str, model_cls: type[BaseModel]) -> dict[str, Any]:
        """Extract, repair, and validate JSON from model output.

        Returns a ``dict`` (Pydantic ``model_dump()``) on success.
        Raises ``ValueError`` on unrecoverable parse or validation failure.
        """
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

    def build_retry_messages(
        self,
        previous_text: str,
        error: Exception,
        schema_json: str | None,
    ) -> list[dict[str, str]]:
        """Build corrective messages to append when parsing fails.

        The returned messages are appended to the conversation so the model
        sees its bad output and the parse error, then generates a corrected
        response.
        """
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
