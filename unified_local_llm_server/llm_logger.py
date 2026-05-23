"""Structured stream logger for LLM request/response pairs.

Responsibility
--------------
Writes a human-readable, line-oriented log of every inference call: the full
message history, request parameters, streaming output (split into reasoning
and content sections), tool execution results, and a timing summary.

The log format uses pipe-delimited section markers (``|REQ-MESSAGES|``,
``|REQ-PRM|``, ``[THINKING]``, ``[MESSAGE]``, ``|INFO|``) so it can be
grep'd or parsed programmatically.

Thread safety
-------------
All file writes go through a single ``threading.Lock``.  Async callers
(``pool._call``) interact only through the ``async`` public methods, which
acquire the lock synchronously for the brief write — acceptable because
writes are tiny compared to inference latency.

Layer position
--------------
Cross-cutting utility.  Imported by ``pool.py``; has no imports from the
rest of this package.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def _msg_code(message: dict[str, Any]) -> str:
    """Compute a short deterministic hash for a message dict.

    Used in ``LOG_MSG_REFS`` mode to replace repeated messages with a
    back-reference, keeping logs readable in long multi-round sessions.
    """
    raw = json.dumps(message, sort_keys=True, ensure_ascii=False)
    return "M" + hashlib.sha1(raw.encode()).hexdigest()[:6].upper()


class AsyncLLMLogger:
    """Append-only structured logger for a single inference session.

    Create one instance per log file; pass it to ``LLMProviderPool`` or
    ``load_model`` so it receives callbacks for every request/response cycle.

    Parameters
    ----------
    path:
        Log file path.  Parent directories are created automatically.

    Environment
    -----------
    ``LOG_MSG_REFS=1``
        Deduplicate repeated messages by SHA1 reference.  Useful in long
        tool-call sessions where the same history is sent on each round.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._f = open(self._path, "w", encoding="utf-8")
        self._f.write(f"# log opened {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._f.flush()

        # Per-call state, reset by log_request
        self._call_num = 0
        self._start = 0.0
        self._pending_msg_count = 0
        self._first_chunk_written = False
        self._in_thinking = False
        self._output_tokens = 0
        self._think_tokens = 0
        self._log_end_written = False

        # Contextual label hierarchy (phase / step / sub-phase / attempt)
        self._levels: list[str] = []

        # Message deduplication: sha1 → already written
        self._seen: set[str] = set()
        self._msg_refs_mode = os.environ.get("LOG_MSG_REFS", "").lower() in ("1", "true", "yes")

    # ------------------------------------------------------------------
    # Context management — optional breadcrumb for multi-phase pipelines
    # ------------------------------------------------------------------

    async def set_context(
        self,
        phase: str = "",
        step: str = "",
        sub_phase: str = "",
        attempt: int | None = None,
    ) -> None:
        """Update the context label appended to every log line header.

        Levels are maintained as a hierarchy: phase > step > sub_phase >
        attempt.  Setting a higher level resets lower ones.
        """
        with self._lock:
            if phase:
                self._levels = [phase]
            if step:
                self._levels = self._levels[:1] + [step]
            if sub_phase:
                base = 1
                if len(self._levels) > 1 and self._levels[1].startswith("step:"):
                    base = 2
                self._levels = self._levels[:base] + [sub_phase]
            if attempt is not None:
                self._levels = self._levels + [str(attempt)]
                # Keep only the last numeric level to avoid stacking attempt counters
                last_num_idx = -1
                for i, level in enumerate(self._levels):
                    if level.isdigit():
                        last_num_idx = i
                self._levels = [
                    level
                    for i, level in enumerate(self._levels)
                    if not (level.isdigit() and i < last_num_idx)
                ]

    # ------------------------------------------------------------------
    # Public async logging API — called by pool._call()
    # ------------------------------------------------------------------

    async def log_request(
        self,
        *,
        provider: str,
        model: str | None,
        messages: list[dict[str, Any]],
        payload: dict[str, Any],
        measured_ctx: int | None = None,
    ) -> None:
        """Log the outgoing request: full message history + request parameters."""
        body = dict(payload)
        body.setdefault("model", model)
        body["messages"] = messages
        body["provider"] = provider

        with self._lock:
            self._call_num += 1
            self._start = time.monotonic()
            self._log_end_written = False
            self._pending_msg_count = len(messages)
            self._first_chunk_written = False
            self._in_thinking = False
            self._output_tokens = 0
            self._think_tokens = 0

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            ctx_info = f" [measured_ctx={measured_ctx}]" if measured_ctx is not None else ""
            self._w(f"\n{ts} [INFO] llm.stream: llm_request #{self._call_num}{ctx_info}{self._context_tag()}\n")

            for message in messages:
                self._write_message(message)

            # Log params without the full message list to avoid double-logging
            display_body = dict(body)
            if display_body.get("tools"):
                display_body["tools"] = [
                    tool.get("function", {}).get("name", tool)
                    for tool in display_body["tools"]
                ]
            body_without_messages = {k: v for k, v in display_body.items() if k != "messages"}
            prm_json = json.dumps(body_without_messages, indent=2, ensure_ascii=False).replace("\\n", "\n")
            for line in prm_json.split("\n"):
                self._w(f"|REQ-PRM|{line}\n")
            self._w("\n")
            self._f.flush()

    async def log_response(
        self,
        *,
        provider: str,
        model: str | None,
        text: str,
        parsed: Any | None = None,
        error: str | None = None,
        done_reason: str = "",
    ) -> None:
        """Log a complete (non-streaming) response and close the call record."""
        with self._lock:
            self._write_output_chunk(text, is_thinking=False)
            if parsed is not None:
                parsed_text = json.dumps(parsed, ensure_ascii=False)
                self._w(f"\n|INFO| parsed={parsed_text}\n")
            if error is not None:
                self._w(f"\n|INFO| error={error}\n")
            self._write_end(done_reason=done_reason)
            self._f.flush()

    async def log_output_chunk(self, text: str, is_thinking: bool = False) -> None:
        """Append a streaming output chunk (content or reasoning) to the log."""
        with self._lock:
            self._write_output_chunk(text, is_thinking=is_thinking)
            self._f.flush()

    async def log_info(self, text: str) -> None:
        """Write a free-form ``|INFO|`` line (used for tool results, events, etc.)."""
        with self._lock:
            for line in text.split("\n"):
                if line:
                    self._w(f"|INFO| {line}\n")
            self._f.flush()

    async def log_event(self, **event: Any) -> None:
        """Log a structured event as a JSON ``|INFO|`` line."""
        await self.log_info(json.dumps(event, ensure_ascii=False))

    async def log_end(
        self,
        *,
        ollama_prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        done_reason: str = "",
    ) -> None:
        """Write the timing and token-count footer for the current call.

        Idempotent — safe to call even if ``log_response`` already closed the
        record (the second call is silently ignored via ``_log_end_written``).
        """
        with self._lock:
            self._write_end(
                ollama_prompt_tokens=ollama_prompt_tokens,
                completion_tokens=completion_tokens,
                done_reason=done_reason,
            )
            self._f.flush()

    # ------------------------------------------------------------------
    # Private write helpers (must be called under self._lock)
    # ------------------------------------------------------------------

    def _context_tag(self) -> str:
        if not self._levels:
            return ""
        return f" [{':'.join(self._levels)}]"

    def _write_message(self, message: dict[str, Any]) -> None:
        """Write one message entry, substituting a hash reference if already seen."""
        code = _msg_code(message)
        if self._msg_refs_mode and code in self._seen:
            self._w(f"|REQ-MESSAGES| <{code}>\n")
            return
        self._seen.add(code)
        msg_json = json.dumps(message, indent=2, ensure_ascii=False).replace("\\n", "\n")
        lines = msg_json.split("\n")
        self._w(f"|REQ-MESSAGES| {code}:## {lines[0]}\n")
        for line in lines[1:]:
            self._w(f"|REQ-MESSAGES|{line}\n")

    def _write_output_chunk(self, text: str, is_thinking: bool = False) -> None:
        """Emit a content or reasoning chunk, inserting section headers on transitions."""
        if is_thinking and not self._in_thinking:
            self._w("\n[THINKING]\n")
            self._in_thinking = True
        elif not is_thinking and self._in_thinking:
            self._w("\n[MESSAGE]\n")
            self._in_thinking = False
        elif not self._in_thinking and not self._first_chunk_written:
            self._w("\n[MESSAGE]\n")
        self._first_chunk_written = True
        if is_thinking:
            self._think_tokens += 1
        else:
            self._output_tokens += 1
        self._w(text)

    def _write_end(
        self,
        *,
        ollama_prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        done_reason: str = "",
    ) -> None:
        """Write the timing footer; no-op if already written for this call."""
        if self._log_end_written:
            return
        self._log_end_written = True
        elapsed = time.monotonic() - self._start
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        prompt_info     = f" [prompt_tokens={ollama_prompt_tokens}]" if ollama_prompt_tokens else ""
        completion_info = f" [completion_tokens={completion_tokens}]" if completion_tokens else ""
        # Show raw chunk counts only when the provider didn't return usage data
        think_info = f" [think_tokens={self._think_tokens}]" if self._think_tokens and not completion_tokens else ""
        out_info   = f" [out_tokens={self._output_tokens}]"  if self._output_tokens and not completion_tokens else ""
        reason_info = f" [done_reason={done_reason}]" if done_reason else ""
        self._w(
            f"\n\n{ts} [INFO] llm.stream: llm_response_end #{self._call_num}"
            f" ({elapsed:.1f}s){prompt_info}{completion_info}{think_info}{out_info}{reason_info}{self._context_tag()}\n"
        )
        self._in_thinking = False

    def _w(self, text: str) -> None:
        """Raw unbuffered write — all callers must hold ``self._lock``."""
        self._f.write(text)

    async def close(self) -> None:
        """Flush and close the log file."""
        with self._lock:
            self._f.close()
