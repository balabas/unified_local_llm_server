"""Repetition-loop detection pipeline.

Responsibility
--------------
Detects when a model has entered a runaway repetition loop and signals the
caller to retry with a corrective prompt.  Five complementary detectors
cover the most common failure modes observed in practice:

- **sequence_loop** — a fixed n-gram repeats many times in the tail.
- **last_token_split_loop** — the model repeats the same suffix at the end
  of every "token" (common with greedy decoding at long context).
- **incrementing_sequence_loop** — structurally identical lines where only
  numbers change (e.g. ``"item 1", "item 2", …`` repeating the pattern).
- **numeric_list** — a dense run of numbers with separators (degenerate
  enumeration or counting loops).
- **numbered_block_cycle** — a numbered block repeats but with different
  numbers each time; detected by stripping digits before n-gram matching.

All detectors are tunable via environment variables so thresholds can be
adjusted without code changes.

Layer position
--------------
Pipeline layer.  Instantiated once by ``LLMProviderPool.__init__`` and
called from ``pool._call()`` after each generation round.
No imports from the rest of this package.
"""
from __future__ import annotations

import os
import re

# Tunable thresholds — override via environment variables
_LOOP_WINDOW       = int(os.environ.get("LOOP_WINDOW",            "220"))
_LOOP_LOOKBACK     = int(os.environ.get("LOOP_LOOKBACK",          "4000"))
_LOOP_MIN_HITS     = int(os.environ.get("LOOP_MIN_HITS",          "4"))
_LOOP_MIN_SUM_LEN  = int(os.environ.get("LOOP_MIN_SUM_LEN",       "700"))
_INCR_SEQ_MIN_LINES = int(os.environ.get("LOOP_INCR_SEQ_MIN_LINES", "15"))

_NUMERIC_LIST_RE = re.compile(r"(?:\d+\s*[,\s]\s*){50,}")
_NUMBER_RE       = re.compile(r"\d+")


def _is_sequence_looping(text: str, window: int, lookback: int, min_hits: int) -> bool:
    """Return True if the last ``window`` characters appear ``min_hits`` times
    in the last ``lookback`` characters with sufficient total coverage.

    The coverage check (``count * len(ngram) >= _LOOP_MIN_SUM_LEN``) prevents
    false positives on very short repeated phrases like punctuation.
    """
    try:
        if not text:
            return False
        window   = max(50, int(window))
        lookback = max(window * 2, int(lookback))
        min_hits = max(2, int(min_hits))
        if len(text) < window * 2:
            return False
        tail  = text[-lookback:]
        ngram = text[-window:]
        count = tail.count(ngram)
        return count >= min_hits and count * len(ngram) >= _LOOP_MIN_SUM_LEN
    except Exception:
        return False


def _is_last_token_split_loop(text: str) -> bool:
    """Detect when the penultimate token acts as a splitting delimiter that
    yields identical surrounding segments in the tail of the output."""
    try:
        toks = text.split()
        if not toks or len(toks) < 1000:
            return False
        last_tok = toks[-2]
        if not last_tok:
            return False
        parts = text.split(last_tok)
        if len(parts) < 4:
            return False
        tail_parts = [p.strip() for p in parts[-4:-1]]
        return (tail_parts[0] == tail_parts[1] == tail_parts[2]) and len(tail_parts[0]) > 5
    except Exception:
        return False


def _is_incrementing_sequence_loop(text: str, min_run: int = _INCR_SEQ_MIN_LINES) -> bool:
    """Detect structurally identical lines where only numbers differ.

    Numbers are replaced with ``#`` before comparison so ``"item 1"`` and
    ``"item 2"`` are treated as the same template.  Checks both newline-split
    and comma-split formats.
    """
    try:
        lines = text.splitlines()
        if len(lines) < min_run:
            for sep in ('",', '",\n', '", '):
                parts = text.split(sep)
                if len(parts) >= min_run:
                    lines = parts
                    break
            else:
                return False

        check_lines  = lines[-min_run * 3:] if len(lines) > min_run * 3 else lines
        run_length   = 1
        prev_template = _NUMBER_RE.sub("#", check_lines[0].strip())

        for line in check_lines[1:]:
            template = _NUMBER_RE.sub("#", line.strip())
            if template and template == prev_template:
                run_length += 1
                if run_length >= min_run:
                    return True
            else:
                run_length    = 1
                prev_template = template
        return False
    except Exception:
        return False


def _is_numeric_list_loop(text: str, min_length: int = 2000) -> bool:
    """Detect a dense run of 50+ consecutive numbers — a degenerate counting loop."""
    if len(text) < min_length:
        return False
    tail = text[-min_length:]
    return bool(_NUMERIC_LIST_RE.search(tail))


def _is_numbered_block_cycle(text: str, min_length: int = 1500) -> bool:
    """Detect a repeating block pattern where only embedded numbers change.

    Strips all digits before running the n-gram detector so blocks like
    ``"Step 1: do X"`` and ``"Step 2: do X"`` are treated identically.
    """
    try:
        if len(text) < min_length:
            return False
        stripped = _NUMBER_RE.sub("", text)
        return _is_sequence_looping(stripped, _LOOP_WINDOW, _LOOP_LOOKBACK, _LOOP_MIN_HITS)
    except Exception:
        return False


def check_loop(text: str) -> str | None:
    """Run all detectors and return the first matching reason string, or ``None``.

    The check order is roughly from cheapest/most-specific to most expensive:
    last-token-split first (O(n) split), then general n-gram, then others.
    """
    if _is_last_token_split_loop(text):
        return "last_token_split"
    if _is_sequence_looping(text, _LOOP_WINDOW, _LOOP_LOOKBACK, _LOOP_MIN_HITS):
        return "sequence_loop"
    if _is_numeric_list_loop(text):
        return "numeric_list"
    if _is_incrementing_sequence_loop(text):
        return "incrementing_sequence"
    if _is_numbered_block_cycle(text):
        return "numbered_block_cycle"
    return None


class LoopGuardPipeline:
    """Stateless wrapper exposing the loop-detection logic to ``pool._call()``."""

    def check(self, text: str) -> str | None:
        """Return a reason string if ``text`` is a loop, else ``None``."""
        return check_loop(text)

    def build_retry_messages(self, text: str, reason: str) -> list[dict[str, str]]:
        """Build corrective messages that instruct the model to stop looping.

        Appended to the conversation so the model sees its bad output on retry.
        Only the last 2 000 characters of the looping output are included to
        avoid blowing the context window.
        """
        return [
            {
                "role": "system",
                "content": (
                    "The previous output started looping and must be replaced with a short, "
                    f"non-repetitive answer. Loop reason: {reason}."
                ),
            },
            {
                "role": "user",
                "content": f"Previous looping output:\n{text[-2000:]}",
            },
        ]
