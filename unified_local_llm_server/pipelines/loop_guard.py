from __future__ import annotations

import os
import re

_LOOP_WINDOW = int(os.environ.get("LOOP_WINDOW", "220"))
_LOOP_LOOKBACK = int(os.environ.get("LOOP_LOOKBACK", "4000"))
_LOOP_MIN_HITS = int(os.environ.get("LOOP_MIN_HITS", "4"))
_LOOP_MIN_SUM_LEN = int(os.environ.get("LOOP_MIN_SUM_LEN", "700"))
_INCR_SEQ_MIN_LINES = int(os.environ.get("LOOP_INCR_SEQ_MIN_LINES", "15"))

_NUMERIC_LIST_RE = re.compile(r"(?:\d+\s*[,\s]\s*){50,}")
_NUMBER_RE = re.compile(r"\d+")


def _is_sequence_looping(text: str, window: int, lookback: int, min_hits: int) -> bool:
    try:
        if not text:
            return False
        window = max(50, int(window))
        lookback = max(window * 2, int(lookback))
        min_hits = max(2, int(min_hits))
        if len(text) < window * 2:
            return False
        tail = text[-lookback:]
        ngram = text[-window:]
        count = tail.count(ngram)
        return count >= min_hits and count * len(ngram) >= _LOOP_MIN_SUM_LEN
    except Exception:
        return False


def _is_last_token_split_loop(text: str) -> bool:
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

        check_lines = lines[-min_run * 3:] if len(lines) > min_run * 3 else lines
        run_length = 1
        prev_template = _NUMBER_RE.sub("#", check_lines[0].strip())

        for line in check_lines[1:]:
            template = _NUMBER_RE.sub("#", line.strip())
            if template and template == prev_template:
                run_length += 1
                if run_length >= min_run:
                    return True
            else:
                run_length = 1
                prev_template = template
        return False
    except Exception:
        return False


def _is_numeric_list_loop(text: str, min_length: int = 2000) -> bool:
    if len(text) < min_length:
        return False
    tail = text[-min_length:]
    return bool(_NUMERIC_LIST_RE.search(tail))


def _is_numbered_block_cycle(text: str, min_length: int = 1500) -> bool:
    try:
        if len(text) < min_length:
            return False
        stripped = _NUMBER_RE.sub("", text)
        return _is_sequence_looping(stripped, _LOOP_WINDOW, _LOOP_LOOKBACK, _LOOP_MIN_HITS)
    except Exception:
        return False


def check_loop(text: str) -> str | None:
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
    def check(self, text: str) -> str | None:
        return check_loop(text)

    def build_retry_messages(self, text: str, reason: str) -> list[dict[str, str]]:
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

