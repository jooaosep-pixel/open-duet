"""Text helpers: robust JSON extraction and truncation."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+.-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in ``text``; tolerant of code fences and prose."""
    if text is None:
        raise ValueError("empty response")
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE_RE.findall(text)]
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    # fall back to the outermost {...} or [...] span
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in model output: {text[:200]!r}")


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n[... truncated ...]"


def strip_code_fences(text: str) -> str:
    m = _FENCE_RE.search(text or "")
    if m:
        return m.group(1).strip()
    return (text or "").strip()


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text or "").strip("-").lower()
    return s[:max_len] or "figure"
