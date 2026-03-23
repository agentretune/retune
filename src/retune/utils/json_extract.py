"""Robust JSON extraction from LLM outputs."""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract JSON from LLM output, handling code fences and nested structures.

    Tries in order:
    1. Direct parse of entire text
    2. Extract from markdown code fences (```json ... ```)
    3. Balanced-brace extraction for first { or [ block

    Returns parsed dict/list or None if no valid JSON found.
    """
    text_stripped = text.strip()

    # Step 1: Direct parse
    try:
        result = json.loads(text_stripped)
        if isinstance(result, (dict, list)):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Step 2: Markdown code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, (dict, list)):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Step 3: Balanced-brace extraction
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx:i + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, (dict, list)):
                            return result
                    except (json.JSONDecodeError, ValueError):
                        break

    return None


def extract_json_or_default(text: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract JSON dict from text, returning default if extraction fails."""
    result = extract_json(text)
    if isinstance(result, dict):
        return result
    return default or {}
