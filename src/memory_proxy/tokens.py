from __future__ import annotations

import json
import math
import re
from typing import Any


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
TOKEN_COUNTER_NAME = "approx_json_v1"


def approx_tokens(text: str) -> int:
    total = 0
    index = 0
    while index < len(text):
        char = text[index]
        if CJK_RE.match(char):
            total += 1
            index += 1
            continue
        if char.isspace():
            index += 1
            continue
        matched = ASCII_WORD_RE.match(text, index)
        if matched:
            total += max(1, math.ceil(len(matched.group(0)) / 4))
            index = matched.end()
            continue
        total += 1
        index += 1
    return total


def estimate_payload_tokens(payload: dict[str, Any]) -> int:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return approx_tokens(serialized)
