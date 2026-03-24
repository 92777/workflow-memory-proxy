from __future__ import annotations

import re


SEMANTIC_FILE_PATTERN = re.compile(r"\b[\w./-]+\.[A-Za-z0-9]+\b")
SEMANTIC_NORMALIZE_RE = re.compile(r"[\s`'\"“”‘’(),，。；：！？!\[\]{}<>]+")
MIN_CONTAINMENT_CHARS = 3
MIN_INNER_CONTAINMENT_CHARS = 8


def merge_semantic_list(existing: list[str], incoming: list[str]) -> list[str]:
    merged = list(existing)
    for item in incoming:
        if not item:
            continue
        index = find_semantic_match_index(merged, item)
        if index is None:
            merged.append(item)
            continue
        merged[index] = prefer_richer_text(merged[index], item)
    return merged


def append_semantic_unique(items: list[str], value: str) -> None:
    index = find_semantic_match_index(items, value)
    if index is None:
        items.append(value)
        return
    items[index] = prefer_richer_text(items[index], value)


def remove_semantic_value(items: list[str], value: str) -> None:
    items[:] = [item for item in items if not are_texts_semantically_similar(item, value)]


def prefer_richer_text(existing: str | None, incoming: str | None) -> str | None:
    if not existing:
        return incoming
    if not incoming:
        return existing
    existing_score = _information_score(existing)
    incoming_score = _information_score(incoming)
    if incoming_score >= existing_score:
        return incoming
    return existing


def find_semantic_match_index(items: list[str], value: str) -> int | None:
    for index, item in enumerate(items):
        if are_texts_semantically_similar(item, value):
            return index
    return None


def are_texts_semantically_similar(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    left_norm = _normalize_for_semantic_match(left)
    right_norm = _normalize_for_semantic_match(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    short, long = sorted((left_norm, right_norm), key=len)
    if len(short) < MIN_CONTAINMENT_CHARS:
        return False
    if long.startswith(short) or long.endswith(short):
        return True
    if len(short) >= MIN_INNER_CONTAINMENT_CHARS and short in long:
        return True
    return False


def _normalize_for_semantic_match(text: str) -> str:
    lowered = text.casefold().strip()
    return SEMANTIC_NORMALIZE_RE.sub("", lowered)


def _information_score(text: str) -> tuple[int, int, int, int]:
    normalized = _normalize_for_semantic_match(text)
    file_hint = 1 if SEMANTIC_FILE_PATTERN.search(text) else 0
    newline_hint = 1 if "\n" in text else 0
    digit_hint = 1 if any(char.isdigit() for char in text) else 0
    return (len(normalized), file_hint, newline_hint, digit_hint)


def contains_file_hint(text: str) -> bool:
    return bool(SEMANTIC_FILE_PATTERN.search(text))
