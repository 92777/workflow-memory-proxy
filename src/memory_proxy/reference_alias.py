from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from .openai_api import ChatMessage
from .static_code import CODE_FENCE_RE, summarize_static_code_reference
from .tokens import approx_tokens

URL_RE = re.compile(r"https?://[^\s<>\")]+")
FILE_RE = re.compile(
    r"(?:/[\w./-]+|(?:\./|\.\./)[\w./-]+|(?:\b[\w.-]+/[\w./-]+\.[A-Za-z0-9]+\b)|(?:\b[\w.-]+\.[A-Za-z0-9]+\b))"
)
TRAILING_TRIM_CHARS = ".,;:!?)]}>\"'"
KIND_PREFIX = {
    "file": "FILE",
    "url": "URL",
    "code": "CODE",
}


@dataclass(frozen=True, slots=True)
class _ReferenceOccurrence:
    message_index: int
    kind: str
    start: int
    end: int
    raw_text: str
    canonical: str
    fingerprint: str
    display_text: str


@dataclass(slots=True)
class _AliasGroup:
    kind: str
    fingerprint: str
    occurrences: list[_ReferenceOccurrence]
    alias: str = ""

    def definition_text(self) -> str:
        display = self.occurrences[0].display_text
        return f"{self.alias}{{{self.fingerprint}:{display}}}"

    def reference_text(self) -> str:
        return self.alias

    def saves_tokens(self) -> bool:
        original = sum(approx_tokens(item.raw_text) for item in self.occurrences)
        replaced = approx_tokens(self.definition_text()) + approx_tokens(self.reference_text()) * (
            len(self.occurrences) - 1
        )
        return replaced < original


def alias_repeated_references(messages: list[ChatMessage]) -> list[ChatMessage]:
    if len(messages) < 2:
        return messages

    occurrences = _collect_occurrences(messages)
    groups = _build_alias_groups(occurrences)
    if not groups:
        return messages

    return _apply_aliases(messages, groups)


def _collect_occurrences(messages: list[ChatMessage]) -> list[_ReferenceOccurrence]:
    occurrences: list[_ReferenceOccurrence] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message.content, str):
            continue
        occurrences.extend(_extract_occurrences(message.content, message_index=message_index))
    return occurrences


def _build_alias_groups(occurrences: list[_ReferenceOccurrence]) -> dict[tuple[str, str], _AliasGroup]:
    grouped: dict[tuple[str, str], list[_ReferenceOccurrence]] = {}
    for occurrence in occurrences:
        grouped.setdefault((occurrence.kind, occurrence.fingerprint), []).append(occurrence)

    counters = {"file": 0, "url": 0, "code": 0}
    alias_groups: dict[tuple[str, str], _AliasGroup] = {}
    ordered_groups = sorted(
        (items for items in grouped.values() if len(items) >= 2),
        key=lambda items: (items[0].message_index, items[0].start),
    )
    for items in ordered_groups:
        kind = items[0].kind
        counters[kind] += 1
        group = _AliasGroup(
            kind=kind,
            fingerprint=items[0].fingerprint,
            occurrences=sorted(items, key=lambda item: (item.message_index, item.start)),
            alias=f"{KIND_PREFIX[kind]}_{counters[kind]}",
        )
        if group.saves_tokens():
            alias_groups[(kind, group.fingerprint)] = group
    return alias_groups


def _apply_aliases(messages: list[ChatMessage], groups: dict[tuple[str, str], _AliasGroup]) -> list[ChatMessage]:
    defined: set[tuple[str, str]] = set()
    output: list[ChatMessage] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message.content, str):
            output.append(message)
            continue
        occurrences = [
            item
            for item in _extract_occurrences(message.content, message_index=message_index)
            if (item.kind, item.fingerprint) in groups
        ]
        if not occurrences:
            output.append(message)
            continue

        rewritten: list[str] = []
        cursor = 0
        for occurrence in occurrences:
            rewritten.append(message.content[cursor : occurrence.start])
            group = groups[(occurrence.kind, occurrence.fingerprint)]
            key = (occurrence.kind, occurrence.fingerprint)
            if key in defined:
                rewritten.append(group.reference_text())
            else:
                rewritten.append(group.definition_text())
                defined.add(key)
            cursor = occurrence.end
        rewritten.append(message.content[cursor:])
        output.append(message.model_copy(update={"content": "".join(rewritten)}))
    return output


def _extract_occurrences(text: str, *, message_index: int) -> list[_ReferenceOccurrence]:
    occupied: list[tuple[int, int]] = []
    occurrences: list[_ReferenceOccurrence] = []

    for match in CODE_FENCE_RE.finditer(text):
        full_block = match.group(0)
        summary = summarize_static_code_reference(full_block)
        if summary is None:
            continue
        canonical = _canonicalize_code_block(match.group("lang"), match.group("code"))
        occurrences.append(
            _ReferenceOccurrence(
                message_index=message_index,
                kind="code",
                start=match.start(),
                end=match.end(),
                raw_text=full_block,
                canonical=canonical,
                fingerprint=_md5_short(canonical),
                display_text=summary,
            )
        )
        occupied.append((match.start(), match.end()))

    for match in URL_RE.finditer(text):
        start, end, cleaned = _trim_match(text, match.start(), match.end())
        if not cleaned or _overlaps(start, end, occupied):
            continue
        canonical = cleaned
        occurrences.append(
            _ReferenceOccurrence(
                message_index=message_index,
                kind="url",
                start=start,
                end=end,
                raw_text=cleaned,
                canonical=canonical,
                fingerprint=_md5_short(canonical),
                display_text=cleaned,
            )
        )
        occupied.append((start, end))

    for match in FILE_RE.finditer(text):
        start, end, cleaned = _trim_match(text, match.start(), match.end())
        if not cleaned or _overlaps(start, end, occupied):
            continue
        if "://" in cleaned:
            continue
        if not _looks_like_reference_file(cleaned):
            continue
        canonical = cleaned
        occurrences.append(
            _ReferenceOccurrence(
                message_index=message_index,
                kind="file",
                start=start,
                end=end,
                raw_text=cleaned,
                canonical=canonical,
                fingerprint=_md5_short(canonical),
                display_text=cleaned,
            )
        )
        occupied.append((start, end))

    return sorted(occurrences, key=lambda item: item.start)


def _canonicalize_code_block(language: str | None, code: str | None) -> str:
    normalized_lang = (language or "").strip().casefold()
    normalized_code = "\n".join(line.rstrip() for line in (code or "").strip().splitlines())
    return f"{normalized_lang}\n{normalized_code}"


def _trim_match(text: str, start: int, end: int) -> tuple[int, int, str]:
    while end > start and text[end - 1] in TRAILING_TRIM_CHARS:
        end -= 1
    while start < end and text[start].isspace():
        start += 1
    return start, end, text[start:end]


def _overlaps(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in occupied)


def _looks_like_reference_file(value: str) -> bool:
    if len(value) < 6:
        return False
    if "/" in value or value.startswith(("./", "../")):
        return True
    if "." not in value:
        return False
    stem, _, suffix = value.rpartition(".")
    return bool(stem) and suffix.isalnum()


def _md5_short(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
