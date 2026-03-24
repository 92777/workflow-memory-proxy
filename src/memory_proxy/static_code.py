from __future__ import annotations

import re

from .tokens import approx_tokens

STATIC_CODE_SUMMARY_MARKER = "[static-code-summary]"

CODE_FENCE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_#+.-]*)[^\n]*\n?(?P<code>.*?)```", re.DOTALL)
GENERIC_PATH_RE = re.compile(r"(?:/|\.{1,2}/)?[\w.-]+(?:/[\w.-]+)*\.[A-Za-z0-9]+")
URL_RE = re.compile(r"https?://[^\s<>\")]+")
DIFF_HEADER_RE = re.compile(r"^(?:diff --git|index |@@|--- |\+\+\+ )", re.MULTILINE)
DIFF_FILE_RE = re.compile(r"^(?:--- |\+\+\+ )(?:[ab]/)?(?P<path>\S+)", re.MULTILINE)

LANGUAGE_BY_EXTENSION = {
    "go": "go",
    "java": "java",
    "js": "javascript",
    "jsx": "javascript",
    "json": "json",
    "php": "php",
    "py": "python",
    "rb": "ruby",
    "rs": "rust",
    "sh": "shell",
    "sql": "sql",
    "toml": "toml",
    "ts": "typescript",
    "tsx": "typescript",
    "yaml": "yaml",
    "yml": "yaml",
}

NOTE_FILLER_PREFIXES = (
    "这条已经加进去了",
    "这条也修好了",
    "现在",
    "然后",
    "另外",
    "主要逻辑在",
)

DEFINITION_PATTERNS = (
    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE),
    re.compile(r"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE),
    re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        re.MULTILINE,
    ),
    re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE),
    re.compile(
        r"^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)\s*=>|function\b)",
        re.MULTILINE,
    ),
)


def is_code_heavy_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if CODE_FENCE_RE.search(stripped):
        return True
    if _looks_like_diff(stripped):
        return True
    code_like_lines = _count_code_like_lines(stripped)
    if code_like_lines >= 6:
        return True
    if code_like_lines >= 4 and _collect_file_hints(stripped):
        return True
    return False


def compress_static_code_message(text: str) -> str | None:
    summary = summarize_static_code_reference(text)
    if not summary:
        return None
    stripped = text.strip()
    if approx_tokens(summary) + 8 >= approx_tokens(stripped):
        return None
    return summary


def summarize_static_code_reference(text: str) -> str | None:
    stripped = text.strip()
    if not is_code_heavy_text(stripped):
        return None
    return _build_summary(stripped)


def _build_summary(text: str) -> str:
    files = _collect_file_hints(text)
    languages = _collect_languages(text, files)
    definitions = _collect_definitions(text)
    prose_notes = _collect_prose_notes(text)
    diff_like = _looks_like_diff(text)
    code_line_count = _count_code_lines(text, diff_like=diff_like)

    parts: list[str] = [f"kind={'diff' if diff_like else 'code'}"]
    if files:
        parts.append(f"files={','.join(files[:3])}")
    if languages:
        parts.append(f"lang={','.join(languages[:2])}")
    parts.append(f"lines={code_line_count}")
    if diff_like:
        added, removed, hunks = _diff_stats(text)
        parts.append(f"hunks={hunks}")
        parts.append(f"changes=+{added}/-{removed}")
    else:
        block_count = len(list(CODE_FENCE_RE.finditer(text)))
        if block_count > 0:
            parts.append(f"blocks={block_count}")
    if definitions:
        parts.append(f"defs={','.join(definitions[:4])}")
    if prose_notes and (not files or not definitions):
        parts.append(f"note={prose_notes}")
    return f"{STATIC_CODE_SUMMARY_MARKER} " + "; ".join(parts)


def _looks_like_diff(text: str) -> bool:
    if not DIFF_HEADER_RE.search(text):
        return False
    diff_lines = 0
    for line in text.splitlines():
        if line.startswith(("diff --git", "index ", "@@", "--- ", "+++ ")):
            diff_lines += 1
            continue
        if line.startswith(("+", "-")) and len(line) > 1 and not line.startswith(("+++", "---")):
            diff_lines += 1
    return diff_lines >= 4


def _count_code_like_lines(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("```", "@@", "diff --git", "+++ ", "--- ")):
            count += 1
            continue
        if stripped.startswith(("+", "-")) and len(stripped) > 1:
            count += 1
            continue
        if stripped.endswith(("{", "}", ";", "):")):
            count += 1
            continue
        if re.match(
            r"^(?:class |def |func |from |import |return |if |for |while |const |let |var |export |package )",
            stripped,
        ):
            count += 1
    return count


def _count_code_lines(text: str, *, diff_like: bool) -> int:
    blocks = [match.group("code") for match in CODE_FENCE_RE.finditer(text)]
    if blocks:
        return sum(_count_non_empty_lines(block) for block in blocks)
    if diff_like:
        return sum(
            1
            for line in text.splitlines()
            if line and not line.startswith(("diff --git", "index ", "@@", "--- ", "+++ "))
        )
    return _count_code_like_lines(text)


def _count_non_empty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _collect_languages(text: str, files: list[str]) -> list[str]:
    seen: set[str] = set()
    languages: list[str] = []
    for match in CODE_FENCE_RE.finditer(text):
        lang = match.group("lang").strip().lower()
        if not lang or lang in seen:
            continue
        seen.add(lang)
        languages.append(lang)
    for path in files:
        extension = path.rsplit(".", 1)[-1].lower()
        guessed = LANGUAGE_BY_EXTENSION.get(extension)
        if guessed is None or guessed in seen:
            continue
        seen.add(guessed)
        languages.append(guessed)
    return languages


def _collect_file_hints(text: str) -> list[str]:
    seen: set[str] = set()
    hints: list[str] = []
    url_spans = [match.span() for match in URL_RE.finditer(text)]
    for match in DIFF_FILE_RE.finditer(text):
        path = _clean_path(match.group("path"))
        if path and path not in seen:
            seen.add(path)
            hints.append(path)
    for match in GENERIC_PATH_RE.finditer(text):
        if _span_overlaps_any(match.span(), url_spans):
            continue
        path = _clean_path(match.group(0))
        if path and path not in seen:
            seen.add(path)
            hints.append(path)
    return hints


def _clean_path(value: str) -> str:
    cleaned = value.strip().strip("`'\",:;()[]{}")
    if not cleaned or cleaned in {"/dev/null", "a/dev/null", "b/dev/null"}:
        return ""
    if "://" in cleaned:
        return ""
    return cleaned


def _span_overlaps_any(span: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
    start, end = span
    for range_start, range_end in ranges:
        if start < range_end and end > range_start:
            return True
    return False


def _collect_definitions(text: str) -> list[str]:
    seen: set[str] = set()
    definitions: list[str] = []
    for pattern in DEFINITION_PATTERNS:
        for match in pattern.finditer(text):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            definitions.append(name)
    return definitions


def _collect_prose_notes(text: str) -> str:
    stripped = _strip_fenced_code(text)
    notes: list[str] = []
    for line in stripped.splitlines():
        candidate = " ".join(line.strip().split())
        if not candidate:
            continue
        if candidate.startswith(("diff --git", "index ", "@@", "--- ", "+++ ")):
            continue
        if GENERIC_PATH_RE.fullmatch(candidate):
            continue
        if _count_code_like_lines(candidate) > 0 and len(candidate) < 48:
            continue
        compressed = _compress_prose_note(candidate)
        if not compressed:
            continue
        notes.append(compressed)
        if len(notes) == 2:
            break
    return "|".join(notes)


def _strip_fenced_code(text: str) -> str:
    parts: list[str] = []
    last_index = 0
    for match in CODE_FENCE_RE.finditer(text):
        parts.append(text[last_index : match.start()])
        last_index = match.end()
    parts.append(text[last_index:])
    return "\n".join(parts)


def _diff_stats(text: str) -> tuple[int, int, int]:
    added = 0
    removed = 0
    hunks = 0
    for line in text.splitlines():
        if line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed, hunks


def _compress_prose_note(text: str) -> str:
    lowered = text.casefold()
    tags: list[str] = []

    if "md5" in lowered:
        tags.append("MD5")
    if "短引用" in text or "别名" in text:
        tags.append("短引用别名")
    if "静态代码" in text or "静态摘要" in text:
        tags.append("静态摘要")
    if "报错" in text or "堆栈" in text or "stack trace" in lowered:
        tags.append("报错堆栈")
    if "recent_token_budget" in lowered or "token budget" in lowered or "预算" in text:
        tags.append("预算")

    if tags:
        seen: set[str] = set()
        ordered = [tag for tag in tags if not (tag in seen or seen.add(tag))]
        return "/".join(ordered[:3])

    candidate = text
    for prefix in NOTE_FILLER_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :].lstrip("：:，,。;； ")
            break
    candidate = re.sub(r"\s+", " ", candidate).strip()
    if not candidate:
        return ""
    return candidate[:32]
