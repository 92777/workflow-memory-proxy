from __future__ import annotations

from dataclasses import dataclass, field
import re

from .openai_api import ChatMessage
from .static_code import is_code_heavy_text
from .zh_semantics import (
    contains_workflow_keyword,
    low_value_execution_filler_re,
    meta_confirmation_re,
)

ARTIFACT_RE = re.compile(r"(?:/[\w./-]+)|(?:\b[\w./-]+\.[A-Za-z0-9]+\b)")
PATCH_TARGET_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)
PATCH_MOVE_RE = re.compile(r"^\*\*\* Move to: (.+)$", re.MULTILINE)
SEARCH_RESULT_LINE_RE = re.compile(
    r"^(?P<path>(?:\.{0,2}/)?[\w./-]+\.[A-Za-z0-9]+):\d+(?::\d+)?:",
    re.MULTILINE,
)
LISTING_PATH_LINE_RE = re.compile(
    r"^(?P<path>(?:\.{0,2}/)?[\w./-]+(?:/|(?:\.[A-Za-z0-9]+))?)$",
    re.MULTILINE,
)
ERROR_RE = re.compile(r"(报错|失败|error|failed|traceback|exception)", re.IGNORECASE)
QUESTION_RE = re.compile(r"[?？]$|(是否|要不要|需不需要|为什么|怎么做)")
READ_OP_RE = re.compile(r"(查看|读取|读一下|read|open|cat|sed)", re.IGNORECASE)
SEARCH_OP_RE = re.compile(r"(rg|grep|搜索|检索|查找|find)", re.IGNORECASE)
LIST_OP_RE = re.compile(r"(列出|ls|tree|list)", re.IGNORECASE)
WRITE_OP_RE = re.compile(r"(修改|更新|写入|创建|新增|apply_patch|patched|created|updated|edited)", re.IGNORECASE)
TEST_OP_RE = re.compile(
    r"(tests? passed|测试通过|pytest|unittest|"
    r"(?:测试|校验|验证|验收|回归|联调|灰度|巡检|复核)(?:通过|完成)|"
    r"通过(?:测试|校验|验证|验收|回归|联调|灰度|巡检|复核))",
    re.IGNORECASE,
)
VALIDATION_RUN_OP_RE = re.compile(
    r"(?:(?:再|重新|再次|继续)?(?:跑|做|执行|进行)(?:一(?:次|遍|轮)|一下)?[^。；;\n]{0,24}"
    r"(?:测试|校验|验证|验收|回归|灰度|联调|巡检|复核)|"
    r"(?:再|重新|再次|继续)?(?:验(?!证)|测(?!试)|跑)(?:一(?:下|遍|轮)|一下|一遍|一轮)\s*[^。；;\n]{0,24}"
    r"(?:测试|校验|验证|验收|回归|灰度|联调|巡检|复核)|"
    r"(?:再|重新|再次|继续|先)?(?:补|确认|过)(?:一(?:下|遍|轮)|一下|一遍|一轮)\s*[^。；;\n]{0,24}"
    r"(?:测试|校验|验证|验收|回归|灰度|联调|巡检|复核)|"
    r"(?:再|重新|再次|继续)?(?:测试|校验|验证|验收|回归|灰度|联调|巡检|复核)\s+[^。；;\n]+)",
    re.IGNORECASE,
)
VERIFICATION_HINT_RE = re.compile(r"(测试|校验|验证|验收|回归|联调|灰度|巡检|复核)", re.IGNORECASE)
META_CONFIRM_RE = meta_confirmation_re()
LOW_VALUE_EXECUTION_FILLER_RE = low_value_execution_filler_re()

HIGH = "high"
MEDIUM = "medium"
LOW = "low"


@dataclass(frozen=True, slots=True)
class HistoryPruneResult:
    kept_messages: list[ChatMessage]
    dropped_indexes: list[int]


@dataclass(frozen=True, slots=True)
class OperationSignature:
    kind: str
    target_keys: set[str]
    detail_keys: set[str] = field(default_factory=set)


def has_workflow_signal(message_or_text: ChatMessage | str) -> bool:
    text = (
        message_or_text.content
        if isinstance(message_or_text, ChatMessage)
        else message_or_text
    )
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if ERROR_RE.search(stripped) or QUESTION_RE.search(stripped):
        return False
    return contains_workflow_keyword(stripped)


def is_verification_result_message(message: ChatMessage) -> bool:
    if message.role not in {"assistant", "tool"}:
        return False
    if not isinstance(message.content, str):
        return False
    text = message.content.strip()
    if not text or VALIDATION_RUN_OP_RE.search(text):
        return False
    if TEST_OP_RE.search(text) is None:
        return False
    return _extract_verification_target(text) is not None


def classify_history_fidelity(message: ChatMessage) -> str:
    if not isinstance(message.content, str):
        return HIGH

    text = message.content.strip()
    if not text:
        return LOW

    lower = text.lower()
    if _is_low_value_execution_filler(message, text):
        return LOW
    if ERROR_RE.search(text) or QUESTION_RE.search(text) or is_code_heavy_text(text):
        return HIGH
    if message.role == "user" and ("记住" in text or ARTIFACT_RE.search(text)):
        return HIGH
    if message.role == "user" and contains_workflow_keyword(text):
        return HIGH
    if ARTIFACT_RE.search(text) and (contains_workflow_keyword(text) or WRITE_OP_RE.search(text)):
        return HIGH
    operation = _operation_signature(message, text)
    if operation is not None:
        kind = operation.kind
        if kind in {"write", "test"}:
            return MEDIUM
        return LOW
    if message.role == "assistant" and META_CONFIRM_RE.match(text) and len(lower) <= 48:
        return LOW
    if message.role == "user":
        return MEDIUM
    return MEDIUM


def prune_history_messages(messages: list[ChatMessage]) -> HistoryPruneResult:
    if not messages:
        return HistoryPruneResult(kept_messages=[], dropped_indexes=[])

    seen_read_keys: set[str] = set()
    seen_search_keys: set[str] = set()
    seen_list_keys: set[str] = set()
    seen_write_keys: set[str] = set()
    seen_test_keys: set[str] = set()
    seen_search_detail_keys: set[str] = set()
    seen_list_detail_keys: set[str] = set()
    seen_meta = False
    kept_reversed: list[ChatMessage] = []
    dropped_indexes: list[int] = []

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        text = message.content if isinstance(message.content, str) else ""
        stripped = text.strip()
        if not stripped:
            dropped_indexes.append(index)
            continue

        if _is_low_value_execution_filler(message, stripped):
            dropped_indexes.append(index)
            continue

        if _is_low_value_meta_confirmation(message, stripped):
            if seen_meta:
                dropped_indexes.append(index)
                continue
            seen_meta = True

        signature = _operation_signature(message, stripped)
        if signature is not None:
            kind = signature.kind
            target_keys = signature.target_keys
            detail_keys = signature.detail_keys
            if kind == "read":
                if target_keys & (seen_read_keys | seen_search_keys | seen_list_keys | seen_write_keys):
                    dropped_indexes.append(index)
                    continue
            elif kind == "search":
                if target_keys & (seen_read_keys | seen_write_keys):
                    seen_search_keys.update(target_keys)
                    seen_search_detail_keys.update(detail_keys)
                    dropped_indexes.append(index)
                    continue
                if detail_keys:
                    if detail_keys <= seen_search_detail_keys:
                        dropped_indexes.append(index)
                        continue
                elif target_keys & (seen_search_keys | seen_list_keys):
                    dropped_indexes.append(index)
                    continue
            elif kind == "list":
                if target_keys & (seen_read_keys | seen_write_keys):
                    seen_list_keys.update(target_keys)
                    seen_list_detail_keys.update(detail_keys)
                    dropped_indexes.append(index)
                    continue
                if detail_keys:
                    if detail_keys <= seen_list_detail_keys:
                        dropped_indexes.append(index)
                        continue
                elif target_keys & seen_list_keys:
                    dropped_indexes.append(index)
                    continue
            elif kind == "write":
                if target_keys & seen_write_keys:
                    dropped_indexes.append(index)
                    continue
            elif kind == "test":
                if (target_keys & seen_test_keys) or (target_keys & seen_write_keys):
                    dropped_indexes.append(index)
                    continue

            if kind == "read":
                seen_read_keys.update(target_keys)
            if kind == "search":
                seen_search_keys.update(target_keys)
                seen_search_detail_keys.update(detail_keys)
            if kind == "list":
                seen_list_keys.update(target_keys)
                seen_list_detail_keys.update(detail_keys)
            if kind == "write":
                seen_write_keys.update(target_keys)
            if kind == "test":
                seen_test_keys.update(target_keys)

        kept_reversed.append(message)

    kept_reversed.reverse()
    dropped_indexes.sort()
    return HistoryPruneResult(kept_messages=kept_reversed, dropped_indexes=dropped_indexes)


def _is_low_value_meta_confirmation(message: ChatMessage, text: str) -> bool:
    if message.role != "assistant":
        return False
    if ERROR_RE.search(text) or QUESTION_RE.search(text):
        return False
    if contains_workflow_keyword(text):
        return False
    return bool(META_CONFIRM_RE.match(text))


def _is_low_value_execution_filler(message: ChatMessage, text: str) -> bool:
    if message.role != "assistant":
        return False
    if ERROR_RE.search(text) or QUESTION_RE.search(text):
        return False
    if ARTIFACT_RE.search(text) or is_code_heavy_text(text):
        return False
    if _operation_signature(message, text) is not None:
        return False
    return bool(LOW_VALUE_EXECUTION_FILLER_RE.match(text))


def _operation_signature(message: ChatMessage, text: str) -> OperationSignature | None:
    if message.role not in {"assistant", "tool"}:
        return None
    if ERROR_RE.search(text) or QUESTION_RE.search(text):
        return None

    target_keys = _extract_target_keys(text)
    if WRITE_OP_RE.search(text) or PATCH_TARGET_RE.search(text) or PATCH_MOVE_RE.search(text):
        return OperationSignature("write", target_keys)
    if TEST_OP_RE.search(text):
        return OperationSignature("test", target_keys or _target_keys(_normalize_target(text)))
    if VALIDATION_RUN_OP_RE.search(text):
        run_target = _extract_validation_run_target(text)
        return OperationSignature(
            "test",
            target_keys or _target_keys(run_target or _normalize_target(text)),
        )
    if SEARCH_OP_RE.search(text):
        return OperationSignature("search", target_keys or _target_keys(_normalize_target(text)))
    if LIST_OP_RE.search(text):
        return OperationSignature("list", target_keys or _target_keys(_normalize_target(text)))
    if READ_OP_RE.search(text):
        return OperationSignature("read", target_keys or _target_keys(_normalize_target(text)))
    if message.role == "tool":
        heuristic = _heuristic_tool_output_signature(text)
        if heuristic is not None:
            return heuristic
    return None


def _extract_target_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for matched in PATCH_TARGET_RE.findall(text):
        keys.update(_target_keys(matched))
    for matched in PATCH_MOVE_RE.findall(text):
        keys.update(_target_keys(matched))
    if keys:
        return keys

    for matched in ARTIFACT_RE.findall(text):
        keys.update(_target_keys(matched))
    if keys:
        return keys

    for pattern in (
        re.compile(r"(?:for|关于|针对|围绕|处理)\s+([A-Za-z0-9_./-]+|[\u4e00-\u9fff]{2,16})", re.IGNORECASE),
        re.compile(r"测试通过(?:了)?[:：]?\s*([A-Za-z0-9_./-]+|[\u4e00-\u9fff]{2,16})", re.IGNORECASE),
        re.compile(r"(?:搜索|检索|查找|rg|grep|find)\s+([^\n，。；;]+)", re.IGNORECASE),
        re.compile(r"(?:列出|ls|tree|list)\s+([^\n，。；;]+)", re.IGNORECASE),
    ):
        matched = pattern.search(text)
        if matched:
            keys.update(_target_keys(matched.group(1)))
    verification_target = _extract_verification_target(text)
    if verification_target:
        keys.update(_target_keys(verification_target))
    return keys


def _normalize_target(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip("`'\" ，。；;:")
    return normalized


def _extract_verification_target(text: str) -> str | None:
    patterns = (
        r"(?:测试|校验|验证|验收|回归|联调|灰度|巡检|复核)(?:通过|完成)(?:了)?[:：]?\s*(.+)$",
        r"^(?:现在|目前|刚刚|刚才)?(.+?)(?:已经|已)?通过(?:了)?(?:测试|校验|验证|验收|回归|联调|灰度|巡检|复核)$",
    )
    for pattern in patterns:
        matched = re.search(pattern, text, re.IGNORECASE)
        if not matched:
            continue
        subject = _normalize_target(matched.group(1))
        if subject:
            return subject
    for pattern, require_hint in (
        (r"^(?:现在|目前|刚刚|刚才)?(.+?)(?:已经|已)?(?:通过了|通过)$", False),
        (r"^(?:现在|目前|刚刚|刚才)?(.+?)(?:已经|已)?(?:完成了|完成)$", True),
    ):
        matched = re.search(pattern, text, re.IGNORECASE)
        if not matched:
            continue
        subject = _normalize_target(matched.group(1))
        if not subject:
            continue
        if require_hint and not VERIFICATION_HINT_RE.search(subject):
            continue
        if not require_hint and not (VERIFICATION_HINT_RE.search(subject) or "通过" in text):
            continue
        return subject
    return None


def _extract_validation_run_target(text: str) -> str | None:
    patterns = (
        r"(?:再|重新|再次|继续)?(?:跑|做|执行|进行)(?:一(?:次|遍|轮)|一下)?\s*([^，。；;\n]+?(?:测试|校验|验证|验收|回归|灰度联调|灰度|联调|巡检|复核))",
        r"(?:再|重新|再次|继续)?(?:验(?!证)|测(?!试)|跑)(?:一(?:下|遍|轮)|一下|一遍|一轮)\s*([^，。；;\n]+?(?:测试|校验|验证|验收|回归|灰度联调|灰度|联调|巡检|复核))",
        r"(?:再|重新|再次|继续|先)?(?:补|确认|过)(?:一(?:下|遍|轮)|一下|一遍|一轮)\s*([^，。；;\n]+?(?:测试|校验|验证|验收|回归|灰度联调|灰度|联调|巡检|复核))",
        r"(?:再|重新|再次|继续)?(?:测试|校验|验证|验收|回归|联调|灰度|巡检|复核)\s+([^，。；;\n]+)$",
    )
    for pattern in patterns:
        matched = re.search(pattern, text, re.IGNORECASE)
        if not matched:
            continue
        subject = _normalize_target(matched.group(1))
        if subject:
            return subject
    return None


def _target_keys(target: str) -> set[str]:
    normalized = _normalize_target(target)
    if not normalized:
        return set()

    keys = {normalized.casefold()}
    if "/" in normalized:
        basename = normalized.rstrip("/").split("/")[-1]
        if basename:
            keys.add(basename.casefold())
        keys.update(_path_hierarchy_keys(normalized))
    else:
        keys.update(_identifier_keys(normalized))
    return keys


def _path_hierarchy_keys(path: str) -> set[str]:
    normalized = _normalize_target(path)
    if "/" not in normalized:
        return set()

    keys: set[str] = set()
    stripped = normalized.rstrip("/")
    parts = [part for part in stripped.split("/") if part]
    if not parts:
        return keys

    for depth in range(2, min(len(parts), 4) + 1):
        keys.add("/".join(parts[-depth:]).casefold())

    parent_parts = parts if normalized.endswith("/") else parts[:-1]
    for depth in range(1, min(len(parent_parts), 4) + 1):
        keys.add("/".join(parent_parts[-depth:]).casefold())
    return keys


def _identifier_keys(text: str) -> set[str]:
    return {
        matched.group(0).casefold()
        for matched in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text)
    }


def _heuristic_tool_output_signature(text: str) -> OperationSignature | None:
    search_detail_keys = _search_result_detail_keys(text)
    if search_detail_keys:
        search_paths = [detail.split(":", 1)[0] for detail in search_detail_keys]
        search_target_keys = _path_collection_keys(search_paths)
        for detail in search_detail_keys:
            search_target_keys.update(_identifier_keys(detail))
        return OperationSignature(
            "search",
            search_target_keys,
            detail_keys=search_detail_keys,
        )

    listing_paths = _listing_paths(text)
    if len(listing_paths) >= 2:
        return OperationSignature(
            "list",
            _path_collection_keys(listing_paths),
            detail_keys={_normalize_target(path).casefold() for path in listing_paths},
        )
    return None


def _search_result_detail_keys(text: str) -> set[str]:
    detail_keys: set[str] = set()
    for line in text.splitlines():
        normalized_line = line.strip()
        if not normalized_line:
            continue
        if not SEARCH_RESULT_LINE_RE.match(normalized_line):
            continue
        detail_keys.add(normalized_line.casefold())
    return detail_keys


def _listing_paths(text: str) -> list[str]:
    paths: list[str] = []
    for matched in LISTING_PATH_LINE_RE.finditer(text):
        candidate = matched.group("path")
        if not candidate or ":" in candidate:
            continue
        paths.append(candidate)
    return paths


def _path_collection_keys(paths: list[str]) -> set[str]:
    keys: set[str] = set()
    normalized_paths = [_normalize_target(path) for path in paths if _normalize_target(path)]
    for path in normalized_paths:
        keys.update(_target_keys(path))

    common_parent = _common_parent_path(normalized_paths)
    if common_parent:
        keys.update(_target_keys(common_parent))
    return keys


def _common_parent_path(paths: list[str]) -> str:
    if len(paths) < 2:
        return ""
    split_paths = [path.rstrip("/").split("/") for path in paths if path]
    if len(split_paths) < 2:
        return ""

    shared: list[str] = []
    for parts in zip(*split_paths):
        if len(set(parts)) != 1:
            break
        shared.append(parts[0])
    if not shared:
        return ""
    return "/".join(shared)
