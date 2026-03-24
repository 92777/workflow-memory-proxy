from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal
from urllib.parse import urlsplit

from .models import PromptMemory, WorkingMemory
from .text_merge import (
    are_texts_semantically_similar,
    contains_file_hint,
    merge_semantic_list,
    prefer_richer_text,
)
from .tokens import approx_tokens
from .zh_semantics import is_task_management_text


@dataclass(slots=True)
class PromptMemoryConfig:
    max_tokens: int = 160
    max_item_tokens: int = 48
    max_completed_tasks: int = 1
    max_artifacts: int = 2
    max_state_facts: int = 6
    max_constraints: int = 4
    max_decisions: int = 2
    max_open_tasks: int = 2
    max_pending_verification: int = 1
    max_questions: int = 2


class PromptMemoryBuilder:
    def __init__(self, config: PromptMemoryConfig | None = None) -> None:
        self.config = config or PromptMemoryConfig()
        self.budgeter = PromptMemoryBudgeter(self.config)

    def build(self, memory: WorkingMemory) -> PromptMemory:
        completed_tasks = _take_tail(memory.completed_tasks, self.config.max_completed_tasks)
        promoted_pending_completion = False
        pending_completion = _take_tail(
            memory.pending_verification_tasks,
            self.config.max_completed_tasks,
        )
        if pending_completion:
            if not completed_tasks:
                completed_tasks = pending_completion
                promoted_pending_completion = True
            elif not are_texts_semantically_similar(completed_tasks[-1], pending_completion[-1]):
                if _should_promote_pending_completion(pending_completion[-1], completed_tasks[-1]):
                    completed_tasks = pending_completion
                    promoted_pending_completion = True
            else:
                completed_tasks = [prefer_richer_text(completed_tasks[-1], pending_completion[-1])]
        elif not completed_tasks:
            completed_tasks = _take_tail(
                memory.pending_verification_tasks,
                self.config.max_completed_tasks,
            )
            promoted_pending_completion = bool(completed_tasks)
        open_tasks = _take_tail(memory.open_tasks, self.config.max_open_tasks)
        next_step = None if open_tasks else _last_or_none(memory.current_plan)
        pending_verification = []
        if not open_tasks and not promoted_pending_completion:
            pending_verification = _take_tail(
                memory.pending_verification_tasks,
                self.config.max_pending_verification,
            )
        candidate = PromptMemory(
            goal=memory.primary_goal,
            completed_tasks=completed_tasks,
            artifacts=_select_artifact_items(memory, self.config.max_artifacts),
            state_facts=_select_state_fact_items(
                memory.state_facts,
                self.config.max_state_facts,
            ),
            constraints=_select_signal_items(
                memory.active_constraints,
                self.config.max_constraints,
                prefer_recent=False,
            ),
            decisions=_select_signal_items(
                memory.active_decisions,
                self.config.max_decisions,
                prefer_recent=True,
            ),
            next_step=next_step,
            open_tasks=open_tasks,
            pending_verification=pending_verification,
            open_questions=_select_signal_items(
                memory.open_questions,
                self.config.max_questions,
                prefer_recent=True,
            ),
        )
        return self.budgeter.fit(_compact_cross_field_redundancy(candidate))


@dataclass(frozen=True, slots=True)
class _PromptCandidate:
    field: Literal[
        "goal",
        "completed_tasks",
        "artifacts",
        "state_facts",
        "constraints",
        "decisions",
        "next_step",
        "open_tasks",
        "pending_verification",
        "open_questions",
    ]
    index: int
    value: str

    @property
    def key(self) -> tuple[str, int]:
        return self.field, self.index


class PromptMemoryBudgeter:
    def __init__(self, config: PromptMemoryConfig) -> None:
        self.config = config

    def fit(self, memory: PromptMemory) -> PromptMemory:
        normalized = _normalize_memory(memory, max_item_tokens=self.config.max_item_tokens)
        if self.config.max_tokens <= 0:
            return PromptMemory()
        if _prompt_tokens(normalized) <= self.config.max_tokens:
            return normalized

        selected: dict[tuple[str, int], str] = {}
        for candidate in _iter_candidates(normalized):
            fitted_value = self._fit_candidate(normalized, selected, candidate)
            if fitted_value is None:
                continue
            selected[candidate.key] = fitted_value
        return _materialize_memory(normalized, selected)

    def _fit_candidate(
        self,
        source: PromptMemory,
        selected: dict[tuple[str, int], str],
        candidate: _PromptCandidate,
    ) -> str | None:
        trial = dict(selected)
        trial[candidate.key] = candidate.value
        if _prompt_tokens(_materialize_memory(source, trial)) <= self.config.max_tokens:
            return candidate.value

        if len(candidate.value) <= _MIN_TRUNCATED_CHARS:
            return None

        best: str | None = None
        low = _MIN_TRUNCATED_CHARS
        high = len(candidate.value) - 1
        while low <= high:
            mid = (low + high) // 2
            truncated = _truncate_to_chars(candidate.value, mid)
            if not truncated:
                high = mid - 1
                continue
            trial[candidate.key] = truncated
            if _prompt_tokens(_materialize_memory(source, trial)) <= self.config.max_tokens:
                best = truncated
                low = mid + 1
            else:
                high = mid - 1
        return best


def prompt_memory_to_text(memory: PromptMemory) -> str:
    lines: list[str] = []
    if memory.goal:
        lines.append(f"GOAL: {memory.goal}")
    if memory.completed_tasks:
        lines.append(f"DONE: {' ; '.join(memory.completed_tasks)}")
    if memory.artifacts:
        lines.append(f"ART: {' ; '.join(memory.artifacts)}")
    if memory.open_tasks:
        lines.append(f"TODO: {' ; '.join(memory.open_tasks)}")
    if memory.pending_verification:
        lines.append(f"VERIFY: {' ; '.join(memory.pending_verification)}")
    if memory.next_step:
        lines.append(f"NEXT: {memory.next_step}")
    if memory.decisions:
        lines.append(f"DEC: {' ; '.join(memory.decisions)}")
    if memory.constraints:
        lines.append(f"CONS: {' ; '.join(memory.constraints)}")
    if memory.state_facts:
        lines.append(f"STATE: {' ; '.join(memory.state_facts)}")
    if memory.open_questions:
        lines.append(f"ASK: {' ; '.join(memory.open_questions)}")
    return "\n".join(lines)


def parse_prompt_memory_text(text: str) -> PromptMemory:
    memory = PromptMemory()
    if not isinstance(text, str):
        return memory

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        field, raw_value = line.split(":", 1)
        value = raw_value.strip()
        if not value:
            continue

        if field == "GOAL":
            if memory.goal and are_texts_semantically_similar(memory.goal, value):
                memory.goal = prefer_richer_text(memory.goal, value)
            else:
                memory.goal = value
            continue
        if field == "DONE":
            memory.completed_tasks = merge_semantic_list(memory.completed_tasks, _split_prompt_items(value))
            continue
        if field == "ART":
            memory.artifacts = merge_semantic_list(memory.artifacts, _split_prompt_items(value))
            continue
        if field == "STATE":
            memory.state_facts = _merge_state_fact_items(memory.state_facts, _split_prompt_items(value))
            continue
        if field == "CONS":
            memory.constraints = merge_semantic_list(memory.constraints, _split_prompt_items(value))
            continue
        if field == "DEC":
            memory.decisions = merge_semantic_list(memory.decisions, _split_prompt_items(value))
            continue
        if field == "NEXT":
            if memory.next_step and are_texts_semantically_similar(memory.next_step, value):
                memory.next_step = prefer_richer_text(memory.next_step, value)
            else:
                memory.next_step = value
            continue
        if field in {"TODO", "OPEN"}:
            memory.open_tasks = merge_semantic_list(memory.open_tasks, _split_prompt_items(value))
            continue
        if field == "VERIFY":
            memory.pending_verification = merge_semantic_list(
                memory.pending_verification,
                _split_prompt_items(value),
            )
            continue
        if field == "ASK":
            memory.open_questions = merge_semantic_list(memory.open_questions, _split_prompt_items(value))
    return memory


def merge_prompt_memories(memories: list[PromptMemory] | tuple[PromptMemory, ...]) -> PromptMemory:
    merged = PromptMemory()
    for memory in memories:
        if memory.goal:
            if merged.goal and are_texts_semantically_similar(merged.goal, memory.goal):
                merged.goal = prefer_richer_text(merged.goal, memory.goal)
            else:
                merged.goal = memory.goal
        merged.completed_tasks = merge_semantic_list(merged.completed_tasks, memory.completed_tasks)
        merged.artifacts = merge_semantic_list(merged.artifacts, memory.artifacts)
        merged.state_facts = _merge_state_fact_items(merged.state_facts, memory.state_facts)
        merged.constraints = merge_semantic_list(merged.constraints, memory.constraints)
        merged.decisions = merge_semantic_list(merged.decisions, memory.decisions)
        if memory.next_step:
            if merged.next_step and are_texts_semantically_similar(merged.next_step, memory.next_step):
                merged.next_step = prefer_richer_text(merged.next_step, memory.next_step)
            else:
                merged.next_step = memory.next_step
        merged.open_tasks = merge_semantic_list(merged.open_tasks, memory.open_tasks)
        merged.pending_verification = merge_semantic_list(
            merged.pending_verification,
            memory.pending_verification,
        )
        merged.open_questions = merge_semantic_list(merged.open_questions, memory.open_questions)
    return merged


def _take_head(items: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    return items[:limit]


def _take_tail(items: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    return items[-limit:]


def _last_or_none(items: list[str]) -> str | None:
    if not items:
        return None
    return items[-1]


def _should_promote_pending_completion(pending_task: str, completed_task: str) -> bool:
    pending_has_verify_hint = _looks_like_verification_milestone(pending_task)
    completed_has_verify_hint = _looks_like_verification_milestone(completed_task)
    if pending_has_verify_hint != completed_has_verify_hint:
        return pending_has_verify_hint
    return True


def _looks_like_verification_milestone(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.startswith(("灰度", "联调", "回归", "验收", "复核")):
        return True
    return bool(re.search(r"(测试|校验|验证|验收|回归|联调|灰度|复核)$", stripped))


_MIN_TRUNCATED_CHARS = 6
PROMPT_ERROR_RE = re.compile(r"(报错|失败|error|failed|traceback|exception)", re.IGNORECASE)
PROMPT_URL_RE = re.compile(r"https?://[^\s<>\")]+")
PROMPT_FILE_RE = re.compile(
    r"(?:/[\w./-]+|(?:\./|\.\./)+[\w./-]+|(?:\b[\w.-]+/[\w./-]+\.[A-Za-z0-9]+\b))"
)
RECOMMENDATION_QUESTION_RE = re.compile(r"(推荐|建议|怎么选|哪个好|哪种更好|选哪个)")
PROMPT_REF_TRAILING_CHARS = ".,;:!?)]}>\"'"
NEGATIVE_CONSTRAINT_RE = re.compile(
    r"^(?P<prefix>(?:(?:尽可能|尽量|最好|务必|必须|需要|得)\s*)*)(?P<neg>不要|不能)(?P<body>.+)$"
)
NEGATIVE_CONSTRAINT_ANCHOR_RE = re.compile(
    r"^(?P<anchor>用[^，。；;、]+?)(?P<suffix>(?:来做|来|进行|做).+)$"
)
TRANSIENT_CONSTRAINT_RE = re.compile(
    r"^(?:只(?:输出|回答|回复)|不要解释|每条都写成|引用至少\d+个当前已知事实|"
    r"回答不超过|用两行回答|用一句话|先给结论|再给\d+条理由|只基于本轮事实)",
    re.IGNORECASE,
)
TRANSIENT_QUESTION_RE = re.compile(
    r"^(?:回答|请回答|请列出|列出|给出|只输出|只回答|只回复|做(?:一个)?|反例检查|最后一次记忆核对)",
    re.IGNORECASE,
)


def _select_signal_items(
    items: list[str],
    limit: int,
    *,
    prefer_recent: bool,
) -> list[str]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)

    total = len(items)
    scored = sorted(
        (
            (
                _prompt_item_signal_score(item, index=index, total=total, prefer_recent=prefer_recent),
                index,
            )
            for index, item in enumerate(items)
            if item
        ),
        key=lambda item: (-item[0], -item[1]),
    )
    chosen_indexes = sorted(index for _, index in scored[:limit])
    return [items[index] for index in chosen_indexes]


def _prompt_item_signal_score(
    value: str,
    *,
    index: int,
    total: int,
    prefer_recent: bool,
) -> float:
    score = 0.0
    if contains_file_hint(value):
        score += 1.6
    if PROMPT_ERROR_RE.search(value):
        score += 1.8
    if value.endswith(("?", "？")):
        score += 0.8
    if len(value) >= 48:
        score += 0.4
    if any(char.isdigit() for char in value):
        score += 0.2
    if prefer_recent:
        denominator = max(1, total - 1)
        score += index / denominator * 0.6
    return round(score, 4)


def _compact_cross_field_redundancy(memory: PromptMemory) -> PromptMemory:
    goal = memory.goal
    completed_tasks = _remove_items_similar_to(
        memory.completed_tasks,
        blockers=[item for item in [*memory.open_tasks, *memory.pending_verification, memory.next_step] if item],
    )
    artifacts = _remove_items_similar_to(
        memory.artifacts,
        blockers=[item for item in [goal, *memory.open_tasks, memory.next_step] if item],
    )
    state_facts = _merge_state_fact_items([], memory.state_facts)
    open_tasks = _remove_items_similar_to(memory.open_tasks, blockers=[goal] if goal else [])
    pending_verification = _remove_items_similar_to(
        memory.pending_verification,
        blockers=[*open_tasks, goal] if goal else list(open_tasks),
    )

    next_step = memory.next_step
    if next_step and _matches_any(next_step, [*open_tasks, *pending_verification]):
        next_step = None

    decisions = _remove_items_similar_to(
        memory.decisions,
        blockers=[item for item in [goal, next_step, *open_tasks, *pending_verification] if item],
    )
    constraints = _remove_items_similar_to(
        memory.constraints,
        blockers=[item for item in [goal, next_step] if item],
    )
    constraints = _merge_related_constraints(constraints)
    constraints = [item for item in constraints if _is_persistent_constraint(item)]
    open_questions = _remove_items_similar_to(
        memory.open_questions,
        blockers=[item for item in [goal, next_step, *open_tasks, *pending_verification] if item],
    )
    open_questions = [item for item in open_questions if not _is_transient_question(item)]
    open_questions = _drop_questions_answered_by_decisions(open_questions, decisions)

    return PromptMemory(
        goal=goal,
        completed_tasks=completed_tasks,
        artifacts=artifacts,
        state_facts=state_facts,
        constraints=constraints,
        decisions=decisions,
        next_step=next_step,
        open_tasks=open_tasks,
        pending_verification=pending_verification,
        open_questions=open_questions,
    )


def _remove_items_similar_to(items: list[str], *, blockers: list[str]) -> list[str]:
    filtered: list[str] = []
    for item in items:
        if not item:
            continue
        if _matches_any(item, blockers):
            continue
        if _matches_any(item, filtered):
            existing_index = next(
                index for index, existing in enumerate(filtered) if are_texts_semantically_similar(existing, item)
            )
            preferred = prefer_richer_text(filtered[existing_index], item)
            if preferred:
                filtered[existing_index] = preferred
            continue
        filtered.append(item)
    return filtered


def _merge_related_constraints(items: list[str]) -> list[str]:
    merged: list[str] = []
    for item in items:
        if not item:
            continue
        merged_item = item
        combined = False
        for index, existing in enumerate(merged):
            candidate = _merge_negative_constraint_variants(existing, merged_item)
            if candidate is None:
                continue
            merged[index] = candidate
            combined = True
            break
        if not combined:
            merged.append(merged_item)
    return merged


def _select_state_fact_items(items: dict[str, str], limit: int) -> list[str]:
    if limit <= 0 or not items:
        return []
    pairs = [f"{slot}={value}" for slot, value in items.items() if slot and value]
    if len(pairs) <= limit:
        return pairs
    return pairs[-limit:]


def _select_artifact_items(memory: WorkingMemory, limit: int) -> list[str]:
    if limit <= 0 or not memory.active_artifacts:
        return []
    if _should_suppress_artifacts_for_context(memory):
        return []

    file_like = [item for item in memory.active_artifacts if item and contains_file_hint(item)]
    if not file_like:
        return []
    if len(file_like) <= limit:
        return list(file_like)

    scored = sorted(
        (
            (
                _artifact_signal_score(item, index=index, total=len(file_like)),
                index,
            )
            for index, item in enumerate(file_like)
        ),
        key=lambda item: (-item[0], -item[1]),
    )
    chosen_indexes = sorted(index for _, index in scored[:limit])
    return [file_like[index] for index in chosen_indexes]


def _should_suppress_artifacts_for_context(memory: WorkingMemory) -> bool:
    context_items = [
        memory.primary_goal or "",
        *memory.open_tasks,
        *memory.pending_verification_tasks,
        *memory.completed_tasks[-1:],
        *memory.current_plan[-1:],
        *memory.active_decisions[-2:],
        *memory.active_constraints[-2:],
        *memory.open_questions[-1:],
        *memory.state_facts.values(),
    ]
    if any(contains_file_hint(item) for item in context_items if item):
        return False

    context_text = " ".join(item for item in context_items if item)
    if not context_text:
        return False
    return is_task_management_text(context_text)


def _merge_state_fact_items(existing: list[str], incoming: list[str]) -> list[str]:
    merged: dict[str, str] = {}
    for item in [*existing, *incoming]:
        parsed = _parse_state_fact_item(item)
        if parsed is None:
            continue
        slot, value = parsed
        if slot in merged:
            merged.pop(slot, None)
        merged[slot] = f"{slot}={value}"
    return list(merged.values())


def _parse_state_fact_item(item: str) -> tuple[str, str] | None:
    if "=" not in item:
        return None
    slot, _, value = item.partition("=")
    slot = slot.strip()
    value = value.strip()
    if not slot or not value:
        return None
    return slot, value


def _merge_negative_constraint_variants(left: str, right: str) -> str | None:
    if are_texts_semantically_similar(left, right):
        preferred = prefer_richer_text(left, right)
        return preferred if preferred is not None else left

    left_parts = _parse_negative_constraint(left)
    right_parts = _parse_negative_constraint(right)
    if left_parts is None or right_parts is None:
        return None
    if left_parts["anchor_key"] != right_parts["anchor_key"]:
        return None
    if are_texts_semantically_similar(left_parts["suffix"], right_parts["suffix"]):
        preferred = prefer_richer_text(left, right)
        return preferred if preferred is not None else left

    prefix = prefer_richer_text(left_parts["prefix"], right_parts["prefix"]) or left_parts["prefix"]
    suffixes = _merge_negative_suffixes([left_parts["suffix"], right_parts["suffix"]])
    if len(suffixes) < 2:
        preferred = prefer_richer_text(left, right)
        return preferred if preferred is not None else left

    merged = f"{prefix}{left_parts['anchor_display']}{'或'.join(suffixes)}"
    if approx_tokens(merged) >= approx_tokens(left) + approx_tokens(right):
        return None
    return merged


def _parse_negative_constraint(text: str) -> dict[str, str] | None:
    matched = NEGATIVE_CONSTRAINT_RE.match(text.strip())
    if not matched:
        return None
    body = matched.group("body").strip()
    anchor_match = NEGATIVE_CONSTRAINT_ANCHOR_RE.match(body)
    if not anchor_match:
        return None

    anchor_display = re.sub(r"等+$", "", anchor_match.group("anchor").strip())
    suffix = anchor_match.group("suffix").strip()
    if not anchor_display or not suffix:
        return None

    prefix = f"{matched.group('prefix')}{matched.group('neg')}"
    anchor_key = re.sub(r"\s+", "", anchor_display)
    return {
        "prefix": prefix,
        "anchor_display": anchor_display,
        "anchor_key": anchor_key,
        "suffix": suffix,
    }


def _merge_negative_suffixes(suffixes: list[str]) -> list[str]:
    merged: list[str] = []
    for suffix in suffixes:
        if not suffix:
            continue
        if _matches_any(suffix, merged):
            existing_index = next(
                index for index, existing in enumerate(merged) if are_texts_semantically_similar(existing, suffix)
            )
            preferred = prefer_richer_text(merged[existing_index], suffix)
            if preferred:
                merged[existing_index] = preferred
            continue
        merged.append(suffix)
    return merged


def _drop_questions_answered_by_decisions(open_questions: list[str], decisions: list[str]) -> list[str]:
    if not decisions:
        return open_questions
    return [
        question
        for question in open_questions
        if not _looks_like_recommendation_question(question)
    ]


def _is_persistent_constraint(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return not TRANSIENT_CONSTRAINT_RE.match(stripped)


def _is_transient_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if TRANSIENT_QUESTION_RE.match(stripped):
        return True
    return not stripped.endswith(("?", "？")) and stripped.startswith(("回答", "请回答", "列出", "给出"))


def _looks_like_recommendation_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > 24:
        return False
    return bool(RECOMMENDATION_QUESTION_RE.search(stripped))


def _matches_any(value: str, candidates: list[str]) -> bool:
    return any(candidate and are_texts_semantically_similar(value, candidate) for candidate in candidates)


def _normalize_prompt_value(value: str | None, max_tokens: int) -> str | None:
    if value is None:
        return None
    shortened = _shorten_prompt_references(value)
    return _truncate_to_token_limit(shortened, max_tokens)


def _shorten_prompt_references(text: str) -> str:
    if not text:
        return text

    replacements: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []

    for match in PROMPT_URL_RE.finditer(text):
        start, end, value = _trim_reference_match(text, match.start(), match.end())
        if not value:
            continue
        shortened = _shorten_url_reference(value)
        if shortened == value or approx_tokens(shortened) >= approx_tokens(value):
            continue
        replacements.append((start, end, shortened))
        occupied.append((start, end))

    for match in PROMPT_FILE_RE.finditer(text):
        start, end, value = _trim_reference_match(text, match.start(), match.end())
        if not value or _overlaps_reference(start, end, occupied):
            continue
        if "://" in value:
            continue
        shortened = _shorten_file_reference(value)
        if shortened == value or approx_tokens(shortened) >= approx_tokens(value):
            continue
        replacements.append((start, end, shortened))
        occupied.append((start, end))

    if not replacements:
        return text

    parts: list[str] = []
    cursor = 0
    for start, end, replacement in sorted(replacements, key=lambda item: item[0]):
        parts.append(text[cursor:start])
        parts.append(replacement)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _trim_reference_match(text: str, start: int, end: int) -> tuple[int, int, str]:
    while end > start and text[end - 1] in PROMPT_REF_TRAILING_CHARS:
        end -= 1
    while start < end and text[start].isspace():
        start += 1
    return start, end, text[start:end]


def _overlaps_reference(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in occupied)


def _shorten_file_reference(value: str) -> str:
    if "/" not in value or len(value) < 24:
        return value

    prefix_match = re.match(r"^(?:(?:\.\./)+|\./|/)", value)
    prefix = prefix_match.group(0) if prefix_match else ""
    rest = value[len(prefix) :] if prefix else value
    segments = [segment for segment in rest.split("/") if segment]
    if len(segments) <= 2:
        return value
    tail = "/".join(segments[-2:])
    return f"{prefix}.../{tail}" if prefix else f".../{tail}"


def _shorten_url_reference(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value

    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) <= 2 and not parsed.query and not parsed.fragment:
        return value

    tail = "/".join(segments[-2:]) if len(segments) >= 2 else "/".join(segments)
    short_path = f"/.../{tail}" if tail else ""
    shortened = f"{parsed.scheme}://{parsed.netloc}{short_path}"
    if parsed.query:
        shortened += "?..."
    if parsed.fragment:
        shortened += "#..."
    return shortened


def _normalize_memory(memory: PromptMemory, *, max_item_tokens: int) -> PromptMemory:
    return PromptMemory(
        goal=_normalize_prompt_value(memory.goal, max_item_tokens),
        completed_tasks=_normalize_items(memory.completed_tasks, max_item_tokens),
        artifacts=_normalize_items(memory.artifacts, max_item_tokens),
        state_facts=_normalize_items(memory.state_facts, max_item_tokens),
        constraints=_normalize_items(memory.constraints, max_item_tokens),
        decisions=_normalize_items(memory.decisions, max_item_tokens),
        next_step=_normalize_prompt_value(memory.next_step, max_item_tokens),
        open_tasks=_normalize_items(memory.open_tasks, max_item_tokens),
        pending_verification=_normalize_items(
            memory.pending_verification,
            max_item_tokens,
        ),
        open_questions=_normalize_items(memory.open_questions, max_item_tokens),
    )


def _truncate_to_token_limit(value: str | None, max_tokens: int) -> str | None:
    if value is None:
        return None
    if max_tokens <= 0:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if approx_tokens(stripped) <= max_tokens:
        return stripped

    best: str | None = None
    low = 1
    high = len(stripped)
    while low <= high:
        mid = (low + high) // 2
        candidate = _truncate_to_chars(stripped, mid)
        if not candidate:
            high = mid - 1
            continue
        if approx_tokens(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _truncate_to_chars(value: str, char_limit: int) -> str:
    if char_limit <= 0:
        return ""
    stripped = value.strip()
    if len(stripped) <= char_limit:
        return stripped
    if char_limit <= 3:
        return stripped[:char_limit].strip()
    return f"{stripped[: char_limit - 3].rstrip()}..."


def _split_prompt_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(" ; ") if item.strip()]


def _normalize_items(items: list[str], max_item_tokens: int) -> list[str]:
    normalized: list[str] = []
    for item in items:
        value = _normalize_prompt_value(item, max_item_tokens)
        if value:
            normalized.append(value)
    return normalized


def _prompt_tokens(memory: PromptMemory) -> int:
    text = prompt_memory_to_text(memory)
    if not text:
        return 0
    return approx_tokens(text)


def _iter_candidates(memory: PromptMemory) -> list[_PromptCandidate]:
    candidates: list[_PromptCandidate] = []
    candidates.extend(
        _PromptCandidate("open_tasks", index, value)
        for index, value in reversed(list(enumerate(memory.open_tasks)))
        if value
    )
    candidates.extend(
        _PromptCandidate("pending_verification", index, value)
        for index, value in reversed(list(enumerate(memory.pending_verification)))
        if value
    )
    if memory.next_step:
        candidates.append(_PromptCandidate("next_step", 0, memory.next_step))
    candidates.extend(
        _PromptCandidate("completed_tasks", index, value)
        for index, value in reversed(list(enumerate(memory.completed_tasks)))
        if value
    )
    if memory.goal:
        candidates.append(_PromptCandidate("goal", 0, memory.goal))
    candidates.extend(
        _PromptCandidate("decisions", index, value)
        for index, value in reversed(list(enumerate(memory.decisions)))
        if value
    )
    candidates.extend(
        _PromptCandidate("constraints", index, value)
        for index, value in enumerate(memory.constraints)
        if value
    )
    candidates.extend(
        _PromptCandidate("artifacts", index, value)
        for index, value in reversed(list(enumerate(memory.artifacts)))
        if value
    )
    candidates.extend(
        _PromptCandidate("state_facts", index, value)
        for index, value in reversed(list(enumerate(memory.state_facts)))
        if value
    )
    candidates.extend(
        _PromptCandidate("open_questions", index, value)
        for index, value in reversed(list(enumerate(memory.open_questions)))
        if value
    )
    return candidates


def _materialize_memory(
    source: PromptMemory,
    selected: dict[tuple[str, int], str],
) -> PromptMemory:
    return PromptMemory(
        goal=selected.get(("goal", 0)),
        completed_tasks=_selected_list(source.completed_tasks, selected, "completed_tasks"),
        artifacts=_selected_list(source.artifacts, selected, "artifacts"),
        state_facts=_selected_list(source.state_facts, selected, "state_facts"),
        constraints=_selected_list(source.constraints, selected, "constraints"),
        decisions=_selected_list(source.decisions, selected, "decisions"),
        next_step=selected.get(("next_step", 0)),
        open_tasks=_selected_list(source.open_tasks, selected, "open_tasks"),
        pending_verification=_selected_list(
            source.pending_verification,
            selected,
            "pending_verification",
        ),
        open_questions=_selected_list(source.open_questions, selected, "open_questions"),
    )


def _selected_list(
    items: list[str],
    selected: dict[tuple[str, int], str],
    field: str,
) -> list[str]:
    return [
        selected[(field, index)]
        for index in range(len(items))
        if (field, index) in selected
    ]


def _artifact_signal_score(value: str, *, index: int, total: int) -> float:
    lower = value.casefold()
    score = 0.0
    if contains_file_hint(value):
        score += 1.6
    if lower.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java")):
        score += 1.4
    elif lower.endswith((".md", ".sql", ".yaml", ".yml", ".json", ".toml")):
        score += 1.1
    if any(token in lower for token in ("acceptance", "runbook", "migration", "schema", "prompt")):
        score += 0.6
    denominator = max(1, total - 1)
    score += index / denominator * 0.7
    return round(score, 4)
