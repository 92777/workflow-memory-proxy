from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from .dsl import events_to_dsl, working_memory_to_dsl
from .extractor import RuleBasedExtractor
from .models import Actor, EventAction, EventType, MemoryEvent, PromptMemory, RawMessage, WorkingMemory
from .prompt_builder import PromptMemoryBuilder, PromptMemoryConfig, prompt_memory_to_text
from .reducer import WorkingMemoryReducer
from .static_code import is_code_heavy_text
from .text_merge import are_texts_semantically_similar, contains_file_hint, prefer_richer_text

ERROR_SIGNAL_TOKENS = ("报错", "失败", "error", "failed", "traceback", "exception")
PAIR_USER_PREFIX_RE = re.compile(r"^(?:我们|还是|继续|仍然|请继续|请|还要继续|还得继续)+")
PAIR_ASSISTANT_PREFIX_RE = re.compile(r"^(?:好的|明白|收到|可以|行|那我|我会继续|我会|继续|将继续|会继续)+")
QUESTION_TOKEN_RE = re.compile(r"[A-Za-z0-9_./+-]{2,}|[\u4e00-\u9fff]{2,}")
RECOMMENDATION_QUESTION_RE = re.compile(r"(推荐|建议|怎么选|哪个好|哪种更好|选哪个)")
COMPARISON_QUESTION_RE = re.compile(r"(区别|差别|不同|有什么区别|有什么不同)")
YES_NO_QUESTION_RE = re.compile(r"(是否|能否|可否|要不要|需不需要|适不适合)")
YES_NO_ANSWER_RE = re.compile(r"^(?:结论[:：])?(?:适合|不适合|可以|不可以|能|不能|建议|不建议)")
ANSWER_PREFIXES = (
    "我推荐",
    "推荐",
    "我建议",
    "建议",
    "可以",
    "行",
    "好的",
    "已经",
    "已",
    "这条已经",
    "这个已经",
    "结论",
    "适合",
    "不适合",
    "正式口径",
    "应纠正为",
)
QUESTION_KEYWORD_STOPWORDS = {
    "是否",
    "可以",
    "一下",
    "一个",
    "这个",
    "那个",
    "继续",
    "进行",
    "怎么",
    "什么",
    "问题",
    "对话",
    "引用",
}


class MemoryExtractor(Protocol):
    def extract(self, message: RawMessage, session_id: str, turn_id: str) -> list[MemoryEvent]:
        ...


@dataclass(slots=True)
class CompressionResult:
    recent_messages: list[RawMessage]
    events: list[MemoryEvent]
    event_dsl: str
    memory_dsl: str
    audit_memory_dsl: str
    prompt_memory: PromptMemory
    working_memory: WorkingMemory


class MemoryCompressor:
    def __init__(
        self,
        recent_window: int = 4,
        extractor: MemoryExtractor | None = None,
        prompt_config: PromptMemoryConfig | None = None,
    ) -> None:
        self.recent_window = recent_window
        self.extractor = extractor or RuleBasedExtractor()
        self.reducer = WorkingMemoryReducer()
        self.prompt_builder = PromptMemoryBuilder(prompt_config)

    def compress(
        self,
        messages: list[RawMessage],
        session_id: str = "sess_local",
        *,
        prompt_exclude_message_ids: set[str] | None = None,
    ) -> CompressionResult:
        events: list[MemoryEvent] = []
        history_events: list[MemoryEvent] = []
        excluded = prompt_exclude_message_ids or set()
        history_limit = max(0, len(messages) - self.recent_window)
        kept_history_ids = {
            message.message_id
            for message in collapse_redundant_history_messages(
                messages[:history_limit],
                protected_message_ids=excluded,
            )
        }
        for index, message in enumerate(messages, start=1):
            if index <= history_limit and message.message_id not in kept_history_ids:
                continue
            turn_id = f"turn_{index:04d}"
            extracted = self.extractor.extract(message, session_id=session_id, turn_id=turn_id)
            events.extend(extracted)
            if index <= history_limit and message.message_id not in excluded:
                history_events.extend(extracted)

        question_resolution_events = _build_question_resolution_events(
            messages,
            session_id=session_id,
            history_limit=history_limit,
            excluded_message_ids=excluded,
        )
        events.extend(question_resolution_events)
        history_events.extend(
            event
            for event in question_resolution_events
            if _question_resolution_belongs_to_history(event, messages, history_limit, excluded)
        )

        working_memory = self.reducer.reduce(events)
        history_memory = self.reducer.reduce(history_events)
        prompt_memory = self.prompt_builder.build(history_memory)
        recent_messages = messages[-self.recent_window :]
        return CompressionResult(
            recent_messages=recent_messages,
            events=events,
            event_dsl=events_to_dsl(events),
            memory_dsl=prompt_memory_to_text(prompt_memory),
            audit_memory_dsl=working_memory_to_dsl(
                working_memory,
                include_artifacts=True,
                include_observations=True,
            ),
            prompt_memory=prompt_memory,
            working_memory=working_memory,
        )

def collapse_redundant_history_messages(
    messages: list[RawMessage],
    *,
    protected_message_ids: set[str] | None = None,
) -> list[RawMessage]:
    protected = protected_message_ids or set()
    collapsed_same_role: list[RawMessage] = []
    for message in messages:
        if not collapsed_same_role:
            collapsed_same_role.append(message)
            continue
        previous = collapsed_same_role[-1]
        if not _should_merge_history_messages(previous, message):
            collapsed_same_role.append(message)
            continue
        chosen = _choose_message_variant(previous, message, protected_message_ids=protected)
        if chosen is previous:
            continue
        collapsed_same_role[-1] = chosen

    collapsed_pairs: list[RawMessage] = []
    for message in collapsed_same_role:
        if not collapsed_pairs:
            collapsed_pairs.append(message)
            continue
        collapsed_pairs.append(message)
        if len(collapsed_pairs) < 4:
            continue
        previous_pair = collapsed_pairs[-4:-2]
        current_pair = collapsed_pairs[-2:]
        if not _should_merge_history_turn_pair(previous_pair, current_pair):
            continue
        chosen_pair = _choose_turn_pair_variant(
            previous_pair,
            current_pair,
            protected_message_ids=protected,
        )
        collapsed_pairs = collapsed_pairs[:-4] + chosen_pair
    return collapsed_pairs


def _should_merge_history_messages(previous: RawMessage, current: RawMessage) -> bool:
    if previous.role != current.role:
        return False
    if previous.role == "tool":
        return False
    if not previous.content.strip() or not current.content.strip():
        return False
    if _history_message_signature(previous.content) != _history_message_signature(current.content):
        return False
    return are_texts_semantically_similar(previous.content, current.content)


def _should_merge_history_turn_pair(previous_pair: list[RawMessage], current_pair: list[RawMessage]) -> bool:
    if len(previous_pair) != 2 or len(current_pair) != 2:
        return False
    previous_first, previous_second = previous_pair
    current_first, current_second = current_pair
    if previous_first.role != current_first.role or previous_second.role != current_second.role:
        return False
    if previous_first.role == previous_second.role:
        return False
    if {previous_first.role, previous_second.role} != {"user", "assistant"}:
        return False
    return _turn_pair_messages_similar(previous_first, current_first) and _turn_pair_messages_similar(
        previous_second,
        current_second,
    )


def _choose_message_variant(
    previous: RawMessage,
    current: RawMessage,
    *,
    protected_message_ids: set[str],
) -> RawMessage:
    previous_protected = previous.message_id in protected_message_ids
    current_protected = current.message_id in protected_message_ids
    if previous_protected and current_protected:
        return current
    if previous_protected:
        return previous
    if current_protected:
        return current
    preferred = prefer_richer_text(previous.content, current.content)
    if preferred == current.content:
        return current
    return previous


def _choose_turn_pair_variant(
    previous_pair: list[RawMessage],
    current_pair: list[RawMessage],
    *,
    protected_message_ids: set[str],
) -> list[RawMessage]:
    previous_protected = any(message.message_id in protected_message_ids for message in previous_pair)
    current_protected = any(message.message_id in protected_message_ids for message in current_pair)
    if previous_protected and current_protected:
        return previous_pair + current_pair
    if previous_protected:
        return previous_pair
    if current_protected:
        return current_pair

    previous_score = sum(_history_information_score(message.content) for message in previous_pair)
    current_score = sum(_history_information_score(message.content) for message in current_pair)
    if current_score >= previous_score:
        return current_pair
    return previous_pair


def _history_message_signature(text: str) -> tuple[bool, bool, bool, bool]:
    stripped = text.strip()
    lower = stripped.casefold()
    has_question = stripped.endswith(("?", "？"))
    has_error = any(token in lower for token in ERROR_SIGNAL_TOKENS)
    has_file = contains_file_hint(stripped)
    has_code = is_code_heavy_text(stripped)
    return has_question, has_error, has_file, has_code


def _history_information_score(text: str) -> int:
    normalized = re.sub(r"\s+", "", text)
    score = len(normalized)
    if contains_file_hint(text):
        score += 24
    if is_code_heavy_text(text):
        score += 32
    if any(token in text.casefold() for token in ERROR_SIGNAL_TOKENS):
        score += 18
    if text.strip().endswith(("?", "？")):
        score += 10
    return score


def _turn_pair_messages_similar(previous: RawMessage, current: RawMessage) -> bool:
    if previous.role != current.role:
        return False
    if previous.role == "tool":
        return False
    if _history_message_signature(previous.content) != _history_message_signature(current.content):
        return False
    if are_texts_semantically_similar(previous.content, current.content):
        return True
    previous_norm = _normalize_for_turn_pair_merge(previous.content, previous.role)
    current_norm = _normalize_for_turn_pair_merge(current.content, current.role)
    if not previous_norm or not current_norm:
        return False
    short, long = sorted((previous_norm, current_norm), key=len)
    return len(short) >= 6 and short in long


def _normalize_for_turn_pair_merge(text: str, role: str) -> str:
    normalized = re.sub(r"[\s`'\"“”‘’(),，。；：！？!\[\]{}<>]+", "", text.casefold())
    if role == "user":
        return PAIR_USER_PREFIX_RE.sub("", normalized)
    if role == "assistant":
        return PAIR_ASSISTANT_PREFIX_RE.sub("", normalized)
    return normalized


def _build_question_resolution_events(
    messages: list[RawMessage],
    *,
    session_id: str,
    history_limit: int,
    excluded_message_ids: set[str],
) -> list[MemoryEvent]:
    events: list[MemoryEvent] = []
    for index in range(len(messages) - 1):
        question_message = messages[index]
        answer_message = messages[index + 1]
        if question_message.role not in {Actor.USER, Actor.SYSTEM}:
            continue
        if answer_message.role != Actor.ASSISTANT:
            continue
        question_text = question_message.content.strip()
        answer_text = answer_message.content.strip()
        if not _looks_like_question_text(question_text):
            continue
        if not _assistant_answer_matches_question(question_text, answer_text):
            continue
        if question_message.message_id in excluded_message_ids:
            continue
        events.append(
            MemoryEvent(
                event_id=f"evt_qresolve_{index + 1:04d}",
                session_id=session_id,
                turn_id=f"turn_{index + 2:04d}",
                source_message_ids=[question_message.message_id, answer_message.message_id],
                actor=Actor.ASSISTANT,
                type=EventType.QUESTION,
                action=EventAction.RESOLVE,
                status="answered",
                subject=question_text,
                confidence=0.86,
                details={"source": "adjacent_qa_resolution"},
            )
        )
    return events


def _question_resolution_belongs_to_history(
    event: MemoryEvent,
    messages: list[RawMessage],
    history_limit: int,
    excluded_message_ids: set[str],
) -> bool:
    if history_limit <= 0 or not event.source_message_ids:
        return False
    question_id = event.source_message_ids[0]
    for index, message in enumerate(messages[:history_limit], start=1):
        if message.message_id == question_id:
            return message.message_id not in excluded_message_ids
    return False


def _looks_like_question_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.endswith(("?", "？")) or any(token in stripped for token in ("是否", "要不要", "需不需要"))


def _assistant_answer_matches_question(question_text: str, answer_text: str) -> bool:
    if not answer_text:
        return False
    if _looks_like_recommendation_question(question_text) and answer_text.startswith(ANSWER_PREFIXES):
        return True
    if _looks_like_comparison_question(question_text) and _looks_like_comparison_answer(answer_text):
        return True
    if _looks_like_yes_no_question(question_text) and YES_NO_ANSWER_RE.match(answer_text.strip()):
        return True
    if not answer_text.startswith(ANSWER_PREFIXES):
        return False
    question_keywords = _extract_question_keywords(question_text)
    answer_keywords = _extract_question_keywords(answer_text)
    return bool(question_keywords & answer_keywords)


def _looks_like_recommendation_question(text: str) -> bool:
    return bool(RECOMMENDATION_QUESTION_RE.search(text))


def _looks_like_comparison_question(text: str) -> bool:
    return bool(COMPARISON_QUESTION_RE.search(text))


def _looks_like_yes_no_question(text: str) -> bool:
    return bool(YES_NO_QUESTION_RE.search(text))


def _looks_like_comparison_answer(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 18:
        return False
    return any(token in stripped for token in ("区别", "不同", "前者", "后者", "一个", "另一", "追求", "而"))


def _extract_question_keywords(text: str) -> set[str]:
    keywords: set[str] = set()
    for token in QUESTION_TOKEN_RE.findall(text.casefold()):
        normalized = token.strip("`'\".,;:!?()[]{}<>")
        if len(normalized) < 2 or normalized in QUESTION_KEYWORD_STOPWORDS:
            continue
        keywords.add(normalized)
    return keywords
