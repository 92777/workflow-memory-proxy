from __future__ import annotations

from functools import lru_cache
import json
import re
from importlib import resources


def _resource_path() -> str:
    return str(resources.files("memory_proxy").joinpath("resources/zh_semantics.json"))


@lru_cache(maxsize=1)
def _load_semantics() -> dict[str, list[str]]:
    with open(_resource_path(), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: [str(item) for item in value] for key, value in payload.items() if isinstance(value, list)}


def _alternation(items: list[str]) -> str:
    return "|".join(sorted((re.escape(item) for item in items if item), key=len, reverse=True))


def _literal_list(key: str) -> list[str]:
    return list(_load_semantics().get(key, []))


GENERIC_TASK_SUBJECTS = frozenset(_literal_list("generic_task_subjects"))
GENERIC_TASK_AGGREGATE_MARKERS = tuple(_literal_list("generic_task_aggregate_markers"))


@lru_cache(maxsize=1)
def workflow_re() -> re.Pattern[str]:
    return re.compile(_alternation(_literal_list("workflow_keywords")), re.IGNORECASE)


@lru_cache(maxsize=1)
def task_management_re() -> re.Pattern[str]:
    return re.compile(_alternation(_literal_list("task_management_keywords")), re.IGNORECASE)


@lru_cache(maxsize=1)
def meta_confirmation_re() -> re.Pattern[str]:
    prefixes = _alternation(_literal_list("meta_confirmation_prefixes"))
    return re.compile(rf"^(?:{prefixes})[，,、 ]*", re.IGNORECASE)


@lru_cache(maxsize=1)
def low_value_execution_filler_re() -> re.Pattern[str]:
    prefixes = _alternation(_literal_list("meta_confirmation_prefixes"))
    subjects = _alternation(_literal_list("low_value_execution_subjects"))
    actors = _alternation(_literal_list("low_value_execution_actors"))
    modifiers = _alternation(_literal_list("low_value_execution_modifiers"))
    verbs = "|".join(_literal_list("low_value_execution_verb_patterns"))
    return re.compile(
        rf"^(?:(?:{prefixes})[，,、 ]*)?"
        rf"(?:(?:{subjects})[，,、 ]*)?"
        rf"(?:(?:{actors})[，,、 ]*)?"
        rf"(?:(?:{modifiers})\s*)?"
        rf"(?:(?:来)\s*)?"
        rf"(?:{verbs})(?:吧|哈|啊|哦|喔|呢)?[。！!]*$",
        re.IGNORECASE,
    )


def contains_workflow_keyword(text: str) -> bool:
    return bool(workflow_re().search(text))


def is_task_management_text(text: str) -> bool:
    return bool(task_management_re().search(text))


def is_generic_task_subject(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text.strip("`'\" ：:-，。；;"))
    if not normalized:
        return True
    if normalized in GENERIC_TASK_SUBJECTS:
        return True
    if normalized.endswith(("任务都", "条任务都")) and any(
        marker in normalized for marker in GENERIC_TASK_AGGREGATE_MARKERS
    ):
        return True
    return bool(
        re.match(r"^(?:全部|所有)?任务都?$", normalized, re.IGNORECASE)
        or re.match(
            r"^(?:你)?(?:目前|当前|最近)*(?:的)?(?:\d+|[一二三四五六七八九十两百]+)?条任务都?$",
            normalized,
            re.IGNORECASE,
        )
        or re.match(r"^(?:任务|条任务|最近任务)(?:都)?$", normalized, re.IGNORECASE)
    )
