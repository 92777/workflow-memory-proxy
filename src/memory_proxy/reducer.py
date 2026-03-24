from __future__ import annotations

from copy import deepcopy
import re

from .models import EventType, MemoryEvent, WorkingMemory
from .text_merge import append_semantic_unique, are_texts_semantically_similar, prefer_richer_text, remove_semantic_value

TASK_ARTIFACT_PATTERN = re.compile(r"(?:/[\w./-]+)|(?:\b[\w./-]+\.[A-Za-z0-9]+\b)")


class WorkingMemoryReducer:
    def reduce(self, events: list[MemoryEvent]) -> WorkingMemory:
        memory = WorkingMemory()
        for event in events:
            self._apply(memory, event)
        return memory

    def reduce_from(self, base_memory: WorkingMemory, events: list[MemoryEvent]) -> WorkingMemory:
        memory = deepcopy(base_memory)
        for event in events:
            self._apply(memory, event)
        return memory

    def _apply(self, memory: WorkingMemory, event: MemoryEvent) -> None:
        if event.type == EventType.GOAL:
            if event.status == "active":
                if memory.primary_goal and are_texts_semantically_similar(memory.primary_goal, event.subject):
                    memory.primary_goal = prefer_richer_text(memory.primary_goal, event.subject)
                else:
                    memory.primary_goal = event.subject
            elif event.status in {"resolved", "superseded", "invalidated", "stale"}:
                if memory.primary_goal and are_texts_semantically_similar(memory.primary_goal, event.subject):
                    memory.primary_goal = None
            return

        if event.type == EventType.STATE:
            slot = _state_event_slot(event)
            if not slot:
                return
            if event.status == "active":
                value = _state_event_value(event)
                if not value:
                    return
                if slot in memory.state_facts:
                    memory.state_facts.pop(slot, None)
                memory.state_facts[slot] = value
            elif event.status in {"resolved", "superseded", "invalidated", "stale"}:
                memory.state_facts.pop(slot, None)
            return

        if event.type == EventType.CONSTRAINT:
            if event.status == "active":
                append_semantic_unique(memory.active_constraints, event.subject)
            elif event.status in {"resolved", "superseded", "invalidated", "stale"}:
                remove_semantic_value(memory.active_constraints, event.subject)
            return

        if event.type == EventType.DECISION:
            if event.status == "active":
                append_semantic_unique(memory.active_decisions, event.subject)
            elif event.status in {"resolved", "superseded", "invalidated", "stale"}:
                remove_semantic_value(memory.active_decisions, event.subject)
            return

        if event.type == EventType.PLAN:
            if event.status == "active":
                append_semantic_unique(memory.current_plan, event.subject)
            elif event.status in {"resolved", "superseded", "invalidated", "stale"}:
                remove_semantic_value(memory.current_plan, event.subject)
            return

        if event.type == EventType.ARTIFACT:
            if event.status == "active":
                _append_unique(memory.active_artifacts, event.subject)
            elif event.status in {"resolved", "superseded", "invalidated", "stale"}:
                _remove_value(memory.active_artifacts, event.subject)
            return

        if event.type == EventType.OBSERVATION:
            if event.status == "active":
                _append_unique(memory.key_observations, event.subject)
            elif event.status in {"resolved", "superseded", "invalidated", "stale"}:
                _remove_value(memory.key_observations, event.subject)
            return

        if event.type == EventType.QUESTION:
            if event.status == "open":
                append_semantic_unique(memory.open_questions, event.subject)
            elif event.status in {"answered", "resolved", "superseded", "invalidated", "stale"}:
                remove_semantic_value(memory.open_questions, event.subject)
            return

        if event.type != EventType.TASK:
            return

        if event.status in {"proposed", "in_progress", "blocked"}:
            _move_task_subject(event.subject, target=memory.open_tasks, others=[
                memory.pending_verification_tasks,
                memory.completed_tasks,
            ])
            return

        if event.status == "claimed_done":
            if any(_task_subject_matches(item, event.subject) for item in memory.completed_tasks):
                _move_task_subject(event.subject, target=memory.completed_tasks, others=[
                    memory.open_tasks,
                    memory.pending_verification_tasks,
                ])
                return
            _move_task_subject(event.subject, target=memory.pending_verification_tasks, others=[
                memory.open_tasks,
                memory.completed_tasks,
            ])
            return

        if event.status == "verified_done":
            _move_task_subject(event.subject, target=memory.completed_tasks, others=[
                memory.open_tasks,
                memory.pending_verification_tasks,
            ])
            return

        if event.status in {"resolved", "invalidated", "failed", "stale"}:
            canonical = _canonical_task_subject(event.subject)
            for items in (
                memory.open_tasks,
                memory.pending_verification_tasks,
                memory.completed_tasks,
            ):
                _remove_canonical_task(items, canonical)


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _remove_value(items: list[str], value: str) -> None:
    while value in items:
        items.remove(value)


def _move_subject(value: str, target: list[str], others: list[list[str]]) -> None:
    for items in others:
        _remove_value(items, value)
    _append_unique(target, value)


def _move_task_subject(value: str, target: list[str], others: list[list[str]]) -> None:
    merged_value = value
    for items in [target, *others]:
        for item in items:
            if _task_subject_matches(item, value):
                preferred = prefer_richer_text(item, value)
                if preferred:
                    merged_value = preferred
    canonical = _canonical_task_subject(merged_value)
    for items in others:
        _remove_canonical_task(items, canonical)
    _remove_canonical_task(target, canonical)
    target.append(canonical)


def _remove_canonical_task(items: list[str], canonical: str) -> None:
    remaining = [item for item in items if not _task_subject_matches(item, canonical)]
    items[:] = remaining


def _canonical_task_subject(subject: str) -> str:
    matched = TASK_ARTIFACT_PATTERN.search(subject)
    if matched:
        return matched.group(0)
    return subject


def _task_subject_matches(existing: str, incoming: str) -> bool:
    existing_canonical = _canonical_task_subject(existing)
    incoming_canonical = _canonical_task_subject(incoming)
    if existing_canonical == incoming_canonical:
        return True
    return are_texts_semantically_similar(existing_canonical, incoming_canonical)


def _state_event_slot(event: MemoryEvent) -> str:
    raw_slot = str(event.details.get("slot", "")).strip()
    if raw_slot:
        return raw_slot
    if "=" not in event.subject:
        return ""
    slot, _, _ = event.subject.partition("=")
    return slot.strip()


def _state_event_value(event: MemoryEvent) -> str:
    raw_value = str(event.details.get("value", "")).strip()
    if raw_value:
        return raw_value
    if "=" not in event.subject:
        return ""
    _, _, value = event.subject.partition("=")
    return value.strip()
