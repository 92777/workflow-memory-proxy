from __future__ import annotations

import re
from typing import Iterable

from .models import EventAction, EventType, MemoryEvent, WorkingMemory


TYPE_TO_DSL = {
    EventType.GOAL: "GOAL",
    EventType.STATE: "STATE",
    EventType.CONSTRAINT: "CONS",
    EventType.PLAN: "PLAN",
    EventType.TASK: "TASK",
    EventType.DECISION: "DEC",
    EventType.ARTIFACT: "ART",
    EventType.OBSERVATION: "OBS",
    EventType.QUESTION: "QUES",
}

DSL_TO_TYPE = {value: key for key, value in TYPE_TO_DSL.items()}


def escape_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|")


def unescape_value(value: str) -> str:
    value = value.replace("\\|", "\x00")
    value = value.replace("\\\\", "\\")
    return value.replace("\x00", "|")


def event_to_dsl(event: MemoryEvent) -> str:
    type_code = TYPE_TO_DSL[event.type]
    parts = [f"{type_code}[{event.event_id}]: {escape_value(event.subject)}"]
    attrs = {
        "actor": event.actor,
        "status": event.status,
        "action": event.action,
        "confidence": f"{event.confidence:.2f}",
    }
    if event.supersedes:
        attrs["supersedes"] = event.supersedes
    for key, value in event.details.items():
        attrs[str(key)] = str(value)
    for key, value in attrs.items():
        parts.append(f"{key}={escape_value(value)}")
    return " | ".join(parts)


def events_to_dsl(events: Iterable[MemoryEvent]) -> str:
    return "\n".join(event_to_dsl(event) for event in events)


def parse_dsl_line(line: str) -> dict[str, object]:
    line = line.strip()
    matched = re.match(r"^(?P<type>[A-Z]+)\[(?P<event_id>[^\]]+)\]:\s*(?P<body>.+)$", line)
    if not matched:
        raise ValueError(f"Invalid DSL line: {line}")

    parts = _split_escaped_fields(matched.group("body"))
    subject = unescape_value(parts[0].strip())
    attrs: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"Invalid DSL attribute: {part}")
        key, value = part.split("=", 1)
        attrs[key.strip()] = unescape_value(value.strip())

    return {
        "type": DSL_TO_TYPE[matched.group("type")],
        "event_id": matched.group("event_id"),
        "subject": subject,
        "attrs": attrs,
    }


def working_memory_to_dsl(
    memory: WorkingMemory,
    *,
    include_artifacts: bool = False,
    include_observations: bool = False,
) -> str:
    lines: list[str] = []
    if memory.primary_goal:
        lines.append(f"GOAL[goal_1]: {escape_value(memory.primary_goal)} | status=active")
    for index, (slot, value) in enumerate(memory.state_facts.items(), start=1):
        lines.append(f"STATE[s_{index}]: {escape_value(f'{slot}={value}')} | status=active")
    for index, item in enumerate(memory.active_constraints, start=1):
        lines.append(f"CONS[c_{index}]: {escape_value(item)} | status=active")
    for index, item in enumerate(memory.active_decisions, start=1):
        lines.append(f"DEC[d_{index}]: {escape_value(item)} | status=active")
    for index, item in enumerate(memory.current_plan, start=1):
        lines.append(f"PLAN[p_{index}]: {escape_value(item)} | status=active")
    for index, item in enumerate(memory.open_tasks, start=1):
        lines.append(f"TASK[t_open_{index}]: {escape_value(item)} | status=in_progress")
    for index, item in enumerate(memory.pending_verification_tasks, start=1):
        lines.append(f"TASK[t_pending_{index}]: {escape_value(item)} | status=claimed_done")
    for index, item in enumerate(memory.completed_tasks, start=1):
        lines.append(f"TASK[t_done_{index}]: {escape_value(item)} | status=verified_done")
    if include_artifacts:
        for index, item in enumerate(memory.active_artifacts, start=1):
            lines.append(f"ART[a_{index}]: {escape_value(item)} | status=active")
    if include_observations:
        for index, item in enumerate(memory.key_observations, start=1):
            lines.append(f"OBS[o_{index}]: {escape_value(item)} | status=active")
    for index, item in enumerate(memory.open_questions, start=1):
        lines.append(f"QUES[q_{index}]: {escape_value(item)} | status=open")
    return "\n".join(lines)


def _split_escaped_fields(body: str) -> list[str]:
    fields: list[str] = []
    current: list[str] = []
    escaped = False
    for char in body:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == "|" and current[-1:] == [" "] and body:
            if current:
                current.pop()
            fields.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    fields.append("".join(current).strip())
    return fields
