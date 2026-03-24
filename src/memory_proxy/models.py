from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Actor(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class EventType(StrEnum):
    GOAL = "goal"
    STATE = "state"
    CONSTRAINT = "constraint"
    PLAN = "plan"
    TASK = "task"
    DECISION = "decision"
    ARTIFACT = "artifact"
    OBSERVATION = "observation"
    QUESTION = "question"


class EventAction(StrEnum):
    ADD = "add"
    UPDATE = "update"
    RESOLVE = "resolve"
    INVALIDATE = "invalidate"
    VERIFY = "verify"


@dataclass(slots=True)
class RawMessage:
    message_id: str
    role: str
    content: str


@dataclass(slots=True)
class MemoryEvent:
    event_id: str
    session_id: str
    turn_id: str
    source_message_ids: list[str]
    actor: str
    type: str
    action: str
    status: str
    subject: str
    details: dict[str, str | int | float | bool] = field(default_factory=dict)
    confidence: float = 0.8
    supersedes: str | None = None


@dataclass(slots=True)
class WorkingMemory:
    primary_goal: str | None = None
    state_facts: dict[str, str] = field(default_factory=dict)
    active_constraints: list[str] = field(default_factory=list)
    active_decisions: list[str] = field(default_factory=list)
    current_plan: list[str] = field(default_factory=list)
    open_tasks: list[str] = field(default_factory=list)
    pending_verification_tasks: list[str] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    active_artifacts: list[str] = field(default_factory=list)
    key_observations: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PromptMemory:
    goal: str | None = None
    completed_tasks: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    state_facts: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    next_step: str | None = None
    open_tasks: list[str] = field(default_factory=list)
    pending_verification: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
