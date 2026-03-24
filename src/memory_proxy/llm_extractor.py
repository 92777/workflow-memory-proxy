from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from typing import Protocol

from .models import Actor, EventAction, EventType, MemoryEvent, RawMessage


ALLOWED_ACTORS = {actor.value for actor in Actor}
ALLOWED_TYPES = {event_type.value for event_type in EventType}
ALLOWED_ACTIONS = {action.value for action in EventAction}
ALLOWED_STATUSES = {
    "active",
    "superseded",
    "invalidated",
    "resolved",
    "stale",
    "open",
    "answered",
    "proposed",
    "in_progress",
    "claimed_done",
    "verified_done",
    "blocked",
    "failed",
}


@dataclass(slots=True)
class ExtractionPrompt:
    system: str
    user: str


class LLMExtractionClient(Protocol):
    def complete(self, prompt: ExtractionPrompt) -> str:
        ...


class JsonLLMExtractor:
    def __init__(self, client: LLMExtractionClient, min_confidence: float = 0.6) -> None:
        self.client = client
        self.min_confidence = min_confidence
        self._event_counter = itertools.count(1)

    def extract(self, message: RawMessage, session_id: str, turn_id: str) -> list[MemoryEvent]:
        prompt = self.build_prompt(message)
        raw_output = self.client.complete(prompt)
        payload = self.parse_response(raw_output)
        actor = self._normalize_actor(message.role)
        events: list[MemoryEvent] = []
        for item in payload.get("events", []):
            event = self._to_event(
                item=item,
                actor=actor,
                session_id=session_id,
                turn_id=turn_id,
                source_message_id=message.message_id,
            )
            if event:
                events.append(event)
        return events

    def build_prompt(self, message: RawMessage) -> ExtractionPrompt:
        system = (
            "You extract durable memory events from one conversation message.\n"
            "Return JSON only. Do not explain.\n"
            "Extract only high-value information for long-running agent tasks.\n"
            "Skip chit-chat and low-signal filler.\n"
            "Allowed actor values: user, assistant, tool, system.\n"
            "Allowed type values: goal, constraint, plan, task, decision, artifact, observation, question.\n"
            "Allowed action values: add, update, resolve, invalidate, verify.\n"
            "Allowed status values: active, superseded, invalidated, resolved, stale, open, answered, "
            "proposed, in_progress, claimed_done, verified_done, blocked, failed.\n"
            "For assistant self-reported completions, use task + claimed_done.\n"
            "Use verified_done only when the message is a tool result or explicit user verification.\n"
            "The response schema is: "
            '{"events":[{"actor":"assistant","type":"task","action":"update","status":"claimed_done",'
            '"subject":"implement session storage","confidence":0.91,"details":{"kind":"optional"}}]}'
        )
        user = (
            f"role={message.role}\n"
            f"message_id={message.message_id}\n"
            "message:\n"
            f"{message.content}\n\n"
            "Return only the JSON object. If nothing is worth storing, return {\"events\": []}."
        )
        return ExtractionPrompt(system=system, user=user)

    def parse_response(self, raw_output: str) -> dict[str, object]:
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("LLM extractor response must be a JSON object")
        events = payload.get("events", [])
        if not isinstance(events, list):
            raise ValueError("LLM extractor response must contain an events array")
        return payload

    def _to_event(
        self,
        item: object,
        actor: str,
        session_id: str,
        turn_id: str,
        source_message_id: str,
    ) -> MemoryEvent | None:
        if not isinstance(item, dict):
            return None
        event_actor = str(item.get("actor") or actor).lower()
        event_type = str(item.get("type") or "").lower()
        action = str(item.get("action") or EventAction.ADD).lower()
        status = str(item.get("status") or "active").lower()
        subject = str(item.get("subject") or "").strip()
        confidence = float(item.get("confidence", 0.8))
        details = item.get("details") or {}

        if (
            event_actor not in ALLOWED_ACTORS
            or event_type not in ALLOWED_TYPES
            or action not in ALLOWED_ACTIONS
            or status not in ALLOWED_STATUSES
            or not subject
            or confidence < self.min_confidence
        ):
            return None
        if not isinstance(details, dict):
            details = {}

        event_id = f"evt_llm_{next(self._event_counter):04d}"
        return MemoryEvent(
            event_id=event_id,
            session_id=session_id,
            turn_id=turn_id,
            source_message_ids=[source_message_id],
            actor=event_actor,
            type=event_type,
            action=action,
            status=status,
            subject=subject,
            details={str(key): _normalize_detail_value(value) for key, value in details.items()},
            confidence=confidence,
        )

    def _normalize_actor(self, role: str) -> str:
        role = role.lower()
        if role in ALLOWED_ACTORS:
            return role
        return Actor.USER


class HybridExtractor:
    def __init__(self, *extractors: object) -> None:
        self.extractors = list(extractors)

    def extract(self, message: RawMessage, session_id: str, turn_id: str) -> list[MemoryEvent]:
        merged: dict[tuple[str, str, str], MemoryEvent] = {}
        for extractor in self.extractors:
            events = extractor.extract(message, session_id=session_id, turn_id=turn_id)
            for event in events:
                key = (event.type, event.status, event.subject)
                existing = merged.get(key)
                if existing is None or event.confidence > existing.confidence:
                    merged[key] = event
        return list(merged.values())


def _normalize_detail_value(value: object) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
