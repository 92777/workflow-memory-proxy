from __future__ import annotations

import base64
import binascii
from dataclasses import replace
from dataclasses import dataclass
import json
import re
import time
from typing import Any
from uuid import uuid4

from .compressor import MemoryCompressor
from .config import ProxySettings
from .dsl import working_memory_to_dsl
from .history_pruner import (
    HIGH,
    LOW,
    MEDIUM,
    classify_history_fidelity,
    has_workflow_signal,
    is_verification_result_message,
    prune_history_messages,
)
from .openai_api import ChatCompletionsRequest, ChatMessage, ResponsesRequest
from .models import RawMessage, WorkingMemory
from .prompt_builder import (
    PromptMemoryConfig,
    merge_prompt_memories,
    parse_prompt_memory_text,
    prompt_memory_to_text,
)
from .reference_alias import alias_repeated_references
from .static_code import compress_static_code_message, is_code_heavy_text
from .store import SQLiteMemoryStore
from .tokens import TOKEN_COUNTER_NAME, approx_tokens, estimate_payload_tokens

INSTRUCTION_ROLES = {"system", "developer"}
UNSAFE_MESSAGE_KEYS = {"tool_calls", "function_call", "audio", "refusal"}
PROXY_COMPACTION_PREFIX = "mcpx:1:"
PROXY_COMPACTION_MARKER = "[memory-proxy-compaction:v1]"
SALIENT_FILE_PATTERN = re.compile(r"\b[\w./-]+\.[A-Za-z0-9]+\b")
SALIENT_ERROR_TOKENS = (
    "报错",
    "失败",
    "error",
    "failed",
    "traceback",
    "exception",
    "stack trace",
)
SUMMARY_FRIENDLY_GOAL_RE = re.compile(
    r"^(?:我想做(?:一个)?|我想|我要做|我要|希望做|希望|目标是|想做(?:一个)?).+"
)


@dataclass(slots=True)
class PreparedChatRequest:
    api_kind: str
    payload: dict[str, Any]
    original_payload: dict[str, Any]
    compressed: bool
    compression_reason: str
    session_id: str
    request_id: str
    prompt_memory: str = ""
    dropped_message_count: int = 0
    recent_message_count: int = 0
    stored: bool = False
    snapshot_id: str | None = None
    estimated_input_tokens_before: int | None = None
    estimated_input_tokens_after: int | None = None
    estimated_savings_pct: float | None = None
    token_counter: str | None = None
    recalled_memory: str = ""


@dataclass(slots=True)
class PreparedCompactionResponse:
    body: dict[str, Any]
    original_payload: dict[str, Any]
    compacted_payload: dict[str, Any]
    compressed: bool
    compression_reason: str
    session_id: str
    request_id: str
    prompt_memory: str = ""
    dropped_message_count: int = 0
    recent_message_count: int = 0
    snapshot_id: str | None = None
    estimated_input_tokens_before: int | None = None
    estimated_input_tokens_after: int | None = None
    estimated_savings_pct: float | None = None
    token_counter: str | None = None


@dataclass(slots=True)
class _ResponseInputProjection:
    items: list[dict[str, Any]]
    prefix_items: list[dict[str, Any]]
    tail_messages: list[ChatMessage]
    tail_message_indexes: list[int]
    prefix_boundary_index: int
    proxy_compaction_indexes: list[int]
    proxy_compaction_memory: list[str]
    late_instruction_items: list[dict[str, Any]]
    late_instruction_indexes: list[int]
    late_instruction_texts: list[str]
    has_pending_function_calls: bool

    def recent_items(self, recent_message_count: int) -> list[dict[str, Any]]:
        if recent_message_count <= 0 or not self.tail_messages:
            return []
        first_message_offset = max(0, len(self.tail_messages) - recent_message_count)
        start_index = self.tail_message_indexes[first_message_offset]
        while start_index > self.prefix_boundary_index and not _is_response_input_message_item(self.items[start_index - 1]):
            start_index -= 1
        return [
            item
            for index, item in enumerate(self.items[start_index:], start=start_index)
            if index not in self.proxy_compaction_indexes and index not in self.late_instruction_indexes
        ]

    def messages_after_compaction(self) -> int:
        if not self.proxy_compaction_indexes:
            return len(self.tail_messages)
        boundary = max(self.proxy_compaction_indexes)
        return sum(1 for index in self.tail_message_indexes if index > boundary)

    def compaction_items(
        self,
        *,
        salient_messages: list[ChatMessage],
        recent_message_count: int,
    ) -> list[dict[str, Any]]:
        output = [_response_input_message_from_chat_message(message) for message in salient_messages]
        output.extend(self.recent_items(recent_message_count))
        return output


class ChatProxyService:
    def __init__(
        self,
        settings: ProxySettings,
        compressor: MemoryCompressor | None = None,
        store: SQLiteMemoryStore | None = None,
    ) -> None:
        self.settings = settings
        prompt_config = PromptMemoryConfig(max_tokens=settings.prompt_memory_max_tokens)
        self.prompt_config = prompt_config
        self.compressor = compressor or MemoryCompressor(
            recent_window=settings.recent_window,
            prompt_config=prompt_config,
        )
        self.history_compressor = MemoryCompressor(
            recent_window=0,
            extractor=self.compressor.extractor,
            prompt_config=prompt_config,
        )
        self.store = store

    def prepare_chat_request(
        self,
        request: ChatCompletionsRequest,
        *,
        session_id: str | None = None,
    ) -> PreparedChatRequest:
        original_payload = request.to_payload()
        payload = dict(original_payload)
        resolved_session_id = session_id or f"sess_{uuid4().hex[:12]}"
        request_id = f"req_{uuid4().hex[:12]}"
        original_messages = [message.model_dump(exclude_none=True) for message in request.messages]
        recalled = self._load_recalled_memory(resolved_session_id)

        if not self.settings.compression_enabled:
            return PreparedChatRequest(
                api_kind="chat_completions",
                payload=payload,
                original_payload=original_payload,
                compressed=False,
                compression_reason="compression_disabled",
                session_id=resolved_session_id,
                request_id=request_id,
                estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                estimated_input_tokens_after=estimate_payload_tokens(payload),
                estimated_savings_pct=0.0,
                token_counter=TOKEN_COUNTER_NAME,
            )

        prefix_count = _count_instruction_prefix(request.messages)
        if _has_instruction_message_after_prefix(request.messages, prefix_count):
            return PreparedChatRequest(
                api_kind="chat_completions",
                payload=payload,
                original_payload=original_payload,
                compressed=False,
                compression_reason="instruction_after_prefix",
                session_id=resolved_session_id,
                request_id=request_id,
                estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                estimated_input_tokens_after=estimate_payload_tokens(payload),
                estimated_savings_pct=0.0,
                token_counter=TOKEN_COUNTER_NAME,
            )

        prefix = original_messages[:prefix_count]
        tail = request.messages[prefix_count:]
        compaction = self._compact_messages(
            tail,
            session_id=resolved_session_id,
            request_id=request_id,
        )
        if not compaction.compressed:
            if compaction.reason in {"history_too_short", "empty_prompt_memory"} and recalled is not None:
                payload["messages"] = prefix + [_build_memory_message(self.settings.memory_system_prompt, recalled.memory_dsl)] + original_messages[prefix_count:]
                return PreparedChatRequest(
                    api_kind="chat_completions",
                    payload=payload,
                    original_payload=original_payload,
                    compressed=False,
                    compression_reason="session_memory_recalled",
                    session_id=resolved_session_id,
                    request_id=request_id,
                    snapshot_id=recalled.snapshot_id,
                    prompt_memory=recalled.memory_dsl,
                    recent_message_count=len(tail),
                    estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                    estimated_input_tokens_after=estimate_payload_tokens(payload),
                    estimated_savings_pct=_estimate_savings_pct(original_payload, payload),
                    token_counter=TOKEN_COUNTER_NAME,
                    recalled_memory=recalled.memory_dsl,
                )
            return PreparedChatRequest(
                api_kind="chat_completions",
                payload=payload,
                original_payload=original_payload,
                compressed=False,
                compression_reason=compaction.reason,
                session_id=resolved_session_id,
                request_id=request_id,
                estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                estimated_input_tokens_after=estimate_payload_tokens(payload),
                estimated_savings_pct=0.0,
                token_counter=TOKEN_COUNTER_NAME,
            )

        memory_message = {
            "role": "system",
            "content": f"{self.settings.memory_system_prompt}\n\n{compaction.prompt_memory}",
        }
        payload["messages"] = (
            prefix
            + [memory_message]
            + [message.model_dump(exclude_none=True) for message in compaction.retained_messages]
        )
        stored, snapshot_id = self._persist_compaction(
            session_id=resolved_session_id,
            request_id=request_id,
            upstream_model=request.model,
            compaction=compaction,
        )
        return PreparedChatRequest(
            api_kind="chat_completions",
            payload=payload,
            original_payload=original_payload,
            compressed=True,
            compression_reason="history_compacted",
            session_id=resolved_session_id,
            request_id=request_id,
            prompt_memory=compaction.prompt_memory,
            dropped_message_count=compaction.dropped_message_count,
            recent_message_count=len(compaction.retained_messages),
            stored=stored,
            snapshot_id=snapshot_id,
            estimated_input_tokens_before=estimate_payload_tokens(original_payload),
            estimated_input_tokens_after=estimate_payload_tokens(payload),
            estimated_savings_pct=_estimate_savings_pct(original_payload, payload),
            token_counter=TOKEN_COUNTER_NAME,
        )

    def prepare_responses_request(
        self,
        request: ResponsesRequest,
        *,
        session_id: str | None = None,
    ) -> PreparedChatRequest:
        original_payload = request.to_payload()
        payload = dict(original_payload)
        resolved_session_id = session_id or f"sess_{uuid4().hex[:12]}"
        request_id = f"req_{uuid4().hex[:12]}"
        recalled = self._load_recalled_memory(resolved_session_id)

        if not self.settings.compression_enabled:
            return PreparedChatRequest(
                api_kind="responses",
                payload=payload,
                original_payload=original_payload,
                compressed=False,
                compression_reason="compression_disabled",
                session_id=resolved_session_id,
                request_id=request_id,
                estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                estimated_input_tokens_after=estimate_payload_tokens(payload),
                estimated_savings_pct=0.0,
                token_counter=TOKEN_COUNTER_NAME,
            )

        input_projection = _project_response_input(request.input)
        if input_projection is None:
            return PreparedChatRequest(
                api_kind="responses",
                payload=payload,
                original_payload=original_payload,
                compressed=False,
                compression_reason="unsupported_input_shape",
                session_id=resolved_session_id,
                request_id=request_id,
                estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                estimated_input_tokens_after=estimate_payload_tokens(payload),
                estimated_savings_pct=0.0,
                token_counter=TOKEN_COUNTER_NAME,
            )

        compacted_memory = _merge_memory_blocks(
            input_projection.proxy_compaction_memory,
            budgeter=self.compressor.prompt_builder.budgeter,
        )
        prefix = input_projection.prefix_items
        tail = input_projection.tail_messages
        instruction_text = _merge_text_blocks([request.instructions, *input_projection.late_instruction_texts])
        if input_projection.has_pending_function_calls:
            if compacted_memory or input_projection.late_instruction_texts:
                payload["instructions"] = _merge_memory_instructions(
                    instruction_text,
                    compacted_memory,
                    self.settings.memory_system_prompt,
                )
                payload["input"] = prefix + input_projection.recent_items(len(tail))
            return PreparedChatRequest(
                api_kind="responses",
                payload=payload,
                original_payload=original_payload,
                compressed=False,
                compression_reason="pending_function_call",
                session_id=resolved_session_id,
                request_id=request_id,
                prompt_memory=compacted_memory,
                recent_message_count=len(tail),
                estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                estimated_input_tokens_after=estimate_payload_tokens(payload),
                estimated_savings_pct=_estimate_savings_pct(original_payload, payload),
                token_counter=TOKEN_COUNTER_NAME,
                recalled_memory=compacted_memory,
            )

        if compacted_memory and len(tail) <= (
            self.settings.recent_window + self.settings.salient_history_messages + 1
        ):
            payload["instructions"] = _merge_memory_instructions(
                instruction_text,
                compacted_memory,
                self.settings.memory_system_prompt,
            )
            payload["input"] = prefix + input_projection.recent_items(len(tail))
            return PreparedChatRequest(
                api_kind="responses",
                payload=payload,
                original_payload=original_payload,
                compressed=False,
                compression_reason="compaction_item_rehydrated",
                session_id=resolved_session_id,
                request_id=request_id,
                prompt_memory=compacted_memory,
                recent_message_count=len(tail),
                estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                estimated_input_tokens_after=estimate_payload_tokens(payload),
                estimated_savings_pct=_estimate_savings_pct(original_payload, payload),
                token_counter=TOKEN_COUNTER_NAME,
                recalled_memory=compacted_memory,
            )
        compaction = self._compact_messages(
            tail,
            session_id=resolved_session_id,
            request_id=request_id,
        )
        if not compaction.compressed:
            if compaction.reason in {"history_too_short", "empty_prompt_memory"} and recalled is not None:
                combined_memory = _merge_memory_blocks(
                    [compacted_memory, recalled.memory_dsl],
                    budgeter=self.compressor.prompt_builder.budgeter,
                )
                payload["instructions"] = _merge_memory_instructions(
                    instruction_text,
                    combined_memory,
                    self.settings.memory_system_prompt,
                )
                payload["input"] = prefix + input_projection.recent_items(len(tail))
                return PreparedChatRequest(
                    api_kind="responses",
                    payload=payload,
                    original_payload=original_payload,
                    compressed=False,
                    compression_reason="session_memory_recalled",
                    session_id=resolved_session_id,
                    request_id=request_id,
                    snapshot_id=recalled.snapshot_id,
                    prompt_memory=recalled.memory_dsl,
                    recent_message_count=len(tail),
                    estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                    estimated_input_tokens_after=estimate_payload_tokens(payload),
                    estimated_savings_pct=_estimate_savings_pct(original_payload, payload),
                    token_counter=TOKEN_COUNTER_NAME,
                    recalled_memory=combined_memory,
                )
            if input_projection.late_instruction_texts:
                payload["instructions"] = instruction_text
                payload["input"] = prefix + input_projection.recent_items(len(tail))
                return PreparedChatRequest(
                    api_kind="responses",
                    payload=payload,
                    original_payload=original_payload,
                    compressed=True,
                    compression_reason="instruction_tail_normalized",
                    session_id=resolved_session_id,
                    request_id=request_id,
                    recent_message_count=len(tail),
                    estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                    estimated_input_tokens_after=estimate_payload_tokens(payload),
                    estimated_savings_pct=_estimate_savings_pct(original_payload, payload),
                    token_counter=TOKEN_COUNTER_NAME,
                )
            return PreparedChatRequest(
                api_kind="responses",
                payload=payload,
                original_payload=original_payload,
                compressed=False,
                compression_reason=compaction.reason,
                session_id=resolved_session_id,
                request_id=request_id,
                estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                estimated_input_tokens_after=estimate_payload_tokens(payload),
                estimated_savings_pct=0.0,
                token_counter=TOKEN_COUNTER_NAME,
            )

        merged_memory = _merge_memory_blocks(
            [compacted_memory, compaction.prompt_memory],
            budgeter=self.compressor.prompt_builder.budgeter,
        )
        payload["instructions"] = _merge_memory_instructions(
            instruction_text,
            merged_memory,
            self.settings.memory_system_prompt,
        )
        payload["input"] = prefix + input_projection.compaction_items(
            salient_messages=compaction.salient_messages,
            recent_message_count=compaction.recent_suffix_count,
        )
        stored, snapshot_id = self._persist_compaction(
            session_id=resolved_session_id,
            request_id=request_id,
            upstream_model=request.model,
            compaction=compaction,
        )
        return PreparedChatRequest(
            api_kind="responses",
            payload=payload,
            original_payload=original_payload,
            compressed=True,
            compression_reason="history_compacted",
            session_id=resolved_session_id,
            request_id=request_id,
            prompt_memory=merged_memory,
            dropped_message_count=compaction.dropped_message_count,
            recent_message_count=len(compaction.retained_messages),
            stored=stored,
            snapshot_id=snapshot_id,
            estimated_input_tokens_before=estimate_payload_tokens(original_payload),
            estimated_input_tokens_after=estimate_payload_tokens(payload),
            estimated_savings_pct=_estimate_savings_pct(original_payload, payload),
            token_counter=TOKEN_COUNTER_NAME,
        )

    def compact_responses_request(
        self,
        request: ResponsesRequest,
        *,
        session_id: str | None = None,
    ) -> PreparedCompactionResponse:
        original_payload = request.to_payload()
        resolved_session_id = session_id or f"sess_{uuid4().hex[:12]}"
        request_id = f"cmp_{uuid4().hex[:12]}"
        input_projection = _project_response_input(request.input)

        if input_projection is None:
            output_items = request.input if isinstance(request.input, list) else []
            compacted_payload = {**original_payload, "input": output_items}
            return PreparedCompactionResponse(
                body=_build_compaction_response_body(request_id, output_items),
                original_payload=original_payload,
                compacted_payload=compacted_payload,
                compressed=False,
                compression_reason="unsupported_input_shape",
                session_id=resolved_session_id,
                request_id=request_id,
                estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                estimated_input_tokens_after=estimate_payload_tokens(compacted_payload),
                estimated_savings_pct=_estimate_savings_pct(original_payload, compacted_payload),
                token_counter=TOKEN_COUNTER_NAME,
            )

        prefix = input_projection.prefix_items
        tail = input_projection.tail_messages
        merged_existing_memory = _merge_memory_blocks(
            input_projection.proxy_compaction_memory,
            budgeter=self.compressor.prompt_builder.budgeter,
        )
        normalized_prefix = prefix + input_projection.late_instruction_items
        if input_projection.has_pending_function_calls:
            output_items = request.input if isinstance(request.input, list) else []
            compacted_payload = {**original_payload, "input": output_items}
            return PreparedCompactionResponse(
                body=_build_compaction_response_body(request_id, output_items),
                original_payload=original_payload,
                compacted_payload=compacted_payload,
                compressed=False,
                compression_reason="pending_function_call",
                session_id=resolved_session_id,
                request_id=request_id,
                estimated_input_tokens_before=estimate_payload_tokens(original_payload),
                estimated_input_tokens_after=estimate_payload_tokens(compacted_payload),
                estimated_savings_pct=_estimate_savings_pct(original_payload, compacted_payload),
                token_counter=TOKEN_COUNTER_NAME,
            )
        compaction = self._compact_messages(
            tail,
            session_id=resolved_session_id,
            request_id=request_id,
        )

        compressed = False
        compression_reason = compaction.reason
        prompt_memory = ""
        dropped_message_count = 0
        recent_message_count = len(tail)
        if compaction.compressed:
            prompt_memory = _merge_memory_blocks(
                [merged_existing_memory, compaction.prompt_memory],
                budgeter=self.compressor.prompt_builder.budgeter,
            )
            output_items = normalized_prefix + [_build_proxy_compaction_item(prompt_memory, request_id)] + input_projection.compaction_items(
                salient_messages=compaction.salient_messages,
                recent_message_count=compaction.recent_suffix_count,
            )
            compressed = True
            compression_reason = "history_compacted"
            dropped_message_count = compaction.dropped_message_count
            recent_message_count = len(compaction.retained_messages)
        elif merged_existing_memory:
            prompt_memory = merged_existing_memory
            output_items = normalized_prefix + [_build_proxy_compaction_item(prompt_memory, request_id)] + input_projection.recent_items(
                len(tail)
            )
            compression_reason = "compaction_item_repacked"
        elif input_projection.late_instruction_items:
            output_items = normalized_prefix + input_projection.recent_items(len(tail))
            compression_reason = "instruction_tail_normalized"
        else:
            output_items = request.input if isinstance(request.input, list) else []

        compacted_payload = {**original_payload, "input": output_items}
        return PreparedCompactionResponse(
            body=_build_compaction_response_body(request_id, output_items),
            original_payload=original_payload,
            compacted_payload=compacted_payload,
            compressed=compressed,
            compression_reason=compression_reason,
            session_id=resolved_session_id,
            request_id=request_id,
            prompt_memory=prompt_memory,
            dropped_message_count=dropped_message_count,
            recent_message_count=recent_message_count,
            estimated_input_tokens_before=estimate_payload_tokens(original_payload),
            estimated_input_tokens_after=estimate_payload_tokens(compacted_payload),
            estimated_savings_pct=_estimate_savings_pct(original_payload, compacted_payload),
            token_counter=TOKEN_COUNTER_NAME,
        )

    def capture_response_memory(
        self,
        *,
        api_kind: str,
        session_id: str,
        request_id: str,
        response_body: dict[str, Any] | None,
    ) -> str | None:
        if self.store is None or response_body is None:
            return None

        response_text = _extract_response_text(api_kind, response_body)
        if not response_text:
            return None

        response_turn_id = f"{request_id}:response"
        response_message = RawMessage(
            message_id=f"{request_id}_assistant_response",
            role="assistant",
            content=response_text,
        )
        extracted_events = self.compressor.extractor.extract(
            response_message,
            session_id=session_id,
            turn_id=response_turn_id,
        )
        if not extracted_events:
            return None

        latest_snapshot = self.store.get_latest_working_memory_snapshot(session_id)
        base_memory = WorkingMemory()
        if latest_snapshot is not None:
            loaded_memory = self.store.load_snapshot_memory(latest_snapshot["snapshot_id"])
            if loaded_memory is not None:
                base_memory = loaded_memory

        merged_memory = self.compressor.reducer.reduce_from(base_memory, extracted_events)
        recall_prompt = prompt_memory_to_text(self.compressor.prompt_builder.build(merged_memory)).strip()
        if not recall_prompt:
            return None

        self.store.insert_raw_message(session_id, response_turn_id, response_message)
        stored_events = [
            replace(
                event,
                turn_id=response_turn_id,
                details={**event.details, "_source": "assistant_response"},
            )
            for event in extracted_events
        ]
        self.store.insert_events(stored_events)
        snapshot_id = self.store.insert_working_memory_snapshot(
            session_id=session_id,
            turn_id=response_turn_id,
            memory=merged_memory,
            memory_dsl=recall_prompt,
            metadata={
                "source": "response_memory",
                "audit_memory_dsl": working_memory_to_dsl(
                    merged_memory,
                    include_artifacts=True,
                    include_observations=True,
                ),
                "response_preview": response_text[:800],
            },
        )
        return snapshot_id

    def _compact_messages(
        self,
        messages: list[ChatMessage],
        *,
        session_id: str,
        request_id: str,
    ) -> "_CompactionResult":
        recent_count = _select_recent_suffix_count(
            messages,
            max_recent_messages=self.settings.recent_window,
            min_recent_messages=self.settings.recent_min_messages,
            token_budget=self.settings.recent_token_budget,
        )
        history_count = len(messages) - recent_count
        if history_count < self.settings.min_history_messages:
            return _CompactionResult(False, "history_too_short", "", 0, messages, [], recent_count, None, [])

        history = messages[:history_count]
        recent = messages[history_count:]
        if not history or not all(_is_safe_for_compression(message) for message in history):
            return _CompactionResult(False, "unsafe_history_message", "", 0, messages, [], recent_count, None, [])
        pruned_history = prune_history_messages(history)
        compression_history = pruned_history.kept_messages

        raw_messages = [
            RawMessage(
                message_id=f"{request_id}_msg_{index:04d}",
                role=message.role,
                content=str(message.content),
            )
            for index, message in enumerate(compression_history, start=1)
        ]
        redundant_salient_indexes = _find_redundant_salient_history_indexes(
            compression_history,
            recent_messages=recent,
        )
        salient_indexes = _select_salient_history_indexes(
            compression_history,
            keep_count=self.settings.salient_history_messages,
            min_compressed_messages=self.settings.min_history_messages,
            excluded_indexes=redundant_salient_indexes,
        )
        prompt_exclude_message_ids = {raw_messages[index].message_id for index in salient_indexes}
        compression = self.history_compressor.compress(
            raw_messages,
            session_id=session_id,
            prompt_exclude_message_ids=prompt_exclude_message_ids,
        )
        prompt_memory = compression.memory_dsl.strip()
        if not prompt_memory:
            return _CompactionResult(False, "empty_prompt_memory", "", 0, messages, [], recent_count, None, raw_messages)
        salient_history = _prepare_salient_history_messages(
            compression_history,
            selected_indexes=salient_indexes,
            token_budget=self.settings.salient_history_token_budget,
        )
        retained_messages = salient_history + recent
        return _CompactionResult(
            True,
            "history_compacted",
            prompt_memory,
            len(history) - len(salient_history),
            retained_messages,
            salient_history,
            len(recent),
            compression,
            raw_messages,
        )

    def _persist_compaction(
        self,
        *,
        session_id: str,
        request_id: str,
        upstream_model: str,
        compaction: "_CompactionResult",
    ) -> tuple[bool, str | None]:
        if self.store is None or compaction.compression is None:
            return False, None

        self.store.upsert_session(session_id, client="proxy", upstream_model=upstream_model)
        for raw_message in compaction.history_raw_messages:
            self.store.insert_raw_message(session_id, request_id, raw_message)

        stored_events = [
            replace(
                event,
                turn_id=request_id,
                details={**event.details, "_local_turn_id": event.turn_id},
            )
            for event in compaction.compression.events
        ]
        self.store.insert_events(stored_events)
        snapshot_id = self.store.insert_working_memory_snapshot(
            session_id=session_id,
            turn_id=request_id,
            memory=compaction.compression.working_memory,
            memory_dsl=compaction.compression.memory_dsl,
            metadata={
                "prompt_memory": compaction.compression.memory_dsl,
                "audit_memory_dsl": compaction.compression.audit_memory_dsl,
                "dropped_message_count": compaction.dropped_message_count,
            },
        )
        return True, snapshot_id

    def _load_recalled_memory(self, session_id: str) -> "_SessionRecall | None":
        if self.store is None:
            return None
        snapshot = self.store.get_latest_working_memory_snapshot(session_id)
        if snapshot is None:
            return None
        memory_dsl = (snapshot["memory_dsl"] or "").strip()
        if not memory_dsl:
            return None
        return _SessionRecall(
            snapshot_id=snapshot["snapshot_id"],
            memory_dsl=memory_dsl,
        )


@dataclass(slots=True)
class _CompactionResult:
    compressed: bool
    reason: str
    prompt_memory: str
    dropped_message_count: int
    retained_messages: list[ChatMessage]
    salient_messages: list[ChatMessage]
    recent_suffix_count: int
    compression: Any | None
    history_raw_messages: list[RawMessage]


@dataclass(slots=True)
class _SessionRecall:
    snapshot_id: str
    memory_dsl: str


@dataclass(frozen=True, slots=True)
class _SalientCandidate:
    index: int
    message: ChatMessage
    score: float
    token_cost: int


def _count_instruction_prefix(messages: list[ChatMessage]) -> int:
    count = 0
    for message in messages:
        if message.role not in INSTRUCTION_ROLES:
            break
        count += 1
    return count


def _has_instruction_message_after_prefix(messages: list[ChatMessage], prefix_count: int) -> bool:
    return any(message.role in INSTRUCTION_ROLES for message in messages[prefix_count:])


def _is_safe_for_compression(message: ChatMessage) -> bool:
    if not isinstance(message.content, str):
        return False
    dumped = message.model_dump(exclude_none=True)
    return not any(key in dumped for key in UNSAFE_MESSAGE_KEYS)


def _select_recent_suffix_count(
    messages: list[ChatMessage],
    *,
    max_recent_messages: int,
    min_recent_messages: int,
    token_budget: int,
) -> int:
    if not messages:
        return 0

    max_keep = min(len(messages), max(1, max_recent_messages))
    min_keep = min(max_keep, max(1, min_recent_messages))
    selected = 0
    used_tokens = 0

    for message in reversed(messages):
        if selected >= max_keep:
            break
        message_tokens = _estimated_chat_message_tokens(message)
        if selected < min_keep:
            selected += 1
            used_tokens += message_tokens
            continue
        if token_budget > 0 and used_tokens + message_tokens <= token_budget:
            selected += 1
            used_tokens += message_tokens
            continue
        break
    return selected


def _estimated_chat_message_tokens(message: ChatMessage) -> int:
    content = message.content if isinstance(message.content, str) else ""
    return approx_tokens(content) + 4


def _select_salient_history_indexes(
    messages: list[ChatMessage],
    *,
    keep_count: int,
    min_compressed_messages: int,
    excluded_indexes: set[int] | None = None,
) -> list[int]:
    if keep_count <= 0 or not messages:
        return []

    max_keep = min(keep_count, max(0, len(messages) - max(1, min_compressed_messages)))
    if max_keep <= 0:
        return []

    scored: list[tuple[float, int]] = []
    for index, message in enumerate(messages):
        if excluded_indexes and index in excluded_indexes:
            continue
        score = _salient_history_score(message, index=index, total=len(messages))
        if score <= 0:
            continue
        scored.append((score, index))

    if not scored:
        return []

    chosen = sorted(scored, key=lambda item: (-item[0], -item[1]))[:max_keep]
    return sorted(index for _, index in chosen)


def _salient_history_score(message: ChatMessage, *, index: int, total: int) -> float:
    content = message.content if isinstance(message.content, str) else ""
    text = content.strip()
    if not text:
        return 0.0

    fidelity = classify_history_fidelity(message)
    if fidelity == LOW:
        return 0.0

    lower = text.lower()
    signal_score = 0.0

    if fidelity == HIGH:
        signal_score += 2.6
    elif fidelity == MEDIUM:
        signal_score += 1.0

    if _has_salient_artifact_hint(text):
        signal_score += 3.2
    if any(token in lower for token in SALIENT_ERROR_TOKENS):
        signal_score += 2.8
    if is_code_heavy_text(text):
        signal_score += 3.4
    if text.endswith(("?", "？")):
        signal_score += 0.7
    if len(text) >= 96:
        signal_score += 0.6
    if "\n" in text:
        signal_score += 0.4

    if signal_score <= 0:
        return 0.0

    score = signal_score

    if message.role == "user":
        score += 0.5
    elif message.role == "tool":
        score += 0.7
    elif message.role == "assistant":
        score += 0.2

    recency_denominator = max(1, total - 1)
    score += index / recency_denominator * 0.5
    return round(score, 4)


def _find_redundant_salient_history_indexes(
    history: list[ChatMessage],
    *,
    recent_messages: list[ChatMessage],
) -> set[int]:
    if not history:
        return set()

    redundant: set[int] = set()
    for index, message in enumerate(history):
        if _is_summary_friendly_workflow_message(message):
            redundant.add(index)
            continue
        if not is_verification_result_message(message):
            continue
        later_messages = history[index + 1 :] + recent_messages
        if any(has_workflow_signal(candidate) for candidate in later_messages):
            redundant.add(index)
    return redundant


def _is_summary_friendly_workflow_message(message: ChatMessage) -> bool:
    if not isinstance(message.content, str):
        return False
    text = message.content.strip()
    if not text:
        return False
    if not has_workflow_signal(text) and SUMMARY_FRIENDLY_GOAL_RE.search(text) is None:
        return False
    if _has_salient_artifact_hint(text):
        return False
    if is_code_heavy_text(text):
        return False
    lower = text.lower()
    if any(token in lower for token in SALIENT_ERROR_TOKENS):
        return False
    if text.endswith(("?", "？")):
        return False
    return True


def _has_salient_artifact_hint(text: str) -> bool:
    if SALIENT_FILE_PATTERN.search(text):
        return True
    return any(prefix in text for prefix in ("/Users/", "/tmp/", "./", "../"))


def _compress_salient_history_message(message: ChatMessage) -> ChatMessage:
    if not isinstance(message.content, str):
        return message
    compressed = compress_static_code_message(message.content)
    if compressed is None:
        return message
    return message.model_copy(update={"content": compressed})


def _prepare_salient_history_messages(
    history: list[ChatMessage],
    *,
    selected_indexes: list[int],
    token_budget: int,
) -> list[ChatMessage]:
    candidates = [
        _SalientCandidate(
            index=index,
            message=_compress_salient_history_message(history[index]),
            score=_salient_history_score(history[index], index=index, total=len(history)),
            token_cost=_estimated_chat_message_tokens(_compress_salient_history_message(history[index])),
        )
        for index in selected_indexes
    ]
    admitted = _fit_salient_history_candidates(candidates, token_budget=token_budget)
    if not admitted:
        return []
    ordered_messages = [candidate.message for candidate in sorted(admitted, key=lambda item: item.index)]
    aliased = alias_repeated_references(ordered_messages)
    compressed = [_compress_salient_history_message(message) for message in aliased]
    return _dedupe_retained_history_messages(compressed)


def _fit_salient_history_candidates(
    candidates: list[_SalientCandidate],
    *,
    token_budget: int,
) -> list[_SalientCandidate]:
    if not candidates:
        return []
    if token_budget <= 0:
        return candidates

    guaranteed = max(candidates, key=lambda item: (item.score, item.index))
    remaining = [item for item in candidates if item != guaranteed]
    budget = max(0, token_budget - guaranteed.token_cost)
    chosen = _maximize_candidate_signal_under_budget(remaining, budget=budget)
    return [guaranteed, *chosen]


def _maximize_candidate_signal_under_budget(
    candidates: list[_SalientCandidate],
    *,
    budget: int,
) -> list[_SalientCandidate]:
    if not candidates or budget <= 0:
        return []

    best_by_cost: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for candidate in candidates:
        score_value = max(1, int(round(candidate.score * 100)))
        updates: dict[int, tuple[int, tuple[int, ...]]] = {}
        for used_cost, (used_score, used_indexes) in best_by_cost.items():
            next_cost = used_cost + candidate.token_cost
            if next_cost > budget:
                continue
            proposal = (used_score + score_value, used_indexes + (candidate.index,))
            current = updates.get(next_cost) or best_by_cost.get(next_cost)
            if current is None or proposal > current:
                updates[next_cost] = proposal
        best_by_cost.update(updates)

    _, selected_indexes = max(
        best_by_cost.values(),
        key=lambda item: (item[0], -len(item[1]), item[1]),
    )
    selected_set = set(selected_indexes)
    return [candidate for candidate in candidates if candidate.index in selected_set]


def _dedupe_retained_history_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    if len(messages) < 2:
        return messages

    seen: set[tuple[str, str]] = set()
    deduped_reversed: list[ChatMessage] = []
    for message in reversed(messages):
        normalized = _normalize_retained_message_content(message)
        if normalized:
            key = (message.role, normalized)
            if key in seen:
                continue
            seen.add(key)
        deduped_reversed.append(message)
    return list(reversed(deduped_reversed))


def _normalize_retained_message_content(message: ChatMessage) -> str:
    if not isinstance(message.content, str):
        return ""
    return re.sub(r"\s+", " ", message.content).strip()


def _response_input_message_from_chat_message(message: ChatMessage) -> dict[str, Any]:
    return {
        "type": "message",
        "role": message.role,
        "content": str(message.content),
    }


def _project_response_input(value: Any) -> _ResponseInputProjection | None:
    if not isinstance(value, list):
        return None
    items: list[dict[str, Any]] = []
    prefix_items: list[dict[str, Any]] = []
    tail_messages: list[ChatMessage] = []
    tail_message_indexes: list[int] = []
    proxy_compaction_indexes: list[int] = []
    proxy_compaction_memory: list[str] = []
    late_instruction_items: list[dict[str, Any]] = []
    late_instruction_indexes: list[int] = []
    late_instruction_texts: list[str] = []
    in_prefix = True
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            return None
        items.append(item)
        decoded_compaction = _decode_proxy_compaction_item(item)
        if decoded_compaction is not None:
            proxy_compaction_indexes.append(index)
            proxy_compaction_memory.append(decoded_compaction)
            in_prefix = False
            continue
        if not _is_response_input_message_item(item):
            in_prefix = False
            continue

        role = item.get("role")
        if not isinstance(role, str):
            return None
        content = _response_input_message_text(item)
        if content is None:
            return None

        message = ChatMessage(role=role, content=content)
        if in_prefix and message.role in INSTRUCTION_ROLES:
            prefix_items.append(item)
            continue
        if message.role in INSTRUCTION_ROLES:
            in_prefix = False
            late_instruction_items.append(item)
            late_instruction_indexes.append(index)
            late_instruction_texts.append(content)
            continue

        in_prefix = False
        tail_messages.append(message)
        tail_message_indexes.append(index)

    return _ResponseInputProjection(
        items=items,
        prefix_items=prefix_items,
        tail_messages=tail_messages,
        tail_message_indexes=tail_message_indexes,
        prefix_boundary_index=len(prefix_items),
        proxy_compaction_indexes=proxy_compaction_indexes,
        proxy_compaction_memory=proxy_compaction_memory,
        late_instruction_items=late_instruction_items,
        late_instruction_indexes=late_instruction_indexes,
        late_instruction_texts=late_instruction_texts,
        has_pending_function_calls=_has_pending_function_calls(items),
    )


def _is_response_input_message_item(item: dict[str, Any]) -> bool:
    item_type = item.get("type")
    if item_type is not None and item_type != "message":
        return False
    return isinstance(item.get("role"), str)


def _response_input_message_text(item: dict[str, Any]) -> str | None:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            return None
        block_type = block.get("type")
        if block_type not in {"input_text", "output_text", "text"}:
            return None
        text = block.get("text")
        if not isinstance(text, str):
            return None
        stripped = text.strip()
        if stripped:
            text_parts.append(stripped)
    return "\n\n".join(text_parts)


def _estimate_savings_pct(original_payload: dict[str, Any], payload: dict[str, Any]) -> float:
    before = estimate_payload_tokens(original_payload)
    after = estimate_payload_tokens(payload)
    if before <= 0:
        return 0.0
    return round((before - after) / before * 100, 2)


def _build_compaction_response_body(
    compaction_id: str,
    output_items: list[dict[str, Any]],
    *,
    object_type: str = "memory_proxy.compaction",
) -> dict[str, Any]:
    output_payload = {"output": output_items}
    output_tokens = estimate_payload_tokens(output_payload)
    return {
        "id": compaction_id,
        "object": object_type,
        "created_at": int(time.time()),
        "output": output_items,
        "usage": {
            "input_tokens": output_tokens,
            "output_tokens": 0,
            "total_tokens": output_tokens,
        },
    }


def _build_proxy_compaction_item(memory_dsl: str, compaction_id: str) -> dict[str, Any]:
    return {
        "id": f"{compaction_id}_item",
        "type": "message",
        "role": "developer",
        "content": [
            {
                "type": "input_text",
                "text": _render_proxy_compaction_message(memory_dsl),
            }
        ],
    }


def _encode_proxy_compaction_memory(memory_dsl: str) -> str:
    payload = json.dumps({"v": 1, "memory_dsl": memory_dsl}, ensure_ascii=False).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"{PROXY_COMPACTION_PREFIX}{encoded}"


def _decode_proxy_compaction_item(item: dict[str, Any]) -> str | None:
    legacy = _decode_legacy_proxy_compaction_item(item)
    if legacy is not None:
        return legacy

    if not _is_response_input_message_item(item):
        return None
    content = _response_input_message_text(item)
    if content is None:
        return None
    return _decode_proxy_compaction_message_text(content)


def _decode_legacy_proxy_compaction_item(item: dict[str, Any]) -> str | None:
    if item.get("type") != "compaction":
        return None
    encrypted_content = item.get("encrypted_content")
    if not isinstance(encrypted_content, str) or not encrypted_content.startswith(PROXY_COMPACTION_PREFIX):
        return None
    encoded = encrypted_content[len(PROXY_COMPACTION_PREFIX) :]
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return None
    if not isinstance(payload, dict) or payload.get("v") != 1:
        return None
    memory_dsl = payload.get("memory_dsl")
    if not isinstance(memory_dsl, str):
        return None
    stripped = memory_dsl.strip()
    return stripped or None


def _render_proxy_compaction_message(memory_dsl: str) -> str:
    stripped = memory_dsl.strip()
    return f"{PROXY_COMPACTION_MARKER}\n{stripped}" if stripped else PROXY_COMPACTION_MARKER


def _decode_proxy_compaction_message_text(content: str) -> str | None:
    stripped = content.strip()
    if not stripped.startswith(PROXY_COMPACTION_MARKER):
        return None
    remainder = stripped[len(PROXY_COMPACTION_MARKER) :].lstrip()
    return remainder or None


def _has_pending_function_calls(items: list[dict[str, Any]]) -> bool:
    pending_call_ids: set[str] = set()
    malformed_function_call = False
    for item in items:
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                malformed_function_call = True
                continue
            pending_call_ids.add(call_id.strip())
            continue
        if item_type != "function_call_output":
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str):
            continue
        pending_call_ids.discard(call_id.strip())
    return malformed_function_call or bool(pending_call_ids)


def _merge_memory_blocks(blocks: list[str] | tuple[str, ...], *, budgeter: Any | None = None) -> str:
    prompt_memories = []
    for block in blocks:
        if not isinstance(block, str):
            continue
        stripped = block.strip()
        if not stripped:
            continue
        prompt_memories.append(parse_prompt_memory_text(stripped))

    if not prompt_memories:
        return ""

    merged = merge_prompt_memories(prompt_memories)
    if budgeter is not None:
        merged = budgeter.fit(merged)
    return prompt_memory_to_text(merged)


def _merge_text_blocks(blocks: list[str] | tuple[str, ...]) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        if not isinstance(block, str):
            continue
        stripped = block.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        merged.append(stripped)
    return "\n\n".join(merged)


def _merge_memory_instructions(
    instructions: str | None,
    memory_dsl: str,
    memory_system_prompt: str,
) -> str:
    instruction_text = instructions.strip() if isinstance(instructions, str) else ""
    memory_text = memory_dsl.strip()
    if not memory_text:
        return instruction_text
    memory_block = f"{memory_system_prompt}\n\n{memory_text}"
    return f"{instruction_text}\n\n{memory_block}".strip() if instruction_text else memory_block


def _build_memory_message(memory_system_prompt: str, memory_dsl: str) -> dict[str, str]:
    return {
        "role": "system",
        "content": f"{memory_system_prompt}\n\n{memory_dsl}",
    }


def _extract_response_text(api_kind: str, response_body: dict[str, Any]) -> str:
    if api_kind == "responses":
        output = response_body.get("output")
        if not isinstance(output, list):
            return ""
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
        return "\n".join(texts).strip()

    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        return "\n".join(text_parts).strip()
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)
