from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import ValidationError

from .config import ProxySettings
from .dashboard import render_dashboard_html
from .extractor_factory import build_memory_extractor
from .openai_api import ChatCompletionsRequest, ResponsesRequest
from .compressor import MemoryCompressor
from .prompt_builder import PromptMemoryConfig
from .proxy_service import ChatProxyService, PreparedChatRequest, PreparedCompactionResponse
from .store import SQLiteMemoryStore
from .upstream import UpstreamOpenAIClient, UpstreamProxyError

COMPACT_ALIAS_SUNSET = "Wed, 30 Sep 2026 00:00:00 GMT"
PRIVATE_COMPACT_PATH = "/v1/proxy/responses/compact"
LEGACY_COMPACT_PATH = "/v1/responses/compact"


def create_app(
    settings: ProxySettings | None = None,
    *,
    service: ChatProxyService | None = None,
    upstream_client: UpstreamOpenAIClient | None = None,
) -> FastAPI:
    settings = settings or ProxySettings.from_env()
    upstream_client = upstream_client or UpstreamOpenAIClient(settings)
    store = None
    if settings.store_enabled:
        store = SQLiteMemoryStore(settings.store_db_path)
        store.init_db()
    if service is None:
        extractor = build_memory_extractor(settings)
        compressor = MemoryCompressor(
            recent_window=settings.recent_window,
            extractor=extractor,
            prompt_config=PromptMemoryConfig(max_tokens=settings.prompt_memory_max_tokens),
        )
        service = ChatProxyService(settings, compressor=compressor, store=store)

    app = FastAPI(title="Memory Compression Proxy", version="0.1.0")
    app.state.memory_store = store
    app.state.proxy_settings = settings

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=307)

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(render_dashboard_html())

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "compression_enabled": settings.compression_enabled,
            "recent_window": settings.recent_window,
            "recent_min_messages": settings.recent_min_messages,
            "recent_token_budget": settings.recent_token_budget,
            "salient_history_messages": settings.salient_history_messages,
            "salient_history_token_budget": settings.salient_history_token_budget,
            "min_history_messages": settings.min_history_messages,
            "prompt_memory_max_tokens": settings.prompt_memory_max_tokens,
            "store_enabled": settings.store_enabled,
            "store_db_path": settings.store_db_path if settings.store_enabled else None,
            "store_max_requests": settings.store_max_requests if settings.store_enabled else None,
            "session_auto_continue_enabled": settings.session_auto_continue_enabled,
            "session_stitching_window_seconds": settings.session_stitching_window_seconds,
            "legacy_compact_alias": {
                "path": LEGACY_COMPACT_PATH,
                "deprecated": True,
                "successor": PRIVATE_COMPACT_PATH,
                "sunset": COMPACT_ALIAS_SUNSET,
            },
        }

    @app.get("/api/dashboard/summary")
    async def dashboard_summary() -> JSONResponse:
        if store is None:
            return JSONResponse(
                {
                    "store_enabled": False,
                    "store_max_requests": None,
                    "summary": {
                        "sessions": 0,
                        "requests": 0,
                        "compressed_requests": 0,
                        "avg_estimated_savings_pct": None,
                    },
                }
            )
        return JSONResponse(
            {
                "store_enabled": True,
                "store_max_requests": settings.store_max_requests,
                "summary": store.get_dashboard_summary(),
            }
        )

    @app.get("/api/dashboard/sessions")
    async def dashboard_sessions(limit: int = Query(30, ge=1, le=200)) -> JSONResponse:
        if store is None:
            return JSONResponse({"store_enabled": False, "items": []})
        return JSONResponse(
            {
                "store_enabled": True,
                "items": [_session_row_to_dict(row) for row in store.list_sessions(limit=limit)],
            }
        )

    @app.get("/api/dashboard/requests")
    async def dashboard_requests(
        session_id: str | None = Query(default=None),
        limit: int = Query(40, ge=1, le=200),
    ) -> JSONResponse:
        if store is None:
            return JSONResponse({"store_enabled": False, "items": []})
        rows = store.list_request_audits(session_id=session_id, limit=limit)
        return JSONResponse(
            {
                "store_enabled": True,
                "items": [_request_row_to_dict(row, detailed=False) for row in rows],
            }
        )

    @app.get("/api/dashboard/requests/{request_id}")
    async def dashboard_request_detail(request_id: str) -> JSONResponse:
        if store is None:
            raise HTTPException(status_code=409, detail="store_disabled")
        row = store.get_request_audit(request_id)
        if row is None:
            raise HTTPException(status_code=404, detail="request_not_found")

        request_data = _request_row_to_dict(row, detailed=True)
        snapshot = None
        if row["snapshot_id"]:
            snapshot_row = store.get_working_memory_snapshot(row["snapshot_id"])
            if snapshot_row is not None:
                snapshot = _snapshot_row_to_dict(snapshot_row)
        raw_messages = [_row_to_dict(item) for item in store.list_turn_raw_messages(row["session_id"], row["request_id"])]
        events = [_event_row_to_dict(item) for item in store.list_turn_memory_events(row["session_id"], row["request_id"])]
        return JSONResponse(
            {
                "request": request_data,
                "snapshot": snapshot,
                "raw_messages": raw_messages,
                "events": events,
            }
        )

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        try:
            result = await upstream_client.list_models(
                forwarded_auth=request.headers.get("authorization"),
            )
        except UpstreamProxyError as error:
            raise HTTPException(status_code=error.status_code, detail=error.detail) from error
        return JSONResponse(
            status_code=result.status_code,
            content=result.body,
            headers={"content-type": result.content_type},
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request_body: ChatCompletionsRequest, request: Request):
        client_fingerprint = _client_fingerprint(request)
        prepared = service.prepare_chat_request(
            request_body,
            session_id=_resolve_session_id(
                request=request,
                payload=request_body.to_payload(),
                api_kind="chat_completions",
                store=store,
                settings=settings,
            ),
        )
        headers = _proxy_headers(prepared)
        try:
            if request_body.stream:
                streamed = await upstream_client.stream_chat_completions(
                    prepared.payload,
                    forwarded_auth=request.headers.get("authorization"),
                )
                _store_request_audit(
                    store,
                    prepared,
                    upstream_model=request_body.model,
                    status_code=streamed.status_code,
                    response_body=None,
                    max_requests=settings.store_max_requests,
                    client_fingerprint=client_fingerprint,
                )
                return StreamingResponse(
                    streamed.chunks,
                    status_code=streamed.status_code,
                    media_type=streamed.content_type,
                    headers=headers,
                )

            result = await upstream_client.chat_completions(
                prepared.payload,
                forwarded_auth=request.headers.get("authorization"),
            )
        except UpstreamProxyError as error:
            raise HTTPException(status_code=error.status_code, detail=error.detail) from error

        _store_request_audit(
            store,
            prepared,
            upstream_model=request_body.model,
            status_code=result.status_code,
            response_body=result.body,
            max_requests=settings.store_max_requests,
            client_fingerprint=client_fingerprint,
        )
        if 200 <= result.status_code < 400:
            service.capture_response_memory(
                api_kind=prepared.api_kind,
                session_id=prepared.session_id,
                request_id=prepared.request_id,
                response_body=result.body,
            )
        headers["content-type"] = result.content_type
        return JSONResponse(
            status_code=result.status_code,
            content=result.body,
            headers=headers,
        )

    @app.post("/v1/responses")
    async def responses(request_body: ResponsesRequest, request: Request):
        client_fingerprint = _client_fingerprint(request)
        prepared = service.prepare_responses_request(
            request_body,
            session_id=_resolve_session_id(
                request=request,
                payload=request_body.to_payload(),
                api_kind="responses",
                store=store,
                settings=settings,
            ),
        )
        headers = _proxy_headers(prepared)
        try:
            if request_body.stream:
                streamed = await upstream_client.stream_responses(
                    prepared.payload,
                    forwarded_auth=request.headers.get("authorization"),
                )
                _store_request_audit(
                    store,
                    prepared,
                    upstream_model=request_body.model,
                    status_code=streamed.status_code,
                    response_body=None,
                    max_requests=settings.store_max_requests,
                    client_fingerprint=client_fingerprint,
                )
                return StreamingResponse(
                    streamed.chunks,
                    status_code=streamed.status_code,
                    media_type=streamed.content_type,
                    headers=headers,
                )

            result = await upstream_client.responses(
                prepared.payload,
                forwarded_auth=request.headers.get("authorization"),
            )
        except UpstreamProxyError as error:
            raise HTTPException(status_code=error.status_code, detail=error.detail) from error

        _store_request_audit(
            store,
            prepared,
            upstream_model=request_body.model,
            status_code=result.status_code,
            response_body=result.body,
            max_requests=settings.store_max_requests,
            client_fingerprint=client_fingerprint,
        )
        if 200 <= result.status_code < 400:
            service.capture_response_memory(
                api_kind=prepared.api_kind,
                session_id=prepared.session_id,
                request_id=prepared.request_id,
                response_body=result.body,
            )
        headers["content-type"] = result.content_type
        return JSONResponse(
            status_code=result.status_code,
            content=result.body,
            headers=headers,
        )

    @app.websocket("/v1/responses")
    async def responses_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        client_fingerprint = _scope_client_fingerprint(
            client_host=websocket.client.host if websocket.client else "",
            headers=websocket.headers,
        )
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break

                text = message.get("text")
                if text is None and message.get("bytes") is not None:
                    try:
                        text = message["bytes"].decode("utf-8")
                    except UnicodeDecodeError:
                        await websocket.send_json({"type": "error", "error": {"message": "invalid_binary_payload"}})
                        continue
                if not text:
                    continue

                try:
                    envelope = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "error": {"message": "invalid_json"}})
                    continue

                if isinstance(envelope, dict) and envelope.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                payload = _coerce_websocket_responses_payload(envelope)
                if payload is None:
                    await websocket.send_json({"type": "error", "error": {"message": "invalid_request_shape"}})
                    continue

                try:
                    request_body = ResponsesRequest.model_validate(payload)
                except ValidationError as error:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": {"message": "invalid_request", "detail": error.errors(include_url=False)},
                        }
                    )
                    continue

                prepared = service.prepare_responses_request(
                    request_body,
                    session_id=_resolve_scope_session_id(
                        headers=websocket.headers,
                        payload=request_body.to_payload(),
                        api_kind="responses",
                        store=store,
                        settings=settings,
                        client_fingerprint=client_fingerprint,
                    ),
                )

                if request_body.stream:
                    try:
                        streamed = await upstream_client.stream_responses(
                            prepared.payload,
                            forwarded_auth=websocket.headers.get("authorization"),
                        )
                    except UpstreamProxyError as error:
                        await websocket.send_json(
                            {"type": "error", "error": {"message": error.detail, "status_code": error.status_code}}
                        )
                        continue

                    _store_request_audit(
                        store,
                        prepared,
                        upstream_model=request_body.model,
                        status_code=streamed.status_code,
                        response_body=None,
                        max_requests=settings.store_max_requests,
                        client_fingerprint=client_fingerprint,
                    )
                    async for chunk in streamed.chunks:
                        await websocket.send_text(chunk.decode("utf-8", errors="replace"))
                    continue

                try:
                    result = await upstream_client.responses(
                        prepared.payload,
                        forwarded_auth=websocket.headers.get("authorization"),
                    )
                except UpstreamProxyError as error:
                    await websocket.send_json(
                        {"type": "error", "error": {"message": error.detail, "status_code": error.status_code}}
                    )
                    continue

                _store_request_audit(
                    store,
                    prepared,
                    upstream_model=request_body.model,
                    status_code=result.status_code,
                    response_body=result.body,
                    max_requests=settings.store_max_requests,
                    client_fingerprint=client_fingerprint,
                )
                if 200 <= result.status_code < 400:
                    service.capture_response_memory(
                        api_kind=prepared.api_kind,
                        session_id=prepared.session_id,
                        request_id=prepared.request_id,
                        response_body=result.body,
                    )
                await websocket.send_json(
                    {
                        "type": "response",
                        "response": result.body,
                        "proxy": _proxy_headers(prepared),
                    }
                )
        except WebSocketDisconnect:
            return

    @app.post(PRIVATE_COMPACT_PATH)
    async def responses_compact_private(request_body: ResponsesRequest, request: Request) -> JSONResponse:
        prepared = service.compact_responses_request(
            request_body,
            session_id=_resolve_session_id(
                request=request,
                payload=request_body.to_payload(),
                api_kind="responses",
                store=store,
                settings=settings,
            ),
        )
        headers = _proxy_headers(prepared)
        headers["content-type"] = "application/json"
        return JSONResponse(
            status_code=200,
            content=prepared.body,
            headers=headers,
        )

    @app.post(LEGACY_COMPACT_PATH)
    async def responses_compact_legacy(request_body: ResponsesRequest, request: Request) -> JSONResponse:
        prepared = service.compact_responses_request(
            request_body,
            session_id=_resolve_session_id(
                request=request,
                payload=request_body.to_payload(),
                api_kind="responses",
                store=store,
                settings=settings,
            ),
        )
        headers = _proxy_headers(prepared)
        headers["content-type"] = "application/json"
        headers["deprecation"] = "true"
        headers["sunset"] = COMPACT_ALIAS_SUNSET
        headers["link"] = f'<{PRIVATE_COMPACT_PATH}>; rel="successor-version"'
        return JSONResponse(
            status_code=200,
            content=_with_response_object(
                _with_legacy_warning(prepared.body),
                "response.compaction",
            ),
            headers=headers,
        )

    return app


def main() -> None:
    import uvicorn

    settings = ProxySettings.from_env()
    uvicorn.run(
        create_app(settings),
        host="0.0.0.0",
        port=8000,
    )


def _proxy_headers(prepared: PreparedChatRequest | PreparedCompactionResponse) -> dict[str, str]:
    headers = {
        "x-memory-proxy-compressed": str(prepared.compressed).lower(),
        "x-memory-proxy-reason": prepared.compression_reason,
        "x-memory-proxy-history-dropped": str(prepared.dropped_message_count),
        "x-memory-proxy-session-id": prepared.session_id,
        "x-memory-proxy-request-id": prepared.request_id,
        "x-memory-proxy-estimated-savings-pct": str(prepared.estimated_savings_pct or 0.0),
        "x-memory-proxy-estimated-before-tokens": str(prepared.estimated_input_tokens_before or 0),
        "x-memory-proxy-estimated-after-tokens": str(prepared.estimated_input_tokens_after or 0),
    }
    if prepared.snapshot_id:
        headers["x-memory-proxy-snapshot-id"] = prepared.snapshot_id
    return headers


def _with_response_object(body: dict[str, Any], object_type: str) -> dict[str, Any]:
    return {**body, "object": object_type}


def _with_legacy_warning(body: dict[str, Any]) -> dict[str, Any]:
    return {
        **body,
        "warning": {
            "code": "deprecated_endpoint",
            "message": (
                f"{LEGACY_COMPACT_PATH} is deprecated; use {PRIVATE_COMPACT_PATH} "
                f"before {COMPACT_ALIAS_SUNSET}."
            ),
            "successor": PRIVATE_COMPACT_PATH,
            "sunset": COMPACT_ALIAS_SUNSET,
        },
    }


def _store_request_audit(
    store: SQLiteMemoryStore | None,
    prepared: PreparedChatRequest,
    *,
    upstream_model: str,
    status_code: int,
    response_body: dict[str, Any] | None,
    max_requests: int,
    client_fingerprint: str | None = None,
) -> None:
    if store is None:
        return
    store.upsert_session(
        prepared.session_id,
        client="proxy",
        upstream_model=upstream_model,
    )
    usage = _extract_usage(prepared.api_kind, response_body)
    upstream_input_tokens, upstream_output_tokens, upstream_total_tokens = _usage_numbers(prepared.api_kind, usage)
    store.insert_request_audit(
        request_id=prepared.request_id,
        session_id=prepared.session_id,
        api_kind=prepared.api_kind,
        upstream_model=upstream_model,
        compressed=prepared.compressed,
        compression_reason=prepared.compression_reason,
        dropped_message_count=prepared.dropped_message_count,
        recent_message_count=prepared.recent_message_count,
        snapshot_id=prepared.snapshot_id,
        original_payload=prepared.original_payload,
        forwarded_payload=prepared.payload,
        prompt_memory=prepared.prompt_memory,
        estimated_input_tokens_before=prepared.estimated_input_tokens_before,
        estimated_input_tokens_after=prepared.estimated_input_tokens_after,
        estimated_savings_pct=prepared.estimated_savings_pct,
        token_counter=prepared.token_counter,
        upstream_usage=usage,
        upstream_input_tokens=upstream_input_tokens,
        upstream_output_tokens=upstream_output_tokens,
        upstream_total_tokens=upstream_total_tokens,
        client_fingerprint=client_fingerprint,
        upstream_response_id=_extract_response_id(response_body),
        status_code=status_code,
        response_preview=_extract_response_preview(prepared.api_kind, response_body),
    )
    store.prune_request_history(max_requests)


def _resolve_session_id(
    *,
    request: Request,
    payload: dict[str, Any],
    api_kind: str,
    store: SQLiteMemoryStore | None,
    settings: ProxySettings,
) -> str | None:
    return _resolve_scope_session_id(
        headers=request.headers,
        payload=payload,
        api_kind=api_kind,
        store=store,
        settings=settings,
        client_fingerprint=_client_fingerprint(request),
    )


def _resolve_scope_session_id(
    *,
    headers: Any,
    payload: dict[str, Any],
    api_kind: str,
    store: SQLiteMemoryStore | None,
    settings: ProxySettings,
    client_fingerprint: str | None,
) -> str | None:
    header_session = _extract_header_session_hint(headers, api_kind=api_kind)
    if header_session:
        return header_session

    body_session = _extract_payload_session_hint(payload)
    if body_session:
        return body_session

    if store is None:
        return None

    previous_response_id = _clean_session_hint(_extract_string_field(payload, "previous_response_id"))
    if api_kind == "responses" and previous_response_id:
        previous_session = store.find_session_id_by_upstream_response_id(previous_response_id)
        if previous_session:
            return previous_session

    if not settings.session_auto_continue_enabled or not client_fingerprint:
        return None
    return store.find_recent_session_by_client_fingerprint(
        client_fingerprint,
        max_age_seconds=settings.session_stitching_window_seconds,
    )


def _extract_header_session_hint(headers: Any, *, api_kind: str) -> str | None:
    for key in ("x-session-id",):
        value = _clean_session_hint(headers.get(key))
        if value:
            return value

    if api_kind == "responses":
        for key in ("session_id", "conversation_id"):
            value = _clean_session_hint(headers.get(key))
            if value:
                return value

    return None


def _extract_usage(api_kind: str, response_body: dict[str, Any] | None) -> dict[str, Any] | None:
    if not response_body or not isinstance(response_body, dict):
        return None
    usage = response_body.get("usage")
    return usage if isinstance(usage, dict) else None


def _usage_numbers(api_kind: str, usage: dict[str, Any] | None) -> tuple[int | None, int | None, int | None]:
    if usage is None:
        return None, None, None
    if api_kind == "responses":
        return _maybe_int(usage.get("input_tokens")), _maybe_int(usage.get("output_tokens")), _maybe_int(
            usage.get("total_tokens")
        )
    return _maybe_int(usage.get("prompt_tokens")), _maybe_int(usage.get("completion_tokens")), _maybe_int(
        usage.get("total_tokens")
    )


def _extract_response_preview(api_kind: str, response_body: dict[str, Any] | None) -> str | None:
    if not response_body or not isinstance(response_body, dict):
        return None
    if api_kind == "responses":
        output = response_body.get("output")
        if not isinstance(output, list):
            return None
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
        if texts:
            return "\n".join(texts)[:1200]
        return None

    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content[:1200]
    return json.dumps(content, ensure_ascii=False)[:1200] if content is not None else None


def _extract_response_id(response_body: dict[str, Any] | None) -> str | None:
    if not response_body or not isinstance(response_body, dict):
        return None
    value = response_body.get("id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _extract_payload_session_hint(payload: dict[str, Any]) -> str | None:
    for key in ("session_id", "conversation_id", "thread_id"):
        value = _clean_session_hint(_extract_string_field(payload, key))
        if value:
            return value
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("session_id", "conversation_id", "thread_id", "client_session_id", "codex_session_id"):
            value = _clean_session_hint(_extract_string_field(metadata, key))
            if value:
                return value
    return None


def _extract_string_field(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _clean_session_hint(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) <= 160:
        return cleaned
    return f"sess_ext_{hashlib.sha256(cleaned.encode('utf-8')).hexdigest()[:24]}"


def _client_fingerprint(request: Request) -> str | None:
    return _scope_client_fingerprint(
        client_host=request.client.host if request.client else "",
        headers=request.headers,
    )


def _scope_client_fingerprint(*, client_host: str, headers: Any) -> str | None:
    parts = [
        client_host,
        headers.get("user-agent", ""),
        headers.get("authorization", ""),
    ]
    if not any(parts):
        return None
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"cfp_{digest[:24]}"


def _coerce_websocket_responses_payload(envelope: Any) -> dict[str, Any] | None:
    if not isinstance(envelope, dict):
        return None
    for candidate in (envelope, envelope.get("request"), envelope.get("payload"), envelope.get("params")):
        if isinstance(candidate, dict) and "model" in candidate and "input" in candidate:
            return candidate
    return None


def _session_row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "session_id": row["session_id"],
        "client": row["client"],
        "upstream_model": row["upstream_model"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "request_count": row["request_count"],
        "compressed_request_count": row["compressed_request_count"],
        "last_request_at": row["last_request_at"],
    }


def _request_row_to_dict(row: Any, *, detailed: bool) -> dict[str, Any]:
    data = {
        "request_id": row["request_id"],
        "session_id": row["session_id"],
        "api_kind": row["api_kind"],
        "upstream_model": row["upstream_model"],
        "compressed": bool(row["compressed"]),
        "compression_reason": row["compression_reason"],
        "dropped_message_count": row["dropped_message_count"],
        "recent_message_count": row["recent_message_count"],
        "snapshot_id": row["snapshot_id"],
        "prompt_memory": row["prompt_memory"],
        "estimated_input_tokens_before": row["estimated_input_tokens_before"],
        "estimated_input_tokens_after": row["estimated_input_tokens_after"],
        "estimated_savings_pct": row["estimated_savings_pct"],
        "token_counter": row["token_counter"],
        "upstream_input_tokens": row["upstream_input_tokens"],
        "upstream_output_tokens": row["upstream_output_tokens"],
        "upstream_total_tokens": row["upstream_total_tokens"],
        "status_code": row["status_code"],
        "response_preview": row["response_preview"],
        "created_at": row["created_at"],
    }
    if detailed:
        data["original_payload"] = json.loads(row["original_payload_json"])
        data["forwarded_payload"] = json.loads(row["forwarded_payload_json"])
        data["upstream_usage"] = json.loads(row["upstream_usage_json"]) if row["upstream_usage_json"] else None
    return data


def _snapshot_row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "snapshot_id": row["snapshot_id"],
        "session_id": row["session_id"],
        "turn_id": row["turn_id"],
        "memory_json": json.loads(row["memory_json"]),
        "memory_dsl": row["memory_dsl"],
        "created_at": row["created_at"],
    }


def _event_row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "event_id": row["event_id"],
        "session_id": row["session_id"],
        "turn_id": row["turn_id"],
        "source_message_ids": json.loads(row["source_message_ids"]),
        "actor": row["actor"],
        "type": row["type"],
        "action": row["action"],
        "status": row["status"],
        "subject": row["subject"],
        "details": json.loads(row["details_json"]),
        "confidence": row["confidence"],
        "supersedes": row["supersedes"],
        "created_at": row["created_at"],
    }


def _row_to_dict(row: Any) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
