from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .config import ProxySettings

MODEL_ALIASES = {
    "gpt-5-mini": "gpt-5.4-mini",
    "gpt-5-nano": "gpt-5.4-mini",
    "gpt-4.1-nano": "gpt-5.4-mini",
    "gpt-4o-mini": "gpt-5.4-mini",
    "gpt-4-turbo": "gpt-5",
    "gpt-4": "gpt-5",
    "gpt-3.5-turbo": "gpt-5.4-mini",
}


class UpstreamProxyError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(slots=True)
class UpstreamJSONResponse:
    status_code: int
    body: dict[str, Any]
    content_type: str


@dataclass(slots=True)
class UpstreamStreamResponse:
    status_code: int
    content_type: str
    chunks: AsyncIterator[bytes]


class UpstreamOpenAIClient:
    def __init__(
        self,
        settings: ProxySettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def chat_completions(
        self,
        payload: dict[str, Any],
        *,
        forwarded_auth: str | None,
    ) -> UpstreamJSONResponse:
        payload = _normalize_model_payload(payload)
        async with self._client() as client:
            response = await client.post(
                "chat/completions",
                json=payload,
                headers=self._headers(forwarded_auth),
            )
        if response.status_code >= 400:
            raise UpstreamProxyError(response.status_code, response.text)
        return UpstreamJSONResponse(
            status_code=response.status_code,
            body=response.json(),
            content_type=response.headers.get("content-type", "application/json"),
        )

    async def stream_chat_completions(
        self,
        payload: dict[str, Any],
        *,
        forwarded_auth: str | None,
    ) -> UpstreamStreamResponse:
        payload = _normalize_model_payload(payload)
        client = self._client()
        request = client.build_request(
            "POST",
            "chat/completions",
            json=payload,
            headers=self._headers(forwarded_auth),
        )
        response = await client.send(request, stream=True)
        if response.status_code >= 400:
            body = await response.aread()
            await response.aclose()
            await client.aclose()
            raise UpstreamProxyError(response.status_code, body.decode("utf-8", errors="replace"))

        async def iterator() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return UpstreamStreamResponse(
            status_code=response.status_code,
            content_type=response.headers.get("content-type", "text/event-stream"),
            chunks=iterator(),
        )

    async def responses(
        self,
        payload: dict[str, Any],
        *,
        forwarded_auth: str | None,
    ) -> UpstreamJSONResponse:
        payload = _normalize_model_payload(payload)
        async with self._client() as client:
            response = await client.post(
                "responses",
                json=payload,
                headers=self._headers(forwarded_auth),
            )
        if response.status_code >= 400:
            raise UpstreamProxyError(response.status_code, response.text)
        return UpstreamJSONResponse(
            status_code=response.status_code,
            body=response.json(),
            content_type=response.headers.get("content-type", "application/json"),
        )

    async def stream_responses(
        self,
        payload: dict[str, Any],
        *,
        forwarded_auth: str | None,
    ) -> UpstreamStreamResponse:
        payload = _normalize_model_payload(payload)
        client = self._client()
        request = client.build_request(
            "POST",
            "responses",
            json=payload,
            headers=self._headers(forwarded_auth),
        )
        response = await client.send(request, stream=True)
        if response.status_code >= 400:
            body = await response.aread()
            await response.aclose()
            await client.aclose()
            raise UpstreamProxyError(response.status_code, body.decode("utf-8", errors="replace"))

        async def iterator() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return UpstreamStreamResponse(
            status_code=response.status_code,
            content_type=response.headers.get("content-type", "text/event-stream"),
            chunks=iterator(),
        )

    async def list_models(self, *, forwarded_auth: str | None) -> UpstreamJSONResponse:
        async with self._client() as client:
            response = await client.get("models", headers=self._headers(forwarded_auth))
        if response.status_code >= 400:
            raise UpstreamProxyError(response.status_code, response.text)
        return UpstreamJSONResponse(
            status_code=response.status_code,
            body=response.json(),
            content_type=response.headers.get("content-type", "application/json"),
        )

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.settings.upstream_base_url,
            timeout=self.settings.timeout_seconds,
            transport=self.transport,
        )

    def _headers(self, forwarded_auth: str | None) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        auth_value = forwarded_auth or _bearer(self.settings.upstream_api_key)
        if auth_value:
            headers["authorization"] = auth_value
        return headers


def _bearer(api_key: str | None) -> str | None:
    if not api_key:
        return None
    return f"Bearer {api_key}"


def _normalize_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.get("model")
    if not isinstance(model, str):
        return payload
    normalized_model = MODEL_ALIASES.get(model, model)
    if normalized_model == model:
        return payload
    updated = dict(payload)
    updated["model"] = normalized_model
    return updated
