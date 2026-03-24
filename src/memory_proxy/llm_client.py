from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .llm_extractor import ExtractionPrompt, LLMExtractionClient


@dataclass(slots=True)
class OpenAICompatLLMClient(LLMExtractionClient):
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 60.0

    def complete(self, prompt: ExtractionPrompt) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "stream": False,
        }
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(base_url=_normalize_base_url(self.base_url), timeout=self.timeout_seconds) as client:
            response = client.post("chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM extractor response is missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("LLM extractor response is missing message content")
        content = message.get("content")
        text = _coerce_message_text(content)
        if not text:
            raise ValueError("LLM extractor returned empty content")
        return text


def _coerce_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            text_parts.append(text.strip())
            continue
        inner = item.get("content")
        if isinstance(inner, str) and inner.strip():
            text_parts.append(inner.strip())
    return "\n".join(text_parts).strip()


def _normalize_base_url(value: str) -> str:
    return value.rstrip("/") + "/"
