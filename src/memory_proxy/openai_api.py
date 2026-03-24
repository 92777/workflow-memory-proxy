from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    name: str | None = None
    tool_call_id: str | None = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False

    model_config = ConfigDict(extra="allow")

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class ResponsesRequest(BaseModel):
    model: str
    input: Any
    instructions: str | None = None
    stream: bool = False

    model_config = ConfigDict(extra="allow")

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)
