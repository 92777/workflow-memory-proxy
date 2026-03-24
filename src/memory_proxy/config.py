from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MEMORY_SYSTEM_PROMPT = "Older summary. Recent messages override. Reply naturally."


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(slots=True)
class ProxySettings:
    upstream_base_url: str = "https://api.openai.com/v1/"
    upstream_api_key: str | None = None
    compression_enabled: bool = True
    recent_window: int = 2
    recent_min_messages: int = 1
    recent_token_budget: int = 120
    salient_history_messages: int = 1
    salient_history_token_budget: int = 96
    min_history_messages: int = 2
    prompt_memory_max_tokens: int = 160
    store_enabled: bool = False
    store_db_path: str = str(Path("memory_proxy.db").resolve())
    store_max_requests: int = 100
    session_auto_continue_enabled: bool = True
    session_stitching_window_seconds: int = 1800
    timeout_seconds: float = 60.0
    memory_system_prompt: str = DEFAULT_MEMORY_SYSTEM_PROMPT
    extractor_mode: str = "rule"
    extractor_llm_base_url: str | None = None
    extractor_llm_api_key: str | None = None
    extractor_llm_model: str | None = None
    extractor_llm_timeout_seconds: float = 30.0
    extractor_llm_min_confidence: float = 0.75

    @classmethod
    def from_env(cls) -> "ProxySettings":
        return cls(
            upstream_base_url=_normalize_base_url(
                os.getenv("MCPROXY_UPSTREAM_BASE_URL", "https://api.openai.com/v1/")
            ),
            upstream_api_key=os.getenv("MCPROXY_UPSTREAM_API_KEY"),
            compression_enabled=_env_bool("MCPROXY_COMPRESSION_ENABLED", True),
            recent_window=int(os.getenv("MCPROXY_RECENT_WINDOW", "2")),
            recent_min_messages=int(os.getenv("MCPROXY_RECENT_MIN_MESSAGES", "1")),
            recent_token_budget=int(os.getenv("MCPROXY_RECENT_TOKEN_BUDGET", "120")),
            salient_history_messages=int(os.getenv("MCPROXY_SALIENT_HISTORY_MESSAGES", "1")),
            salient_history_token_budget=int(os.getenv("MCPROXY_SALIENT_HISTORY_TOKEN_BUDGET", "96")),
            min_history_messages=int(os.getenv("MCPROXY_MIN_HISTORY_MESSAGES", "2")),
            prompt_memory_max_tokens=int(os.getenv("MCPROXY_PROMPT_MEMORY_MAX_TOKENS", "160")),
            store_enabled=_env_bool("MCPROXY_STORE_ENABLED", False),
            store_db_path=os.getenv("MCPROXY_STORE_DB_PATH", str(Path("memory_proxy.db").resolve())),
            store_max_requests=int(os.getenv("MCPROXY_STORE_MAX_REQUESTS", "100")),
            session_auto_continue_enabled=_env_bool("MCPROXY_SESSION_AUTO_CONTINUE_ENABLED", True),
            session_stitching_window_seconds=int(
                os.getenv("MCPROXY_SESSION_STITCHING_WINDOW_SECONDS", "1800")
            ),
            timeout_seconds=float(os.getenv("MCPROXY_TIMEOUT_SECONDS", "60")),
            memory_system_prompt=os.getenv(
                "MCPROXY_MEMORY_SYSTEM_PROMPT",
                DEFAULT_MEMORY_SYSTEM_PROMPT,
            ),
            extractor_mode=os.getenv("MCPROXY_EXTRACTOR_MODE", "rule").strip().lower(),
            extractor_llm_base_url=_optional_base_url(os.getenv("MCPROXY_EXTRACTOR_LLM_BASE_URL")),
            extractor_llm_api_key=os.getenv("MCPROXY_EXTRACTOR_LLM_API_KEY"),
            extractor_llm_model=_optional_str(os.getenv("MCPROXY_EXTRACTOR_LLM_MODEL")),
            extractor_llm_timeout_seconds=float(os.getenv("MCPROXY_EXTRACTOR_LLM_TIMEOUT_SECONDS", "30")),
            extractor_llm_min_confidence=float(os.getenv("MCPROXY_EXTRACTOR_LLM_MIN_CONFIDENCE", "0.75")),
        )


def _normalize_base_url(value: str) -> str:
    value = value.rstrip("/") + "/"
    return value


def _optional_base_url(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return _normalize_base_url(value)


def _optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
