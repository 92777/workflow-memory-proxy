from __future__ import annotations

from dataclasses import asdict
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_proxy import MemoryCompressor, RawMessage  # noqa: E402


def main() -> None:
    compressor = MemoryCompressor(recent_window=2)
    messages = [
        RawMessage(
            message_id="msg_001",
            role="user",
            content="我想做一个 OpenAI 兼容代理，重点是压缩记忆来减少 token 消耗。",
        ),
        RawMessage(
            message_id="msg_002",
            role="assistant",
            content="我已经完成了会话存储，下一步准备实现 prompt builder。",
        ),
        RawMessage(
            message_id="msg_003",
            role="tool",
            content="3 tests passed for 会话存储",
        ),
    ]
    result = compressor.compress(messages, session_id="sess_demo")

    print("== Event DSL ==")
    print(result.event_dsl)
    print("\n== Prompt Memory ==")
    print(result.memory_dsl)
    print("\n== Audit Memory DSL ==")
    print(result.audit_memory_dsl)
    print("\n== Working Memory JSON ==")
    print(json.dumps(asdict(result.working_memory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
