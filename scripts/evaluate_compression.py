from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_proxy import MemoryCompressor, RawMessage  # noqa: E402


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")


def build_token_counter(use_tiktoken: bool) -> tuple[str, Callable[[str], int]]:
    if not use_tiktoken:
        return "approx:v1", approx_tokens

    import tiktoken  # type: ignore

    encoding = tiktoken.get_encoding("o200k_base")

    def count_with_tiktoken(text: str) -> int:
        return len(encoding.encode(text))

    return "tiktoken:o200k_base", count_with_tiktoken


def approx_tokens(text: str) -> int:
    total = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if CJK_RE.match(ch):
            total += 1
            i += 1
            continue
        if ch.isspace():
            i += 1
            continue
        matched = ASCII_WORD_RE.match(text, i)
        if matched:
            total += max(1, math.ceil(len(matched.group(0)) / 4))
            i = matched.end()
            continue
        total += 1
        i += 1
    return total


def serialize_messages(messages: list[RawMessage]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def build_scenarios() -> dict[str, list[RawMessage]]:
    return {
        "planning_loop": [
            RawMessage("m1", "user", "我想做一个 OpenAI 兼容代理，重点是压缩记忆来减少 token 消耗。"),
            RawMessage("m2", "assistant", "我建议第一版先使用 Python + FastAPI。"),
            RawMessage("m3", "user", "可以，必须支持 OpenAI-compatible 接口，而且不要和现有 CLIProxyAPI 混在一起。"),
            RawMessage("m4", "assistant", "我已经完成了方案文档，下一步准备实现 Memory DSL。"),
            RawMessage("m5", "tool", "tests passed for 文档生成器"),
            RawMessage("m6", "user", "接下来重点是先梳理语法规则，再做记忆压缩器。"),
            RawMessage("m7", "assistant", "我已经完成了 Memory DSL v0.1，下一步会实现规则抽取器。"),
            RawMessage("m8", "tool", "3 tests passed for Memory DSL parser"),
        ],
        "coding_loop": [
            RawMessage("m1", "user", "我要做一个 event-sourced memory proxy for agents。"),
            RawMessage("m2", "assistant", "我建议先定义 Session、RawMessage、MemoryEvent、WorkingMemory。"),
            RawMessage("m3", "user", "重点是区分 AI 说完成了，和工具真正验证完成。"),
            RawMessage("m4", "assistant", "我已经完成了 models.py，下一步准备实现 reducer.py。"),
            RawMessage("m5", "tool", "tests passed for models.py dataclasses"),
            RawMessage("m6", "assistant", "我已经完成了 reducer.py，下一步准备实现 dsl.py。"),
            RawMessage("m7", "tool", "1 test failed for dsl.py parser: invalid escaped pipe handling"),
            RawMessage("m8", "assistant", "dsl.py 这里报错了，我下一步准备修 parser 的转义逻辑。"),
            RawMessage("m9", "assistant", "我已经完成了 dsl.py parser 修复。"),
            RawMessage("m10", "tool", "4 tests passed for dsl.py parser"),
            RawMessage("m11", "assistant", "我已经完成了 store.py，下一步准备实现 schema.sql。"),
            RawMessage("m12", "tool", "created /workspace/workflow-memory-proxy/storage/schema.sql"),
        ],
        "messy_natural": [
            RawMessage("m1", "user", "我现在比较担心的是咱们怎么实现这个压缩，就是把用户的话转换成结构化语言。"),
            RawMessage("m2", "assistant", "这个担心很正常，而且我觉得你正好抓到了项目最难、也最核心的一层。"),
            RawMessage("m3", "assistant", "我建议先别想着把所有自然语言都结构化，只抽高价值信息。"),
            RawMessage("m4", "user", "嗯，这样会不会漏掉重要上下文？"),
            RawMessage("m5", "assistant", "有这个风险，所以原文要保留，事件要带 source_message_id 和 confidence。"),
            RawMessage("m6", "user", "那语言咱们选择英语还是什么比较适合？"),
            RawMessage("m7", "assistant", "我建议字段名和状态名用英语，subject 可以保留原语言。"),
            RawMessage("m8", "user", "好，那咱们先实现语法规则和记忆压缩器。"),
            RawMessage("m9", "assistant", "我已经完成了第一版骨架，下一步准备接 LLM 抽取器。"),
            RawMessage("m10", "tool", "7 tests passed for memory compressor"),
        ],
    }


def evaluate_scenario(name: str, messages: list[RawMessage], recent_window: int, counter: Callable[[str], int]) -> dict[str, object]:
    compressor = MemoryCompressor(recent_window=recent_window)
    result = compressor.compress(messages, session_id=f"sess_{name}")
    raw_all = serialize_messages(messages)
    current_prompt = f"MEMORY\n{result.memory_dsl}\nRECENT\n{serialize_messages(result.recent_messages)}".strip()

    history = messages[:-recent_window] if recent_window else messages
    recent = messages[-recent_window:] if recent_window else []
    history_result = MemoryCompressor(recent_window=0).compress(history, session_id=f"sess_{name}_history") if history else None
    realistic_prompt = (
        (f"MEMORY\n{history_result.memory_dsl}\n" if history_result and history_result.memory_dsl else "")
        + (f"RECENT\n{serialize_messages(recent)}" if recent else "")
    ).strip()

    raw_tokens = counter(raw_all)
    current_tokens = counter(current_prompt)
    realistic_tokens = counter(realistic_prompt)

    return {
        "scenario": name,
        "message_count": len(messages),
        "events": len(result.events),
        "raw_tokens": raw_tokens,
        "current_prompt_tokens": current_tokens,
        "current_savings_pct": round((raw_tokens - current_tokens) / raw_tokens * 100, 2),
        "history_only_prompt_tokens": realistic_tokens,
        "history_only_savings_pct": round((raw_tokens - realistic_tokens) / raw_tokens * 100, 2),
        "current_memory_lines": len([line for line in result.memory_dsl.splitlines() if line.strip()]),
        "history_memory_lines": len(
            [line for line in (history_result.memory_dsl if history_result else "").splitlines() if line.strip()]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory compression effectiveness on sample conversations.")
    parser.add_argument("--recent-window", type=int, default=4, help="Number of recent raw messages kept in prompt.")
    parser.add_argument(
        "--use-tiktoken",
        action="store_true",
        help="Use tiktoken for counting. Disabled by default because some environments may hang on import.",
    )
    args = parser.parse_args()

    counter_name, counter = build_token_counter(use_tiktoken=args.use_tiktoken)
    scenarios = build_scenarios()
    results = [evaluate_scenario(name, messages, args.recent_window, counter) for name, messages in scenarios.items()]

    print(f"token_counter={counter_name}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
