from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from memory_proxy import (  # noqa: E402
    ProxySettings,
    SQLiteMemoryStore,
    UpstreamOpenAIClient,
    create_app,
)


def build_upstream_app() -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        return JSONResponse({"object": "list", "data": [{"id": "gpt-test"}]})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        if payload.get("stream"):
            chunks = [
                b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                b"data: [DONE]\n\n",
            ]

            async def iterator():
                for chunk in chunks:
                    yield chunk

            return StreamingResponse(iterator(), media_type="text/event-stream")

        return JSONResponse(
            {
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "我已经完成了会话存储。下一步准备实现代理路由。",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                "echo_messages": payload["messages"],
            }
        )

    @app.post("/v1/responses")
    async def responses(request: Request):
        payload = await request.json()
        if payload.get("stream"):
            chunks = [
                b"event: response.output_text.delta\ndata: {\"delta\":\"hi\"}\n\n",
                b"data: [DONE]\n\n",
            ]

            async def iterator():
                for chunk in chunks:
                    yield chunk

            return StreamingResponse(iterator(), media_type="text/event-stream")

        return JSONResponse(
            {
                "id": "resp_test",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "我已经完成了会话存储。下一步准备实现代理路由。",
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
                "echo_input": payload.get("input"),
                "echo_instructions": payload.get("instructions"),
            }
        )

    return app


class ProxyServerTests(unittest.TestCase):
    def make_client(self, **settings_overrides: object) -> TestClient:
        defaults = {
            "upstream_base_url": "http://upstream/v1/",
            "compression_enabled": True,
            "recent_window": 2,
            "min_history_messages": 2,
        }
        defaults.update(settings_overrides)
        settings = ProxySettings(**defaults)
        upstream = build_upstream_app()
        upstream_client = UpstreamOpenAIClient(
            settings,
            transport=httpx.ASGITransport(app=upstream),
        )
        app = create_app(
            settings,
            upstream_client=upstream_client,
        )
        return TestClient(app)

    def test_models_endpoint_passthrough(self) -> None:
        client = self.make_client()
        response = client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "gpt-test")

    def test_from_env_uses_real_default_memory_prompt(self) -> None:
        original = os.environ.pop("MCPROXY_MEMORY_SYSTEM_PROMPT", None)
        try:
            settings = ProxySettings.from_env()
        finally:
            if original is not None:
                os.environ["MCPROXY_MEMORY_SYSTEM_PROMPT"] = original
        self.assertEqual(
            settings.memory_system_prompt,
            "Older summary. Recent messages override. Reply naturally.",
        )

    def test_dashboard_page_renders(self) -> None:
        client = self.make_client()
        response = client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Memory Proxy Dashboard", response.text)
        self.assertIn('id="lang-toggle"', response.text)
        self.assertIn('id="compact-alias-status"', response.text)
        self.assertIn("记忆压缩代理面板", response.text)

    def test_health_reports_legacy_compact_alias_status(self) -> None:
        client = self.make_client()
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["recent_min_messages"], 1)
        self.assertEqual(body["recent_token_budget"], 120)
        self.assertEqual(body["salient_history_token_budget"], 96)
        self.assertEqual(body["legacy_compact_alias"]["path"], "/v1/responses/compact")
        self.assertTrue(body["legacy_compact_alias"]["deprecated"])
        self.assertEqual(body["legacy_compact_alias"]["successor"], "/v1/proxy/responses/compact")
        self.assertEqual(body["legacy_compact_alias"]["sunset"], "Wed, 30 Sep 2026 00:00:00 GMT")

    def test_chat_endpoint_compresses_history_conservatively(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "assistant", "content": "建议第一版先使用 Python + FastAPI。"},
                {"role": "user", "content": "重点是压缩记忆，但不要明显影响精度。"},
                {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现代理路由。"},
                {"role": "user", "content": "继续实现 /v1/chat/completions。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "true")
        self.assertTrue(response.headers["x-memory-proxy-session-id"].startswith("sess_"))
        self.assertTrue(response.headers["x-memory-proxy-request-id"].startswith("req_"))
        echoed = response.json()["echo_messages"]
        self.assertEqual(echoed[0]["role"], "system")
        self.assertEqual(echoed[0]["content"], "You are a helpful assistant.")
        self.assertEqual(echoed[1]["role"], "system")
        self.assertIn("Older summary. Recent messages override.", echoed[1]["content"])
        self.assertIn("GOAL:", echoed[1]["content"])
        self.assertEqual(echoed[-1]["content"], "继续实现 /v1/chat/completions。")
        self.assertLess(len(echoed), len(payload["messages"]))

    def test_chat_endpoint_keeps_salient_history_message_raw(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "assistant", "content": "建议第一版先使用 Python + FastAPI。"},
                {"role": "user", "content": "必须尽量不影响精度。"},
                {"role": "assistant", "content": "请记住文件 /tmp/migration-plan.md 里有迁移清单。"},
                {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现代理路由。"},
                {"role": "user", "content": "继续实现 /v1/chat/completions。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "true")
        self.assertEqual(response.headers["x-memory-proxy-history-dropped"], "3")
        echoed = response.json()["echo_messages"]
        self.assertTrue(
            any(
                item.get("role") == "assistant"
                and item.get("content") == "请记住文件 /tmp/migration-plan.md 里有迁移清单。"
                for item in echoed
            )
        )
        self.assertEqual(echoed[-1]["content"], "继续实现 /v1/chat/completions。")

    def test_chat_endpoint_uses_recent_token_budget_to_shrink_recent_suffix(self) -> None:
        client = self.make_client(recent_window=2, recent_min_messages=1, recent_token_budget=28)
        long_status = (
            "我已经完成了非常详细的联调记录，包含大量重复说明、重复说明、重复说明、"
            "重复说明，并且把每个步骤都完整展开写出来。"
        )
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "assistant", "content": "建议第一版先使用 Python + FastAPI。"},
                {"role": "user", "content": "必须尽量不影响精度。"},
                {"role": "assistant", "content": long_status},
                {"role": "user", "content": "继续实现 /v1/chat/completions。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        echoed = response.json()["echo_messages"]
        self.assertFalse(any(item.get("content") == long_status for item in echoed if item.get("role") == "assistant"))
        self.assertEqual(echoed[-1]["content"], "继续实现 /v1/chat/completions。")

    def test_chat_endpoint_limits_salient_history_by_token_budget(self) -> None:
        client = self.make_client(
            recent_window=2,
            salient_history_messages=2,
            salient_history_token_budget=30,
        )
        long_reference = (
            "请记住文件 /workspace/workflow-memory-proxy/src/memory_proxy/nested/very_long_path/proxy_service.py "
            "里有很长的迁移清单和详细备注。"
        )
        short_error = "这里出现报错了吗？"
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "assistant", "content": "建议第一版先使用 Python + FastAPI。"},
                {"role": "user", "content": long_reference},
                {"role": "user", "content": short_error},
                {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现代理路由。"},
                {"role": "user", "content": "继续实现 /v1/chat/completions。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        echoed = response.json()["echo_messages"]
        retained = [item.get("content") for item in echoed if item.get("role") == "user"]
        self.assertIn(long_reference, retained)
        self.assertNotIn(short_error, retained)

    def test_chat_endpoint_static_compresses_old_code_history(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "assistant", "content": "建议第一版先使用 Python + FastAPI。"},
                {
                    "role": "assistant",
                    "content": """请记住 /tmp/server.py 的关键实现：
```python
from fastapi import FastAPI

class ProxyServer:
    def build_app(self) -> FastAPI:
        app = FastAPI()
        return app
```
""",
                },
                {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现代理路由。"},
                {"role": "user", "content": "继续实现 /v1/chat/completions。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        echoed = response.json()["echo_messages"]
        summaries = [
            item["content"]
            for item in echoed
            if item.get("role") == "assistant" and "[static-code-summary]" in str(item.get("content"))
        ]
        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertIn("files=/tmp/server.py", summary)
        self.assertIn("lang=python", summary)
        self.assertIn("ProxyServer", summary)
        self.assertIn("build_app", summary)
        self.assertNotIn("app = FastAPI()", summary)

    def test_chat_endpoint_prunes_redundant_agent_operation_logs(self) -> None:
        client = self.make_client(salient_history_messages=2, min_history_messages=1)
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "assistant", "content": "搜索 demo.py 里的 build_app 实现。"},
                {"role": "tool", "content": "tests passed for demo.py"},
                {"role": "assistant", "content": "已更新 /tmp/demo.py 并补上测试。"},
                {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现代理路由。"},
                {"role": "user", "content": "继续实现 /v1/chat/completions。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        echoed = response.json()["echo_messages"]
        retained = [str(item.get("content")) for item in echoed]
        self.assertIn("已更新 /tmp/demo.py 并补上测试。", retained)
        self.assertNotIn("搜索 demo.py 里的 build_app 实现。", retained)
        self.assertNotIn("tests passed for demo.py", retained)

    def test_chat_endpoint_keeps_only_latest_write_log_for_same_file(self) -> None:
        client = self.make_client(salient_history_messages=2, min_history_messages=1, recent_window=1)
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "assistant", "content": "created /tmp/demo.py"},
                {"role": "assistant", "content": "updated /tmp/demo.py"},
                {"role": "assistant", "content": "edited /tmp/demo.py"},
                {"role": "user", "content": "继续实现 /v1/chat/completions。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        echoed = response.json()["echo_messages"]
        retained = [str(item.get("content")) for item in echoed]
        self.assertIn("edited /tmp/demo.py", retained)
        self.assertNotIn("created /tmp/demo.py", retained)
        self.assertNotIn("updated /tmp/demo.py", retained)

    def test_chat_endpoint_prunes_redundant_directory_listing_output(self) -> None:
        client = self.make_client(salient_history_messages=0, min_history_messages=1, recent_window=1)
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "tool", "content": "src/memory_proxy/\nsrc/memory_proxy/proxy_service.py"},
                {
                    "role": "tool",
                    "content": (
                        "src/memory_proxy/\n"
                        "src/memory_proxy/proxy_service.py\n"
                        "src/memory_proxy/history_pruner.py"
                    ),
                },
                {"role": "user", "content": "继续实现 /v1/chat/completions。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "true")
        echoed = response.json()["echo_messages"]
        retained = [str(item.get("content")) for item in echoed]
        self.assertNotIn("src/memory_proxy/\nsrc/memory_proxy/proxy_service.py", retained)
        self.assertNotIn("src/memory_proxy/\nsrc/memory_proxy/proxy_service.py\nsrc/memory_proxy/history_pruner.py", retained)

    def test_chat_endpoint_prunes_directory_listing_after_later_same_dir_file_read(self) -> None:
        client = self.make_client(salient_history_messages=0, min_history_messages=1, recent_window=1)
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想优化 coding-agent 的压缩逻辑。"},
                {
                    "role": "tool",
                    "content": (
                        "src/memory_proxy/\n"
                        "src/memory_proxy/proxy_service.py\n"
                        "src/memory_proxy/history_pruner.py"
                    ),
                },
                {
                    "role": "assistant",
                    "content": "查看 /workspace/workflow-memory-proxy/src/memory_proxy/prompt_builder.py 的现有实现。",
                },
                {"role": "user", "content": "继续收紧目录级工具噪音。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "true")
        echoed = response.json()["echo_messages"]
        retained = [str(item.get("content")) for item in echoed]
        self.assertNotIn(
            "src/memory_proxy/\nsrc/memory_proxy/proxy_service.py\nsrc/memory_proxy/history_pruner.py",
            retained,
        )
        self.assertEqual(response.headers["x-memory-proxy-history-dropped"], "3")
        self.assertTrue(any("GOAL: 优化 coding-agent 的压缩逻辑" in item for item in retained))

    def test_chat_endpoint_prunes_symbol_search_trace_after_final_file_read(self) -> None:
        client = self.make_client(salient_history_messages=0, min_history_messages=1, recent_window=1)
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想继续优化 coding-agent 的压缩逻辑。"},
                {"role": "assistant", "content": "搜索 build_prompt_memory 的实现。"},
                {"role": "tool", "content": "src/memory_proxy/prompt_builder.py:90:def build_prompt_memory("},
                {
                    "role": "assistant",
                    "content": "查看 /workspace/workflow-memory-proxy/src/memory_proxy/prompt_builder.py 的现有实现。",
                },
                {"role": "user", "content": "继续压定位工具痕迹。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "true")
        self.assertEqual(response.headers["x-memory-proxy-history-dropped"], "4")
        echoed = response.json()["echo_messages"]
        retained = [str(item.get("content")) for item in echoed]
        self.assertNotIn("搜索 build_prompt_memory 的实现。", retained)
        self.assertNotIn("src/memory_proxy/prompt_builder.py:90:def build_prompt_memory(", retained)
        self.assertTrue(any("GOAL: 继续优化 coding-agent 的压缩逻辑" in item for item in retained))

    def test_chat_endpoint_dedupes_identical_salient_history_messages(self) -> None:
        client = self.make_client(salient_history_messages=2)
        repeated = "请记住文件 /tmp/migration-plan.md 里有迁移清单。"
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "assistant", "content": repeated},
                {"role": "user", "content": "重点是压缩记忆，但不要明显影响精度。"},
                {"role": "assistant", "content": repeated},
                {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现代理路由。"},
                {"role": "user", "content": "继续实现 /v1/chat/completions。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-history-dropped"], "3")
        echoed = response.json()["echo_messages"]
        self.assertEqual(
            sum(1 for item in echoed if item.get("role") == "assistant" and item.get("content") == repeated),
            1,
        )

    def test_chat_endpoint_aliases_repeated_long_file_reference(self) -> None:
        client = self.make_client(salient_history_messages=2)
        repeated_path = "/workspace/workflow-memory-proxy/src/memory_proxy/proxy_service.py"
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "assistant", "content": f"请先检查 {repeated_path} 的现有实现。"},
                {"role": "user", "content": "重点是压缩记忆，但不要明显影响精度。"},
                {"role": "assistant", "content": f"继续基于 {repeated_path} 调整压缩逻辑。"},
                {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现代理路由。"},
                {"role": "user", "content": "继续实现 /v1/chat/completions。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        echoed = response.json()["echo_messages"]
        retained = [item["content"] for item in echoed if item.get("role") == "assistant"]
        self.assertTrue(any("FILE_1{" in text and repeated_path in text for text in retained))
        self.assertTrue(any("FILE_1" in text and repeated_path not in text for text in retained))

    def test_chat_endpoint_aliases_repeated_code_block_reference(self) -> None:
        client = self.make_client(salient_history_messages=2)
        code_block = """```python
class ProxyServer:
    def build_app(self):
        return "ok"
```"""
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "assistant", "content": f"先看这段代码：\n{code_block}"},
                {"role": "user", "content": "重点是压缩记忆，但不要明显影响精度。"},
                {"role": "assistant", "content": f"后续继续基于这段代码调整：\n{code_block}"},
                {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现代理路由。"},
                {"role": "user", "content": "继续实现 /v1/chat/completions。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        echoed = response.json()["echo_messages"]
        retained = [item["content"] for item in echoed if item.get("role") == "assistant"]
        self.assertTrue(any("CODE_1{" in text and "[static-code-summary]" in text for text in retained))
        self.assertTrue(any("CODE_1" in text and "```python" not in text for text in retained))

    def test_chat_endpoint_skips_short_history(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，请问要做什么？"},
                {"role": "user", "content": "帮我继续。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "false")
        self.assertEqual(response.headers["x-memory-proxy-reason"], "history_too_short")
        echoed = response.json()["echo_messages"]
        self.assertEqual(echoed, payload["messages"])

    def test_chat_endpoint_skips_unsafe_history_messages(self) -> None:
        client = self.make_client(min_history_messages=1, recent_window=1)
        payload = {
            "model": "gpt-test",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "请根据图片回答"}],
                },
                {"role": "assistant", "content": "好的，我会保留多模态内容。"},
            ],
        }
        response = client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "false")
        self.assertEqual(response.headers["x-memory-proxy-reason"], "unsafe_history_message")

    def test_streaming_requests_are_proxied(self) -> None:
        client = self.make_client(min_history_messages=10)
        payload = {
            "model": "gpt-test",
            "stream": True,
            "messages": [
                {"role": "user", "content": "请流式回复一声 hi"},
            ],
        }
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            body = b"".join(response.iter_bytes())
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"data: [DONE]", body)

    def test_responses_endpoint_compresses_simple_input_messages(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "input": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "assistant", "content": "建议第一版先使用 Python + FastAPI。"},
                {"role": "user", "content": "必须尽量不影响精度，而且不要和现有 CLIProxyAPI 混在一起。"},
                {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现 /v1/responses 代理。"},
                {"role": "user", "content": "继续实现真实上游联调，并补测试。"},
            ],
        }
        response = client.post("/v1/responses", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "true")
        self.assertTrue(response.headers["x-memory-proxy-session-id"].startswith("sess_"))
        body = response.json()
        self.assertIn("Older summary. Recent messages override.", body["echo_instructions"])
        self.assertIn("GOAL:", body["echo_instructions"])
        echoed = body["echo_input"]
        self.assertEqual(echoed[0]["role"], "system")
        self.assertEqual(echoed[-1]["content"], "继续实现真实上游联调，并补测试。")
        self.assertLess(len(echoed), len(payload["input"]))

    def test_responses_endpoint_keeps_salient_history_message(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "input": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                {"role": "assistant", "content": "建议第一版先使用 Python + FastAPI。"},
                {"role": "user", "content": "必须尽量不影响精度。"},
                {"role": "assistant", "content": "请记住文件 /tmp/migration-plan.md 里有迁移清单。"},
                {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现 /v1/responses 代理。"},
                {"role": "user", "content": "继续实现真实上游联调，并补测试。"},
            ],
        }
        response = client.post("/v1/responses", json=payload)
        self.assertEqual(response.status_code, 200)
        echoed = response.json()["echo_input"]
        self.assertTrue(
            any(
                item.get("role") == "assistant"
                and item.get("content") == "请记住文件 /tmp/migration-plan.md 里有迁移清单。"
                for item in echoed
            )
        )

    def test_responses_endpoint_skips_unsupported_input_shape(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "input": "Reply with exactly OK",
        }
        response = client.post("/v1/responses", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "false")
        self.assertEqual(response.headers["x-memory-proxy-reason"], "unsupported_input_shape")

    def test_responses_endpoint_compresses_codex_style_mixed_input(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "我想做一个 OpenAI 兼容代理。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "建议第一版先使用 Python + FastAPI。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "必须尽量不影响精度，而且不要和现有 CLIProxyAPI 混在一起。"}],
                },
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": None,
                    "encrypted_content": "opaque",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "我已经完成了方案文档，下一步准备实现 /v1/responses 代理。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "继续实现真实上游联调，并补测试。"}],
                },
            ],
        }
        response = client.post("/v1/responses", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "true")
        body = response.json()
        self.assertIn("Older summary. Recent messages override.", body["echo_instructions"])
        self.assertIn("GOAL:", body["echo_instructions"])
        echoed = body["echo_input"]
        self.assertEqual(echoed[0]["role"], "developer")
        self.assertTrue(any(item.get("type") == "reasoning" for item in echoed))
        self.assertEqual(
            echoed[-1]["content"][0]["text"],
            "继续实现真实上游联调，并补测试。",
        )
        self.assertLess(len(echoed), len(payload["input"]))

    def test_proxy_responses_compact_endpoint_returns_proxy_compaction_item(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "我想做一个 OpenAI 兼容代理。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "建议第一版先使用 Python + FastAPI。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "必须尽量不影响精度，而且不要和现有 CLIProxyAPI 混在一起。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "我已经完成了方案文档，下一步准备实现 /v1/responses 代理。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "继续实现真实上游联调，并补测试。"}],
                },
            ],
        }
        response = client.post("/v1/proxy/responses/compact", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "true")
        body = response.json()
        self.assertEqual(body["object"], "memory_proxy.compaction")
        output = body["output"]
        compaction_messages = [
            item
            for item in output
            if item.get("type") == "message"
            and item.get("role") == "developer"
            and any(
                isinstance(block, dict)
                and isinstance(block.get("text"), str)
                and block["text"].startswith("[memory-proxy-compaction:v1]")
                for block in item.get("content", [])
            )
        ]
        self.assertEqual(len(compaction_messages), 1)
        self.assertFalse(any("encrypted_content" in item for item in output))
        self.assertEqual(output[0]["role"], "developer")
        self.assertLess(len(output), len(payload["input"]))

    def test_responses_compact_legacy_alias_is_deprecated(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "我想做一个 OpenAI 兼容代理。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "建议第一版先使用 Python + FastAPI。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "必须尽量不影响精度，而且不要和现有 CLIProxyAPI 混在一起。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "我已经完成了方案文档，下一步准备实现 /v1/responses 代理。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "继续实现真实上游联调，并补测试。"}],
                },
            ],
        }
        response = client.post("/v1/responses/compact", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["deprecation"], "true")
        self.assertEqual(response.headers["sunset"], "Wed, 30 Sep 2026 00:00:00 GMT")
        self.assertIn("/v1/proxy/responses/compact", response.headers["link"])
        body = response.json()
        self.assertEqual(body["object"], "response.compaction")
        self.assertEqual(body["warning"]["code"], "deprecated_endpoint")
        self.assertEqual(body["warning"]["successor"], "/v1/proxy/responses/compact")
        self.assertEqual(body["warning"]["sunset"], "Wed, 30 Sep 2026 00:00:00 GMT")

    def test_responses_endpoint_skips_compaction_when_function_call_is_pending(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "先熟悉一下仓库。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "我先看下目录结构。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "顺便查一下测试。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "我先跑一遍基础检查。"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_pending_1",
                    "name": "exec_command",
                    "arguments": "{\"cmd\":\"pwd\"}",
                },
            ],
        }
        response = client.post("/v1/responses", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "false")
        self.assertEqual(response.headers["x-memory-proxy-reason"], "pending_function_call")
        body = response.json()
        self.assertEqual(body["echo_input"], payload["input"])
        self.assertIsNone(body["echo_instructions"])

    def test_proxy_responses_compact_endpoint_skips_when_function_call_is_pending(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "先熟悉一下仓库。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "我先看下目录结构。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "顺便查一下测试。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "我先跑一遍基础检查。"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_pending_1",
                    "name": "exec_command",
                    "arguments": "{\"cmd\":\"pwd\"}",
                },
            ],
        }
        response = client.post("/v1/proxy/responses/compact", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "false")
        self.assertEqual(response.headers["x-memory-proxy-reason"], "pending_function_call")
        body = response.json()
        self.assertEqual(body["object"], "memory_proxy.compaction")
        self.assertEqual(body["output"], payload["input"])

    def test_responses_endpoint_rehydrates_proxy_compaction_item(self) -> None:
        client = self.make_client()
        original_payload = {
            "model": "gpt-test",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "我想做一个 OpenAI 兼容代理。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "建议第一版先使用 Python + FastAPI。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "必须尽量不影响精度，而且不要和现有 CLIProxyAPI 混在一起。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "我已经完成了方案文档，下一步准备实现 /v1/responses 代理。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "继续实现真实上游联调，并补测试。"}],
                },
            ],
        }
        compacted = client.post("/v1/proxy/responses/compact", json=original_payload).json()["output"]
        follow_up_payload = {
            "model": "gpt-test",
            "input": compacted
            + [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "那下一步做什么？"}],
                }
            ],
        }
        response = client.post("/v1/responses", json=follow_up_payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-reason"], "compaction_item_rehydrated")
        body = response.json()
        self.assertIn("Older summary. Recent messages override.", body["echo_instructions"])
        self.assertIn("GOAL:", body["echo_instructions"])
        self.assertTrue(
            all(
                not any(
                    isinstance(block, dict)
                    and isinstance(block.get("text"), str)
                    and block["text"].startswith("[memory-proxy-compaction:v1]")
                    for block in item.get("content", [])
                )
                for item in body["echo_input"]
                if isinstance(item, dict)
            )
        )

    def test_responses_endpoint_merges_prompt_memory_without_duplicate_fields(self) -> None:
        client = self.make_client()
        original_payload = {
            "model": "gpt-test",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "我想做一个 OpenAI 兼容代理。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "建议第一版先使用 Python + FastAPI。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "必须尽量不影响精度。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "我已经完成了方案文档，下一步准备实现 /v1/responses 代理。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "继续实现真实上游联调。"}],
                },
            ],
        }
        compacted = client.post("/v1/proxy/responses/compact", json=original_payload).json()["output"]
        follow_up_payload = {
            "model": "gpt-test",
            "input": compacted
            + [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "我们还是在做 OpenAI 兼容代理。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "建议第一版继续使用 Python + FastAPI。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "必须继续尽量不影响精度。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "我已经完成了一轮联调，下一步准备补 dashboard。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "继续推进并告诉我当前结论。"}],
                },
            ],
        }
        response = client.post("/v1/responses", json=follow_up_payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "true")
        instructions = response.json()["echo_instructions"]
        self.assertEqual(instructions.count("GOAL:"), 1)
        self.assertEqual(instructions.count("CONS:"), 1)
        self.assertEqual(instructions.count("NEXT:"), 1)

    def test_responses_endpoint_normalizes_late_instruction_messages(self) -> None:
        client = self.make_client()
        payload = {
            "model": "gpt-test",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "我们在做 memory proxy。"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "第一版先做 OpenAI 兼容代理。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "重点是减少 tokens，同时别太影响精度。"}],
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Always keep user-facing answers in natural language."}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "我已经完成了方案文档，下一步准备接真实上游。"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "继续实现并补测试。"}],
                },
            ],
        }
        response = client.post("/v1/responses", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-memory-proxy-compressed"], "true")
        body = response.json()
        self.assertIn("Always keep user-facing answers in natural language.", body["echo_instructions"])
        self.assertTrue(
            all(
                not (
                    item.get("role") == "developer"
                    and any(
                        isinstance(part, dict)
                        and part.get("text") == "Always keep user-facing answers in natural language."
                        for part in item.get("content", [])
                    )
                )
                for item in body["echo_input"]
            )
        )

    def test_responses_streaming_requests_are_proxied(self) -> None:
        client = self.make_client(min_history_messages=10)
        payload = {
            "model": "gpt-test",
            "stream": True,
            "input": [{"role": "user", "content": "请流式回复一声 hi"}],
        }
        with client.stream("POST", "/v1/responses", json=payload) as response:
            body = b"".join(response.iter_bytes())
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"data: [DONE]", body)

    def test_responses_websocket_accepts_json_requests(self) -> None:
        client = self.make_client(min_history_messages=10)
        with client.websocket_connect("/v1/responses") as websocket:
            websocket.send_json(
                {
                    "model": "gpt-test",
                    "input": [{"role": "user", "content": "请简单回复 hi"}],
                }
            )
            message = websocket.receive_json()
        self.assertEqual(message["type"], "response")
        self.assertEqual(message["response"]["object"], "response")
        self.assertEqual(message["proxy"]["x-memory-proxy-reason"], "history_too_short")

    def test_responses_codex_style_input_can_recall_session_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = ProxySettings(
                upstream_base_url="http://upstream/v1/",
                compression_enabled=True,
                recent_window=2,
                min_history_messages=4,
                store_enabled=True,
                store_db_path=f"{temp_dir}/memory_proxy.db",
                session_auto_continue_enabled=False,
            )
            upstream = build_upstream_app()
            upstream_client = UpstreamOpenAIClient(
                settings,
                transport=httpx.ASGITransport(app=upstream),
            )
            app = create_app(settings, upstream_client=upstream_client)
            client = TestClient(app)

            conv_id = "conv_responses_recall"
            first_response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-test",
                    "metadata": {"conversation_id": conv_id},
                    "input": [
                        {
                            "type": "message",
                            "role": "developer",
                            "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                        },
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "先给我一个当前进度更新。"}],
                        },
                    ],
                },
            )
            self.assertEqual(first_response.status_code, 200)
            self.assertEqual(first_response.headers["x-memory-proxy-reason"], "history_too_short")

            store = app.state.memory_store
            self.assertIsNotNone(store)
            latest_snapshot = store.get_latest_working_memory_snapshot(conv_id)
            self.assertIsNotNone(latest_snapshot)
            self.assertIn("DONE:", latest_snapshot["memory_dsl"])
            self.assertIn("NEXT:", latest_snapshot["memory_dsl"])

            second_response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-test",
                    "metadata": {"conversation_id": conv_id},
                    "input": [
                        {
                            "type": "message",
                            "role": "developer",
                            "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                        },
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "那下一步做什么？"}],
                        },
                    ],
                },
            )
            self.assertEqual(second_response.status_code, 200)
            self.assertEqual(second_response.headers["x-memory-proxy-reason"], "session_memory_recalled")
            body = second_response.json()
            self.assertIn("Older summary. Recent messages override.", body["echo_instructions"])
            self.assertIn("DONE:", body["echo_instructions"])
            self.assertIn("NEXT:", body["echo_instructions"])

    def test_store_enabled_persists_compaction_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = ProxySettings(
                upstream_base_url="http://upstream/v1/",
                compression_enabled=True,
                recent_window=2,
                min_history_messages=2,
                store_enabled=True,
                store_db_path=f"{temp_dir}/memory_proxy.db",
            )
            upstream = build_upstream_app()
            upstream_client = UpstreamOpenAIClient(
                settings,
                transport=httpx.ASGITransport(app=upstream),
            )
            app = create_app(settings, upstream_client=upstream_client)
            client = TestClient(app)
            payload = {
                "model": "gpt-test",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                    {"role": "assistant", "content": "建议第一版先使用 Python + FastAPI。"},
                    {"role": "user", "content": "必须尽量不影响精度，而且不要和现有 CLIProxyAPI 混在一起。"},
                    {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现代理路由。"},
                    {"role": "user", "content": "继续实现 /v1/chat/completions。"},
                ],
            }
            response = client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"x-session-id": "sess_test_persist"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-memory-proxy-session-id"], "sess_test_persist")
            self.assertTrue(response.headers["x-memory-proxy-snapshot-id"].startswith("snap_"))

            store = app.state.memory_store
            self.assertIsNotNone(store)
            raw_rows = store.list_raw_messages("sess_test_persist")
            event_rows = store.list_memory_events("sess_test_persist")
            snapshot_rows = store.list_working_memory_snapshots("sess_test_persist")
            audit_rows = store.list_request_audits(session_id="sess_test_persist")
            self.assertGreaterEqual(len(raw_rows), 1)
            self.assertGreaterEqual(len(event_rows), 1)
            self.assertGreaterEqual(len(snapshot_rows), 1)
            self.assertEqual(len(audit_rows), 1)
            self.assertTrue(audit_rows[0]["compressed"])

    def test_dashboard_api_returns_request_detail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = ProxySettings(
                upstream_base_url="http://upstream/v1/",
                compression_enabled=True,
                recent_window=2,
                min_history_messages=2,
                store_enabled=True,
                store_db_path=f"{temp_dir}/memory_proxy.db",
            )
            upstream = build_upstream_app()
            upstream_client = UpstreamOpenAIClient(
                settings,
                transport=httpx.ASGITransport(app=upstream),
            )
            app = create_app(settings, upstream_client=upstream_client)
            client = TestClient(app)
            payload = {
                "model": "gpt-test",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                    {"role": "assistant", "content": "建议第一版先使用 Python + FastAPI。"},
                    {"role": "user", "content": "重点是压缩记忆，但不要明显影响精度。"},
                    {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现代理路由。"},
                    {"role": "user", "content": "继续实现 /v1/chat/completions。"},
                ],
            }
            response = client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"x-session-id": "sess_dashboard"},
            )
            self.assertEqual(response.status_code, 200)
            request_id = response.headers["x-memory-proxy-request-id"]

            summary = client.get("/api/dashboard/summary")
            self.assertEqual(summary.status_code, 200)
            self.assertEqual(summary.json()["summary"]["requests"], 1)

            sessions = client.get("/api/dashboard/sessions")
            self.assertEqual(sessions.status_code, 200)
            self.assertEqual(sessions.json()["items"][0]["session_id"], "sess_dashboard")

            requests = client.get("/api/dashboard/requests")
            self.assertEqual(requests.status_code, 200)
            self.assertEqual(requests.json()["items"][0]["request_id"], request_id)

            detail = client.get(f"/api/dashboard/requests/{request_id}")
            self.assertEqual(detail.status_code, 200)
            body = detail.json()
            self.assertEqual(body["request"]["request_id"], request_id)
            self.assertTrue(body["request"]["compressed"])
            self.assertNotEqual(
                json.dumps(body["request"]["original_payload"], ensure_ascii=False),
                json.dumps(body["request"]["forwarded_payload"], ensure_ascii=False),
            )
            self.assertIsNotNone(body["snapshot"])
            self.assertGreaterEqual(len(body["raw_messages"]), 1)
            self.assertGreaterEqual(len(body["events"]), 1)

    def test_responses_accepts_codex_session_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = ProxySettings(
                upstream_base_url="http://upstream/v1/",
                compression_enabled=True,
                recent_window=2,
                min_history_messages=4,
                store_enabled=True,
                store_db_path=f"{temp_dir}/memory_proxy.db",
                session_auto_continue_enabled=False,
            )
            upstream = build_upstream_app()
            upstream_client = UpstreamOpenAIClient(
                settings,
                transport=httpx.ASGITransport(app=upstream),
            )
            app = create_app(settings, upstream_client=upstream_client)
            client = TestClient(app)

            first_response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-test",
                    "input": [
                        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "先说下当前状态。"}]}
                    ],
                },
                headers={"Session_id": "sess_codex_alias"},
            )
            self.assertEqual(first_response.status_code, 200)
            self.assertEqual(first_response.headers["x-memory-proxy-session-id"], "sess_codex_alias")

            second_response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-test",
                    "input": [
                        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "那下一步呢？"}]}
                    ],
                },
                headers={"Conversation_id": "sess_codex_alias"},
            )
            self.assertEqual(second_response.status_code, 200)
            self.assertEqual(second_response.headers["x-memory-proxy-session-id"], "sess_codex_alias")
            self.assertEqual(second_response.headers["x-memory-proxy-reason"], "session_memory_recalled")

    def test_store_prunes_old_request_audits_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = ProxySettings(
                upstream_base_url="http://upstream/v1/",
                compression_enabled=True,
                recent_window=2,
                min_history_messages=2,
                store_enabled=True,
                store_db_path=f"{temp_dir}/memory_proxy.db",
                store_max_requests=2,
            )
            upstream = build_upstream_app()
            upstream_client = UpstreamOpenAIClient(
                settings,
                transport=httpx.ASGITransport(app=upstream),
            )
            app = create_app(settings, upstream_client=upstream_client)
            client = TestClient(app)
            payload = {
                "model": "gpt-test",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "我想做一个 OpenAI 兼容代理。"},
                    {"role": "assistant", "content": "建议第一版先使用 Python + FastAPI。"},
                    {"role": "user", "content": "重点是压缩记忆，但不要明显影响精度。"},
                    {"role": "assistant", "content": "我已经完成了方案文档，下一步准备实现代理路由。"},
                    {"role": "user", "content": "继续实现 /v1/chat/completions。"},
                ],
            }

            first_response = client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"x-session-id": "sess_prune_1"},
            )
            second_response = client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"x-session-id": "sess_prune_2"},
            )
            third_response = client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"x-session-id": "sess_prune_3"},
            )

            self.assertEqual(first_response.status_code, 200)
            self.assertEqual(second_response.status_code, 200)
            self.assertEqual(third_response.status_code, 200)

            first_request_id = first_response.headers["x-memory-proxy-request-id"]
            first_snapshot_id = first_response.headers["x-memory-proxy-snapshot-id"]

            store = app.state.memory_store
            self.assertIsNotNone(store)
            remaining_audits = store.list_request_audits(limit=10)
            remaining_ids = [row["request_id"] for row in remaining_audits]
            self.assertEqual(len(remaining_ids), 2)
            self.assertNotIn(first_request_id, remaining_ids)
            self.assertIsNone(store.get_request_audit(first_request_id))
            self.assertEqual(store.list_turn_raw_messages("sess_prune_1", first_request_id), [])
            self.assertEqual(store.list_turn_memory_events("sess_prune_1", first_request_id), [])
            self.assertIsNone(store.get_working_memory_snapshot(first_snapshot_id))
            self.assertEqual(store.list_turn_raw_messages("sess_prune_1", f"{first_request_id}:response"), [])
            self.assertEqual(store.list_turn_memory_events("sess_prune_1", f"{first_request_id}:response"), [])

    def test_short_history_can_recall_latest_session_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = ProxySettings(
                upstream_base_url="http://upstream/v1/",
                compression_enabled=True,
                recent_window=2,
                min_history_messages=4,
                store_enabled=True,
                store_db_path=f"{temp_dir}/memory_proxy.db",
            )
            upstream = build_upstream_app()
            upstream_client = UpstreamOpenAIClient(
                settings,
                transport=httpx.ASGITransport(app=upstream),
            )
            app = create_app(settings, upstream_client=upstream_client)
            client = TestClient(app)

            first_payload = {
                "model": "gpt-test",
                "messages": [
                    {"role": "user", "content": "先给我一个当前进度更新。"},
                ],
            }
            first_response = client.post(
                "/v1/chat/completions",
                json=first_payload,
                headers={"x-session-id": "sess_recall"},
            )
            self.assertEqual(first_response.status_code, 200)
            self.assertEqual(first_response.headers["x-memory-proxy-reason"], "history_too_short")

            store = app.state.memory_store
            self.assertIsNotNone(store)
            latest_snapshot = store.get_latest_working_memory_snapshot("sess_recall")
            self.assertIsNotNone(latest_snapshot)
            self.assertIn("DONE:", latest_snapshot["memory_dsl"])
            self.assertIn("NEXT:", latest_snapshot["memory_dsl"])
            self.assertGreaterEqual(
                len(store.list_turn_memory_events("sess_recall", f"{first_response.headers['x-memory-proxy-request-id']}:response")),
                1,
            )

            second_payload = {
                "model": "gpt-test",
                "messages": [
                    {"role": "user", "content": "那下一步做什么？"},
                ],
            }
            second_response = client.post(
                "/v1/chat/completions",
                json=second_payload,
                headers={"x-session-id": "sess_recall"},
            )
            self.assertEqual(second_response.status_code, 200)
            self.assertEqual(second_response.headers["x-memory-proxy-compressed"], "false")
            self.assertEqual(second_response.headers["x-memory-proxy-reason"], "session_memory_recalled")
            echoed = second_response.json()["echo_messages"]
            self.assertEqual(echoed[0]["role"], "system")
            self.assertIn("Older summary. Recent messages override.", echoed[0]["content"])
            self.assertIn("DONE:", echoed[0]["content"])
            self.assertIn("NEXT:", echoed[0]["content"])
            self.assertEqual(echoed[-1]["content"], "那下一步做什么？")

    def test_missing_header_can_reuse_single_recent_session_by_client_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = ProxySettings(
                upstream_base_url="http://upstream/v1/",
                compression_enabled=True,
                recent_window=2,
                min_history_messages=4,
                store_enabled=True,
                store_db_path=f"{temp_dir}/memory_proxy.db",
            )
            upstream = build_upstream_app()
            upstream_client = UpstreamOpenAIClient(
                settings,
                transport=httpx.ASGITransport(app=upstream),
            )
            app = create_app(settings, upstream_client=upstream_client)
            client = TestClient(app)

            first_response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [{"role": "user", "content": "先说下当前状态。"}]},
            )
            self.assertEqual(first_response.status_code, 200)
            first_session_id = first_response.headers["x-memory-proxy-session-id"]
            self.assertTrue(first_session_id.startswith("sess_"))

            second_response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [{"role": "user", "content": "那下一步呢？"}]},
            )
            self.assertEqual(second_response.status_code, 200)
            self.assertEqual(second_response.headers["x-memory-proxy-session-id"], first_session_id)
            self.assertEqual(second_response.headers["x-memory-proxy-reason"], "session_memory_recalled")

    def test_body_metadata_conversation_id_can_drive_session_recall(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = ProxySettings(
                upstream_base_url="http://upstream/v1/",
                compression_enabled=True,
                recent_window=2,
                min_history_messages=4,
                store_enabled=True,
                store_db_path=f"{temp_dir}/memory_proxy.db",
                session_auto_continue_enabled=False,
            )
            upstream = build_upstream_app()
            upstream_client = UpstreamOpenAIClient(
                settings,
                transport=httpx.ASGITransport(app=upstream),
            )
            app = create_app(settings, upstream_client=upstream_client)
            client = TestClient(app)

            first_response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-test",
                    "metadata": {"conversation_id": "conv_demo_001"},
                    "messages": [{"role": "user", "content": "先说下当前状态。"}],
                },
            )
            self.assertEqual(first_response.status_code, 200)
            self.assertEqual(first_response.headers["x-memory-proxy-session-id"], "conv_demo_001")

            second_response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-test",
                    "metadata": {"conversation_id": "conv_demo_001"},
                    "messages": [{"role": "user", "content": "那下一步呢？"}],
                },
            )
            self.assertEqual(second_response.status_code, 200)
            self.assertEqual(second_response.headers["x-memory-proxy-session-id"], "conv_demo_001")
            self.assertEqual(second_response.headers["x-memory-proxy-reason"], "session_memory_recalled")

    def test_auto_continue_skips_ambiguous_recent_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = ProxySettings(
                upstream_base_url="http://upstream/v1/",
                compression_enabled=True,
                recent_window=2,
                min_history_messages=4,
                store_enabled=True,
                store_db_path=f"{temp_dir}/memory_proxy.db",
            )
            upstream = build_upstream_app()
            upstream_client = UpstreamOpenAIClient(
                settings,
                transport=httpx.ASGITransport(app=upstream),
            )
            app = create_app(settings, upstream_client=upstream_client)
            client = TestClient(app)

            for session_id in ("sess_explicit_a", "sess_explicit_b"):
                response = client.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-test", "messages": [{"role": "user", "content": "先说下当前状态。"}]},
                    headers={"x-session-id": session_id},
                )
                self.assertEqual(response.status_code, 200)

            third_response = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [{"role": "user", "content": "那下一步呢？"}]},
            )
            self.assertEqual(third_response.status_code, 200)
            self.assertEqual(third_response.headers["x-memory-proxy-reason"], "history_too_short")
            self.assertNotIn(
                third_response.headers["x-memory-proxy-session-id"],
                {"sess_explicit_a", "sess_explicit_b"},
            )

    def test_store_init_db_migrates_old_request_audit_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = f"{temp_dir}/memory_proxy.db"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE request_audits (
                    request_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    api_kind TEXT NOT NULL,
                    upstream_model TEXT,
                    compressed INTEGER NOT NULL DEFAULT 0,
                    compression_reason TEXT NOT NULL,
                    dropped_message_count INTEGER NOT NULL DEFAULT 0,
                    recent_message_count INTEGER NOT NULL DEFAULT 0,
                    snapshot_id TEXT,
                    original_payload_json TEXT NOT NULL,
                    forwarded_payload_json TEXT NOT NULL,
                    prompt_memory TEXT NOT NULL DEFAULT '',
                    estimated_input_tokens_before INTEGER,
                    estimated_input_tokens_after INTEGER,
                    estimated_savings_pct REAL,
                    token_counter TEXT,
                    upstream_usage_json TEXT,
                    upstream_input_tokens INTEGER,
                    upstream_output_tokens INTEGER,
                    upstream_total_tokens INTEGER,
                    status_code INTEGER,
                    response_preview TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.commit()
            connection.close()

            store = SQLiteMemoryStore(db_path)
            store.init_db()

            connection = sqlite3.connect(db_path)
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(request_audits)").fetchall()
            }
            connection.close()
            self.assertIn("client_fingerprint", columns)
            self.assertIn("upstream_response_id", columns)


if __name__ == "__main__":
    unittest.main()
