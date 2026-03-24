from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_proxy import ChatMessage, MemoryCompressor, RawMessage  # noqa: E402
from memory_proxy.compressor import collapse_redundant_history_messages  # noqa: E402
from memory_proxy.config import ProxySettings  # noqa: E402
from memory_proxy.dsl import event_to_dsl, parse_dsl_line, working_memory_to_dsl  # noqa: E402
from memory_proxy.extractor_factory import build_memory_extractor  # noqa: E402
from memory_proxy.extractor import RuleBasedExtractor  # noqa: E402
from memory_proxy.history_pruner import classify_history_fidelity, prune_history_messages  # noqa: E402
from memory_proxy.llm_extractor import (  # noqa: E402
    ExtractionPrompt,
    HybridExtractor,
    JsonLLMExtractor,
)
from memory_proxy.models import MemoryEvent, WorkingMemory  # noqa: E402
from memory_proxy.prompt_builder import (  # noqa: E402
    PromptMemoryBuilder,
    PromptMemoryConfig,
    merge_prompt_memories,
    parse_prompt_memory_text,
    prompt_memory_to_text,
)
from memory_proxy.reference_alias import alias_repeated_references  # noqa: E402
from memory_proxy.reducer import WorkingMemoryReducer  # noqa: E402
from memory_proxy.static_code import (  # noqa: E402
    compress_static_code_message,
    is_code_heavy_text,
    summarize_static_code_reference,
)
from memory_proxy.store import SQLiteMemoryStore  # noqa: E402
from memory_proxy.tokens import approx_tokens  # noqa: E402
from memory_proxy.zh_semantics import contains_workflow_keyword, is_task_management_text  # noqa: E402


class StubLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[ExtractionPrompt] = []

    def complete(self, prompt: ExtractionPrompt) -> str:
        self.prompts.append(prompt)
        return self.response


class MemoryCompressorTests(unittest.TestCase):
    def test_user_message_extracts_goal_and_constraint(self) -> None:
        compressor = MemoryCompressor()
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_001",
                    role="user",
                    content="我想做一个 OpenAI 兼容代理，重点是压缩记忆来减少 token 消耗。",
                )
            ]
        )
        event_types = [event.type for event in result.events]
        self.assertIn("goal", event_types)
        self.assertIn("constraint", event_types)
        self.assertEqual(result.working_memory.primary_goal, "OpenAI 兼容代理")

    def test_user_message_extracts_goal_from_conversational_setup(self) -> None:
        compressor = MemoryCompressor()
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_001b",
                    role="user",
                    content="我们现在做教育排课这个，先把排课回放助手骨架搭起来。",
                )
            ]
        )
        self.assertEqual(result.working_memory.primary_goal, "排课回放助手")

    def test_assistant_message_extracts_task_and_plan(self) -> None:
        compressor = MemoryCompressor()
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_002",
                    role="assistant",
                    content="我已经完成了会话存储，下一步准备实现 prompt builder。",
                )
            ]
        )
        statuses = {(event.type, event.status, event.subject) for event in result.events}
        self.assertIn(("task", "claimed_done", "会话存储"), statuses)
        self.assertIn(("plan", "active", "实现 prompt builder"), statuses)
        self.assertIn("会话存储", result.working_memory.pending_verification_tasks)

    def test_assistant_status_field_does_not_extract_generic_done_subject(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_assistant_status_field",
                    role="assistant",
                    content="1. AI 家教报表配置\n状态：已完成",
                )
            ]
        )
        statuses = {(event.type, event.status, event.subject) for event in result.events}
        self.assertNotIn(("task", "claimed_done", "状态"), statuses)
        self.assertNotIn("状态", result.working_memory.pending_verification_tasks)

    def test_assistant_aggregate_done_summary_does_not_extract_generic_task_subject(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_assistant_aggregate_done",
                    role="assistant",
                    content="你目前最近的 20 条任务都已经完成。",
                )
            ]
        )
        statuses = {(event.type, event.status, event.subject) for event in result.events}
        self.assertNotIn(("task", "claimed_done", "你目前最近的 20 条任务都"), statuses)
        self.assertEqual(result.working_memory.pending_verification_tasks, [])

    def test_assistant_message_extracts_recommendation_as_decision(self) -> None:
        compressor = MemoryCompressor()
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_recommend",
                    role="assistant",
                    content="我推荐默认走 cross-proxy-compatible。",
                )
            ]
        )
        statuses = {(event.type, event.status, event.subject) for event in result.events}
        self.assertIn(("decision", "active", "默认走 cross-proxy-compatible"), statuses)
        self.assertIn("默认走 cross-proxy-compatible", result.working_memory.active_decisions)

    def test_user_message_extracts_future_task_and_constraint(self) -> None:
        compressor = MemoryCompressor()
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_020",
                    role="user",
                    content="接下来还要补测试，并且区分 AI 自述完成和工具验证完成。",
                )
            ]
        )
        statuses = {(event.type, event.status, event.subject) for event in result.events}
        self.assertIn(("task", "proposed", "补测试"), statuses)
        self.assertIn(("constraint", "active", "区分 AI 自述完成和工具验证完成"), statuses)
        self.assertIn("补测试", result.working_memory.open_tasks)

    def test_user_imperative_message_extracts_todo_task(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_user_todo",
                    role="user",
                    content="你帮我把记忆压缩服务部署到 docker 干了。",
                )
            ]
        )
        statuses = {(event.type, event.status, event.subject) for event in result.events}
        self.assertIn(("task", "proposed", "记忆压缩服务部署到 docker"), statuses)
        self.assertIn("TODO: 记忆压缩服务部署到 docker", result.memory_dsl)

    def test_user_task_batch_with_uniform_done_suffix_does_not_extract_generic_done_subject(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_user_task_batch",
                    role="user",
                    content="帮我创建4个任务，1.ai家教报表配置 4h；2.产品研发沟通会 1h。3.平台问题支撑 1h，4.安徽话机文档整理。均已完成。",
                )
            ]
        )
        statuses = {(event.type, event.status, event.subject) for event in result.events}
        self.assertIn(("task", "proposed", "创建4个任务"), statuses)
        self.assertNotIn(("task", "claimed_done", "均"), statuses)
        self.assertNotIn("均", result.working_memory.pending_verification_tasks)

    def test_user_status_update_extracts_done_and_next(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_user_progress",
                    role="user",
                    content="现在代理兼容层已经完成了，下一步做压缩算法。",
                )
            ]
        )
        self.assertIn("代理兼容层", result.working_memory.pending_verification_tasks)
        self.assertIn("DONE: 代理兼容层", result.memory_dsl)
        self.assertIn("TODO: 做压缩算法", result.memory_dsl)

    def test_user_message_extracts_decision_from_follow_previous_plan(self) -> None:
        extractor = RuleBasedExtractor()
        events = extractor.extract(
            RawMessage(
                "msg_user_decision",
                "user",
                "还是按之前那个方案。",
            ),
            session_id="sess_user_decision",
            turn_id="turn_user_decision",
        )
        decisions = [event.subject for event in events if event.type == "decision"]
        self.assertIn("按之前那个方案", decisions)

    def test_user_message_extracts_decision_from_colloquial_keep_plan(self) -> None:
        extractor = RuleBasedExtractor()
        events = extractor.extract(
            RawMessage(
                "msg_user_decision_keep",
                "user",
                "先按逐校导入方案走。",
            ),
            session_id="sess_user_decision_keep",
            turn_id="turn_user_decision_keep",
        )
        decisions = [event.subject for event in events if event.type == "decision"]
        self.assertIn("按逐校导入方案", decisions)

    def test_user_message_extracts_decision_from_change_phrase(self) -> None:
        extractor = RuleBasedExtractor()
        events = extractor.extract(
            RawMessage(
                "msg_user_decision_change",
                "user",
                "改成按年级批量导入方案吧。",
            ),
            session_id="sess_user_decision_change",
            turn_id="turn_user_decision_change",
        )
        decisions = [event.subject for event in events if event.type == "decision"]
        self.assertIn("按年级批量导入方案", decisions)

    def test_user_message_extracts_decision_resolution_from_do_not_return_phrase(self) -> None:
        extractor = RuleBasedExtractor()
        events = extractor.extract(
            RawMessage(
                "msg_user_decision_resolution",
                "user",
                "不要再回到按旧方案。",
            ),
            session_id="sess_user_decision_resolution",
            turn_id="turn_user_decision_resolution",
        )
        statuses = {(event.type, event.status, event.subject) for event in events}
        self.assertIn(("decision", "superseded", "按旧方案"), statuses)

    def test_state_extractor_skips_decision_like_clause(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_decision_state_skip",
                    role="user",
                    content="还是按批量导入方案。",
                )
            ]
        )
        self.assertEqual(result.working_memory.state_facts, {})
        self.assertIn("按批量导入方案", result.working_memory.active_decisions)

    def test_user_negative_constraint_keeps_negation(self) -> None:
        extractor = RuleBasedExtractor()
        events = extractor.extract(
            RawMessage(
                "msg_neg_constraint",
                "user",
                "尽可能不要用大模型等进行压缩。",
            ),
            session_id="sess_neg_constraint",
            turn_id="turn_neg_constraint",
        )
        constraints = [event.subject for event in events if event.type == "constraint"]
        self.assertIn("尽可能不要用大模型等进行压缩", constraints)
        self.assertNotIn("用大模型等进行压缩", constraints)

    def test_user_conjunction_constraint_keeps_negative_clause(self) -> None:
        extractor = RuleBasedExtractor()
        events = extractor.extract(
            RawMessage(
                "msg_clause_constraint",
                "user",
                "重点是压缩记忆，但不要明显影响精度。",
            ),
            session_id="sess_clause_constraint",
            turn_id="turn_clause_constraint",
        )
        constraints = [event.subject for event in events if event.type == "constraint"]
        self.assertIn("压缩记忆", constraints)
        self.assertIn("不要明显影响精度", constraints)

    def test_user_message_extracts_current_state_updates(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_state",
                    role="user",
                    content=(
                        "项目代号“赤霄”；回滚负责人从陈策改成周芮；"
                        "节点总上限18个；ID方案确定用ULID；欧盟数据必须留在欧盟。"
                    ),
                )
            ]
        )

        self.assertEqual(result.working_memory.state_facts.get("代号"), "赤霄")
        self.assertEqual(result.working_memory.state_facts.get("当前回滚负责人"), "周芮")
        self.assertEqual(result.working_memory.state_facts.get("节点总上限"), "18个")
        self.assertEqual(result.working_memory.state_facts.get("ID方案"), "ULID")
        self.assertEqual(result.working_memory.state_facts.get("欧盟数据规则"), "留在欧盟")
        self.assertIn("STATE:", result.memory_dsl)

    def test_answered_conclusion_question_is_removed_from_open_questions(self) -> None:
        compressor = MemoryCompressor()
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_q",
                    role="user",
                    content="基于当前约束，回答是否适合做跨区域强一致写入。先给结论，再给2条理由。",
                ),
                RawMessage(
                    message_id="msg_a",
                    role="assistant",
                    content="结论：不适合。理由：禁用两阶段提交且跨区一致写入代价高。",
                ),
            ]
        )
        self.assertEqual(result.working_memory.open_questions, [])

    def test_tool_message_verifies_task(self) -> None:
        compressor = MemoryCompressor()
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_002",
                    role="assistant",
                    content="我已经完成了会话存储。",
                ),
                RawMessage(
                    message_id="msg_003",
                    role="tool",
                    content="3 tests passed for 会话存储",
                ),
            ]
        )
        self.assertIn("会话存储", result.working_memory.completed_tasks)
        self.assertNotIn("会话存储", result.working_memory.pending_verification_tasks)

    def test_user_message_with_verified_status_marks_task_completed(self) -> None:
        compressor = MemoryCompressor()
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_user_verify",
                    role="user",
                    content="现在灰度联调已经通过了，下一步做上线回归。",
                ),
            ]
        )
        self.assertIn("灰度联调", result.working_memory.completed_tasks)
        self.assertIn("做上线回归", result.working_memory.open_tasks)
        self.assertEqual(result.working_memory.pending_verification_tasks, [])

    def test_tool_message_acceptance_pass_verifies_task(self) -> None:
        compressor = MemoryCompressor()
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_acceptance_done",
                    role="assistant",
                    content="我已经完成了导入校验。",
                ),
                RawMessage(
                    message_id="msg_acceptance_verify",
                    role="tool",
                    content="验收通过: 导入校验",
                ),
            ]
        )
        self.assertIn("导入校验", result.working_memory.completed_tasks)
        self.assertNotIn("导入校验", result.working_memory.pending_verification_tasks)

    def test_claimed_done_after_verified_done_keeps_task_completed(self) -> None:
        reducer = WorkingMemoryReducer()
        memory = reducer.reduce(
            [
                MemoryEvent(
                    event_id="evt_task_001",
                    session_id="sess_reduce",
                    turn_id="turn_001",
                    source_message_ids=["msg_001"],
                    actor="assistant",
                    type="task",
                    action="update",
                    status="claimed_done",
                    subject="接口清单",
                ),
                MemoryEvent(
                    event_id="evt_task_002",
                    session_id="sess_reduce",
                    turn_id="turn_002",
                    source_message_ids=["msg_002"],
                    actor="tool",
                    type="task",
                    action="verify",
                    status="verified_done",
                    subject="接口清单",
                ),
                MemoryEvent(
                    event_id="evt_task_003",
                    session_id="sess_reduce",
                    turn_id="turn_003",
                    source_message_ids=["msg_003"],
                    actor="user",
                    type="task",
                    action="update",
                    status="claimed_done",
                    subject="接口清单",
                ),
            ]
        )
        self.assertEqual(memory.completed_tasks, ["接口清单"])
        self.assertEqual(memory.pending_verification_tasks, [])

    def test_compress_can_exclude_selected_messages_from_prompt_memory(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_goal",
                    role="user",
                    content="我想做一个 OpenAI 兼容代理。",
                ),
                RawMessage(
                    message_id="msg_task",
                    role="user",
                    content="接下来还要补测试。",
                ),
            ],
            prompt_exclude_message_ids={"msg_goal"},
        )
        self.assertEqual(result.working_memory.primary_goal, "OpenAI 兼容代理")
        self.assertNotIn("GOAL:", result.memory_dsl)
        self.assertIn("TODO:", result.memory_dsl)

    def test_static_code_summary_compresses_code_heavy_text(self) -> None:
        text = """请记住 /tmp/server.py 的关键实现：
```python
from fastapi import FastAPI

class ProxyServer:
    def build_app(self) -> FastAPI:
        app = FastAPI()
        return app
```
"""
        self.assertTrue(is_code_heavy_text(text))
        summary = compress_static_code_message(text)
        self.assertIsNotNone(summary)
        if summary is None:
            self.fail("expected static code summary")
        self.assertIn("[static-code-summary]", summary)
        self.assertIn("files=/tmp/server.py", summary)
        self.assertIn("lang=python", summary)
        self.assertIn("ProxyServer", summary)
        self.assertIn("build_app", summary)
        self.assertNotIn("app = FastAPI()", summary)
        self.assertLess(approx_tokens(summary), approx_tokens(text))

    def test_reference_aliases_only_when_it_saves_tokens(self) -> None:
        repeated_path = "/workspace/workflow-memory-proxy/src/memory_proxy/proxy_service.py"
        messages = [
            ChatMessage(role="assistant", content=f"请先查看 {repeated_path} 的现有实现。"),
            ChatMessage(role="user", content=f"继续基于 {repeated_path} 调整压缩逻辑。"),
        ]
        aliased = alias_repeated_references(messages)
        self.assertIn("FILE_1{", aliased[0].content)
        self.assertIn(repeated_path, aliased[0].content)
        self.assertIn("FILE_1", aliased[1].content)
        self.assertNotIn(repeated_path, aliased[1].content)

    def test_reference_aliases_repeated_code_block_with_static_summary(self) -> None:
        code_block = """```python
class ProxyServer:
    def build_app(self):
        return "ok"
```"""
        messages = [
            ChatMessage(role="assistant", content=f"先看这段代码：\n{code_block}"),
            ChatMessage(role="user", content=f"后续继续基于这段代码调整：\n{code_block}"),
        ]
        aliased = alias_repeated_references(messages)
        self.assertIn("CODE_1{", aliased[0].content)
        self.assertIn("[static-code-summary]", aliased[0].content)
        self.assertIn("ProxyServer", aliased[0].content)
        self.assertIn("CODE_1", aliased[1].content)
        self.assertNotIn("```python", aliased[1].content)

    def test_summarize_static_code_reference_returns_summary_without_budget_gate(self) -> None:
        text = """```python
app = FastAPI()
```"""
        summary = summarize_static_code_reference(text)
        self.assertIsNotNone(summary)
        self.assertIn("[static-code-summary]", summary)

    def test_static_code_summary_excludes_url_hosts_from_file_hints(self) -> None:
        text = """请记住 /workspace/workflow-memory-proxy/src/memory_proxy/proxy_service.py
和 https://developers.openai.com/api/docs/guides/compaction 的差异：
```bash
MCPROXY_RECENT_WINDOW=2
MCPROXY_SALIENT_HISTORY_MESSAGES=1
```"""
        summary = summarize_static_code_reference(text)
        self.assertIsNotNone(summary)
        if summary is None:
            self.fail("expected static code summary")
        self.assertIn("/workspace/workflow-memory-proxy/src/memory_proxy/proxy_service.py", summary)
        self.assertNotIn("files=/developers.openai.com", summary)
        self.assertNotIn(",/developers.openai.com", summary)

    def test_static_code_summary_compresses_chinese_prose_note_to_keywords(self) -> None:
        text = """这条已经加进去了。旧历史保留链路会先扫描重复的文件路径、URL、代码块，做规范化后算 MD5；只有首次定义加后续短引用真的更省 token 时，才会启用别名。
```bash
MCPROXY_RECENT_WINDOW=2
```"""
        summary = summarize_static_code_reference(text)
        self.assertIsNotNone(summary)
        if summary is None:
            self.fail("expected static code summary")
        self.assertIn("note=MD5/短引用别名", summary)
        self.assertNotIn("旧历史保留链路会先扫描重复的文件路径", summary)

    def test_history_pruner_drops_superseded_file_reads(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="先查看 /tmp/demo.py 的现有实现。"),
                ChatMessage(role="assistant", content="继续读取 /tmp/demo.py 的现有实现。"),
                ChatMessage(role="assistant", content="已更新 /tmp/demo.py 并补上测试。"),
            ]
        )
        self.assertEqual(len(result.kept_messages), 1)
        self.assertEqual(result.kept_messages[0].content, "已更新 /tmp/demo.py 并补上测试。")
        self.assertEqual(result.dropped_indexes, [0, 1])

    def test_history_pruner_drops_search_logs_superseded_by_later_write(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="搜索 demo.py 里的 build_app 实现。"),
                ChatMessage(role="assistant", content="rg demo.py src/memory_proxy"),
                ChatMessage(role="assistant", content="已更新 /tmp/demo.py 并补上测试。"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["已更新 /tmp/demo.py 并补上测试。"])
        self.assertEqual(result.dropped_indexes, [0, 1])

    def test_history_pruner_drops_stale_test_pass_after_later_write(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="tool", content="tests passed for demo.py"),
                ChatMessage(role="assistant", content="已更新 /tmp/demo.py 并补上测试。"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["已更新 /tmp/demo.py 并补上测试。"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_drops_stale_acceptance_pass_after_later_write(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="tool", content="验收通过: demo_acceptance.md"),
                ChatMessage(role="assistant", content="updated /tmp/demo_acceptance.md"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["updated /tmp/demo_acceptance.md"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_keeps_only_latest_verification_pass_for_same_target(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="tool", content="验收通过: demo_acceptance.md"),
                ChatMessage(role="tool", content="灰度通过: demo_acceptance.md"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["灰度通过: demo_acceptance.md"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_drops_validation_run_command_after_later_pass(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="收到，我先再做一次导入校验。"),
                ChatMessage(role="tool", content="验收通过: 导入校验"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["验收通过: 导入校验"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_keeps_only_latest_validation_run_for_same_target(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="再做一次上线回归。"),
                ChatMessage(role="assistant", content="重新做一次上线回归。"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["重新做一次上线回归。"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_drops_shorthand_validation_run_after_later_pass(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="收到，我先再验一下导入校验。"),
                ChatMessage(role="tool", content="验证通过: 导入校验"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["验证通过: 导入校验"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_keeps_only_latest_shorthand_validation_run_for_same_target(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="再测一轮上线回归。"),
                ChatMessage(role="assistant", content="再跑一下上线回归。"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["再跑一下上线回归。"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_drops_colloquial_validation_run_after_later_pass(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="行，我先补一下导入校验。"),
                ChatMessage(role="tool", content="验证完成: 导入校验"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["验证完成: 导入校验"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_keeps_only_latest_colloquial_validation_run_for_same_target(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="我先确认一轮上线回归。"),
                ChatMessage(role="assistant", content="我先过一遍上线回归。"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["我先过一遍上线回归。"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_drops_targetless_execution_filler_before_search(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="我先看一下。"),
                ChatMessage(role="assistant", content="搜索 build_prompt_memory 的实现。"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["搜索 build_prompt_memory 的实现。"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_drops_targetless_execution_filler_before_write(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="这个我接着弄。"),
                ChatMessage(role="assistant", content="已更新 /tmp/demo.py 并补上测试。"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["已更新 /tmp/demo.py 并补上测试。"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_drops_loose_execution_filler_before_search(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="我先跟一下。"),
                ChatMessage(role="assistant", content="搜索 build_prompt_memory 的实现。"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["搜索 build_prompt_memory 的实现。"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_drops_loose_execution_filler_with_lai_before_write(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="这个我来处理。"),
                ChatMessage(role="assistant", content="已更新 /tmp/demo.py 并补上测试。"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["已更新 /tmp/demo.py 并补上测试。"])
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_keeps_only_latest_write_for_same_file(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="created /tmp/demo.py"),
                ChatMessage(role="assistant", content="updated /tmp/demo.py"),
                ChatMessage(role="assistant", content="edited /tmp/demo.py"),
            ]
        )
        self.assertEqual([message.content for message in result.kept_messages], ["edited /tmp/demo.py"])
        self.assertEqual(result.dropped_indexes, [0, 1])

    def test_history_pruner_patch_overrides_previous_write_for_same_file(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="updated /tmp/demo.py"),
                ChatMessage(
                    role="assistant",
                    content="*** Begin Patch\n*** Update File: /tmp/demo.py\n+print('ok')\n*** End Patch",
                ),
            ]
        )
        self.assertEqual(
            [message.content for message in result.kept_messages],
            ["*** Begin Patch\n*** Update File: /tmp/demo.py\n+print('ok')\n*** End Patch"],
        )
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_drops_redundant_directory_listing_output(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="tool", content="src/memory_proxy/\nsrc/memory_proxy/proxy_service.py"),
                ChatMessage(
                    role="tool",
                    content=(
                        "src/memory_proxy/\n"
                        "src/memory_proxy/proxy_service.py\n"
                        "src/memory_proxy/history_pruner.py"
                    ),
                ),
            ]
        )
        self.assertEqual(
            [message.content for message in result.kept_messages],
            ["src/memory_proxy/\nsrc/memory_proxy/proxy_service.py\nsrc/memory_proxy/history_pruner.py"],
        )
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_drops_narrower_search_summary_superseded_by_fuller_result(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(
                    role="tool",
                    content=(
                        "src/memory_proxy/proxy_service.py:10:def build_app():\n"
                        "src/memory_proxy/proxy_service.py:15:return app"
                    ),
                ),
                ChatMessage(
                    role="tool",
                    content=(
                        "src/memory_proxy/proxy_service.py:10:def build_app():\n"
                        "src/memory_proxy/proxy_service.py:15:return app\n"
                        "src/memory_proxy/proxy_service.py:22:def route_chat():"
                    ),
                ),
            ]
        )
        self.assertEqual(
            [message.content for message in result.kept_messages],
            [
                "src/memory_proxy/proxy_service.py:10:def build_app():\n"
                "src/memory_proxy/proxy_service.py:15:return app\n"
                "src/memory_proxy/proxy_service.py:22:def route_chat():"
            ],
        )
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_keeps_distinct_search_summary_hits_for_same_file(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(
                    role="tool",
                    content=(
                        "src/memory_proxy/proxy_service.py:10:def build_app():\n"
                        "src/memory_proxy/proxy_service.py:15:return app"
                    ),
                ),
                ChatMessage(
                    role="tool",
                    content=(
                        "src/memory_proxy/proxy_service.py:48:def route_chat():\n"
                        "src/memory_proxy/proxy_service.py:60:return response"
                    ),
                ),
            ]
        )
        self.assertEqual(
            [message.content for message in result.kept_messages],
            [
                "src/memory_proxy/proxy_service.py:10:def build_app():\n"
                "src/memory_proxy/proxy_service.py:15:return app",
                "src/memory_proxy/proxy_service.py:48:def route_chat():\n"
                "src/memory_proxy/proxy_service.py:60:return response",
            ],
        )
        self.assertEqual(result.dropped_indexes, [])

    def test_history_pruner_drops_search_summary_after_later_file_read(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(
                    role="tool",
                    content=(
                        "src/memory_proxy/proxy_service.py:120:def _compact_messages\n"
                        "src/memory_proxy/proxy_service.py:200:return PreparedChatRequest"
                    ),
                ),
                ChatMessage(role="assistant", content="查看 /workspace/workflow-memory-proxy/src/memory_proxy/proxy_service.py 的现有实现。"),
            ]
        )
        self.assertEqual(
            [message.content for message in result.kept_messages],
            ["查看 /workspace/workflow-memory-proxy/src/memory_proxy/proxy_service.py 的现有实现。"],
        )
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_drops_directory_listing_after_later_same_dir_file_read(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(
                    role="tool",
                    content="src/memory_proxy/\nsrc/memory_proxy/proxy_service.py\nsrc/memory_proxy/history_pruner.py",
                ),
                ChatMessage(
                    role="assistant",
                    content="查看 /workspace/workflow-memory-proxy/src/memory_proxy/prompt_builder.py 的现有实现。",
                ),
            ]
        )
        self.assertEqual(
            [message.content for message in result.kept_messages],
            ["查看 /workspace/workflow-memory-proxy/src/memory_proxy/prompt_builder.py 的现有实现。"],
        )
        self.assertEqual(result.dropped_indexes, [0])

    def test_history_pruner_keeps_directory_listing_for_unrelated_directory_read(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(
                    role="tool",
                    content="src/memory_proxy/\nsrc/memory_proxy/proxy_service.py\nsrc/memory_proxy/history_pruner.py",
                ),
                ChatMessage(
                    role="assistant",
                    content="查看 /workspace/other_project/src/worker/task_runner.py 的现有实现。",
                ),
            ]
        )
        self.assertEqual(
            [message.content for message in result.kept_messages],
            [
                "src/memory_proxy/\nsrc/memory_proxy/proxy_service.py\nsrc/memory_proxy/history_pruner.py",
                "查看 /workspace/other_project/src/worker/task_runner.py 的现有实现。",
            ],
        )
        self.assertEqual(result.dropped_indexes, [])

    def test_history_pruner_drops_symbol_only_search_command_after_search_and_read(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="搜索 build_prompt_memory 的实现。"),
                ChatMessage(
                    role="tool",
                    content="src/memory_proxy/prompt_builder.py:90:def build_prompt_memory(",
                ),
                ChatMessage(
                    role="assistant",
                    content="查看 /workspace/workflow-memory-proxy/src/memory_proxy/prompt_builder.py 的现有实现。",
                ),
            ]
        )
        self.assertEqual(
            [message.content for message in result.kept_messages],
            ["查看 /workspace/workflow-memory-proxy/src/memory_proxy/prompt_builder.py 的现有实现。"],
        )
        self.assertEqual(result.dropped_indexes, [0, 1])

    def test_history_pruner_drops_listing_command_after_listing_and_read(self) -> None:
        result = prune_history_messages(
            [
                ChatMessage(role="assistant", content="列出 src/memory_proxy 目录。"),
                ChatMessage(
                    role="tool",
                    content="src/memory_proxy/\nsrc/memory_proxy/proxy_service.py\nsrc/memory_proxy/prompt_builder.py",
                ),
                ChatMessage(
                    role="assistant",
                    content="查看 /workspace/workflow-memory-proxy/src/memory_proxy/prompt_builder.py 的现有实现。",
                ),
            ]
        )
        self.assertEqual(
            [message.content for message in result.kept_messages],
            ["查看 /workspace/workflow-memory-proxy/src/memory_proxy/prompt_builder.py 的现有实现。"],
        )
        self.assertEqual(result.dropped_indexes, [0, 1])

    def test_history_fidelity_marks_user_workflow_high_and_read_log_low(self) -> None:
        self.assertEqual(
            classify_history_fidelity(
                ChatMessage(role="user", content="你帮我把 docker 部署修好，然后继续做压缩算法。")
            ),
            "high",
        )
        self.assertEqual(
            classify_history_fidelity(
                ChatMessage(role="assistant", content="查看 /tmp/demo.py 的现有实现。")
            ),
            "low",
        )

    def test_static_zh_semantics_detects_task_management_language(self) -> None:
        self.assertTrue(contains_workflow_keyword("把这 4 个任务设置成已完成状态。"))
        self.assertTrue(is_task_management_text("你刚才创建的任务创建成功了吗？"))

    def test_event_dsl_round_trip(self) -> None:
        compressor = MemoryCompressor()
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_002",
                    role="assistant",
                    content="我已经完成了会话存储。",
                )
            ]
        )
        line = event_to_dsl(result.events[0])
        parsed = parse_dsl_line(line)
        self.assertEqual(parsed["type"], "task")
        self.assertEqual(parsed["subject"], "会话存储")

    def test_collapse_redundant_history_messages_keeps_richer_variant(self) -> None:
        collapsed = collapse_redundant_history_messages(
            [
                RawMessage("msg_001", "user", "必须不要明显影响精度。"),
                RawMessage("msg_002", "user", "我们必须不要明显影响精度。"),
                RawMessage("msg_003", "assistant", "下一步实现 /v1/responses。"),
            ]
        )
        self.assertEqual([message.message_id for message in collapsed], ["msg_002", "msg_003"])

    def test_collapse_redundant_history_messages_keeps_different_signal_types_apart(self) -> None:
        collapsed = collapse_redundant_history_messages(
            [
                RawMessage("msg_010", "user", "要不要接长期记忆？"),
                RawMessage("msg_011", "user", "接长期记忆。"),
            ]
        )
        self.assertEqual([message.message_id for message in collapsed], ["msg_010", "msg_011"])

    def test_collapse_redundant_history_messages_collapses_repeated_user_assistant_turns(self) -> None:
        collapsed = collapse_redundant_history_messages(
            [
                RawMessage("msg_020", "user", "必须不要明显影响精度。"),
                RawMessage("msg_021", "assistant", "明白，我会保证不明显影响精度。"),
                RawMessage("msg_022", "user", "我们必须不要明显影响精度。"),
                RawMessage("msg_023", "assistant", "好的，我会继续保证不明显影响精度。"),
            ]
        )
        self.assertEqual([message.message_id for message in collapsed], ["msg_022", "msg_023"])

    def test_collapse_redundant_history_messages_keeps_protected_turn_pair(self) -> None:
        collapsed = collapse_redundant_history_messages(
            [
                RawMessage("msg_030", "user", "必须不要明显影响精度。"),
                RawMessage("msg_031", "assistant", "明白，我会保证不明显影响精度。"),
                RawMessage("msg_032", "user", "我们必须不要明显影响精度。"),
                RawMessage("msg_033", "assistant", "好的，我会继续保证不明显影响精度。"),
            ],
            protected_message_ids={"msg_030", "msg_031"},
        )
        self.assertEqual([message.message_id for message in collapsed], ["msg_030", "msg_031"])

    def test_llm_extractor_builds_prompt_and_parses_json(self) -> None:
        client = StubLLMClient(
            '{"events":[{"type":"decision","action":"add","status":"active","subject":"第一版使用 Python + FastAPI","confidence":0.91}]}'
        )
        extractor = JsonLLMExtractor(client=client)
        events = extractor.extract(
            RawMessage(
                message_id="msg_010",
                role="assistant",
                content="我建议第一版先使用 Python + FastAPI。",
            ),
            session_id="sess_001",
            turn_id="turn_001",
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "decision")
        self.assertEqual(events[0].subject, "第一版使用 Python + FastAPI")
        self.assertIn("Return JSON only", client.prompts[0].system)

    def test_hybrid_extractor_prefers_higher_confidence_event(self) -> None:
        client = StubLLMClient(
            '{"events":[{"type":"goal","action":"add","status":"active","subject":"OpenAI 兼容代理","confidence":0.98}]}'
        )
        hybrid = HybridExtractor(RuleBasedExtractor(), JsonLLMExtractor(client=client))
        events = hybrid.extract(
            RawMessage(
                message_id="msg_011",
                role="user",
                content="我想做一个 OpenAI 兼容代理。",
            ),
            session_id="sess_001",
            turn_id="turn_001",
        )
        goals = [event for event in events if event.type == "goal"]
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].confidence, 0.98)

    def test_sqlite_store_persists_messages_events_and_snapshot(self) -> None:
        compressor = MemoryCompressor()
        result = compressor.compress(
            [
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
            ],
            session_id="sess_test",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMemoryStore(f"{temp_dir}/memory.db")
            store.init_db()
            store.upsert_session("sess_test", client="unit-test", upstream_model="demo-model")
            for index, message in enumerate(result.recent_messages, start=1):
                store.insert_raw_message("sess_test", f"turn_{index:04d}", message)
            store.insert_events(result.events)
            snapshot_id = store.insert_working_memory_snapshot(
                "sess_test",
                "turn_0002",
                result.working_memory,
                result.memory_dsl,
            )
            self.assertTrue(snapshot_id.startswith("snap_"))
            rows = store.list_memory_events("sess_test")
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(rows[0]["session_id"], "sess_test")

    def test_working_memory_prompt_view_excludes_artifacts_and_observations_by_default(self) -> None:
        compressor = MemoryCompressor()
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_001",
                    role="assistant",
                    content="我已经完成了 store.py，下一步准备实现 schema.sql。",
                ),
                RawMessage(
                    message_id="msg_002",
                    role="tool",
                    content="created /workspace/workflow-memory-proxy/storage/schema.sql",
                ),
            ]
        )
        prompt_view = working_memory_to_dsl(result.working_memory)
        full_view = working_memory_to_dsl(
            result.working_memory,
            include_artifacts=True,
            include_observations=True,
        )
        self.assertNotIn("ART[", prompt_view)
        self.assertNotIn("OBS[", prompt_view)
        self.assertIn("ART[", full_view)

    def test_english_messages_can_produce_prompt_memory(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage("m1", "user", "I want to build an OpenAI-compatible proxy."),
                RawMessage("m2", "assistant", "Use Python and FastAPI first."),
                RawMessage(
                    "m3",
                    "user",
                    "The main goal is to compress memory without hurting accuracy too much.",
                ),
                RawMessage(
                    "m4",
                    "assistant",
                    "I finished the design doc and will implement proxy routing next.",
                ),
            ],
            session_id="sess_en",
        )
        self.assertTrue(result.memory_dsl.strip())
        self.assertIn("GOAL:", result.memory_dsl)

    def test_assistant_meta_commentary_does_not_become_open_question(self) -> None:
        extractor = RuleBasedExtractor()
        events = extractor.extract(
            RawMessage(
                "m_assistant_meta",
                "assistant",
                "这样能直接回答你“先试 4B/8B 是否顺手、要不要改现有代理层”。",
            ),
            session_id="sess_meta",
            turn_id="turn_meta",
        )
        self.assertFalse(any(event.type == "question" for event in events))

    def test_compressor_marks_adjacent_recommendation_question_answered(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage("msg_q1", "user", "你推荐呢？"),
                RawMessage("msg_a1", "assistant", "我推荐默认走 cross-proxy-compatible。"),
            ],
            session_id="sess_qresolve_recommend",
        )
        self.assertEqual(result.working_memory.open_questions, [])
        self.assertIn("默认走 cross-proxy-compatible", result.working_memory.active_decisions)

    def test_compressor_marks_adjacent_keyword_overlap_question_answered(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage("msg_q2", "user", "是否可以再增加 md5 对引用的文件或者代码片段、网址等进行校验？"),
                RawMessage(
                    "msg_a2",
                    "assistant",
                    "这条已经加进去了。旧历史保留链路会先扫描重复的文件路径、URL、代码块，做规范化后算 MD5。",
                ),
            ],
            session_id="sess_qresolve_keywords",
        )
        self.assertEqual(result.working_memory.open_questions, [])

    def test_compressor_marks_adjacent_comparison_question_answered(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage("msg_q3", "user", "这俩有什么区别吗？"),
                RawMessage(
                    "msg_a3",
                    "assistant",
                    "OpenAI exact-compatible 追求和官方协议完全对齐，cross-proxy-compatible 追求穿过各种半兼容代理也不炸。",
                ),
            ],
            session_id="sess_qresolve_compare",
        )
        self.assertEqual(result.working_memory.open_questions, [])

    def test_merge_prompt_memories_dedupes_overlapping_fields(self) -> None:
        older = parse_prompt_memory_text(
            "GOAL: 做 OpenAI 兼容代理\n"
            "CONS: 不要明显影响精度\n"
            "DEC: 先用 Python + FastAPI\n"
            "ASK: 要不要接长期记忆？"
        )
        newer = parse_prompt_memory_text(
            "GOAL: 做 OpenAI 兼容代理\n"
            "CONS: 不要明显影响精度 ; 不要混进 CLIProxyAPI\n"
            "DEC: 先用 Python + FastAPI ; 保留 recent window\n"
            "NEXT: 实现 /v1/responses 代理"
        )
        merged = merge_prompt_memories([older, newer])
        merged_text = prompt_memory_to_text(merged)

        self.assertEqual(merged_text.count("GOAL:"), 1)
        self.assertEqual(merged_text.count("DEC:"), 1)
        self.assertIn("CONS: 不要明显影响精度 ; 不要混进 CLIProxyAPI", merged_text)
        self.assertIn("DEC: 先用 Python + FastAPI ; 保留 recent window", merged_text)
        self.assertIn("ASK: 要不要接长期记忆？", merged_text)
        self.assertIn("NEXT: 实现 /v1/responses 代理", merged_text)

    def test_merge_prompt_memories_semantically_dedupes_variants(self) -> None:
        older = parse_prompt_memory_text(
            "CONS: 不要混进 CLIProxyAPI\n"
            "OPEN: 补测试\n"
            "NEXT: 实现 /v1/responses 代理"
        )
        newer = parse_prompt_memory_text(
            "CONS: 必须不要混进 CLIProxyAPI\n"
            "OPEN: 继续补测试\n"
            "NEXT: 下一步继续实现 /v1/responses 代理"
        )
        merged = merge_prompt_memories([older, newer])
        merged_text = prompt_memory_to_text(merged)

        self.assertEqual(merged_text.count("CONS:"), 1)
        self.assertEqual(merged_text.count("TODO:"), 1)
        self.assertIn("CONS: 必须不要混进 CLIProxyAPI", merged_text)
        self.assertIn("TODO: 继续补测试", merged_text)
        self.assertIn("NEXT: 下一步继续实现 /v1/responses 代理", merged_text)

    def test_reducer_can_remove_answered_question_and_superseded_decision(self) -> None:
        reducer = WorkingMemoryReducer()
        memory = reducer.reduce(
            [
                MemoryEvent(
                    event_id="evt_001",
                    session_id="sess_reduce",
                    turn_id="turn_001",
                    source_message_ids=["msg_001"],
                    actor="user",
                    type="decision",
                    action="add",
                    status="active",
                    subject="先用 Python + FastAPI",
                ),
                MemoryEvent(
                    event_id="evt_002",
                    session_id="sess_reduce",
                    turn_id="turn_001",
                    source_message_ids=["msg_001"],
                    actor="user",
                    type="question",
                    action="add",
                    status="open",
                    subject="要不要接长期记忆？",
                ),
                MemoryEvent(
                    event_id="evt_003",
                    session_id="sess_reduce",
                    turn_id="turn_002",
                    source_message_ids=["msg_002"],
                    actor="assistant",
                    type="decision",
                    action="invalidate",
                    status="superseded",
                    subject="先用 Python + FastAPI",
                ),
                MemoryEvent(
                    event_id="evt_004",
                    session_id="sess_reduce",
                    turn_id="turn_002",
                    source_message_ids=["msg_002"],
                    actor="assistant",
                    type="question",
                    action="resolve",
                    status="answered",
                    subject="要不要接长期记忆？",
                ),
            ]
        )
        self.assertEqual(memory.active_decisions, [])
        self.assertEqual(memory.open_questions, [])

    def test_reducer_merges_repeated_task_variants(self) -> None:
        reducer = WorkingMemoryReducer()
        memory = reducer.reduce(
            [
                MemoryEvent(
                    event_id="evt_101",
                    session_id="sess_reduce",
                    turn_id="turn_001",
                    source_message_ids=["msg_001"],
                    actor="user",
                    type="task",
                    action="add",
                    status="proposed",
                    subject="补测试",
                ),
                MemoryEvent(
                    event_id="evt_102",
                    session_id="sess_reduce",
                    turn_id="turn_002",
                    source_message_ids=["msg_002"],
                    actor="user",
                    type="task",
                    action="add",
                    status="proposed",
                    subject="继续补测试",
                ),
            ]
        )
        self.assertEqual(memory.open_tasks, ["继续补测试"])

    def test_reducer_updates_state_slot_by_latest_value(self) -> None:
        reducer = WorkingMemoryReducer()
        memory = reducer.reduce(
            [
                MemoryEvent(
                    event_id="evt_s001",
                    session_id="sess_reduce",
                    turn_id="turn_001",
                    source_message_ids=["msg_001"],
                    actor="user",
                    type="state",
                    action="add",
                    status="active",
                    subject="当前回滚负责人=陈策",
                    details={"slot": "当前回滚负责人", "value": "陈策"},
                ),
                MemoryEvent(
                    event_id="evt_s002",
                    session_id="sess_reduce",
                    turn_id="turn_002",
                    source_message_ids=["msg_002"],
                    actor="user",
                    type="state",
                    action="add",
                    status="active",
                    subject="当前回滚负责人=周芮",
                    details={"slot": "当前回滚负责人", "value": "周芮"},
                ),
            ]
        )
        self.assertEqual(memory.state_facts, {"当前回滚负责人": "周芮"})

    def test_prompt_memory_builder_respects_budget_priority(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=14, max_item_tokens=24))
        memory = WorkingMemory(
            primary_goal="实现代理",
            active_constraints=["不要动 CLIProxyAPI"],
            active_decisions=["先用 hybrid"],
            completed_tasks=["接好会话存储"],
            open_tasks=["做独立预算器"],
            open_questions=["小模型是否值得接入"],
        )
        prompt_memory = builder.build(memory)
        prompt_text = prompt_memory_to_text(prompt_memory)

        self.assertLessEqual(approx_tokens(prompt_text), 14)
        self.assertIn("TODO: 做独立预算器", prompt_text)
        self.assertNotIn("DONE:", prompt_text)
        self.assertNotIn("CONS:", prompt_text)
        self.assertNotIn("DEC:", prompt_text)
        self.assertNotIn("ASK:", prompt_text)

    def test_prompt_memory_builder_truncates_long_entries_to_fit_budget(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=14, max_item_tokens=10))
        memory = WorkingMemory(primary_goal="这是一个非常长的目标描述，需要被裁剪后再进入 prompt memory")

        prompt_memory = builder.build(memory)
        prompt_text = prompt_memory_to_text(prompt_memory)

        self.assertLessEqual(approx_tokens(prompt_text), 14)
        self.assertTrue(prompt_text.startswith("GOAL: "))
        self.assertIn("...", prompt_text)

    def test_prompt_memory_builder_compacts_cross_field_redundancy(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=80, max_item_tokens=32))
        memory = WorkingMemory(
            primary_goal="实现 OpenAI 兼容代理",
            active_constraints=["不要明显影响精度"],
            active_decisions=["继续实现 /v1/responses 代理"],
            current_plan=["下一步继续实现 /v1/responses 代理"],
            open_tasks=["实现 /v1/responses 代理"],
            pending_verification_tasks=["实现 /v1/responses 代理"],
            open_questions=["要不要继续实现 /v1/responses 代理？"],
        )

        prompt_memory = builder.build(memory)
        prompt_text = prompt_memory_to_text(prompt_memory)

        self.assertIn("TODO: 实现 /v1/responses 代理", prompt_text)
        self.assertNotIn("NEXT:", prompt_text)
        self.assertNotIn("DEC:", prompt_text)
        self.assertNotIn("VERIFY:", prompt_text)
        self.assertNotIn("ASK:", prompt_text)

    def test_prompt_memory_builder_prefers_latest_pending_completion_over_older_verified_done(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=80, max_item_tokens=32, max_completed_tasks=1))
        memory = WorkingMemory(
            completed_tasks=["回放脚本"],
            pending_verification_tasks=["灰度联调"],
            open_tasks=["上线回归"],
        )

        prompt_memory = builder.build(memory)
        prompt_text = prompt_memory_to_text(prompt_memory)

        self.assertIn("DONE: 灰度联调", prompt_text)
        self.assertIn("TODO: 上线回归", prompt_text)
        self.assertNotIn("DONE: 回放脚本", prompt_text)

    def test_prompt_memory_builder_keeps_latest_verified_milestone_over_older_pending_completion(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=80, max_item_tokens=32, max_completed_tasks=1))
        memory = WorkingMemory(
            completed_tasks=["灰度联调"],
            pending_verification_tasks=["排课规则梳理"],
            open_tasks=["上线回归"],
        )

        prompt_memory = builder.build(memory)
        prompt_text = prompt_memory_to_text(prompt_memory)

        self.assertIn("DONE: 灰度联调", prompt_text)
        self.assertIn("TODO: 上线回归", prompt_text)
        self.assertNotIn("DONE: 排课规则梳理", prompt_text)

    def test_prompt_memory_builder_includes_artifacts_for_agent_workflow(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=120, max_item_tokens=40, max_artifacts=2))
        memory = WorkingMemory(
            open_tasks=["做上线回归"],
            active_artifacts=[
                "/workspace/demo/demo_runbook.md",
                "/workspace/demo/demo_pipeline.py",
                "/workspace/demo/demo_acceptance.md",
            ],
        )

        prompt_memory = builder.build(memory)
        prompt_text = prompt_memory_to_text(prompt_memory)

        self.assertIn("ART:", prompt_text)
        self.assertIn(".../demo/demo_pipeline.py", prompt_text)
        self.assertIn(".../demo/demo_acceptance.md", prompt_text)
        self.assertNotIn(".../demo/demo_runbook.md", prompt_text)

    def test_parse_prompt_memory_text_preserves_artifacts(self) -> None:
        parsed = parse_prompt_memory_text(
            "DONE: 灰度联调\nART: /.../demo/demo_pipeline.py ; /.../demo/demo_acceptance.md\nTODO: 做上线回归"
        )
        self.assertEqual(
            parsed.artifacts,
            ["/.../demo/demo_pipeline.py", "/.../demo/demo_acceptance.md"],
        )

    def test_prompt_memory_builder_keeps_artifacts_low_priority_under_tight_budget(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=14, max_item_tokens=32, max_artifacts=2))
        memory = WorkingMemory(
            completed_tasks=["灰度联调"],
            open_tasks=["做上线回归"],
            active_decisions=["按批量方案"],
            active_constraints=["不能改动老师端入口"],
            active_artifacts=["/workspace/demo/demo_acceptance.md"],
        )

        prompt_memory = builder.build(memory)
        prompt_text = prompt_memory_to_text(prompt_memory)

        self.assertIn("TODO:", prompt_text)
        self.assertNotIn("ART:", prompt_text)

    def test_prompt_memory_builder_suppresses_stale_artifacts_for_task_management_context(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=120, max_item_tokens=40, max_artifacts=2))
        memory = WorkingMemory(
            open_tasks=["创建4个任务"],
            active_constraints=["设置成已完成状态"],
            open_questions=["你刚才创建的任务创建成功了吗？"],
            active_artifacts=[
                "/workspace/demo/ai-tutor-backend-dev-spec.md",
                "/workspace/demo/ai-tutor-usage-test-case.md",
            ],
        )

        prompt_memory = builder.build(memory)
        prompt_text = prompt_memory_to_text(prompt_memory)

        self.assertIn("TODO: 创建4个任务", prompt_text)
        self.assertNotIn("ART:", prompt_text)

    def test_prompt_memory_builder_prefers_signal_rich_constraints(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=120, max_item_tokens=40, max_constraints=2))
        memory = WorkingMemory(
            active_constraints=[
                "保持回答自然",
                "注意 /workspace/workflow-memory-proxy/src/memory_proxy/proxy_service.py",
                "这里出现报错时要优先保留堆栈",
                "文案要稳定",
            ]
        )

        prompt_memory = builder.build(memory)
        prompt_text = prompt_memory_to_text(prompt_memory)

        self.assertIn("proxy_service.py", prompt_text)
        self.assertIn("报错", prompt_text)
        self.assertNotIn("保持回答自然 ; 文案要稳定", prompt_text)

    def test_prompt_memory_builder_merges_related_negative_constraints(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=120, max_item_tokens=48, max_constraints=4))
        memory = WorkingMemory(
            active_constraints=[
                "尽可能少引入新技术栈和语言",
                "尽可能不要用大模型来做分析",
                "不要用大模型等进行压缩",
            ]
        )

        prompt_memory = builder.build(memory)
        self.assertEqual(len(prompt_memory.constraints), 2)
        self.assertIn("尽可能少引入新技术栈和语言", prompt_memory.constraints)
        self.assertIn("尽可能不要用大模型来做分析或进行压缩", prompt_memory.constraints)

    def test_prompt_memory_builder_drops_answered_recommendation_questions(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=120, max_item_tokens=40, max_questions=2))
        memory = WorkingMemory(
            active_decisions=["默认走 cross-proxy-compatible"],
            open_questions=["你推荐呢？", "是否可以再增加 md5 对引用进行校验？"],
        )

        prompt_memory = builder.build(memory)
        self.assertNotIn("你推荐呢？", prompt_memory.open_questions)
        self.assertIn("是否可以再增加 md5 对引用进行校验？", prompt_memory.open_questions)

    def test_prompt_memory_builder_keeps_state_facts_and_filters_transient_meta(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=120, max_item_tokens=40, max_state_facts=4))
        memory = WorkingMemory(
            state_facts={
                "当前回滚负责人": "周芮",
                "ID方案": "ULID",
                "节点总上限": "18个",
            },
            active_constraints=["引用至少3个当前已知事实", "不能新增任何禁用技术"],
            open_questions=["回答是否适合做跨区域强一致写入", "是否需要补一轮冷备演练？"],
        )

        prompt_memory = builder.build(memory)
        prompt_text = prompt_memory_to_text(prompt_memory)

        self.assertIn("STATE: 当前回滚负责人=周芮 ; ID方案=ULID ; 节点总上限=18个", prompt_text)
        self.assertIn("CONS: 不能新增任何禁用技术", prompt_text)
        self.assertNotIn("引用至少3个当前已知事实", prompt_text)
        self.assertNotIn("回答是否适合做跨区域强一致写入", prompt_text)
        self.assertIn("是否需要补一轮冷备演练？", prompt_text)

    def test_state_slot_extraction_supports_education_domain(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_edu",
                    role="user",
                    content="学段高中；年级高二；班主任从王敏改成李青；课程方案确定用分层走班。",
                )
            ]
        )
        self.assertEqual(result.working_memory.state_facts.get("学段"), "高中")
        self.assertEqual(result.working_memory.state_facts.get("年级"), "高二")
        self.assertEqual(result.working_memory.state_facts.get("班主任"), "李青")
        self.assertEqual(result.working_memory.state_facts.get("课程方案"), "分层走班")

    def test_state_slot_extraction_supports_food_safety_domain(self) -> None:
        compressor = MemoryCompressor(recent_window=0)
        result = compressor.compress(
            [
                RawMessage(
                    message_id="msg_food",
                    role="user",
                    content="批次2026-A12；产地山东；风险等级橙色；处理时限24小时；召回级别二级。",
                )
            ]
        )
        self.assertEqual(result.working_memory.state_facts.get("批次"), "2026-A12")
        self.assertEqual(result.working_memory.state_facts.get("产地"), "山东")
        self.assertEqual(result.working_memory.state_facts.get("风险等级"), "橙色")
        self.assertEqual(result.working_memory.state_facts.get("处理时限"), "24小时")
        self.assertEqual(result.working_memory.state_facts.get("召回级别"), "二级")

    def test_prompt_memory_builder_shortens_long_file_references(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=120, max_item_tokens=40, max_constraints=2))
        long_path = "/workspace/workflow-memory-proxy/src/memory_proxy/proxy_service.py"
        memory = WorkingMemory(active_constraints=[f"重点检查 {long_path}"])

        prompt_memory = builder.build(memory)
        prompt_text = prompt_memory_to_text(prompt_memory)

        self.assertIn(".../memory_proxy/proxy_service.py", prompt_text)
        self.assertNotIn(long_path, prompt_text)

    def test_prompt_memory_builder_shortens_long_urls(self) -> None:
        builder = PromptMemoryBuilder(PromptMemoryConfig(max_tokens=120, max_item_tokens=40, max_decisions=2))
        long_url = "https://developers.openai.com/api/docs/guides/compaction?view=full"
        memory = WorkingMemory(active_decisions=[f"参考 {long_url}"])

        prompt_memory = builder.build(memory)
        prompt_text = prompt_memory_to_text(prompt_memory)

        self.assertIn("https://developers.openai.com/.../guides/compaction?...", prompt_text)
        self.assertNotIn(long_url, prompt_text)

    def test_build_memory_extractor_defaults_to_rule(self) -> None:
        extractor = build_memory_extractor(ProxySettings())
        self.assertIsInstance(extractor, RuleBasedExtractor)

    def test_build_memory_extractor_ignores_llm_mode_and_stays_rule(self) -> None:
        settings = ProxySettings(
            extractor_mode="hybrid",
            extractor_llm_base_url="http://127.0.0.1:11434/v1/",
            extractor_llm_model="qwen2.5:3b",
        )
        extractor = build_memory_extractor(settings)
        self.assertIsInstance(extractor, RuleBasedExtractor)

    def test_openai_compat_llm_client_reads_chat_completion_text(self) -> None:
        from memory_proxy.llm_client import OpenAICompatLLMClient  # noqa: E402

        class DummyResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"events":[{"type":"goal","action":"add","status":"active","subject":"测试目标","confidence":0.91}]}'
                            }
                        }
                    ]
                }

        class DummyClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.calls: list[tuple[str, dict[str, object], dict[str, str]]] = []

            def __enter__(self) -> "DummyClient":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def post(self, path: str, json: dict[str, object], headers: dict[str, str]) -> DummyResponse:
                self.calls.append((path, json, headers))
                return DummyResponse()

        with patch("memory_proxy.llm_client.httpx.Client", DummyClient):
            client = OpenAICompatLLMClient(
                base_url="http://127.0.0.1:11434/v1/",
                model="qwen2.5:3b",
            )
            output = client.complete(
                ExtractionPrompt(system="Return JSON only.", user="message: hello")
            )
        self.assertIn('"events"', output)


if __name__ == "__main__":
    unittest.main()
