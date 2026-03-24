# 抽取器与存储设计

## 目标

这一层负责把原始对话转换成可持久化、可回放、可压缩的记忆事件，并为后续接入真实模型抽取器留出稳定接口。

当前设计遵循：

- 原始消息永远保留
- 结构化事件追加写入
- `WorkingMemory` 从事件流重建
- LLM 只负责“填受控 schema”，不负责发明新语法

## 抽取管线

推荐流程：

1. 接收 `RawMessage`
2. 规则抽取器先跑一遍，拿高精度事件
3. 如有需要，再让 LLM 抽取器补充事件
4. 对两路结果做去重与合并
5. 把结果写入 `memory_events`
6. reducer 重建 `WorkingMemory`

当前代码里已经有：

- `RuleBasedExtractor`
- `JsonLLMExtractor`
- `HybridExtractor`

其中 `HybridExtractor` 会按 `(type, status, subject)` 去重，优先保留更高 `confidence` 的事件。

## LLM 抽取器约束

LLM 抽取器只输出 JSON：

```json
{
  "events": [
    {
      "actor": "assistant",
      "type": "task",
      "action": "update",
      "status": "claimed_done",
      "subject": "实现会话存储",
      "confidence": 0.91,
      "details": {
        "kind": "implementation"
      }
    }
  ]
}
```

约束：

- 只允许固定 `actor`
- 只允许固定 `type`
- 只允许固定 `action`
- 只允许固定 `status`
- `subject` 不能为空
- `confidence` 低于阈值的事件直接丢弃

## Prompt 设计原则

系统提示词强调这几点：

- 只提取长期任务有价值的信息
- 跳过闲聊和无效 filler
- `assistant` 自述完成只能是 `claimed_done`
- `verified_done` 只能来自工具结果或明确验证
- 返回 JSON，不返回解释

这类 prompt 的核心不是“让模型理解世界”，而是“让模型填固定表单”。

## SQLite 表设计

建议最少保留 4 张表：

### sessions

保存会话级信息。

字段建议：

- `session_id`
- `client`
- `upstream_model`
- `created_at`
- `updated_at`

### raw_messages

保存原始消息，作为最终证据层。

字段建议：

- `message_id`
- `session_id`
- `turn_id`
- `role`
- `content`
- `token_count`
- `created_at`

### memory_events

保存抽取后的结构化事件。

字段建议：

- `event_id`
- `session_id`
- `turn_id`
- `source_message_ids`
- `actor`
- `type`
- `action`
- `status`
- `subject`
- `details_json`
- `confidence`
- `supersedes`
- `created_at`

### working_memory_snapshots

保存 reducer 产出的聚合状态，方便调试、回滚和离线评估。

字段建议：

- `snapshot_id`
- `session_id`
- `turn_id`
- `memory_json`
- `memory_dsl`
- `created_at`

## 当前建议

第一版先把规则抽取和 LLM 抽取都写成“单消息输入 -> 事件数组输出”的统一接口。

这样后面无论你：

- 用本地模型
- 用云模型
- 用 OpenAI 兼容上游

都只是替换 `LLMExtractionClient`，不会影响 reducer、DSL 和存储层。
