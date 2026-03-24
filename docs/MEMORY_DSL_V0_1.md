# Memory DSL v0.1

## 目标

`Memory DSL` 是给模型和系统内部共同使用的一层受控记忆语言。

设计目标：

- 比自然语言更紧凑
- 比自由缩写更稳定
- 比 JSON 更适合直接放入 prompt
- 能无损映射回结构化 `MemoryEvent`

## 设计原则

- 底层事实来源始终是 `MemoryEvent`
- DSL 只是面向模型的紧凑表达层
- 事件类型和状态名固定为英文关键字
- 事件的 `subject` 可以保留用户原语言
- DSL 必须可解析、可校验、可回放

## 支持的事件类型

- `GOAL`
- `CONS`
- `PLAN`
- `TASK`
- `DEC`
- `ART`
- `OBS`
- `QUES`

## 支持的 actor

- `user`
- `assistant`
- `tool`
- `system`

## 支持的 action

- `add`
- `update`
- `resolve`
- `invalidate`
- `verify`

## 关键状态

通用状态：

- `active`
- `superseded`
- `invalidated`
- `resolved`
- `stale`
- `open`
- `answered`

`TASK` 特有状态：

- `proposed`
- `in_progress`
- `claimed_done`
- `verified_done`
- `blocked`
- `failed`
- `superseded`

## 语法

单行格式：

```text
TYPE[event_id]: subject | actor=assistant | status=claimed_done | action=update | confidence=0.91
```

其中：

- `TYPE` 是固定大写关键字
- `event_id` 是事件标识
- `subject` 是简短语义主体
- 后续属性是可选键值对

## 最小属性集

推荐属性：

- `actor`
- `status`
- `action`
- `confidence`

可选属性：

- `ref`：原始消息引用
- `supersedes`：被覆盖的旧事件 id
- `kind`：观察类子类型，如 `test_result`

## 转义规则

- 字面量中的 `\` 写成 `\\`
- 字面量中的 `|` 写成 `\|`
- 行内属性之间使用 ` | ` 分隔
- parser 只按未转义的 `|` 分隔字段

## 示例

```text
GOAL[g1]: build OpenAI-compatible proxy | actor=user | status=active
CONS[c1]: 优先减少 token 消耗 | actor=user | status=active
DEC[d1]: stack=python-fastapi | actor=assistant | status=active
TASK[t1]: 实现会话存储 | actor=assistant | status=claimed_done
PLAN[p1]: 实现 prompt builder | actor=assistant | status=active
OBS[o1]: session storage tests passed | actor=tool | status=active | kind=test_result
QUES[q1]: 第一版是否需要向量检索 | actor=user | status=open
```

## 与 MemoryEvent 的映射

`MemoryEvent.type -> DSL TYPE`

- `goal -> GOAL`
- `constraint -> CONS`
- `plan -> PLAN`
- `task -> TASK`
- `decision -> DEC`
- `artifact -> ART`
- `observation -> OBS`
- `question -> QUES`

`MemoryEvent` 推荐字段：

```json
{
  "event_id": "evt_001",
  "session_id": "sess_001",
  "source_message_ids": ["msg_001"],
  "actor": "assistant",
  "type": "task",
  "action": "update",
  "status": "claimed_done",
  "subject": "实现会话存储",
  "details": {},
  "confidence": 0.91
}
```

## 用法建议

- 存储层：保留 `MemoryEvent`
- Prompt 层：优先放 `Memory DSL`
- 调试层：同时展示原始消息、事件和 DSL

## v0.1 边界

本版本暂不覆盖：

- 嵌套结构
- 多值列表属性
- 自动事实冲突消解
- 长文档分块压缩
