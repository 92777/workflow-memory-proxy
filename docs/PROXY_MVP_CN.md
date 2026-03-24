# OpenAI 兼容代理 MVP

## 当前能力

这一版已经支持：

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/proxy/responses/compact`
- `POST /v1/responses/compact` 兼容别名，已废弃
- `WS /v1/responses`
- `stream=true` 的流式透传
- 保守压缩历史消息
- 可选 SQLite 持久化压缩产物、审计记录和 snapshot
- 基于 `x-session-id` / `conversation_id` 的会话记忆续接

## 压缩策略

为了尽量不影响精度，当前默认策略是：

- 只压缩旧消息
- 最近 `N` 条消息保留原文
- 如有必要，会额外保留少量“高价值旧消息”原文，例如带文件路径、长多行细节或错误信息的历史消息
- 前缀 `system/developer` 指令不压缩
- 如果历史太短，直接透传
- 如果历史里出现非字符串内容或复杂结构，直接透传
- 把旧历史先抽成 working memory，再按 token 预算裁成 prompt memory

默认 `prompt memory` 会优先保留：

1. `GOAL`
2. `OPEN`
3. `VERIFY`
4. `NEXT`
5. `CONS`
6. `DEC`
7. `ASK`

代理会额外注入一条 `system` 消息，把旧历史压成简短的 prompt memory，例如：

```text
GOAL: 实现 OpenAI 兼容代理
CONS: 不要明显影响精度
OPEN: 实现 /v1/chat/completions
ASK: 是否值得接长期记忆
```

## 环境变量

- `MCPROXY_UPSTREAM_BASE_URL`
- `MCPROXY_UPSTREAM_API_KEY`
- `MCPROXY_COMPRESSION_ENABLED`
- `MCPROXY_RECENT_WINDOW`
- `MCPROXY_SALIENT_HISTORY_MESSAGES`
- `MCPROXY_MIN_HISTORY_MESSAGES`
- `MCPROXY_PROMPT_MEMORY_MAX_TOKENS`
- `MCPROXY_STORE_ENABLED`
- `MCPROXY_STORE_DB_PATH`
- `MCPROXY_STORE_MAX_REQUESTS`
- `MCPROXY_TIMEOUT_SECONDS`
- `MCPROXY_MEMORY_SYSTEM_PROMPT`
- `MCPROXY_SESSION_AUTO_CONTINUE_ENABLED`
- `MCPROXY_SESSION_STITCHING_WINDOW_SECONDS`
- `MCPROXY_EXTRACTOR_MODE`
- `MCPROXY_EXTRACTOR_LLM_BASE_URL`
- `MCPROXY_EXTRACTOR_LLM_API_KEY`
- `MCPROXY_EXTRACTOR_LLM_MODEL`
- `MCPROXY_EXTRACTOR_LLM_TIMEOUT_SECONDS`
- `MCPROXY_EXTRACTOR_LLM_MIN_CONFIDENCE`

## 本地启动

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/uvicorn memory_proxy.server:create_app --factory --host 0.0.0.0 --port 8000
```

或者：

```bash
.venv/bin/python -m memory_proxy.server
```

## 响应头

代理会返回几个辅助头，方便你观察压缩行为：

- `x-memory-proxy-compressed`
- `x-memory-proxy-reason`
- `x-memory-proxy-history-dropped`
- `x-memory-proxy-session-id`
- `x-memory-proxy-request-id`
- `x-memory-proxy-snapshot-id`
- `x-memory-proxy-estimated-before-tokens`
- `x-memory-proxy-estimated-after-tokens`
- `x-memory-proxy-estimated-savings-pct`

`GET /health` 现在也会带上 `legacy_compact_alias` 状态，dashboard 顶部会直接显示旧 `compact` 别名是否已废弃以及 sunset 时间。

## 会话衔接

如果你想让代理跨轮次复用同一份压缩记忆，当前优先支持这些会话提示：

- `x-session-id`
- `session_id` / `conversation_id` 请求体字段
- `thread_id` 请求体字段
- `metadata.session_id` / `metadata.conversation_id`
- `metadata.thread_id` / `metadata.client_session_id` / `metadata.codex_session_id`

为了兼容挂在 `CLIProxyAPI` 后面的 Codex `/v1/responses` 链路，代理也会识别它透传下来的 `Session_id` / `Conversation_id` 头。

如果开启存储，`/v1/responses` 还会尝试根据 `previous_response_id` 找回上一次会话；如果你没有显式传 session，代理也可以在单一最近会话场景下按客户端指纹自动续接。

## `/v1/proxy/responses/compact`

这条接口适合“先压缩，再把压缩结果交给别的中间层或客户端”。

它会返回一个代理私有的 `memory_proxy.compaction` 对象，并在 `output` 里插入一条带 `[memory-proxy-compaction:v1]` 标记的标准 `message` 项。后续再把这份 `output` 作为 `/v1/responses` 的输入发回来时，代理会自动把这条压缩消息还原成 `instructions` 里的记忆块，不会把它原样继续转发给上游。

这样即使客户端误把 compact 结果直接发给别的 OpenAI 兼容中间层，也只会把它当成普通开发者消息处理，不会再触发上游对 `encrypted_content` 的校验错误。

旧的 `POST /v1/responses/compact` 仍然保留一段过渡期，方便老客户端迁移；它会继续返回旧的 `response.compaction` 对象名，但响应头会带 `Deprecation`、`Sunset` 和 `Link`，响应体里也会附带一个 `warning` 字段，提示你切到新的私有路径。当前 sunset 时间是 `2026-09-30 00:00:00 GMT`。

## 当前边界

这一版还没有做：

- 压缩质量自动评估
- 基于向量检索的细节回捞
- 长期记忆分层和召回排序
- 更细的多模态压缩策略
- 流式响应后的 assistant 记忆回写
- 对任意复杂 `responses.input` 结构的深度压缩

## 当前真实联调结果

基于本地上游 `gpt-5.4-mini` 的一组真实多轮消息测试：

- `chat/completions`：`prompt_tokens` 从 `187` 降到 `145`
- `responses`：`input_tokens` 从 `185` 降到 `145`

这两组测试里，压缩后的回答仍然保留了核心目标、关键约束和下一步动作。

如果开启 `MCPROXY_STORE_ENABLED=1`，代理还会把被压缩的原始历史、抽取事件和 working memory snapshot 写入 SQLite，便于后续调试和回放。
