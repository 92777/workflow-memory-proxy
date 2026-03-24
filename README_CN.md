# workflow-memory-proxy

规则驱动的工作流记忆压缩代理，面向 `Agent / openClaw / 编程助手` 的多轮对话场景。

如果你是第一次进入这个仓库，建议先看：

- GitHub 首页简版说明：[README.md](README.md)
- 当前 MVP 说明：[docs/PROXY_MVP_CN.md](docs/PROXY_MVP_CN.md)
- 记忆 DSL：[docs/MEMORY_DSL_V0_1.md](docs/MEMORY_DSL_V0_1.md)

当前实现重点已经从“通用摘要”收敛到“工作流状态压缩”：

- `GOAL`
- `DONE`
- `ART`
- `TODO`
- `DEC`
- `CONS`

也就是说，这个项目现在更像是一个把旧对话压成“任务状态”的代理层，而不是普通聊天摘要器。

## 1. 项目定位

这是一个独立于现有业务仓库的 AI 代理方案草案，目标是做一个“带记忆压缩层的模型代理”。

一句话定义：

`CLI / 编辑器 / Agent 客户端 -> Proxy -> 记忆压缩与缓存层 -> 上游模型`

核心目标：

- 持续降低上下文 token 消耗
- 在压缩历史对话的同时尽量保持回答质量
- 为后续 CLI、编辑器插件或多模型接入提供统一中间层

## 2. 当前技术结论

如果当前最优先的目标是“压缩记忆来节省 token”，推荐第一版使用 `Python + FastAPI`。

原因：

- 这个问题本质上更偏摘要、抽取、检索、重排、评估
- Python 更适合快速试验不同的记忆压缩策略
- 后面如果需要更强的 CLI 包装体验，再在外部增加 Node.js CLI 也不晚

建议选型：

- API/代理层：`FastAPI`
- 数据模型：`Pydantic`
- 上游请求：`httpx`
- 本地存储：`SQLite`
- 检索扩展：`pgvector` / `faiss` / `qdrant`

## 3. 关键判断

不要只做“对话全文总结”。

如果把历史对话简单压成一段总结，前期虽然省 token，但很容易出现：

- 关键约束被总结丢失
- 已做决策和原因被抹平
- 文件路径、命令、报错等细节难以保真
- 模型在错误摘要基础上继续推理，越跑越偏

更稳的做法是把记忆拆成多层。

## 4. 推荐的三层记忆模型

### 4.1 Recent Window

最近几轮原始对话保留原文，不压缩。

建议：

- 保留最近 4 到 12 轮
- 按 token 预算动态裁剪
- 最新的代码片段、报错、用户要求尽量保留原样

### 4.2 Working Memory

把较老的对话压缩成结构化状态，而不是一段自然语言摘要。

推荐字段：

- `current_goal`
- `constraints`
- `decisions`
- `artifacts`
- `open_questions`
- `next_steps`

作用：

- 低成本维持任务连续性
- 让模型理解当前目标和历史上下文
- 把“聊天记录”转成“任务状态”

### 4.3 Retrieval Memory

旧细节不长期驻留在 prompt 里，需要时再检索回填。

适合放进检索层的内容：

- 文件摘要
- 工具结果摘要
- 长日志片段
- 长文档切片
- 关键决策记录

原则：

- 默认不进 prompt
- 只有和当前请求相关时才取回

## 5. 缓存层建议

缓存和记忆不是一回事，但建议一起做。

### 5.1 Exact Cache

同模型、同参数、同规范化输入时直接命中。

### 5.2 Tool Cache

缓存工具层产物，往往比缓存最终回答更稳定。

适合缓存：

- 文件摘要
- 文档解析结果
- 搜索结果
- 稳定 API 响应

### 5.3 Semantic Cache

可以做，但不建议第一版就依赖。

原因：

- 语义相近不代表上下文等价
- 在代码和 agent 场景里误命中成本很高

## 6. 推荐请求流程

1. 接收客户端请求
2. 保存原始消息
3. 估算本轮 token 预算
4. 保留最近窗口原文
5. 把较老消息压入 working memory
6. 根据当前问题检索 retrieval memory
7. 组装最终 prompt
8. 请求上游模型
9. 保存输出并更新记忆、缓存、索引
10. 记录压缩率、命中率和延迟

## 7. Prompt 组装原则

建议优先级：

1. 系统规则和不可压缩指令
2. 最近几轮原文
3. working memory
4. 按需检索的历史细节
5. 必要的工具结果摘要

不建议压缩的内容：

- 系统提示词
- 工具 schema
- 当前正在编辑的关键代码片段
- 本轮直接相关的报错原文

## 8. MVP 范围

第一版建议只做闭环，不要过早复杂化。

MVP 功能：

- OpenAI 兼容代理接口
- 会话与消息持久化
- rolling working memory
- recent window + working memory 的 prompt 拼装
- 文件/工具摘要缓存
- token 用量和压缩率统计

先不做：

- 复杂 rerank
- 多层长期记忆
- 语义缓存主流程
- 可视化后台
- 多租户权限体系

## 9. 目录结构建议

```text
memory-compression-proxy/
  app/
    main.py
    api/
      routes_chat.py
      routes_sessions.py
    core/
      config.py
      tokenizer.py
      prompt_builder.py
      proxy_client.py
    memory/
      compressor.py
      working_memory.py
      retrieval.py
      policies.py
    cache/
      exact_cache.py
      tool_cache.py
    storage/
      models.py
      repo_sessions.py
      repo_messages.py
      repo_memory.py
    schemas/
      chat.py
      memory.py
      metrics.py
    services/
      session_service.py
      chat_service.py
      metrics_service.py
  tests/
  scripts/
  README_CN.md
```

## 10. 核心数据结构草案

### 10.1 Session

```json
{
  "session_id": "sess_xxx",
  "created_at": "2026-03-21T00:00:00Z",
  "client": "cli",
  "model": "upstream-model-id"
}
```

### 10.2 Message

```json
{
  "message_id": "msg_xxx",
  "session_id": "sess_xxx",
  "role": "user",
  "content": "请帮我继续修这个问题",
  "token_count": 128,
  "created_at": "2026-03-21T00:00:00Z"
}
```

### 10.3 WorkingMemory

```json
{
  "session_id": "sess_xxx",
  "current_goal": "实现代理中的记忆压缩",
  "constraints": [
    "优先减少 token 消耗",
    "尽量避免摘要漂移"
  ],
  "decisions": [
    {
      "decision": "第一版使用 Python + FastAPI",
      "reason": "更适合快速实验压缩策略"
    }
  ],
  "artifacts": [
    {
      "type": "doc",
      "value": "memory-compression-proxy/README_CN.md"
    }
  ],
  "open_questions": [
    "第一版是否需要向量检索"
  ],
  "next_steps": [
    "实现会话持久化",
    "实现 prompt builder",
    "实现 working memory 更新逻辑"
  ]
}
```

## 11. 推荐实施阶段

### 阶段 1

- 固定保留 recent window
- 旧消息压成 working memory
- 不引入向量检索

### 阶段 2

- 对长工具输出和长文档摘要做分片
- 增加基础检索
- 只在需要时回填细节

### 阶段 3

- 评估不同压缩策略
- 增加更细的记忆槽位
- 尝试语义缓存或 rerank

## 12. 评估指标

建议最少记录：

- `input_tokens_before`
- `input_tokens_after`
- `compression_ratio`
- `cache_hit_rate`
- `retrieval_hit_rate`
- `latency_ms`
- `answer_quality_score`

压缩率不是唯一指标，质量是否漂移同样重要。

## 13. 主要风险

- 摘要漂移：压缩结果和原始上下文不一致
- 过度压缩：省了 token，但失去关键信息
- 缓存污染：旧结果在新上下文下不再成立
- 代码保真不足：代码、diff、日志、报错有时必须保留原文

## 14. 当前推荐落地顺序

1. 搭建最小代理接口
2. 保存完整会话和消息
3. 加入 token 统计
4. 实现 recent window + working memory 的 prompt builder
5. 实现基础缓存
6. 增加基础检索
7. 做压缩质量评估

## 15. 下一步

如果继续往下做，最合适的动作是：

1. 先在这个独立目录里初始化一个 `Python + FastAPI` 项目骨架
2. 定义 `Session / Message / WorkingMemory` 模型
3. 写一个最小版 `prompt_builder`
4. 接一个上游模型接口跑通闭环
5. 再把记忆更新逻辑细化成可迭代策略

---

这份文档目前是独立项目说明，不依附于 `CLIProxyAPI`。如果你愿意，下一步我可以直接在这个新目录里继续给你生成项目骨架。
