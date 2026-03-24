# workflow-memory-proxy

面向 `Agent / openClaw / 编程助手` 的规则驱动工作流记忆压缩代理。

这个项目的目标不是把长对话简单总结成一段自然语言，而是把多轮上下文压成更适合继续工作的任务状态，例如：

```text
GOAL: 做一个排课回放助手
DONE: 做灰度联调
ART: /.../education_scheduler_runbook.md ; /.../education_scheduler_pipeline.py
TODO: 做上线回归
DEC: 按年级批量导入方案
CONS: 保留中文学段和班级术语 ; 不要接外部 SaaS 排课引擎
```

也就是说，它更关心这些问题：

- 已经做完了什么
- 当前下一步要做什么
- 现在生效的决策是什么
- 哪些约束不能破坏
- 哪些文件或工件还重要

## 项目特点

- 纯规则、静态压缩主链路
- 不依赖大模型来做压缩
- 不依赖 embedding 或向量库
- 尽量减少部署复杂度
- 优先面向中文、多轮、agent 工作流对话

当前实现重点已经收敛到：

- `GOAL`
- `DONE`
- `ART`
- `TODO`
- `DEC`
- `CONS`

## 当前能力

- OpenAI-compatible 代理接口
  - `POST /v1/chat/completions`
  - `POST /v1/responses`
  - `POST /v1/proxy/responses/compact`
- 基于规则的工作流状态抽取
- 最近窗口保留 + 旧历史压缩
- coding-agent 场景的历史去噪
  - `read / search / list / write / test`
- 静态代码重内容压缩
- 路径、代码块、URL 的重复引用别名
- health 和 dashboard
- 多业务方向评测脚本与回归测试

## 适用场景

- openClaw 一类需要长时间连续协作的 agent
- 编程助手、CLI 助手、工作流执行代理
- 希望降低 token 成本，同时尽量维持任务连续性的场景

当前不把自己定位成：

- 通用聊天摘要器
- 保证零失真的长期记忆系统
- 已经可以直接宣称稳定正式版的通用方案

## 快速启动

本地运行：

```bash
cd workflow-memory-proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
memory-proxy
```

Docker 运行：

```bash
cd workflow-memory-proxy
docker compose up --build
```

默认本地接口：

- `GET /health`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/proxy/responses/compact`

## 当前验证结果

仓库内置了多业务方向工作流回放评测脚本：

- [`scripts/evaluate_multidomain_workflows.py`](scripts/evaluate_multidomain_workflows.py)

最新一轮已落盘报告：

- 20 个业务方向
- 每个场景 24 轮用户请求
- 最终轮平均压缩率 `86.83%`
- `scenarios_with_issues = 0`

报告文件：

- [`testdata/eval_reports/memory_proxy_multidomain_eval_round5.json`](testdata/eval_reports/memory_proxy_multidomain_eval_round5.json)

## 文档入口

- 更完整的中文背景与设计说明：[`README_CN.md`](README_CN.md)
- MVP 说明：[`docs/PROXY_MVP_CN.md`](docs/PROXY_MVP_CN.md)
- 记忆 DSL：[`docs/MEMORY_DSL_V0_1.md`](docs/MEMORY_DSL_V0_1.md)
- 抽取与存储说明：[`docs/EXTRACTION_AND_STORAGE_CN.md`](docs/EXTRACTION_AND_STORAGE_CN.md)
- Dashboard 与 Docker 说明：[`docs/WEB_DASHBOARD_DOCKER_CN.md`](docs/WEB_DASHBOARD_DOCKER_CN.md)

## 当前建议

当前更适合：

- 内测
- RC 验证
- shadow traffic
- 受控 agent 工作负载

当前还不建议直接宣称：

- 任意对话都能零精度损失
- 已经达到通用稳定正式版
