from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from memory_proxy import ProxySettings, UpstreamOpenAIClient, create_app  # noqa: E402


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    slug: str
    domain: str
    goal: str
    task_seed: str
    task_plan: str
    task_mapping: str
    task_validation: str
    task_replay: str
    task_gray: str
    final_todo: str
    decision_old: str
    decision_new: str
    constraint_primary: str
    constraint_secondary: str
    constraint_third: str
    symbol_plan: str
    symbol_validation: str
    symbol_release: str

    @property
    def workspace_dir(self) -> str:
        return f"/workspace/{self.slug}"

    @property
    def code_file(self) -> str:
        return f"{self.workspace_dir}/{self.slug}_pipeline.py"

    @property
    def artifact_plan(self) -> str:
        return f"{self.workspace_dir}/{self.slug}_runbook.md"

    @property
    def artifact_acceptance(self) -> str:
        return f"{self.workspace_dir}/{self.slug}_acceptance.md"


def build_upstream_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        payload = await request.json()
        return JSONResponse(
            {
                "id": "chatcmpl_eval",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "收到，我会继续推进。",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
                "echo_messages": payload["messages"],
            }
        )

    return app


def make_client() -> TestClient:
    settings = ProxySettings(
        upstream_base_url="http://upstream/v1/",
        compression_enabled=True,
        recent_window=2,
        recent_min_messages=1,
        min_history_messages=2,
        salient_history_messages=1,
        recent_token_budget=120,
        salient_history_token_budget=96,
        prompt_memory_max_tokens=160,
    )
    upstream = build_upstream_app()
    upstream_client = UpstreamOpenAIClient(
        settings,
        transport=httpx.ASGITransport(app=upstream),
    )
    app = create_app(settings, upstream_client=upstream_client)
    return TestClient(app)


def build_scenarios() -> list[ScenarioSpec]:
    return [
        ScenarioSpec("education_scheduler", "教育排课", "排课回放助手", "课程骨架", "排课规则梳理", "班级字段映射", "导入校验", "回放脚本", "灰度联调", "上线回归", "按逐校导入方案", "按年级批量导入方案", "必须保留中文学段和班级术语", "不要接外部 SaaS 排课引擎", "不能改动老师端入口", "build_schedule_frame", "validate_grade_batch", "run_release_check"),
        ScenarioSpec("food_safety_recall", "食品安全召回", "召回工单压缩助手", "批次骨架", "召回字段梳理", "批次字段映射", "召回校验", "回放脚本", "灰度联调", "上线回归", "按人工复核清单方案", "按风险等级批量召回方案", "必须保留批次和产地字段", "不要接外部 风险 SaaS", "不能改动门店端入口", "build_recall_frame", "validate_batch_risk", "run_food_release_check"),
        ScenarioSpec("logistics_dispatch", "物流调度", "调度回放助手", "站点骨架", "运单规则梳理", "站点字段映射", "路由校验", "回放脚本", "灰度联调", "上线回归", "按人工锁单方案", "按区域批量派车方案", "必须保留站点和时间窗字段", "不要接外部 调度 SaaS", "不能改动司机端入口", "build_dispatch_frame", "validate_route_batch", "run_dispatch_release_check"),
        ScenarioSpec("clinic_scheduling", "门诊排班", "排班压缩助手", "科室骨架", "班次规则梳理", "科室字段映射", "排班校验", "回放脚本", "灰度联调", "上线回归", "按人工锁号方案", "按科室批量排班方案", "必须保留科室和班次字段", "不要接外部 排班 SaaS", "不能改动医生端入口", "build_clinic_frame", "validate_shift_batch", "run_clinic_release_check"),
        ScenarioSpec("ecommerce_refund", "电商退款", "退款回放助手", "退款骨架", "退款规则梳理", "订单字段映射", "退款校验", "回放脚本", "灰度联调", "上线回归", "按人工复核退款方案", "按订单批量退款方案", "必须保留订单号和退款原因字段", "不要接外部 财务 SaaS", "不能改动商家端入口", "build_refund_frame", "validate_order_batch", "run_refund_release_check"),
        ScenarioSpec("factory_qc", "制造质检", "质检回放助手", "质检骨架", "质检规则梳理", "工序字段映射", "质检校验", "回放脚本", "灰度联调", "上线回归", "按人工复核质检方案", "按工序批量质检方案", "必须保留产线和工序字段", "不要接外部 质检 SaaS", "不能改动车间端入口", "build_qc_frame", "validate_process_batch", "run_qc_release_check"),
        ScenarioSpec("support_knowledge", "客服知识库", "知识库回放助手", "知识骨架", "问答规则梳理", "标签字段映射", "问答校验", "回放脚本", "灰度联调", "上线回归", "按人工审核问答方案", "按标签批量上架方案", "必须保留问题分类和知识标签", "不要接外部 知识 SaaS", "不能改动客服端入口", "build_knowledge_frame", "validate_tag_batch", "run_knowledge_release_check"),
        ScenarioSpec("hotel_booking", "酒店预订", "预订回放助手", "房态骨架", "房态规则梳理", "房型字段映射", "预订校验", "回放脚本", "灰度联调", "上线回归", "按人工锁房方案", "按房型批量预订方案", "必须保留房型和入住日期字段", "不要接外部 酒店 SaaS", "不能改动前台端入口", "build_booking_frame", "validate_room_batch", "run_booking_release_check"),
        ScenarioSpec("live_ops", "直播运营", "直播回放助手", "直播骨架", "排期规则梳理", "场次字段映射", "场次校验", "回放脚本", "灰度联调", "上线回归", "按人工排场方案", "按场次批量排期方案", "必须保留主播和场次字段", "不要接外部 直播 SaaS", "不能改动主播端入口", "build_live_frame", "validate_show_batch", "run_live_release_check"),
        ScenarioSpec("recruiting_pipeline", "招聘流程", "招聘回放助手", "岗位骨架", "流程规则梳理", "候选人字段映射", "流程校验", "回放脚本", "灰度联调", "上线回归", "按人工筛选方案", "按岗位批量推进方案", "必须保留岗位和候选人字段", "不要接外部 招聘 SaaS", "不能改动面试官端入口", "build_hiring_frame", "validate_candidate_batch", "run_hiring_release_check"),
        ScenarioSpec("property_workorder", "物业工单", "工单回放助手", "工单骨架", "工单规则梳理", "楼栋字段映射", "工单校验", "回放脚本", "灰度联调", "上线回归", "按人工派单方案", "按楼栋批量派单方案", "必须保留楼栋和工单类别", "不要接外部 工单 SaaS", "不能改动住户端入口", "build_workorder_frame", "validate_building_batch", "run_workorder_release_check"),
        ScenarioSpec("procurement_supply", "供应链采购", "采购回放助手", "采购骨架", "采购规则梳理", "供应商字段映射", "采购校验", "回放脚本", "灰度联调", "上线回归", "按人工比价方案", "按供应商批量采购方案", "必须保留供应商和批次字段", "不要接外部 采购 SaaS", "不能改动采购端入口", "build_procurement_frame", "validate_supplier_batch", "run_procurement_release_check"),
        ScenarioSpec("agri_sensor", "农业传感", "传感回放助手", "传感骨架", "告警规则梳理", "地块字段映射", "告警校验", "回放脚本", "灰度联调", "上线回归", "按人工复核告警方案", "按地块批量告警方案", "必须保留地块和设备字段", "不要接外部 农业 SaaS", "不能改动巡检端入口", "build_sensor_frame", "validate_field_batch", "run_sensor_release_check"),
        ScenarioSpec("parking_control", "城市停车", "停车回放助手", "停车骨架", "计费规则梳理", "车场字段映射", "计费校验", "回放脚本", "灰度联调", "上线回归", "按人工复核扣费方案", "按车场批量计费方案", "必须保留车场和时段字段", "不要接外部 停车 SaaS", "不能改动车主端入口", "build_parking_frame", "validate_lot_batch", "run_parking_release_check"),
        ScenarioSpec("insurance_claim", "保险理赔", "理赔回放助手", "理赔骨架", "理赔规则梳理", "保单字段映射", "理赔校验", "回放脚本", "灰度联调", "上线回归", "按人工复核理赔方案", "按保单批量理赔方案", "必须保留保单和理赔单字段", "不要接外部 理赔 SaaS", "不能改动查勘端入口", "build_claim_frame", "validate_policy_batch", "run_claim_release_check"),
        ScenarioSpec("contract_review", "法务合同", "合同回放助手", "合同骨架", "审核规则梳理", "条款字段映射", "审核校验", "回放脚本", "灰度联调", "上线回归", "按人工复核条款方案", "按合同批量审核方案", "必须保留合同编号和条款类别", "不要接外部 合同 SaaS", "不能改动法务端入口", "build_contract_frame", "validate_clause_batch", "run_contract_release_check"),
        ScenarioSpec("soc_triage", "安全运营", "告警回放助手", "告警骨架", "分流规则梳理", "告警字段映射", "分流校验", "回放脚本", "灰度联调", "上线回归", "按人工复核告警方案", "按告警批量分流方案", "必须保留告警等级和来源字段", "不要接外部 安全 SaaS", "不能改动值班端入口", "build_soc_frame", "validate_alert_batch", "run_soc_release_check"),
        ScenarioSpec("video_moderation", "短视频审核", "审核回放助手", "审核骨架", "审核规则梳理", "标签字段映射", "审核校验", "回放脚本", "灰度联调", "上线回归", "按人工复核标签方案", "按标签批量审核方案", "必须保留审核标签和场景字段", "不要接外部 审核 SaaS", "不能改动审核端入口", "build_moderation_frame", "validate_scene_batch", "run_moderation_release_check"),
        ScenarioSpec("energy_ops", "能源运维", "运维回放助手", "运维骨架", "巡检规则梳理", "站点字段映射", "巡检校验", "回放脚本", "灰度联调", "上线回归", "按人工派单方案", "按站点批量巡检方案", "必须保留站点和设备字段", "不要接外部 运维 SaaS", "不能改动巡检端入口", "build_energy_frame", "validate_station_batch", "run_energy_release_check"),
        ScenarioSpec("library_catalog", "图书编目", "编目回放助手", "编目骨架", "编目规则梳理", "馆藏字段映射", "编目校验", "回放脚本", "灰度联调", "上线回归", "按人工复核编目方案", "按馆藏批量编目方案", "必须保留馆藏和分类字段", "不要接外部 编目 SaaS", "不能改动馆员端入口", "build_catalog_frame", "validate_catalog_batch", "run_catalog_release_check"),
    ]


def scripted_turns(spec: ScenarioSpec) -> list[tuple[str, list[dict[str, str]]]]:
    return [
        (
            f"我要给{spec.domain}做一个{spec.goal}，你帮我把方案骨架先搭起来。",
            [{"role": "assistant", "content": f"我先梳理{spec.task_plan}和{spec.task_mapping}。"}],
        ),
        (
            spec.constraint_primary,
            [
                {"role": "assistant", "content": f"我已经完成了{spec.task_seed}，下一步准备做{spec.task_plan}。"},
                {"role": "tool", "content": f"tests passed for {spec.task_seed}"},
            ],
        ),
        (
            f"你帮我把{spec.task_plan}干了。",
            [
                {"role": "assistant", "content": f"搜索 {spec.symbol_plan} 的实现。"},
                {"role": "tool", "content": f"{spec.code_file}:32:def {spec.symbol_plan}("},
                {"role": "assistant", "content": f"查看 {spec.code_file} 的现有实现。"},
                {"role": "assistant", "content": f"我已经完成了{spec.task_plan}，下一步准备做{spec.task_mapping}。"},
            ],
        ),
        (
            f"还是按{spec.decision_old}。",
            [{"role": "assistant", "content": f"收到，先按{spec.decision_old}。"}],
        ),
        (
            f"请记住文件 {spec.artifact_plan} 里有当前回放清单。",
            [{"role": "assistant", "content": f"请记住文件 {spec.artifact_plan} 里有当前回放清单。"}],
        ),
        (
            f"你帮我把{spec.task_mapping}干了。",
            [
                {"role": "assistant", "content": f"列出 {spec.workspace_dir} 目录。"},
                {
                    "role": "tool",
                    "content": f"{spec.workspace_dir}/\n{spec.code_file}\n{spec.artifact_plan}",
                },
                {"role": "assistant", "content": f"我已经完成了{spec.task_mapping}，下一步准备做{spec.task_validation}。"},
                {"role": "tool", "content": f"tests passed for {spec.task_mapping}"},
            ],
        ),
        (
            spec.constraint_secondary,
            [{"role": "assistant", "content": "收到，我会继续遵守。"}],
        ),
        (
            f"现在{spec.task_mapping}已经完成了，下一步做{spec.task_validation}。",
            [{"role": "assistant", "content": f"查看 {spec.code_file} 的现有实现。"}],
        ),
        (
            f"还是按{spec.decision_new}。",
            [{"role": "assistant", "content": f"收到，后续改按{spec.decision_new}。"}],
        ),
        (
            f"你帮我把{spec.task_validation}干了。",
            [
                {"role": "assistant", "content": f"搜索 {spec.symbol_validation} 的实现。"},
                {"role": "tool", "content": f"{spec.code_file}:88:def {spec.symbol_validation}("},
                {"role": "assistant", "content": f"查看 {spec.code_file} 的现有实现。"},
                {"role": "assistant", "content": f"我已经完成了{spec.task_validation}，下一步准备做{spec.task_replay}。"},
            ],
        ),
        (
            f"现在{spec.task_validation}已经通过了，下一步做{spec.task_replay}。",
            [{"role": "assistant", "content": f"收到，我会继续推进{spec.task_replay}。"}],
        ),
        (
            f"请记住文件 {spec.artifact_acceptance} 里有验收口径。",
            [{"role": "assistant", "content": f"请记住文件 {spec.artifact_acceptance} 里有验收口径。"}],
        ),
        (
            f"你帮我把{spec.task_replay}干了。",
            [
                {
                    "role": "assistant",
                    "content": f"*** Begin Patch\n*** Update File: {spec.code_file}\n+def {spec.symbol_validation}():\n+    return 'ok'\n*** End Patch",
                },
                {"role": "assistant", "content": f"我已经完成了{spec.task_replay}，下一步准备做{spec.task_gray}。"},
                {"role": "tool", "content": f"tests passed for {spec.task_replay}"},
            ],
        ),
        (
            spec.constraint_third,
            [{"role": "assistant", "content": "明白，我会继续遵守。"}],
        ),
        (
            f"现在{spec.task_replay}已经完成了，下一步做{spec.task_gray}。",
            [{"role": "assistant", "content": f"继续读取 {spec.code_file} 的现有实现。"}],
        ),
        (
            f"你帮我把{spec.task_gray}干了。",
            [
                {"role": "assistant", "content": f"搜索 {spec.symbol_release} 的实现。"},
                {"role": "tool", "content": f"{spec.code_file}:120:def {spec.symbol_release}("},
                {"role": "assistant", "content": f"查看 {spec.code_file} 的现有实现。"},
                {"role": "assistant", "content": f"我已经完成了{spec.task_gray}，下一步准备做{spec.final_todo}。"},
            ],
        ),
        (
            f"验收通过: {spec.task_gray}",
            [{"role": "assistant", "content": f"收到，{spec.task_gray}已经验收通过。"}],
        ),
        (
            f"还是按{spec.decision_new}，不要再回到{spec.decision_old}。",
            [{"role": "assistant", "content": f"收到，继续按{spec.decision_new}。"}],
        ),
        (
            f"现在{spec.task_gray}已经通过了，下一步做{spec.final_todo}。",
            [{"role": "assistant", "content": "我会继续推进。"}],
        ),
        (
            f"列出 {spec.workspace_dir} 目录。",
            [
                {
                    "role": "tool",
                    "content": f"{spec.workspace_dir}/\n{spec.code_file}\n{spec.artifact_acceptance}",
                },
                {"role": "assistant", "content": f"查看 {spec.artifact_acceptance} 的现有实现。"},
            ],
        ),
        (
            f"搜索 {spec.symbol_release} 的实现。",
            [
                {"role": "tool", "content": f"{spec.code_file}:120:def {spec.symbol_release}("},
                {"role": "assistant", "content": f"查看 {spec.code_file} 的现有实现。"},
            ],
        ),
        (
            "继续。",
            [{"role": "assistant", "content": "好的，我继续。"}],
        ),
        (
            "最后一次记忆核对前，不要解释，只保留当前任务状态。",
            [{"role": "assistant", "content": "收到。"}],
        ),
        (
            "最后一次记忆核对：只输出当前 DONE/TODO/DEC/CONS/ART，不要解释。",
            [],
        ),
    ]


def _extract_memory_text(echo_messages: list[dict[str, Any]]) -> str:
    for item in echo_messages:
        if item.get("role") != "system":
            continue
        content = str(item.get("content") or "")
        if "Older summary. Recent messages override." not in content:
            continue
        if "\n\n" in content:
            return content.split("\n\n", 1)[1].strip()
        return content.strip()
    return ""


def _combined_forwarded_text(echo_messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(item.get("content") or "") for item in echo_messages)


def _scenario_issues(spec: ScenarioSpec, combined_text: str, memory_text: str) -> list[str]:
    issues: list[str] = []
    primary_constraint_core = spec.constraint_primary
    for prefix in ("必须", "需要", "得", "优先", "重点是"):
        if primary_constraint_core.startswith(prefix):
            primary_constraint_core = primary_constraint_core[len(prefix) :]
            break
    primary_constraint_core = primary_constraint_core.strip()
    checks = [
        (spec.goal, "goal_missing"),
        (spec.decision_new, "latest_decision_missing"),
        (primary_constraint_core, "primary_constraint_missing"),
        (spec.constraint_secondary, "secondary_constraint_missing"),
        (spec.final_todo, "final_todo_missing"),
        (spec.task_gray, "latest_done_missing"),
    ]
    for needle, code in checks:
        if needle not in combined_text and needle not in memory_text:
            issues.append(code)

    artifact_needles = {
        Path(spec.artifact_acceptance).name,
        Path(spec.code_file).name,
        Path(spec.artifact_plan).name,
    }
    if not any(needle in memory_text for needle in artifact_needles):
        issues.append("artifact_missing")

    decision_lines = [line for line in memory_text.splitlines() if line.startswith("DEC:")]
    if any(spec.decision_old in line for line in decision_lines):
        issues.append("stale_decision_retained")
    verify_lines = [line for line in memory_text.splitlines() if line.startswith("VERIFY:")]
    if any(spec.task_validation in line or spec.task_gray in line for line in verify_lines):
        issues.append("stale_verify_retained")

    noise_checks = [
        ("tests passed", "stale_test_noise_retained"),
        ("验收通过:", "stale_test_noise_retained"),
        (f"搜索 {spec.symbol_release} 的实现。", "symbol_search_trace_retained"),
        (f"{spec.code_file}:120:def {spec.symbol_release}(", "search_result_trace_retained"),
        (f"列出 {spec.workspace_dir} 目录。", "listing_trace_retained"),
    ]
    for needle, code in noise_checks:
        if needle in combined_text:
            issues.append(code)
    return issues


def evaluate_scenario(client: TestClient, spec: ScenarioSpec) -> dict[str, Any]:
    history: list[dict[str, str]] = [{"role": "system", "content": "You are a helpful assistant."}]
    savings: list[float] = []
    compressed_turns = 0
    final_combined_text = ""
    final_memory_text = ""
    final_headers: dict[str, str] = {}

    turns = scripted_turns(spec)
    for turn_index, (user_text, followups) in enumerate(turns, start=1):
        request_messages = [*history, {"role": "user", "content": user_text}]
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": request_messages},
            headers={"x-session-id": f"sess_eval_{spec.slug}"},
        )
        response.raise_for_status()
        headers = dict(response.headers)
        echoed = response.json()["echo_messages"]
        if headers.get("x-memory-proxy-compressed") == "true":
            compressed_turns += 1
        savings.append(float(headers.get("x-memory-proxy-estimated-savings-pct") or 0.0))

        if turn_index == len(turns):
            final_combined_text = _combined_forwarded_text(echoed)
            final_memory_text = _extract_memory_text(echoed)
            final_headers = headers

        history.append({"role": "user", "content": user_text})
        history.extend(followups)

    issues = _scenario_issues(spec, final_combined_text, final_memory_text)
    return {
        "slug": spec.slug,
        "domain": spec.domain,
        "user_turns": len(turns),
        "message_count": len(history),
        "compressed_turns": compressed_turns,
        "avg_estimated_savings_pct": round(sum(savings) / len(savings), 2),
        "final_estimated_savings_pct": float(final_headers.get("x-memory-proxy-estimated-savings-pct") or 0.0),
        "final_history_dropped": int(final_headers.get("x-memory-proxy-history-dropped") or 0),
        "issues": issues,
        "final_memory_text": final_memory_text,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    issue_counts: dict[str, int] = {}
    for result in results:
        for issue in result["issues"]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    return {
        "scenario_count": len(results),
        "avg_estimated_savings_pct": round(sum(item["avg_estimated_savings_pct"] for item in results) / len(results), 2),
        "avg_final_savings_pct": round(sum(item["final_estimated_savings_pct"] for item in results) / len(results), 2),
        "max_final_savings_pct": max(item["final_estimated_savings_pct"] for item in results),
        "min_final_savings_pct": min(item["final_estimated_savings_pct"] for item in results),
        "scenarios_with_issues": sum(1 for item in results if item["issues"]),
        "issue_counts": issue_counts,
    }


def main() -> None:
    client = make_client()
    results = [evaluate_scenario(client, spec) for spec in build_scenarios()]
    summary = summarize(results)
    report = {"summary": summary, "results": results}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
