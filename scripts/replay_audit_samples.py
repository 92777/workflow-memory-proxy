from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_proxy import MemoryCompressor, RawMessage  # noqa: E402
from memory_proxy.prompt_builder import parse_prompt_memory_text  # noqa: E402


SUSPICIOUS_PREFIXES = (
    "了",
    "的",
    "把",
    "动",
    "看",
    "这样",
    "所以",
    "然后",
    "再",
)
META_PHRASES = (
    "我会",
    "我先",
    "我在",
    "我接着",
    "我补",
    "我去看",
    "我查的是",
    "这样能",
    "这样我",
    "下一条我会",
)
NEGATION_LEADERS = ("不要", "不能", "别", "别去", "不要去", "不要动", "不能动", "不要改", "不能改")
CODE_NOISE_RE = re.compile(r"`[^`]*$|^[^`]*`|/\w|\.py\b|\.go\b|\.md\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay exported audit samples through the current compressor.")
    parser.add_argument(
        "--samples",
        default=str(ROOT / "testdata" / "real_audit_samples" / "latest_session_samples.json"),
        help="Exported sample JSON path.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "testdata" / "real_audit_samples" / "latest_session_replay_report.json"),
        help="Output JSON report path.",
    )
    return parser.parse_args()


def load_samples(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_raw_messages(messages: list[dict[str, object]]) -> list[RawMessage]:
    return [
        RawMessage(
            message_id=str(message["message_id"]),
            role=str(message["role"]),
            content=str(message["content"]),
        )
        for message in messages
    ]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).strip("`'\" ")


def looks_like_fragment(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) <= 3:
        return True
    return stripped.startswith(SUSPICIOUS_PREFIXES)


def looks_like_meta(value: str) -> bool:
    stripped = value.strip()
    return any(phrase in stripped for phrase in META_PHRASES)


def has_code_noise(value: str) -> bool:
    stripped = value.strip()
    if stripped.count("`") % 2 == 1:
        return True
    return bool(CODE_NOISE_RE.search(stripped))


def lost_negation(value: str, raw_text: str) -> bool:
    stripped = value.strip()
    if not stripped or stripped.startswith(NEGATION_LEADERS):
        return False
    return any(prefix + stripped in raw_text for prefix in NEGATION_LEADERS)


def collect_issues(current_prompt_memory: dict[str, object], raw_text: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    goal = str(current_prompt_memory.get("goal") or "")
    if goal and looks_like_fragment(goal):
        issues.append({"kind": "fragmented_goal", "value": goal})

    next_step = str(current_prompt_memory.get("next_step") or "")
    if next_step and looks_like_meta(next_step):
        issues.append({"kind": "meta_next_step", "value": next_step})

    for field in ("constraints", "decisions", "open_tasks", "pending_verification", "open_questions"):
        values = current_prompt_memory.get(field) or []
        if not isinstance(values, list):
            continue
        seen_norm: dict[str, str] = {}
        for value in values:
            if not isinstance(value, str):
                continue
            normalized = normalize_text(value)
            if looks_like_fragment(value):
                issues.append({"kind": f"fragmented_{field[:-1] if field.endswith('s') else field}", "value": value})
            if looks_like_meta(value):
                issues.append({"kind": f"meta_{field[:-1] if field.endswith('s') else field}", "value": value})
            if has_code_noise(value):
                issues.append({"kind": f"format_noise_{field[:-1] if field.endswith('s') else field}", "value": value})
            if lost_negation(value, raw_text):
                issues.append({"kind": f"negation_loss_{field[:-1] if field.endswith('s') else field}", "value": value})
            if normalized in seen_norm and seen_norm[normalized] != value:
                issues.append({"kind": f"near_duplicate_{field[:-1] if field.endswith('s') else field}", "value": value})
            seen_norm[normalized] = value

    return dedupe_issues(issues)


def dedupe_issues(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        key = (item["kind"], item["value"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def replay_item(item: dict[str, object]) -> dict[str, object]:
    raw_history_messages = item.get("raw_history_messages") or []
    if not isinstance(raw_history_messages, list):
        raw_history_messages = []
    raw_messages = to_raw_messages(raw_history_messages)
    compressor = MemoryCompressor(recent_window=0)
    result = compressor.compress(raw_messages, session_id=str(item["session_id"]))
    raw_text = "\n".join(f"{message.role}: {message.content}" for message in raw_messages)
    prompt_memory = parse_prompt_memory_text(result.memory_dsl)
    prompt_memory_dict = asdict(prompt_memory)
    issues = collect_issues(prompt_memory_dict, raw_text)
    return {
        "request_id": item["request_id"],
        "created_at": item["created_at"],
        "raw_history_message_count": len(raw_messages),
        "stored_prompt_memory": item.get("stored_prompt_memory") or "",
        "current_prompt_memory": result.memory_dsl,
        "current_prompt_memory_fields": prompt_memory_dict,
        "issues": issues,
    }


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    issue_counter: Counter[str] = Counter()
    for item in results:
        for issue in item.get("issues", []):
            if isinstance(issue, dict):
                issue_counter[str(issue["kind"])] += 1
    return {
        "sample_count": len(results),
        "samples_with_issues": sum(1 for item in results if item.get("issues")),
        "issue_counts": dict(issue_counter.most_common()),
    }


def main() -> None:
    args = parse_args()
    sample_path = Path(args.samples)
    payload = load_samples(sample_path)
    items = payload.get("items") or []
    if not isinstance(items, list):
        raise SystemExit("Invalid sample file: items must be a list.")

    results = [replay_item(item) for item in items if isinstance(item, dict)]
    report = {
        "samples_path": str(sample_path),
        "session": payload.get("session"),
        "summary": summarize(results),
        "items": results,
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
