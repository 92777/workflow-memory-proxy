from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compressed request audit history into reusable test samples.")
    parser.add_argument(
        "--db",
        default=str(ROOT / "data" / "memory_proxy.db"),
        help="Path to SQLite audit database.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session id. Defaults to the most active session in the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of compacted requests to export.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "testdata" / "real_audit_samples" / "latest_session_samples.json"),
        help="Output JSON file path.",
    )
    return parser.parse_args()


def connect(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def resolve_session_id(connection: sqlite3.Connection, requested: str | None) -> str:
    if requested:
        return requested
    row = connection.execute(
        """
        SELECT session_id
        FROM request_audits
        GROUP BY session_id
        ORDER BY COUNT(*) DESC, MAX(created_at) DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise SystemExit("No request_audits rows found.")
    return str(row["session_id"])


def list_request_rows(connection: sqlite3.Connection, session_id: str, limit: int) -> list[sqlite3.Row]:
    rows = connection.execute(
        """
        SELECT
            request_id,
            session_id,
            api_kind,
            upstream_model,
            compressed,
            compression_reason,
            dropped_message_count,
            recent_message_count,
            prompt_memory,
            estimated_input_tokens_before,
            estimated_input_tokens_after,
            estimated_savings_pct,
            created_at
        FROM request_audits
        WHERE session_id = ?
          AND compressed = 1
          AND compression_reason = 'history_compacted'
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    return list(rows)


def fetch_raw_messages(connection: sqlite3.Connection, session_id: str, turn_id: str) -> list[dict[str, str]]:
    rows = connection.execute(
        """
        SELECT message_id, role, content, created_at
        FROM raw_messages
        WHERE session_id = ? AND turn_id = ?
        ORDER BY message_id
        """,
        (session_id, turn_id),
    ).fetchall()
    return [
        {
            "message_id": str(row["message_id"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "created_at": str(row["created_at"]),
        }
        for row in rows
    ]


def fetch_session_summary(connection: sqlite3.Connection, session_id: str) -> dict[str, object]:
    row = connection.execute(
        """
        SELECT
            session_id,
            COUNT(*) AS request_count,
            SUM(CASE WHEN compressed = 1 THEN 1 ELSE 0 END) AS compressed_request_count,
            MIN(created_at) AS first_request_at,
            MAX(created_at) AS last_request_at
        FROM request_audits
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return {"session_id": session_id}
    return {
        "session_id": session_id,
        "request_count": int(row["request_count"] or 0),
        "compressed_request_count": int(row["compressed_request_count"] or 0),
        "first_request_at": row["first_request_at"],
        "last_request_at": row["last_request_at"],
    }


def export_samples(connection: sqlite3.Connection, session_id: str, limit: int) -> dict[str, object]:
    request_rows = list_request_rows(connection, session_id, limit)
    items: list[dict[str, object]] = []
    for row in request_rows:
        raw_messages = fetch_raw_messages(connection, session_id, str(row["request_id"]))
        if not raw_messages:
            continue
        items.append(
            {
                "request_id": str(row["request_id"]),
                "session_id": str(row["session_id"]),
                "api_kind": str(row["api_kind"]),
                "upstream_model": row["upstream_model"],
                "compression_reason": str(row["compression_reason"]),
                "dropped_message_count": int(row["dropped_message_count"] or 0),
                "recent_message_count": int(row["recent_message_count"] or 0),
                "estimated_input_tokens_before": row["estimated_input_tokens_before"],
                "estimated_input_tokens_after": row["estimated_input_tokens_after"],
                "estimated_savings_pct": row["estimated_savings_pct"],
                "created_at": str(row["created_at"]),
                "stored_prompt_memory": str(row["prompt_memory"] or ""),
                "raw_history_messages": raw_messages,
            }
        )
    return {
        "exported_at": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "source_db": str(connection.execute("PRAGMA database_list").fetchone()[2]),
        "session": fetch_session_summary(connection, session_id),
        "items": items,
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_roles(items: Iterable[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for message in item.get("raw_history_messages", []):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "unknown")
            counts[role] = counts.get(role, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    with connect(args.db) as connection:
        session_id = resolve_session_id(connection, args.session_id)
        payload = export_samples(connection, session_id, args.limit)

    output_path = Path(args.out)
    ensure_parent(output_path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    role_counts = iter_roles(payload["items"])
    print(json.dumps(
        {
            "session_id": session_id,
            "samples": len(payload["items"]),
            "output": str(output_path),
            "roles": role_counts,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
