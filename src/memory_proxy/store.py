from __future__ import annotations

from contextlib import contextmanager
from importlib import resources
import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from .models import MemoryEvent, RawMessage, WorkingMemory


class SQLiteMemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def init_db(self) -> None:
        schema = _schema_path().read_text(encoding="utf-8")
        with self.session() as connection:
            connection.executescript(schema)
            self._migrate_request_audits(connection)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def session(self):
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def upsert_session(
        self,
        session_id: str,
        client: str = "unknown",
        upstream_model: str | None = None,
        created_at: str | None = None,
    ) -> None:
        timestamp = created_at or _utc_now()
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO sessions (session_id, client, upstream_model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    client=excluded.client,
                    upstream_model=excluded.upstream_model,
                    updated_at=excluded.updated_at
                """,
                (session_id, client, upstream_model, timestamp, timestamp),
            )

    def insert_raw_message(
        self,
        session_id: str,
        turn_id: str,
        message: RawMessage,
        token_count: int | None = None,
        created_at: str | None = None,
    ) -> None:
        timestamp = created_at or _utc_now()
        with self.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO raw_messages
                (message_id, session_id, turn_id, role, content, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.message_id,
                    session_id,
                    turn_id,
                    message.role,
                    message.content,
                    token_count,
                    timestamp,
                ),
            )

    def insert_events(self, events: list[MemoryEvent], created_at: str | None = None) -> None:
        if not events:
            return
        timestamp = created_at or _utc_now()
        rows = [
            (
                event.event_id,
                event.session_id,
                event.turn_id,
                json.dumps(event.source_message_ids, ensure_ascii=False),
                event.actor,
                event.type,
                event.action,
                event.status,
                event.subject,
                json.dumps(event.details, ensure_ascii=False),
                event.confidence,
                event.supersedes,
                timestamp,
            )
            for event in events
        ]
        with self.session() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO memory_events
                (event_id, session_id, turn_id, source_message_ids, actor, type, action, status,
                 subject, details_json, confidence, supersedes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def insert_working_memory_snapshot(
        self,
        session_id: str,
        turn_id: str,
        memory: WorkingMemory,
        memory_dsl: str,
        snapshot_id: str | None = None,
        created_at: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        timestamp = created_at or _utc_now()
        snapshot_id = snapshot_id or f"snap_{uuid4().hex[:12]}"
        payload = {"working_memory": asdict(memory)}
        if metadata:
            payload["metadata"] = metadata
        with self.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO working_memory_snapshots
                (snapshot_id, session_id, turn_id, memory_json, memory_dsl, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    session_id,
                    turn_id,
                    json.dumps(payload, ensure_ascii=False),
                    memory_dsl,
                    timestamp,
                ),
            )
        return snapshot_id

    def insert_request_audit(
        self,
        *,
        request_id: str,
        session_id: str,
        api_kind: str,
        upstream_model: str | None,
        compressed: bool,
        compression_reason: str,
        dropped_message_count: int,
        recent_message_count: int,
        snapshot_id: str | None,
        original_payload: dict[str, object],
        forwarded_payload: dict[str, object],
        prompt_memory: str,
        estimated_input_tokens_before: int | None,
        estimated_input_tokens_after: int | None,
        estimated_savings_pct: float | None,
        token_counter: str | None,
        upstream_usage: dict[str, object] | None,
        upstream_input_tokens: int | None,
        upstream_output_tokens: int | None,
        upstream_total_tokens: int | None,
        client_fingerprint: str | None = None,
        upstream_response_id: str | None = None,
        status_code: int,
        response_preview: str | None,
        created_at: str | None = None,
    ) -> None:
        timestamp = created_at or _utc_now()
        with self.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO request_audits
                (request_id, session_id, api_kind, upstream_model, compressed, compression_reason,
                 dropped_message_count, recent_message_count, snapshot_id, original_payload_json,
                 forwarded_payload_json, prompt_memory, estimated_input_tokens_before,
                 estimated_input_tokens_after, estimated_savings_pct, token_counter,
                 upstream_usage_json, upstream_input_tokens, upstream_output_tokens,
                 upstream_total_tokens, client_fingerprint, upstream_response_id, status_code,
                 response_preview, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    session_id,
                    api_kind,
                    upstream_model,
                    int(compressed),
                    compression_reason,
                    dropped_message_count,
                    recent_message_count,
                    snapshot_id,
                    json.dumps(original_payload, ensure_ascii=False),
                    json.dumps(forwarded_payload, ensure_ascii=False),
                    prompt_memory,
                    estimated_input_tokens_before,
                    estimated_input_tokens_after,
                    estimated_savings_pct,
                    token_counter,
                    json.dumps(upstream_usage, ensure_ascii=False) if upstream_usage is not None else None,
                    upstream_input_tokens,
                    upstream_output_tokens,
                    upstream_total_tokens,
                    client_fingerprint,
                    upstream_response_id,
                    status_code,
                    response_preview,
                    timestamp,
                ),
            )

    def prune_request_history(self, max_requests: int) -> int:
        if max_requests <= 0:
            return 0
        with self.session() as connection:
            stale_rows = connection.execute(
                """
                SELECT request_id, snapshot_id
                FROM request_audits
                ORDER BY created_at DESC
                LIMIT -1 OFFSET ?
                """,
                (max_requests,),
            ).fetchall()
            if not stale_rows:
                return 0

            stale_request_ids = [row["request_id"] for row in stale_rows]
            stale_turn_ids = stale_request_ids + [f"{request_id}:response" for request_id in stale_request_ids]
            request_placeholders = ",".join("?" for _ in stale_request_ids)
            turn_placeholders = ",".join("?" for _ in stale_turn_ids)

            connection.execute(
                f"DELETE FROM raw_messages WHERE turn_id IN ({turn_placeholders})",
                stale_turn_ids,
            )
            connection.execute(
                f"DELETE FROM memory_events WHERE turn_id IN ({turn_placeholders})",
                stale_turn_ids,
            )
            connection.execute(
                f"DELETE FROM working_memory_snapshots WHERE turn_id IN ({turn_placeholders})",
                stale_turn_ids,
            )
            connection.execute(
                f"DELETE FROM request_audits WHERE request_id IN ({request_placeholders})",
                stale_request_ids,
            )

            connection.execute(
                """
                DELETE FROM sessions
                WHERE session_id NOT IN (
                    SELECT DISTINCT session_id
                    FROM request_audits
                )
                """
            )
        return len(stale_rows)

    def list_sessions(self, limit: int = 50) -> list[sqlite3.Row]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT
                    s.session_id,
                    s.client,
                    s.upstream_model,
                    s.created_at,
                    s.updated_at,
                    COALESCE(COUNT(ra.request_id), 0) AS request_count,
                    COALESCE(SUM(CASE WHEN ra.compressed = 1 THEN 1 ELSE 0 END), 0) AS compressed_request_count,
                    MAX(ra.created_at) AS last_request_at
                FROM sessions s
                LEFT JOIN request_audits ra ON ra.session_id = s.session_id
                GROUP BY s.session_id, s.client, s.upstream_model, s.created_at, s.updated_at
                ORDER BY COALESCE(MAX(ra.created_at), s.updated_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return rows

    def list_request_audits(
        self,
        *,
        limit: int = 50,
        session_id: str | None = None,
    ) -> list[sqlite3.Row]:
        query = """
            SELECT
                request_id,
                session_id,
                api_kind,
                upstream_model,
                compressed,
                compression_reason,
                dropped_message_count,
                recent_message_count,
                snapshot_id,
                original_payload_json,
                forwarded_payload_json,
                prompt_memory,
                estimated_input_tokens_before,
                estimated_input_tokens_after,
                estimated_savings_pct,
                token_counter,
                upstream_usage_json,
                upstream_input_tokens,
                upstream_output_tokens,
                upstream_total_tokens,
                client_fingerprint,
                upstream_response_id,
                status_code,
                response_preview,
                created_at
            FROM request_audits
        """
        params: tuple[object, ...]
        if session_id:
            query += " WHERE session_id = ? ORDER BY created_at DESC LIMIT ?"
            params = (session_id, limit)
        else:
            query += " ORDER BY created_at DESC LIMIT ?"
            params = (limit,)
        with self.session() as connection:
            rows = connection.execute(query, params).fetchall()
        return rows

    def get_request_audit(self, request_id: str) -> sqlite3.Row | None:
        with self.session() as connection:
            row = connection.execute(
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
                    snapshot_id,
                    original_payload_json,
                    forwarded_payload_json,
                    prompt_memory,
                    estimated_input_tokens_before,
                    estimated_input_tokens_after,
                    estimated_savings_pct,
                    token_counter,
                    upstream_usage_json,
                    upstream_input_tokens,
                    upstream_output_tokens,
                    upstream_total_tokens,
                    client_fingerprint,
                    upstream_response_id,
                    status_code,
                    response_preview,
                    created_at
                FROM request_audits
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
        return row

    def find_session_id_by_upstream_response_id(self, response_id: str) -> str | None:
        with self.session() as connection:
            row = connection.execute(
                """
                SELECT session_id
                FROM request_audits
                WHERE upstream_response_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (response_id,),
            ).fetchone()
        return row["session_id"] if row is not None else None

    def find_recent_session_by_client_fingerprint(
        self,
        client_fingerprint: str,
        *,
        max_age_seconds: int,
    ) -> str | None:
        since = (datetime.now(UTC) - timedelta(seconds=max_age_seconds)).isoformat()
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT session_id, MAX(created_at) AS last_seen
                FROM request_audits
                WHERE client_fingerprint = ?
                  AND created_at >= ?
                GROUP BY session_id
                ORDER BY last_seen DESC
                LIMIT 2
                """,
                (client_fingerprint, since),
            ).fetchall()
        if len(rows) != 1:
            return None
        return rows[0]["session_id"]

    def get_working_memory_snapshot(self, snapshot_id: str) -> sqlite3.Row | None:
        with self.session() as connection:
            row = connection.execute(
                """
                SELECT snapshot_id, session_id, turn_id, memory_json, memory_dsl, created_at
                FROM working_memory_snapshots
                WHERE snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchone()
        return row

    def get_latest_working_memory_snapshot(self, session_id: str) -> sqlite3.Row | None:
        with self.session() as connection:
            row = connection.execute(
                """
                SELECT snapshot_id, session_id, turn_id, memory_json, memory_dsl, created_at
                FROM working_memory_snapshots
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return row

    def load_snapshot_memory(self, snapshot_id: str) -> WorkingMemory | None:
        row = self.get_working_memory_snapshot(snapshot_id)
        if row is None:
            return None
        payload = json.loads(row["memory_json"])
        working_memory = payload.get("working_memory", {})
        if not isinstance(working_memory, dict):
            return None
        return WorkingMemory(**working_memory)

    def list_turn_raw_messages(self, session_id: str, turn_id: str) -> list[sqlite3.Row]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT message_id, session_id, turn_id, role, content, token_count, created_at
                FROM raw_messages
                WHERE session_id = ? AND turn_id = ?
                ORDER BY message_id
                """,
                (session_id, turn_id),
            ).fetchall()
        return rows

    def list_turn_memory_events(self, session_id: str, turn_id: str) -> list[sqlite3.Row]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT event_id, session_id, turn_id, source_message_ids, actor, type, action,
                       status, subject, details_json, confidence, supersedes, created_at
                FROM memory_events
                WHERE session_id = ? AND turn_id = ?
                ORDER BY event_id
                """,
                (session_id, turn_id),
            ).fetchall()
        return rows

    def get_dashboard_summary(self) -> dict[str, int | float | None]:
        with self.session() as connection:
            sessions = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            requests = connection.execute("SELECT COUNT(*) FROM request_audits").fetchone()[0]
            compressed_requests = connection.execute(
                "SELECT COUNT(*) FROM request_audits WHERE compressed = 1"
            ).fetchone()[0]
            avg_savings = connection.execute(
                """
                SELECT AVG(estimated_savings_pct)
                FROM request_audits
                WHERE estimated_savings_pct IS NOT NULL
                """
            ).fetchone()[0]
        return {
            "sessions": int(sessions),
            "requests": int(requests),
            "compressed_requests": int(compressed_requests),
            "avg_estimated_savings_pct": round(float(avg_savings), 2) if avg_savings is not None else None,
        }

    def list_memory_events(self, session_id: str) -> list[sqlite3.Row]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT event_id, session_id, turn_id, source_message_ids, actor, type, action,
                       status, subject, details_json, confidence, supersedes, created_at
                FROM memory_events
                WHERE session_id = ?
                ORDER BY turn_id, event_id
                """,
                (session_id,),
            ).fetchall()
        return rows

    def list_raw_messages(self, session_id: str) -> list[sqlite3.Row]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT message_id, session_id, turn_id, role, content, token_count, created_at
                FROM raw_messages
                WHERE session_id = ?
                ORDER BY turn_id, message_id
                """,
                (session_id,),
            ).fetchall()
        return rows

    def list_working_memory_snapshots(self, session_id: str) -> list[sqlite3.Row]:
        with self.session() as connection:
            rows = connection.execute(
                """
                SELECT snapshot_id, session_id, turn_id, memory_json, memory_dsl, created_at
                FROM working_memory_snapshots
                WHERE session_id = ?
                ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()
        return rows

    def _migrate_request_audits(self, connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(request_audits)").fetchall()
        }
        if "client_fingerprint" not in columns:
            connection.execute("ALTER TABLE request_audits ADD COLUMN client_fingerprint TEXT")
        if "upstream_response_id" not in columns:
            connection.execute("ALTER TABLE request_audits ADD COLUMN upstream_response_id TEXT")
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_request_audits_client_fingerprint_created
            ON request_audits(client_fingerprint, created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_request_audits_upstream_response_id
            ON request_audits(upstream_response_id)
            """
        )


def _schema_path() -> Path:
    return Path(str(resources.files("memory_proxy").joinpath("resources/schema.sql")))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
