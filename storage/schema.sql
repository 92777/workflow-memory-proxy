CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    client TEXT NOT NULL DEFAULT 'unknown',
    upstream_model TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_messages_session_turn
    ON raw_messages(session_id, turn_id);

CREATE TABLE IF NOT EXISTS memory_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    source_message_ids TEXT NOT NULL,
    actor TEXT NOT NULL,
    type TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    subject TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0.0,
    supersedes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_events_session_turn
    ON memory_events(session_id, turn_id);

CREATE INDEX IF NOT EXISTS idx_memory_events_session_type_status
    ON memory_events(session_id, type, status);

CREATE TABLE IF NOT EXISTS working_memory_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    memory_json TEXT NOT NULL,
    memory_dsl TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_working_memory_snapshots_session_turn
    ON working_memory_snapshots(session_id, turn_id);

CREATE TABLE IF NOT EXISTS request_audits (
    request_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    api_kind TEXT NOT NULL,
    upstream_model TEXT,
    compressed INTEGER NOT NULL DEFAULT 0,
    compression_reason TEXT NOT NULL,
    dropped_message_count INTEGER NOT NULL DEFAULT 0,
    recent_message_count INTEGER NOT NULL DEFAULT 0,
    snapshot_id TEXT,
    original_payload_json TEXT NOT NULL,
    forwarded_payload_json TEXT NOT NULL,
    prompt_memory TEXT NOT NULL DEFAULT '',
    estimated_input_tokens_before INTEGER,
    estimated_input_tokens_after INTEGER,
    estimated_savings_pct REAL,
    token_counter TEXT,
    upstream_usage_json TEXT,
    upstream_input_tokens INTEGER,
    upstream_output_tokens INTEGER,
    upstream_total_tokens INTEGER,
    client_fingerprint TEXT,
    upstream_response_id TEXT,
    status_code INTEGER,
    response_preview TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (snapshot_id) REFERENCES working_memory_snapshots(snapshot_id)
);

CREATE INDEX IF NOT EXISTS idx_request_audits_session_created
    ON request_audits(session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_request_audits_created
    ON request_audits(created_at DESC);
