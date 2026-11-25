"""SQLite database initialization and helpers."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pinpoint.prompts import (
    PROMPT_V1_ID,
    PROMPT_V2_A_ID,
    PROMPT_V2_B_ID,
    build_prompt,
    prompt_hash,
    schema_json,
)

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS photos (
    id TEXT PRIMARY KEY,
    sha256_sanitized TEXT UNIQUE NOT NULL,
    phash TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    gps_accuracy_m REAL,
    country_iso2 TEXT NOT NULL,
    continent TEXT NOT NULL,
    place_label TEXT,
    scene_tag TEXT NOT NULL DEFAULT 'urban',
    near_border INTEGER NOT NULL DEFAULT 0,
    has_text INTEGER NOT NULL DEFAULT 0,
    has_landmark INTEGER NOT NULL DEFAULT 0,
    landmark_visibility TEXT,
    status TEXT NOT NULL DEFAULT 'pool',
    exclude_reason TEXT,
    taken_date TEXT,
    camera_model TEXT,
    vault_filename TEXT,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS image_sets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    description TEXT,
    commitment_hash TEXT,
    commitment_salt TEXT,
    created_at TEXT NOT NULL,
    frozen_at TEXT,
    expires_at TEXT,
    published_at TEXT,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS set_members (
    set_id TEXT NOT NULL REFERENCES image_sets(id) ON DELETE CASCADE,
    photo_id TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    PRIMARY KEY (set_id, photo_id)
);

CREATE TABLE IF NOT EXISTS model_configs (
    id TEXT PRIMARY KEY,
    vendor TEXT NOT NULL,
    family TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'fast',
    openrouter_id TEXT NOT NULL,
    reasoning_effort TEXT DEFAULT 'minimal',
    params_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    template_text TEXT NOT NULL,
    schema_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    set_id TEXT NOT NULL REFERENCES image_sets(id),
    track TEXT NOT NULL,
    model_config_id TEXT NOT NULL REFERENCES model_configs(id),
    prompt_version_id TEXT NOT NULL REFERENCES prompt_versions(id),
    inspect_log_path TEXT,
    epochs INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    git_commit TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    override_reason TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    photo_id TEXT NOT NULL REFERENCES photos(id),
    epoch INTEGER NOT NULL DEFAULT 1,
    attempt INTEGER NOT NULL DEFAULT 1,
    is_official INTEGER NOT NULL DEFAULT 1,
    raw_response TEXT,
    parse_status TEXT NOT NULL,
    pred_lat REAL,
    pred_lon REAL,
    pred_country_iso2 TEXT,
    pred_confidence REAL,
    reasoning_text TEXT,
    tool_calls_count INTEGER NOT NULL DEFAULT 0,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    tokens_reasoning INTEGER NOT NULL DEFAULT 0,
    tokens_cached INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    distance_km REAL,
    hit_continent INTEGER NOT NULL DEFAULT 0,
    hit_country INTEGER NOT NULL DEFAULT 0,
    hit_100km INTEGER NOT NULL DEFAULT 0,
    hit_10km INTEGER NOT NULL DEFAULT 0,
    hit_1km INTEGER NOT NULL DEFAULT 0,
    hit_100m INTEGER NOT NULL DEFAULT 0,
    hit_10m INTEGER NOT NULL DEFAULT 0,
    tier_points INTEGER NOT NULL DEFAULT 0,
    smooth_score REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    progress_text TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Starter fast-tier models via OpenRouter (verify/edit model IDs or reasoning_effort in UI).
SEED_MODELS: tuple[dict[str, Any], ...] = (
    {
        "id": "anthropic-fast",
        "vendor": "anthropic",
        "family": "claude-haiku",
        "tier": "fast",
        "openrouter_id": "openrouter/anthropic/claude-haiku-5.5",
        "reasoning_effort": "minimal",
        "params_json": json.dumps({"est_usd_per_1m_in": 1.0, "est_usd_per_1m_out": 5.0}),
    },
    {
        "id": "openai-fast",
        "vendor": "openai",
        "family": "gpt-mini",
        "tier": "fast",
        "openrouter_id": "openrouter/openai/gpt-6-luna",
        "reasoning_effort": "minimal",
        "params_json": json.dumps({"est_usd_per_1m_in": 0.4, "est_usd_per_1m_out": 1.6}),
    },
    {
        "id": "google-fast",
        "vendor": "google",
        "family": "gemini-flash",
        "tier": "fast",
        "openrouter_id": "openrouter/google/gemini-4-hydrogen",
        "reasoning_effort": "minimal",
        "params_json": json.dumps({"est_usd_per_1m_in": 0.3, "est_usd_per_1m_out": 2.5}),
    },
    {
        "id": "xai-fast",
        "vendor": "xai",
        "family": "grok-fast",
        "tier": "fast",
        "openrouter_id": "openrouter/x-ai/grok-4-fast",
        "reasoning_effort": "minimal",
        "params_json": json.dumps({"est_usd_per_1m_in": 0.2, "est_usd_per_1m_out": 0.5}),
    },
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    job_cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "progress_text" not in job_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN progress_text TEXT")

    res_cols = {r["name"] for r in conn.execute("PRAGMA table_info(results)").fetchall()}
    if "tokens_reasoning" not in res_cols:
        conn.execute("ALTER TABLE results ADD COLUMN tokens_reasoning INTEGER NOT NULL DEFAULT 0")
    if "tokens_cached" not in res_cols:
        conn.execute("ALTER TABLE results ADD COLUMN tokens_cached INTEGER NOT NULL DEFAULT 0")
    if "stop_reason" not in res_cols:
        conn.execute("ALTER TABLE results ADD COLUMN stop_reason TEXT")
    if "upstream_provider" not in res_cols:
        conn.execute("ALTER TABLE results ADD COLUMN upstream_provider TEXT")

    photo_cols = {r["name"] for r in conn.execute("PRAGMA table_info(photos)").fetchall()}
    if "coord_basis" not in photo_cols:
        conn.execute("ALTER TABLE photos ADD COLUMN coord_basis TEXT NOT NULL DEFAULT 'camera'")
    if "landmark_visibility" not in photo_cols:
        conn.execute("ALTER TABLE photos ADD COLUMN landmark_visibility TEXT")

    run_cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
    if "scorer_version" not in run_cols:
        conn.execute("ALTER TABLE runs ADD COLUMN scorer_version TEXT NOT NULL DEFAULT 'v1'")
    if "harness_manifest_json" not in run_cols:
        conn.execute("ALTER TABLE runs ADD COLUMN harness_manifest_json TEXT")

    prompt_a = build_prompt("A")
    prompt_b = build_prompt("B")
    now = utc_now()
    conn.execute(
        """
        INSERT OR IGNORE INTO prompt_versions (id, sha256, template_text, schema_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (PROMPT_V1_ID, prompt_hash(prompt_a), prompt_a, schema_json(), now),
    )
    for pid, ptext in ((PROMPT_V2_A_ID, prompt_a), (PROMPT_V2_B_ID, prompt_b)):
        conn.execute(
            """
            INSERT OR REPLACE INTO prompt_versions (id, sha256, template_text, schema_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (pid, prompt_hash(ptext), ptext, schema_json(), now),
        )

    seeded = conn.execute("SELECT value FROM app_meta WHERE key = 'seeded_models'").fetchone()
    existing_models = conn.execute("SELECT COUNT(*) AS c FROM model_configs").fetchone()["c"]
    if seeded is None and existing_models == 0:
        for m in SEED_MODELS:
            conn.execute(
                """
                INSERT OR IGNORE INTO model_configs
                    (id, vendor, family, tier, openrouter_id, reasoning_effort, params_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    m["id"],
                    m["vendor"],
                    m["family"],
                    m["tier"],
                    m["openrouter_id"],
                    m["reasoning_effort"],
                    m["params_json"],
                ),
            )
    conn.execute("INSERT OR IGNORE INTO app_meta (key, value) VALUES ('seeded_models', '1')")
    conn.commit()
