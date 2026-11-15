"""Unit tests for set freezing, commitment hashes, and vendor rotation guard."""

from pathlib import Path

import pytest
from PIL import Image

from pinpoint.config import PinpointConfig, normalize_reasoning_effort
from pinpoint.db import connect_db, init_db, utc_now
from pinpoint.ingest import ingest_photo
from pinpoint.runner import ensure_inspect_model_cost, normalize_openrouter_id
from pinpoint.sets import (
    add_photos_to_set,
    check_vendor_rotation_guard,
    compute_commitment_hash,
    create_image_set,
    freeze_set,
)


def test_freeze_commitment_and_rotation_guard(tmp_path: Path):
    cfg = PinpointConfig(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    conn = connect_db(cfg.db_path)
    init_db(conn)

    img_path = cfg.inbox_dir / "PXL_1.jpg"
    Image.new("RGB", (400, 300), color=(10, 20, 30)).save(img_path, format="JPEG")
    pid = ingest_photo(conn, cfg, img_path, lat=48.8566, lon=2.3522, country_input="FR")

    set_id = create_image_set(conn, name="phase0", version=1)
    assert add_photos_to_set(conn, set_id, [pid]) == 1

    commit_hash = freeze_set(conn, set_id)
    row = conn.execute("SELECT commitment_salt, commitment_hash FROM image_sets WHERE id = ?", (set_id,)).fetchone()
    assert compute_commitment_hash(conn, set_id, row["commitment_salt"]) == commit_hash

    with pytest.raises(RuntimeError):
        add_photos_to_set(conn, set_id, [pid])

    # Add a 2nd model from Google and record a completed run on google-fast
    conn.execute(
        """
        INSERT INTO model_configs (id, vendor, family, tier, openrouter_id)
        VALUES ('google-fast-2', 'google', 'gemini-flash-next', 'fast', 'openrouter/google/gemini-2.5-flash-lite')
        """
    )
    conn.execute(
        """
        INSERT INTO runs (id, set_id, track, model_config_id, prompt_version_id, status, started_at)
        VALUES ('run-1', ?, 'A', 'google-fast', 'pinpoint_v1', 'completed', ?)
        """,
        (set_id, utc_now()),
    )
    conn.commit()

    # Same model slug at different reasoning_effort: allowed for thinking sweeps
    conn.execute(
        """
        INSERT INTO model_configs (id, vendor, family, tier, openrouter_id, reasoning_effort)
        VALUES ('google-fast-high', 'google', 'gemini-flash', 'fast', 'openrouter/google/gemini-4-hydrogen', 'high')
        """
    )
    conn.commit()
    assert check_vendor_rotation_guard(conn, set_id, "google-fast-high")["blocked"] is False

    # Same model rerun: allowed
    assert check_vendor_rotation_guard(conn, set_id, "google-fast")["blocked"] is False
    # Different model, same vendor: blocked
    guard = check_vendor_rotation_guard(conn, set_id, "google-fast-2")
    assert guard["blocked"] is True
    assert guard["vendor"] == "google"
    # Different vendor: allowed
    assert check_vendor_rotation_guard(conn, set_id, "openai-fast")["blocked"] is False
    conn.close()


def test_reasoning_effort_normalization():
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        assert normalize_reasoning_effort(effort) == effort
    assert normalize_reasoning_effort("") is None
    assert normalize_reasoning_effort("default") is None
    assert normalize_reasoning_effort(None) is None
    with pytest.raises(ValueError):
        normalize_reasoning_effort("ultra")


def test_normalize_openrouter_id():
    assert normalize_openrouter_id("anthropic/claude-haiku-5.5") == "openrouter/anthropic/claude-haiku-5.5"
    assert normalize_openrouter_id("openrouter/x-ai/grok-4-fast") == "openrouter/x-ai/grok-4-fast"


def test_ensure_inspect_model_cost_registers_unknown_model():
    from inspect_ai.model import get_model_info

    name = "openrouter/anthropic/claude-haiku-5.5-test-custom"
    cost = ensure_inspect_model_cost(name, "anthropic", '{"est_usd_per_1m_in": 1.0, "est_usd_per_1m_out": 5.0}')
    assert cost.input == 1.0 and cost.output == 5.0
    info = get_model_info(name)
    assert info is not None and info.cost is not None and info.cost.input == 1.0
