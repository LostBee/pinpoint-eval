"""End-to-end offline tests using Inspect mockllm, SQLite importer, and FastAPI TestClient."""

from pathlib import Path

from fastapi.testclient import TestClient
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model
from PIL import Image

from pinpoint.config import PinpointConfig
from pinpoint.db import connect_db, init_db
from pinpoint.importer import import_eval_log
from pinpoint.ingest import ingest_photo
from pinpoint.prompts import PROMPT_V1_ID
from pinpoint.runner import (
    enqueue_run_job,
    execute_single_job,
    export_public_release,
    export_summary_files,
)
from pinpoint.sets import add_photos_to_set, create_image_set, freeze_set
from pinpoint.tasks import build_pinpoint_task
from pinpoint.web.app import create_app


def test_inspect_eval_mockllm_and_rerun_history(tmp_path: Path):
    cfg = PinpointConfig(repo_root=tmp_path / "repo", data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    conn = connect_db(cfg.db_path)
    init_db(conn)

    # Create and ingest 2 synthetic photos
    p1 = cfg.inbox_dir / "PXL_a.jpg"
    p2 = cfg.inbox_dir / "PXL_b.jpg"
    Image.new("RGB", (320, 240), color=(200, 50, 50)).save(p1, format="JPEG")
    Image.new("RGB", (320, 240), color=(50, 200, 50)).save(p2, format="JPEG")

    id1 = ingest_photo(conn, cfg, p1, lat=49.6116, lon=6.1319, country_input="LU", place_label="Lux")
    id2 = ingest_photo(conn, cfg, p2, lat=48.8566, lon=2.3522, country_input="FR", place_label="Paris")

    set_id = create_image_set(conn, name="phase0", version=1)
    add_photos_to_set(conn, set_id, [id1, id2])
    freeze_set(conn, set_id)

    mock = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(
                model="mockllm/model",
                content='{"status":"ok","country":"LU","latitude":49.6116,"longitude":6.1319,"confidence":0.95,"reasoning":"Mock 1"}',
            ),
            ModelOutput.from_content(
                model="mockllm/model",
                content='{"status":"ok","country":"FR","latitude":48.8566,"longitude":2.3522,"confidence":0.90,"reasoning":"Mock 2"}',
            ),
        ],
    )

    task = build_pinpoint_task(set_id=set_id, track="A", data_dir=str(cfg.data_dir))
    logs = inspect_eval(
        task,
        model=mock,
        log_dir=str(cfg.logs_dir),
        max_samples=1,
        max_connections=1,
        display="none",
    )
    assert logs and logs[0].status == "success"

    n1 = import_eval_log(
        conn,
        log_path=logs[0].location,
        run_id="run-full",
        set_id=set_id,
        track="A",
        model_config_id="google-fast",
        prompt_version_id=PROMPT_V1_ID,
    )
    assert n1 == 2

    # Rerun single photo id1; official result must remain from run-full
    mock_rerun = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(
                model="mockllm/model",
                content='{"status":"ok","country":"DE","latitude":49.75,"longitude":6.64,"confidence":0.5,"reasoning":"Mock rerun"}',
            ),
        ],
    )
    task2 = build_pinpoint_task(set_id=set_id, track="A", data_dir=str(cfg.data_dir))
    logs2 = inspect_eval(
        task2,
        model=mock_rerun,
        log_dir=str(cfg.logs_dir),
        sample_id=[id1],
        max_samples=1,
        max_connections=1,
        display="none",
    )
    n2 = import_eval_log(
        conn,
        log_path=logs2[0].location,
        run_id="run-single",
        set_id=set_id,
        track="A",
        model_config_id="google-fast",
        prompt_version_id=PROMPT_V1_ID,
    )
    assert n2 == 1

    r1 = conn.execute(
        "SELECT is_official, tier_points FROM results WHERE run_id = 'run-full' AND photo_id = ?",
        (id1,),
    ).fetchone()
    r2 = conn.execute(
        "SELECT is_official, tier_points FROM results WHERE run_id = 'run-single' AND photo_id = ?",
        (id1,),
    ).fetchone()
    assert r1["is_official"] == 1 and r1["tier_points"] == 15
    assert r2["is_official"] == 0 and r2["tier_points"] == 4

    jpath, cpath = export_summary_files(conn, cfg)
    assert jpath.exists() and cpath.exists()
    assert (cfg.exports_dir / "ranking_chart.svg").exists()
    assert (cfg.exports_dir / "map_guesses.svg").exists()
    assert (cfg.exports_dir / "bullseye_error.svg").exists()

    pub_dir = export_public_release(conn, cfg, set_id=set_id)
    assert (pub_dir / "commitment.json").exists()
    assert (pub_dir / "summary.json").exists()
    assert (pub_dir / "ranking_chart.svg").exists()
    assert (pub_dir / "bullseye_error.svg").exists()
    assert (pub_dir / "README.md").exists()

    # Leak check: Option-1 public artifacts must contain zero ground-truth coords, labels, or salt
    for fname in ("commitment.json", "summary.json", "bullseye_error.svg", "README.md"):
        blob = (pub_dir / fname).read_text(encoding="utf-8")
        assert "49.6116" not in blob
        assert "48.8566" not in blob
        assert "Lux" not in blob
        assert "Paris" not in blob
        assert "commitment_salt" not in blob
    conn.close()


def test_execute_single_job_forwards_reasoning_effort(tmp_path: Path, monkeypatch):
    cfg = PinpointConfig(repo_root=tmp_path / "repo", data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    conn = connect_db(cfg.db_path)
    init_db(conn)

    p1 = cfg.inbox_dir / "PXL_1.jpg"
    Image.new("RGB", (320, 240), color=(20, 40, 60)).save(p1, format="JPEG")
    id1 = ingest_photo(conn, cfg, p1, lat=49.6116, lon=6.1319, country_input="LU")
    set_id = create_image_set(conn, name="europe", version=1)
    add_photos_to_set(conn, set_id, [id1])
    freeze_set(conn, set_id)

    conn.execute(
        """
        INSERT INTO model_configs (id, vendor, family, tier, openrouter_id, reasoning_effort)
        VALUES ('google-fast-high', 'google', 'gemini-flash', 'fast', 'openrouter/google/gemini-4-hydrogen', 'high')
        """
    )
    conn.commit()
    job_id = enqueue_run_job(conn, set_id=set_id, track="A", model_config_id="google-fast-high")
    conn.close()

    captured: dict = {}

    def fake_inspect_eval(task, **kwargs):
        captured.update(kwargs)

        class _Log:
            location = str(tmp_path / "fake.eval")

        return [_Log()]

    monkeypatch.setattr("pinpoint.runner.inspect_eval", fake_inspect_eval)
    monkeypatch.setattr("pinpoint.runner.import_eval_log", lambda **kwargs: 1)

    execute_single_job(cfg, job_id)
    assert captured["reasoning_effort"] == "high"
    assert captured["model"] == "openrouter/google/gemini-4-hydrogen"


def test_web_ui_routes_smoke(tmp_path: Path):
    cfg = PinpointConfig(repo_root=tmp_path / "repo", data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    conn = connect_db(cfg.db_path)
    init_db(conn)
    p = cfg.inbox_dir / "PXL_smoke.jpg"
    Image.new("RGB", (320, 240), color=(30, 60, 90)).save(p, format="JPEG")
    ingest_photo(conn, cfg, p, lat=49.6116, lon=6.1319, country_input="LU")
    conn.close()

    app = create_app(cfg=cfg, start_worker=False)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        pool_resp = client.get("/pool")
        assert pool_resp.status_code == 200
        assert "openstreetmap.org/?mlat=49.6116" in pool_resp.text
        assert "google.com/maps?q=49.6116,6.1319" in pool_resp.text
        client.post("/sets/create", data={"name": "europe", "version": "1"})
        sets_resp = client.get("/sets")
        assert sets_resp.status_code == 200
        assert "select-all-btn" in sets_resp.text
        assert "toggle-all-photos" in sets_resp.text
        assert client.get("/runs").status_code == 200
        # Add and delete a custom model config; confirm seed models stay deleted on re-init
        del_resp = client.post("/models/delete", data={"model_id": "xai-fast"})
        assert del_resp.status_code == 200 and "Deleted" in del_resp.text
        conn2 = connect_db(cfg.db_path)
        init_db(conn2)
        assert conn2.execute("SELECT 1 FROM model_configs WHERE id = 'xai-fast'").fetchone() is None
        conn2.close()
        assert client.get("/results").status_code == 200


def test_openrouter_cost_and_telemetry_extraction():
    from types import SimpleNamespace

    from pinpoint.geo import aggregate_run_metrics
    from pinpoint.importer import _extract_tokens_and_cost

    sample = SimpleNamespace(
        model_usage={
            "openrouter/openai/gpt-6.1-terra-pro": SimpleNamespace(
                input_tokens=3748,
                input_tokens_cache_read=1766,
                output_tokens=476,
                reasoning_tokens=305,
                total_cost=0.002826,
            )
        },
        events=[
            SimpleNamespace(
                event="model",
                call=SimpleNamespace(response={"usage": {"cost": 0.0062163}}),
            )
        ],
    )
    t_in, t_out, t_reas, t_cache, cost = _extract_tokens_and_cost(sample, "{}", 0, "A")
    assert t_in == 5514
    assert t_cache == 1766
    assert t_out == 476
    assert t_reas == 305
    assert cost == 0.006216

    agg = aggregate_run_metrics(
        [
            {
                "parse_status": "ok",
                "tier_points": 15,
                "smooth_score": 5000.0,
                "distance_km": 0.2,
                "hit_continent": 1,
                "hit_country": 1,
                "hit_100km": 1,
                "hit_10km": 1,
                "hit_1km": 1,
                "hit_100m": 0,
                "hit_10m": 0,
                "tokens_in": t_in,
                "tokens_cached": t_cache,
                "tokens_out": t_out,
                "tokens_reasoning": t_reas,
                "latency_ms": 9844,
                "cost_usd": cost,
            }
        ]
    )
    assert agg["total_time_s"] == 9.8
    assert agg["mean_latency_s"] == 9.84
    assert agg["mean_tokens_reasoning"] == 305
    assert agg["mean_tokens_in"] == 5514
    assert agg["total_cost_usd"] == 0.006216
