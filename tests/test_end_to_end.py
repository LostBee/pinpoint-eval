"""End-to-end offline tests using Inspect mockllm, SQLite importer, and FastAPI TestClient."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import use_tools
from PIL import Image

from pinpoint.config import PinpointConfig
from pinpoint.db import connect_db, init_db
from pinpoint.importer import (
    collect_incomplete_official_runs,
    collect_official_leaderboard,
    import_eval_log,
)
from pinpoint.ingest import ingest_photo
from pinpoint.prompts import PROMPT_V1_ID, prompt_version_id_for_track
from pinpoint.runner import (
    enqueue_run_job,
    execute_single_job,
    export_audit_bundle,
    export_public_release,
    export_summary_files,
    verify_audit_commitment,
)
from pinpoint.sets import add_photos_to_set, create_image_set, freeze_set
from pinpoint.tasks import (
    build_pinpoint_task,
    capped_web_search,
    pinpoint_json_solver,
    pinpoint_scorer,
)
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
    summary_obj = json.loads((pub_dir / "summary.json").read_text(encoding="utf-8"))
    assert "harness_manifest" in summary_obj and "naive_baseline" in summary_obj
    assert len(summary_obj["official_runs"]) == 1
    assert summary_obj["official_runs"][0]["strict_ci95"] is not None

    # Confidential NDA audit export + commitment verification round-trip
    audit_dir = export_audit_bundle(conn, cfg, set_id=set_id)
    verified = verify_audit_commitment(audit_dir)
    assert verified["verified"] is True and verified["photo_count"] == 2

    # Tampering with an exported image fails verification
    (audit_dir / "images" / f"{id1}.jpg").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_audit_commitment(audit_dir)
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
            status = "success"

        return [_Log()]

    monkeypatch.setattr("pinpoint.runner.inspect_eval", fake_inspect_eval)
    monkeypatch.setattr("pinpoint.runner.import_eval_log", lambda **kwargs: 1)

    execute_single_job(cfg, job_id)
    assert captured["reasoning_effort"] == "high"
    assert captured["model"] == "openrouter/google/gemini-4-hydrogen"
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 16384
    assert captured["timeout"] == 180
    assert "cost_limit" not in captured


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


def test_resume_skips_completed_photos_and_no_autorefresh(tmp_path: Path, monkeypatch):
    from pinpoint.importer import collect_official_leaderboard

    cfg = PinpointConfig(repo_root=tmp_path / "repo", data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    conn = connect_db(cfg.db_path)
    init_db(conn)

    p1 = cfg.inbox_dir / "PXL_r1.jpg"
    p2 = cfg.inbox_dir / "PXL_r2.jpg"
    Image.new("RGB", (320, 240), color=(10, 20, 30)).save(p1, format="JPEG")
    Image.new("RGB", (320, 240), color=(40, 50, 60)).save(p2, format="JPEG")
    id1 = ingest_photo(conn, cfg, p1, lat=49.6116, lon=6.1319, country_input="LU")
    id2 = ingest_photo(conn, cfg, p2, lat=48.8566, lon=2.3522, country_input="FR")

    set_id = create_image_set(conn, name="europe", version=1)
    add_photos_to_set(conn, set_id, [id1, id2])
    freeze_set(conn, set_id)

    # Simulate partial interrupted run that completed only id1
    mock_partial = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(
                model="mockllm/model",
                content='{"status":"ok","country":"LU","latitude":49.6116,"longitude":6.1319,"confidence":0.9,"reasoning":"Done 1"}',
            )
        ],
    )
    task = build_pinpoint_task(set_id=set_id, track="A", data_dir=str(cfg.data_dir))
    logs1 = inspect_eval(
        task,
        model=mock_partial,
        log_dir=str(cfg.logs_dir),
        sample_id=[id1],
        max_samples=1,
        max_connections=1,
        display="none",
    )
    import_eval_log(
        conn,
        log_path=logs1[0].location,
        run_id="run-part-1",
        set_id=set_id,
        track="A",
        model_config_id="google-fast",
        prompt_version_id=PROMPT_V1_ID,
        run_status_override="incomplete",
    )

    job_id = enqueue_run_job(conn, set_id=set_id, track="A", model_config_id="google-fast")
    conn.commit()

    # Ensure /runs has no meta http-equiv="refresh" even when a pending job exists
    app = create_app(cfg=cfg, start_worker=False)
    with TestClient(app) as client:
        runs_html = client.get("/runs").text
        assert 'http-equiv="refresh"' not in runs_html

    # Resuming executes ONLY remaining sample [id2]
    captured: dict = {}

    def resume_inspect_eval(task_obj, **kwargs):
        captured["sample_id"] = kwargs.get("sample_id")
        mock_rest = get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput.from_content(
                    model="mockllm/model",
                    content='{"status":"ok","country":"FR","latitude":48.8566,"longitude":2.3522,"confidence":0.9,"reasoning":"Done 2"}',
                )
            ],
        )
        return inspect_eval(
            task_obj,
            model=mock_rest,
            log_dir=str(cfg.logs_dir),
            sample_id=kwargs.get("sample_id"),
            max_samples=1,
            max_connections=1,
            display="none",
        )

    monkeypatch.setattr("pinpoint.runner.inspect_eval", resume_inspect_eval)
    execute_single_job(cfg, job_id)
    assert captured["sample_id"] == [id2]

    # Leaderboard merges both partial chunks into a single 2-photo official entry
    # First, verify partial run alone was excluded from official leaderboard until id2 finished
    off_a, _ = collect_official_leaderboard(conn, set_id=set_id)
    assert len(off_a) == 1
    assert off_a[0]["agg"]["count"] == 2
    assert off_a[0]["agg"]["strict_mean_points"] == 15.0
    conn.close()


def test_capped_web_search_and_api_error_recovery_and_incomplete_isolation(tmp_path: Path):
    cfg = PinpointConfig(repo_root=tmp_path / "repo", data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    conn = connect_db(cfg.db_path)
    init_db(conn)

    p1 = cfg.inbox_dir / "PXL_t1.jpg"
    p2 = cfg.inbox_dir / "PXL_t2.jpg"
    Image.new("RGB", (320, 240), color=(15, 25, 35)).save(p1, format="JPEG")
    Image.new("RGB", (320, 240), color=(45, 55, 65)).save(p2, format="JPEG")
    id1 = ingest_photo(conn, cfg, p1, lat=49.6116, lon=6.1319, country_input="LU")
    id2 = ingest_photo(conn, cfg, p2, lat=48.8566, lon=2.3522, country_input="FR")
    set_id = create_image_set(conn, name="europe", version=1)
    add_photos_to_set(conn, set_id, [id1, id2])
    freeze_set(conn, set_id)

    # 1. Verify capped_web_search stops executing inner_search after max_calls=2
    inner_calls: list[str] = []

    def fake_search(q: str) -> str:
        inner_calls.append(q)
        return f"result for {q}"

    mock_b = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call("mockllm/model", "web_search", {"query": "q1"}),
            ModelOutput.for_tool_call("mockllm/model", "web_search", {"query": "q2"}),
            ModelOutput.for_tool_call("mockllm/model", "web_search", {"query": "q3"}),
            ModelOutput.from_content(
                model="mockllm/model",
                content='{"status":"ok","country":"LU","latitude":49.6116,"longitude":6.1319,"confidence":0.9,"reasoning":"Found"}',
            ),
        ],
    )
    ds = MemoryDataset(
        [
            Sample(
                id=id1,
                input="Locate",
                target=json.dumps({"lat": 49.6116, "lon": 6.1319, "country_iso2": "LU", "continent": "EU"}),
            )
        ]
    )
    t_b = Task(
        dataset=ds,
        solver=[use_tools(capped_web_search(max_calls=2, inner_search=fake_search)), pinpoint_json_solver()],
        scorer=pinpoint_scorer(data_dir=str(cfg.data_dir)),
        message_limit=12,
    )
    logs_b = inspect_eval(t_b, model=mock_b, log_dir=str(cfg.logs_dir), display="none")
    assert logs_b and logs_b[0].status == "success"
    assert inner_calls == ["q1", "q2"]

    # 2. Simulate an api_error row on id1, verify it stays non-official and incomplete is isolated
    conn.execute(
        """
        INSERT INTO runs (id, set_id, track, model_config_id, prompt_version_id, status, started_at)
        VALUES ('run-err', ?, 'A', 'google-fast', ?, 'failed', '2026-11-24T00:00:00Z')
        """,
        (set_id, prompt_version_id_for_track("A")),
    )
    conn.execute(
        """
        INSERT INTO results (id, run_id, photo_id, epoch, attempt, is_official, parse_status, tier_points, smooth_score)
        VALUES ('res-err', 'run-err', ?, 1, 1, 1, 'api_error', 0, 0.0)
        """,
        (id1,),
    )
    conn.commit()

    # Import valid id1 only -> incomplete (1/2), must supersede legacy api_error and stay out of complete leaderboard
    import_eval_log(
        conn,
        log_path=logs_b[0].location,
        run_id="run-valid-1",
        set_id=set_id,
        track="A",
        model_config_id="google-fast",
        prompt_version_id=prompt_version_id_for_track("A"),
    )
    err_row = conn.execute("SELECT is_official FROM results WHERE id = 'res-err'").fetchone()
    assert err_row["is_official"] == 0
    off_a, _ = collect_official_leaderboard(conn, set_id=set_id)
    assert off_a == []
    inc = collect_incomplete_official_runs(conn, set_id=set_id)
    assert len(inc) == 1 and inc[0]["agg"]["unique_photos"] == 1

    # Complete id2 -> now 2/2 complete and appears on Official leaderboard with score 15.0
    mock_2 = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(
                model="mockllm/model",
                content='{"status":"ok","country":"FR","latitude":48.8566,"longitude":2.3522,"confidence":0.9,"reasoning":"Done 2"}',
            )
        ],
    )
    task_a = build_pinpoint_task(set_id=set_id, track="A", data_dir=str(cfg.data_dir))
    logs_2 = inspect_eval(task_a, model=mock_2, log_dir=str(cfg.logs_dir), sample_id=[id2], display="none")
    import_eval_log(
        conn,
        log_path=logs_2[0].location,
        run_id="run-valid-2",
        set_id=set_id,
        track="A",
        model_config_id="google-fast",
        prompt_version_id=prompt_version_id_for_track("A"),
    )
    off_a2, _ = collect_official_leaderboard(conn, set_id=set_id)
    assert len(off_a2) == 1 and off_a2[0]["agg"]["strict_mean_points"] == 15.0
    assert collect_incomplete_official_runs(conn, set_id=set_id) == []

    # Multi-epoch aggregation check (epochs=2 on openai-fast)
    mock_ep = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(
                model="mockllm/model",
                content='{"status":"ok","country":"LU","latitude":49.6116,"longitude":6.1319,"confidence":0.9,"reasoning":"ep"}',
            ),
            ModelOutput.from_content(
                model="mockllm/model",
                content='{"status":"ok","country":"FR","latitude":48.8566,"longitude":2.3522,"confidence":0.9,"reasoning":"ep"}',
            ),
        ]
        * 2,
    )
    logs_ep = inspect_eval(task_a, model=mock_ep, log_dir=str(cfg.logs_dir), epochs=2, display="none")
    import_eval_log(
        conn,
        log_path=logs_ep[0].location,
        run_id="run-ep2",
        set_id=set_id,
        track="A",
        model_config_id="openai-fast",
        prompt_version_id=prompt_version_id_for_track("A"),
    )
    off_a3, _ = collect_official_leaderboard(conn, set_id=set_id)
    ep_entry = next(x for x in off_a3 if x["run"]["model_config_id"] == "openai-fast")
    assert ep_entry["agg"]["epochs_count"] == 2
    assert ep_entry["agg"]["count"] == 4
    conn.close()


def test_frozen_photo_metadata_tags_editable_without_changing_commitment(tmp_path: Path):
    from pinpoint.sets import compute_commitment_hash

    cfg = PinpointConfig(repo_root=tmp_path / "repo", data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    conn = connect_db(cfg.db_path)
    init_db(conn)
    p = cfg.inbox_dir / "PXL_tag.jpg"
    Image.new("RGB", (320, 240), color=(30, 60, 90)).save(p, format="JPEG")
    pid = ingest_photo(conn, cfg, p, lat=49.6116, lon=6.1319, country_input="LU", scene_tag="urban")
    set_id = create_image_set(conn, name="europe", version=1)
    add_photos_to_set(conn, set_id, [pid])
    orig_hash = freeze_set(conn, set_id)
    salt = conn.execute("SELECT commitment_salt FROM image_sets WHERE id = ?", (set_id,)).fetchone()["commitment_salt"]
    conn.close()

    app = create_app(cfg=cfg, start_worker=False)
    with TestClient(app) as client:
        assert '<option value="landmark">' not in client.get("/").text
        pool_html = client.get("/pool").text
        assert '<option value="landmark">' not in pool_html

        resp = client.post(
            "/pool/update",
            data={
                "photo_id": pid,
                "lat": "10.0",
                "lon": "10.0",
                "country": "DE",
                "place_label": "Sintra wall",
                "scene_tag": "rural",
                "has_landmark": "on",
            },
        )
        assert resp.status_code == 200

    conn2 = connect_db(cfg.db_path)
    row = conn2.execute("SELECT * FROM photos WHERE id = ?", (pid,)).fetchone()
    # Ground truth stayed locked, metadata tags updated, commitment hash unchanged
    assert row["lat"] == 49.6116 and row["lon"] == 6.1319 and row["country_iso2"] == "LU"
    assert row["scene_tag"] == "rural" and row["has_landmark"] == 1 and row["place_label"] == "Sintra wall"
    assert compute_commitment_hash(conn2, set_id, salt) == orig_hash
    conn2.close()
