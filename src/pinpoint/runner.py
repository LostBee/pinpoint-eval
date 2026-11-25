"""Cost estimation, sequential job queue worker, Inspect execution, and exports."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import inspect_ai
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelCost, ModelInfo, get_model_info, set_model_info

from pinpoint.charts import (
    build_bullseye_error_svg,
    build_guess_map_svg,
    build_ranking_bars_svg,
)
from pinpoint.config import (
    SCORER_VERSION,
    PinpointConfig,
    format_model_label,
    get_config,
    normalize_reasoning_effort,
)
from pinpoint.db import connect_db, utc_now
from pinpoint.geo import aggregate_run_metrics, compute_naive_baseline
from pinpoint.importer import (
    collect_official_leaderboard,
    import_eval_log,
    recover_partial_eval_logs,
)
from pinpoint.prompts import build_prompt, prompt_hash, prompt_version_id_for_track
from pinpoint.sets import check_vendor_rotation_guard, freeze_set
from pinpoint.tasks import build_pinpoint_task

logger = logging.getLogger(__name__)


def normalize_openrouter_id(model_id: str) -> str:
    """Ensure OpenRouter slugs carry the `openrouter/` provider prefix for Inspect AI."""
    cleaned = model_id.strip()
    if cleaned and not cleaned.startswith("openrouter/"):
        return f"openrouter/{cleaned}"
    return cleaned


def get_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def estimate_run_cost(
    conn: sqlite3.Connection,
    set_id: str,
    model_config_id: str,
    track: str,
    sample_count_override: int | None = None,
) -> dict[str, Any]:
    """Conservative pre-run cost estimate based on ~1,800 image+prompt input tokens per photo."""
    model = conn.execute(
        "SELECT openrouter_id, params_json FROM model_configs WHERE id = ?",
        (model_config_id,),
    ).fetchone()
    if not model:
        raise ValueError(f"Unknown model_config_id: {model_config_id}")

    n_photos = sample_count_override
    if n_photos is None:
        n_photos = int(
            conn.execute(
                "SELECT COUNT(*) AS c FROM set_members WHERE set_id = ?", (set_id,)
            ).fetchone()["c"]
        )

    params = json.loads(model["params_json"] or "{}")
    p_in = float(params.get("est_usd_per_1m_in", 0.5))
    p_out = float(params.get("est_usd_per_1m_out", 2.0))

    if track.upper() == "A":
        avg_in, avg_out, avg_searches = 1800, 220, 0
    else:
        avg_in, avg_out, avg_searches = 4500, 450, 3

    model_cost = n_photos * ((avg_in / 1_000_000) * p_in + (avg_out / 1_000_000) * p_out)
    search_cost = n_photos * avg_searches * 0.008
    return {
        "n_photos": n_photos,
        "estimated_model_usd": round(model_cost, 4),
        "estimated_search_usd": round(search_cost, 4),
        "estimated_total_usd": round(model_cost + search_cost, 4),
    }


def build_harness_manifest(
    cfg: PinpointConfig,
    track: str | None = None,
    prompt_version_id: str | None = None,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
    epochs: int | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Build a complete, leak-free harness and API configuration manifest."""
    prompt_a = build_prompt("A")
    prompt_b = build_prompt("B")
    manifest: dict[str, Any] = {
        "scorer_version": SCORER_VERSION,
        "inspect_ai_version": getattr(inspect_ai, "__version__", "unknown"),
        "git_commit": git_commit or get_git_commit(cfg.repo_root),
        "temperature": cfg.default_temperature,
        "max_tokens": cfg.default_max_tokens,
        "timeout_s": cfg.default_timeout_s,
        "retry_on_error": 2,
        "cost_limit_usd": cfg.max_run_cost_usd,
        "image_preprocessing": {
            "format": "JPEG",
            "color_space": "sRGB",
            "max_long_edge_px": cfg.max_long_edge,
            "jpeg_quality": cfg.jpeg_quality,
            "metadata_stripped": True,
        },
        "prompts": {
            "A": {
                "id": prompt_version_id_for_track("A"),
                "sha256": prompt_hash(prompt_a),
                "text": prompt_a,
            },
            "B": {
                "id": prompt_version_id_for_track("B"),
                "sha256": prompt_hash(prompt_b),
                "text": prompt_b,
                "max_search_calls": cfg.max_search_calls,
                "tavily_max_results": cfg.tavily_max_results,
                "message_limit": 24,
            },
        },
    }
    if track is not None:
        manifest["track"] = track.upper()
        manifest["message_limit"] = 4 if track.upper() == "A" else 24
    if prompt_version_id is not None:
        manifest["prompt_version_id"] = prompt_version_id
    if model_name is not None:
        manifest["model"] = model_name
    if reasoning_effort is not None:
        manifest["reasoning_effort"] = reasoning_effort
    if epochs is not None:
        manifest["epochs"] = epochs
    return manifest


def enqueue_run_job(
    conn: sqlite3.Connection,
    set_id: str,
    track: str,
    model_config_id: str,
    photo_id: str | None = None,
    override_reason: str | None = None,
    epochs: int = 1,
) -> str:
    """Validate rotation guard and push a sequential job to SQLite queue."""
    guard = check_vendor_rotation_guard(conn, set_id, model_config_id)
    if guard["blocked"] and not (override_reason and override_reason.strip()):
        raise RuntimeError(str(guard["message"]))

    freeze_set(conn, set_id)
    job_id = str(uuid.uuid4())
    now = utc_now()
    payload = {
        "set_id": set_id,
        "track": track.upper(),
        "model_config_id": model_config_id,
        "photo_id": photo_id,
        "override_reason": override_reason.strip() if override_reason else None,
        "epochs": max(1, int(epochs)),
    }
    conn.execute(
        """
        INSERT INTO jobs (id, job_type, payload_json, state, created_at, updated_at)
        VALUES (?, ?, ?, 'pending', ?, ?)
        """,
        (
            job_id,
            "run_photo" if photo_id else "run_set",
            json.dumps(payload),
            now,
            now,
        ),
    )
    conn.commit()
    return job_id


def ensure_inspect_model_cost(
    model_name: str,
    vendor: str,
    params_json: str,
) -> ModelCost:
    """Register ModelInfo + ModelCost in Inspect AI if the model isn't in its static table."""
    params = json.loads(params_json or "{}")
    model_cost = ModelCost(
        input=float(params.get("est_usd_per_1m_in", 0.5)),
        output=float(params.get("est_usd_per_1m_out", 2.0)),
        input_cache_write=0.0,
        input_cache_read=0.0,
    )
    if get_model_info(model_name) is None:
        set_model_info(
            model_name,
            ModelInfo(
                organization=vendor,
                model=model_name,
                context_length=128000,
                output_tokens=32768,
                cost=model_cost,
            ),
        )
    return model_cost


def _official_completed_photo_ids(
    conn: sqlite3.Connection,
    set_id: str,
    track: str,
    model_config_id: str,
) -> set[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT res.photo_id
        FROM results res
        JOIN runs r ON r.id = res.run_id
        WHERE r.set_id = ?
          AND r.track = ?
          AND r.model_config_id = ?
          AND res.is_official = 1
          AND res.parse_status != 'api_error'
        """,
        (set_id, track.upper(), model_config_id),
    ).fetchall()
    return {str(r["photo_id"]) for r in rows}


def _recover_job_progress(
    conn: sqlite3.Connection,
    cfg: PinpointConfig,
    payload: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Import any partial .eval logs for this job and return (all_set_photo_ids, remaining_ids)."""
    set_id = str(payload["set_id"])
    track = str(payload["track"]).upper()
    model_config_id = str(payload["model_config_id"])
    model = conn.execute(
        "SELECT openrouter_id, reasoning_effort FROM model_configs WHERE id = ?",
        (model_config_id,),
    ).fetchone()
    if model:
        recover_partial_eval_logs(
            conn=conn,
            logs_dir=cfg.logs_dir,
            set_id=set_id,
            track=track,
            model_config_id=model_config_id,
            openrouter_id=model["openrouter_id"],
            reasoning_effort=model["reasoning_effort"],
            prompt_version_id=prompt_version_id_for_track(track),
            git_commit=get_git_commit(cfg.repo_root),
        )

    all_ids = [
        str(r["photo_id"])
        for r in conn.execute(
            "SELECT photo_id FROM set_members WHERE set_id = ? ORDER BY position ASC, photo_id ASC",
            (set_id,),
        ).fetchall()
    ]
    done_ids = _official_completed_photo_ids(conn, set_id, track, model_config_id)
    remaining_ids = [pid for pid in all_ids if pid not in done_ids]
    return all_ids, remaining_ids


def execute_single_job(cfg: PinpointConfig, job_id: str) -> str:
    """Run one queued job via Inspect AI (strictly 1 photo at a time) and import results.

    Automatically resumes interrupted set runs from the remaining unevaluated photos
    without re-running already-completed samples.
    """
    conn = connect_db(cfg.db_path)
    try:
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        payload = json.loads(job["payload_json"])
        model = conn.execute(
            "SELECT vendor, openrouter_id, reasoning_effort, params_json FROM model_configs WHERE id = ?",
            (payload["model_config_id"],),
        ).fetchone()
        if not model:
            raise ValueError(f"Unknown model_config_id: {payload['model_config_id']}")

        run_id = str(uuid.uuid4())
        task = build_pinpoint_task(
            set_id=payload["set_id"],
            track=payload["track"],
            data_dir=str(cfg.data_dir),
        )
        sample_id: list[str] | None
        if payload.get("photo_id"):
            sample_id = [payload["photo_id"]]
        else:
            all_ids, remaining_ids = _recover_job_progress(conn, cfg, payload)
            done_count = len(all_ids) - len(remaining_ids)
            should_resume = bool(payload.get("resume")) or (0 < done_count < len(all_ids))
            if should_resume:
                if not remaining_ids:
                    return "resumed-complete"
                sample_id = remaining_ids
                os.environ["PINPOINT_JOB_DONE_OFFSET"] = str(done_count)
                conn.execute(
                    "UPDATE jobs SET progress_text = ?, updated_at = ? WHERE id = ?",
                    (f"{done_count} / {len(all_ids)} photos", utc_now(), job_id),
                )
                conn.commit()
            else:
                sample_id = None

        model_name = normalize_openrouter_id(model["openrouter_id"])
        effort = normalize_reasoning_effort(model["reasoning_effort"])
        epochs = max(1, int(payload.get("epochs") or cfg.default_epochs))
        prompt_ver_id = prompt_version_id_for_track(str(payload["track"]))
        git_rev = get_git_commit(cfg.repo_root)
        manifest = build_harness_manifest(
            cfg=cfg,
            track=str(payload["track"]),
            prompt_version_id=prompt_ver_id,
            model_name=model_name,
            reasoning_effort=effort,
            epochs=epochs,
            git_commit=git_rev,
        )
        model_cost = ensure_inspect_model_cost(
            model_name=model_name,
            vendor=model["vendor"],
            params_json=model["params_json"],
        )

        eval_kwargs: dict[str, Any] = {
            "model": model_name,
            "log_dir": str(cfg.logs_dir),
            "sample_id": sample_id,
            "epochs": epochs,
            "max_samples": 1,
            "max_connections": 1,
            "retry_on_error": 2,
            "temperature": cfg.default_temperature,
            "max_tokens": cfg.default_max_tokens,
            "timeout": cfg.default_timeout_s,
            "model_cost_config": {model_name: model_cost},
            "display": "none",
        }
        if cfg.max_run_cost_usd is not None:
            eval_kwargs["cost_limit"] = cfg.max_run_cost_usd
        if effort is not None:
            eval_kwargs["reasoning_effort"] = effort

        logs = inspect_eval(task, **eval_kwargs)
        if not logs or not logs[0].location:
            if not payload.get("photo_id"):
                _recover_job_progress(conn, cfg, payload)
            raise RuntimeError("Inspect eval interrupted or produced no log location; progress saved.")

        is_complete = logs[0].status == "success"
        import_eval_log(
            conn=conn,
            log_path=logs[0].location,
            run_id=run_id,
            set_id=payload["set_id"],
            track=payload["track"],
            model_config_id=payload["model_config_id"],
            prompt_version_id=prompt_ver_id,
            git_commit=git_rev,
            override_reason=payload.get("override_reason"),
            skip_sample_errors=not is_complete,
            run_status_override=None if is_complete else "incomplete",
            scorer_version=SCORER_VERSION,
            harness_manifest_json=json.dumps(manifest),
        )
        if not is_complete:
            raise RuntimeError(f"Inspect eval ended with status '{logs[0].status}'; completed photos saved.")
        return run_id
    finally:
        conn.close()


class SequentialJobWorker:
    """Single-threaded background worker processing queued jobs one at a time."""

    def __init__(self, cfg: PinpointConfig, poll_interval: float = 1.5) -> None:
        self.cfg = cfg
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _recover_interrupted_running_jobs(self) -> None:
        """Mark stale 'running' jobs left after an unclean shutdown as resumable 'failed'."""
        conn = connect_db(self.cfg.db_path)
        try:
            stale = conn.execute(
                "SELECT id, payload_json FROM jobs WHERE state = 'running'"
            ).fetchall()
            for j in stale:
                payload = json.loads(j["payload_json"])
                prog = None
                if not payload.get("photo_id"):
                    all_ids, remaining_ids = _recover_job_progress(conn, self.cfg, payload)
                    prog = f"{len(all_ids) - len(remaining_ids)} / {len(all_ids)} photos"
                conn.execute(
                    "UPDATE jobs SET state = 'failed', progress_text = COALESCE(?, progress_text), "
                    "error = COALESCE(error, 'Interrupted when server stopped; click Retry / Resume to continue.'), "
                    "updated_at = ? WHERE id = ?",
                    (prog, utc_now(), j["id"]),
                )
            conn.commit()
        finally:
            conn.close()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._recover_interrupted_running_jobs()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="pinpoint-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_next_pending()
            except Exception:
                logger.exception("Unhandled error in SequentialJobWorker loop")
            self._stop.wait(self.poll_interval)

    def run_next_pending(self) -> bool:
        conn = connect_db(self.cfg.db_path)
        try:
            job = conn.execute(
                "SELECT id, payload_json FROM jobs WHERE state = 'pending' ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            if not job:
                return False
            job_id = str(job["id"])
            payload = json.loads(job["payload_json"])
            if payload.get("photo_id"):
                total = 1
            else:
                total = int(
                    conn.execute(
                        "SELECT COUNT(*) AS c FROM set_members WHERE set_id = ?",
                        (payload["set_id"],),
                    ).fetchone()["c"]
                )
            conn.execute(
                "UPDATE jobs SET state = 'running', progress_text = ?, updated_at = ? WHERE id = ?",
                (f"0 / {total} photos", utc_now(), job_id),
            )
            conn.commit()
        finally:
            conn.close()

        env = os.environ.copy()
        env["PINPOINT_JOB_ID"] = job_id
        env["PINPOINT_JOB_TOTAL"] = str(total)

        try:
            # Run each Inspect eval in an isolated subprocess so global eval_async state never leaks across jobs.
            proc = subprocess.run(
                [sys.executable, "-m", "pinpoint.runner", job_id, str(self.cfg.data_dir)],
                cwd=str(self.cfg.repo_root),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                state, err, prog = "done", None, f"{total} / {total} photos"
            else:
                detail = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
                state, err, prog = "failed", detail[-1000:], None
        except (OSError, subprocess.SubprocessError) as exc:
            state, err, prog = "failed", str(exc), None

        conn = connect_db(self.cfg.db_path)
        try:
            if state != "done" and not payload.get("photo_id"):
                all_ids, remaining_ids = _recover_job_progress(conn, self.cfg, payload)
                prog = f"{len(all_ids) - len(remaining_ids)} / {len(all_ids)} photos"
            conn.execute(
                "UPDATE jobs SET state = ?, progress_text = COALESCE(?, progress_text), error = ?, updated_at = ? WHERE id = ?",
                (state, prog, err, utc_now(), job_id),
            )
            conn.commit()
        finally:
            conn.close()
        return True


def export_summary_files(conn: sqlite3.Connection, cfg: PinpointConfig) -> tuple[Path, Path]:
    """Export official run summaries and per-photo results to private local JSON and CSV."""
    runs = conn.execute(
        """
        SELECT r.*, s.name AS set_name, s.version AS set_version,
               mc.vendor, mc.family, mc.openrouter_id, mc.reasoning_effort
        FROM runs r
        JOIN image_sets s ON s.id = r.set_id
        JOIN model_configs mc ON mc.id = r.model_config_id
        ORDER BY r.started_at DESC
        """
    ).fetchall()

    summary_rows: list[dict[str, Any]] = []
    for r in runs:
        res_rows = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM results WHERE run_id = ?", (r["id"],)
            ).fetchall()
        ]
        official_res = [x for x in res_rows if x["is_official"] == 1]
        agg = aggregate_run_metrics(res_rows)
        is_off = len(official_res) > 0
        summary_rows.append(
            {
                "run_id": r["id"],
                "set": f"{r['set_name']} v{r['set_version']}",
                "track": r["track"],
                "vendor": r["vendor"],
                "model_config_id": r["model_config_id"],
                "model_id": r["openrouter_id"],
                "reasoning_effort": r["reasoning_effort"],
                "model_label": format_model_label(r["openrouter_id"], r["reasoning_effort"]),
                "status": r["status"],
                "is_official_run": is_off,
                "started_at": r["started_at"],
                **agg,
            }
        )

    official_a, official_b = collect_official_leaderboard(conn)
    json_path = cfg.exports_dir / "summary.json"
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    ranking_svg = build_ranking_bars_svg(official_a, official_b)
    if ranking_svg:
        (cfg.exports_dir / "ranking_chart.svg").write_text(ranking_svg, encoding="utf-8")

    map_rows = [
        dict(x)
        for x in conn.execute(
            """
            SELECT res.photo_id, res.pred_lat, res.pred_lon, res.pred_country_iso2,
                   res.distance_km, r.track, mc.vendor, mc.openrouter_id AS model_label,
                   mc.reasoning_effort,
                   p.lat AS gt_lat, p.lon AS gt_lon, p.country_iso2 AS gt_country, p.place_label
            FROM results res
            JOIN runs r ON r.id = res.run_id
            JOIN model_configs mc ON mc.id = r.model_config_id
            JOIN photos p ON p.id = res.photo_id
            WHERE res.is_official = 1
            """
        ).fetchall()
    ]
    map_svg = build_guess_map_svg(map_rows, "Official Results — Ground Truth vs Model Guesses")
    if map_svg:
        (cfg.exports_dir / "map_guesses.svg").write_text(map_svg, encoding="utf-8")
    bullseye_svg = build_bullseye_error_svg(map_rows)
    if bullseye_svg:
        (cfg.exports_dir / "bullseye_error.svg").write_text(bullseye_svg, encoding="utf-8")

    csv_path = cfg.exports_dir / "results.csv"
    detail_rows = conn.execute(
        """
        SELECT r.id AS run_id, r.track, r.started_at, mc.vendor, mc.openrouter_id, mc.reasoning_effort,
               res.photo_id, res.epoch, res.is_official, res.parse_status,
               res.pred_country_iso2, res.pred_lat, res.pred_lon, res.pred_confidence,
               res.distance_km, res.tier_points, res.smooth_score,
               res.hit_continent, res.hit_country, res.hit_100km, res.hit_10km, res.hit_1km,
               res.hit_100m, res.hit_10m, res.tool_calls_count,
               res.tokens_in, res.tokens_out, res.tokens_reasoning, res.tokens_cached,
               res.latency_ms, res.cost_usd
        FROM results res
        JOIN runs r ON r.id = res.run_id
        JOIN model_configs mc ON mc.id = r.model_config_id
        ORDER BY r.started_at DESC, res.photo_id ASC
        """
    ).fetchall()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if detail_rows:
            writer.writerow(detail_rows[0].keys())
            for row in detail_rows:
                writer.writerow(list(row))
        else:
            writer.writerow(["run_id", "track", "photo_id", "parse_status", "tier_points"])

    return json_path, csv_path


def export_public_release(
    conn: sqlite3.Connection,
    cfg: PinpointConfig,
    set_id: str | None = None,
) -> Path:
    """Export safe Option-1 public artifacts for git push (zero coords, salt, or photo IDs).

    Writes into `results/<set-name>-v<version>/`: commitment.json, summary.json,
    ranking_chart.svg, bullseye_error.svg, and README.md.
    """
    if set_id:
        set_row = conn.execute("SELECT * FROM image_sets WHERE id = ?", (set_id,)).fetchone()
    else:
        set_row = conn.execute(
            """
            SELECT * FROM image_sets
            ORDER BY (frozen_at IS NOT NULL) DESC, created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if not set_row:
        raise ValueError("No image set found to export publicly.")

    photo_count = int(
        conn.execute(
            "SELECT COUNT(*) AS c FROM set_members WHERE set_id = ?", (set_row["id"],)
        ).fetchone()["c"]
    )
    slug = f"{set_row['name']}-v{set_row['version']}".replace(" ", "-").lower()
    out_dir = cfg.repo_root / "results" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    harness_manifest = build_harness_manifest(cfg)
    gt_rows = [
        dict(x)
        for x in conn.execute(
            """
            SELECT p.id, p.lat, p.lon, p.country_iso2, p.continent
            FROM set_members sm
            JOIN photos p ON p.id = sm.photo_id
            WHERE sm.set_id = ?
            """,
            (set_row["id"],),
        ).fetchall()
    ]
    naive_baseline = compute_naive_baseline(gt_rows)

    commitment = {
        "set_name": set_row["name"],
        "version": set_row["version"],
        "description": set_row["description"],
        "photo_count": photo_count,
        "frozen_at": set_row["frozen_at"],
        "commitment_hash": set_row["commitment_hash"],
        "public_redaction_mode": "option_1_bullseye_only",
        "harness_manifest": harness_manifest,
        "note": "Salt and exact ground-truth coordinates stay private until this set is retired.",
    }
    (out_dir / "commitment.json").write_text(json.dumps(commitment, indent=2), encoding="utf-8")

    official_a, official_b = collect_official_leaderboard(conn, set_id=set_row["id"])
    public_summary: list[dict[str, Any]] = []
    for item in official_a + official_b:
        r, agg = item["run"], item["agg"]
        public_summary.append(
            {
                "set": f"{set_row['name']} v{set_row['version']}",
                "track": r["track"],
                "vendor": r["vendor"],
                "model_config_id": r["model_config_id"],
                "model_id": r["openrouter_id"],
                "reasoning_effort": r["reasoning_effort"],
                "model_label": r["model_label"],
                "started_at": r["started_at"],
                "prompt_versions": r.get("prompt_versions", []),
                "scorer_versions": r.get("scorer_versions", []),
                "upstream_providers": r.get("upstream_providers", []),
                **agg,
            }
        )
    summary_payload = {
        "set": f"{set_row['name']} v{set_row['version']}",
        "photo_count": photo_count,
        "frozen_at": set_row["frozen_at"],
        "commitment_hash": set_row["commitment_hash"],
        "harness_manifest": harness_manifest,
        "naive_baseline": naive_baseline,
        "official_runs": public_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    ranking_svg = build_ranking_bars_svg(official_a, official_b)
    if ranking_svg:
        (out_dir / "ranking_chart.svg").write_text(ranking_svg, encoding="utf-8")

    bullseye_rows = [
        dict(x)
        for x in conn.execute(
            """
            SELECT res.pred_lat, res.pred_lon, res.distance_km,
                   r.track, mc.vendor, mc.openrouter_id AS model_label, mc.reasoning_effort,
                   p.lat AS gt_lat, p.lon AS gt_lon
            FROM results res
            JOIN runs r ON r.id = res.run_id
            JOIN model_configs mc ON mc.id = r.model_config_id
            JOIN photos p ON p.id = res.photo_id
            WHERE r.set_id = ? AND res.is_official = 1 AND res.parse_status != 'api_error'
            """,
            (set_row["id"],),
        ).fetchall()
    ]
    bullseye_svg = build_bullseye_error_svg(
        bullseye_rows,
        f"{set_row['name']} v{set_row['version']} — Relative Distance Error (Truth = 0 km)",
    )
    if bullseye_svg:
        (out_dir / "bullseye_error.svg").write_text(bullseye_svg, encoding="utf-8")

    md_lines = [
        f"# Pinpoint Results — {set_row['name']} v{set_row['version']}",
        "",
        f"- **Photos:** {photo_count}",
        f"- **Frozen At:** `{set_row['frozen_at'] or 'unfrozen'}`",
        f"- **Commitment Hash (`SHA-256`):** `{set_row['commitment_hash'] or 'pending'}`",
        f"- **Scorer Version:** `{SCORER_VERSION}` · **Inspect AI:** `{harness_manifest['inspect_ai_version']}` · **Temp:** `{cfg.default_temperature}` · **Max Tokens:** `{cfg.default_max_tokens}`",
        "- **Public Map Policy:** Relative bullseye error only (ground-truth coordinates withheld while set is active).",
        "",
    ]
    if naive_baseline:
        md_lines.extend(
            [
                (
                    f"- **Naive Constant-Centroid Floor ({naive_baseline['ref_label']}):** "
                    f"**{naive_baseline['strict_mean_points']:.2f}/15** "
                    f"(Median km: {naive_baseline['median_distance_km']})"
                ),
                "",
            ]
        )
    if ranking_svg:
        md_lines.extend(["![Strict Score Ranking](./ranking_chart.svg)", ""])
    if bullseye_svg:
        md_lines.extend(["![Relative Error Bullseye](./bullseye_error.svg)", ""])
    for trk_title, items in (("Track A — Pixels Only", official_a), ("Track B — Web Search", official_b)):
        md_lines.extend([f"## {trk_title}", ""])
        if not items:
            md_lines.extend(["_No official runs recorded yet._", ""])
            continue
        md_lines.extend([
            "| # | Model (Reasoning Effort) | N | Epochs | Strict (/15) [95% CI] | Adjusted (/15) | Median km | Country % | ≤10 km % | ≤1 km % | Trunc % | Cost |",
            "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for i, item in enumerate(items, 1):
            r, a = item["run"], item["agg"]
            lbl = format_model_label(r["openrouter_id"], r.get("reasoning_effort"))
            med = f"{a['median_distance_km']:.3f}" if a["median_distance_km"] is not None else "—"
            ci = a.get("strict_ci95")
            ci_str = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
            tr = a["tier_rates"]
            n_lbl = f"{a.get('unique_photos', a['count'])}/{r.get('set_photo_count', photo_count)}"
            ep_lbl = str(a.get("epochs_count", 1))
            trunc_pct = f"{a.get('truncated_rate', 0.0) * 100:.0f}%"
            md_lines.append(
                f"| {i} | `{lbl}` | {n_lbl} | {ep_lbl} | **{a['strict_mean_points']:.2f}**{ci_str} | {a['adjusted_mean_points']:.2f} | "
                f"{med} | {tr.get('hit_country', 0)*100:.0f}% | {tr.get('hit_10km', 0)*100:.0f}% | "
                f"{tr.get('hit_1km', 0)*100:.0f}% | {trunc_pct} | ${a['total_cost_usd']:.4f} |"
            )
        md_lines.append("")

    (out_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")
    return out_dir


def export_audit_bundle(
    conn: sqlite3.Connection,
    cfg: PinpointConfig,
    set_id: str | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Export a confidential NDA auditor package (sanitized images, ground truth, salt, and .eval logs)."""
    if set_id:
        set_row = conn.execute("SELECT * FROM image_sets WHERE id = ?", (set_id,)).fetchone()
    else:
        set_row = conn.execute(
            "SELECT * FROM image_sets ORDER BY (frozen_at IS NOT NULL) DESC, created_at DESC LIMIT 1"
        ).fetchone()
    if not set_row:
        raise ValueError("No image set found to export for audit.")
    if set_row["frozen_at"] is None or not set_row["commitment_salt"]:
        raise ValueError("Image set must be frozen with a commitment salt before exporting an audit bundle.")

    slug = f"{set_row['name']}-v{set_row['version']}".replace(" ", "-").lower()
    target_dir = (out_dir or (cfg.exports_dir / f"audit-{slug}")).resolve()
    images_out = target_dir / "images"
    logs_out = target_dir / "logs"
    images_out.mkdir(parents=True, exist_ok=True)
    logs_out.mkdir(parents=True, exist_ok=True)

    photo_rows = conn.execute(
        """
        SELECT p.id, p.sha256_sanitized, p.lat, p.lon, p.country_iso2, p.continent,
               p.scene_tag, p.coord_basis, p.gps_accuracy_m, p.near_border, p.has_text,
               p.has_landmark, p.landmark_visibility
        FROM set_members sm
        JOIN photos p ON p.id = sm.photo_id
        WHERE sm.set_id = ?
        ORDER BY sm.position ASC, p.id ASC
        """,
        (set_row["id"],),
    ).fetchall()

    photos_manifest: list[dict[str, Any]] = []
    for r in photo_rows:
        src_img = cfg.images_dir / f"{r['id']}.jpg"
        if not src_img.exists():
            raise FileNotFoundError(f"Missing sanitized image: {src_img}")
        shutil.copy2(src_img, images_out / f"{r['id']}.jpg")
        photos_manifest.append(
            {
                "id": r["id"],
                "sha256": r["sha256_sanitized"],
                "lat": round(float(r["lat"]), 7),
                "lon": round(float(r["lon"]), 7),
                "country_iso2": r["country_iso2"],
                "continent": r["continent"],
                "scene_tag": r["scene_tag"],
                "coord_basis": r["coord_basis"],
                "gps_accuracy_m": r["gps_accuracy_m"],
                "near_border": bool(r["near_border"]),
                "has_text": bool(r["has_text"]),
                "has_landmark": bool(r["has_landmark"]),
                "landmark_visibility": r["landmark_visibility"],
            }
        )

    run_rows = conn.execute(
        "SELECT id, track, model_config_id, prompt_version_id, scorer_version, inspect_log_path, status "
        "FROM runs WHERE set_id = ?",
        (set_row["id"],),
    ).fetchall()
    copied_logs: list[str] = []
    for rr in run_rows:
        lp = rr["inspect_log_path"]
        if lp and Path(lp).exists():
            dst_log = logs_out / Path(lp).name
            shutil.copy2(lp, dst_log)
            copied_logs.append(dst_log.name)

    audit_manifest = {
        "set_id": set_row["id"],
        "set_name": set_row["name"],
        "version": set_row["version"],
        "description": set_row["description"],
        "frozen_at": set_row["frozen_at"],
        "expires_at": set_row["expires_at"],
        "commitment_hash": set_row["commitment_hash"],
        "commitment_salt": set_row["commitment_salt"],
        "harness_manifest": build_harness_manifest(cfg),
        "photos": photos_manifest,
        "eval_logs": copied_logs,
    }
    (target_dir / "audit_manifest.json").write_text(
        json.dumps(audit_manifest, indent=2), encoding="utf-8"
    )
    return target_dir


def verify_audit_commitment(audit_dir: Path) -> dict[str, Any]:
    """Verify sanitized image SHA-256s and recompute pre-run commitment hash from an audit bundle."""
    manifest_path = Path(audit_dir) / "audit_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"audit_manifest.json not found in {audit_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    salt = manifest.get("commitment_salt")
    expected_hash = manifest.get("commitment_hash")
    photos = manifest.get("photos") or []
    if not salt or not expected_hash or not photos:
        raise ValueError("audit_manifest.json is missing salt, commitment_hash, or photos.")

    images_dir = Path(audit_dir) / "images"
    commitment_payload: list[dict[str, Any]] = []
    for p in photos:
        img_path = images_dir / f"{p['id']}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image in audit bundle: {img_path}")
        actual_sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
        if actual_sha != p["sha256"]:
            raise ValueError(
                f"Image SHA-256 mismatch for {p['id']}: expected {p['sha256']}, got {actual_sha}"
            )
        commitment_payload.append(
            {
                "id": p["id"],
                "sha256": p["sha256"],
                "lat": round(float(p["lat"]), 7),
                "lon": round(float(p["lon"]), 7),
                "country_iso2": p["country_iso2"],
                "continent": p["continent"],
            }
        )

    blob = f"{salt}|{json.dumps(commitment_payload, separators=(',', ':'), sort_keys=True)}"
    actual_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"Commitment hash mismatch: expected {expected_hash}, computed {actual_hash}"
        )
    return {
        "verified": True,
        "photo_count": len(photos),
        "commitment_hash": actual_hash,
    }


if __name__ == "__main__":
    job_arg = sys.argv[1]
    data_arg = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    cfg_arg = PinpointConfig(data_dir=data_arg) if data_arg else get_config()
    execute_single_job(cfg_arg, job_arg)
