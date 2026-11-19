"""Cost estimation, sequential job queue worker, Inspect execution, and exports."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import sqlite3
import statistics
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelCost, ModelInfo, get_model_info, set_model_info

from pinpoint.charts import (
    build_bullseye_error_svg,
    build_guess_map_svg,
    build_ranking_bars_svg,
)
from pinpoint.config import (
    PinpointConfig,
    format_model_label,
    get_config,
    normalize_reasoning_effort,
)
from pinpoint.db import connect_db, utc_now
from pinpoint.geo import aggregate_run_metrics
from pinpoint.importer import (
    collect_official_leaderboard,
    import_eval_log,
    recover_partial_eval_logs,
)
from pinpoint.prompts import PROMPT_V1_ID
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


def enqueue_run_job(
    conn: sqlite3.Connection,
    set_id: str,
    track: str,
    model_config_id: str,
    photo_id: str | None = None,
    override_reason: str | None = None,
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
                output_tokens=8192,
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
            prompt_version_id=PROMPT_V1_ID,
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
        model_cost = ensure_inspect_model_cost(
            model_name=model_name,
            vendor=model["vendor"],
            params_json=model["params_json"],
        )

        eval_kwargs: dict[str, Any] = {
            "model": model_name,
            "log_dir": str(cfg.logs_dir),
            "sample_id": sample_id,
            "max_samples": 1,
            "max_connections": 1,
            "retry_on_error": 2,
            "cost_limit": cfg.max_run_cost_usd,
            "model_cost_config": {model_name: model_cost},
            "display": "none",
        }
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
            prompt_version_id=PROMPT_V1_ID,
            git_commit=get_git_commit(cfg.repo_root),
            override_reason=payload.get("override_reason"),
            skip_sample_errors=not is_complete,
            run_status_override=None if is_complete else "incomplete",
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

    commitment = {
        "set_name": set_row["name"],
        "version": set_row["version"],
        "description": set_row["description"],
        "photo_count": photo_count,
        "frozen_at": set_row["frozen_at"],
        "commitment_hash": set_row["commitment_hash"],
        "public_redaction_mode": "option_1_bullseye_only",
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
                **agg,
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(public_summary, indent=2), encoding="utf-8")

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
            WHERE r.set_id = ? AND res.is_official = 1
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
        "- **Public Map Policy:** Relative bullseye error only (ground-truth coordinates withheld while set is active).",
        "",
    ]
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
            "| # | Model (Reasoning Effort) | Strict (/15) | Adjusted (/15) | Median km | Country % | ≤10 km % | ≤1 km % | Total Time | Mean Latency | Reasoning Tok | Cost |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for i, item in enumerate(items, 1):
            r, a = item["run"], item["agg"]
            lbl = format_model_label(r["openrouter_id"], r.get("reasoning_effort"))
            med = f"{a['median_distance_km']:.3f}" if a["median_distance_km"] is not None else "—"
            t_sec = float(a.get("total_time_s") or 0.0)
            t_lbl = f"{t_sec / 60:.1f}m" if t_sec >= 60 else f"{t_sec:.0f}s"
            tr = a["tier_rates"]
            md_lines.append(
                f"| {i} | `{lbl}` | **{a['strict_mean_points']:.2f}** | {a['adjusted_mean_points']:.2f} | "
                f"{med} | {tr.get('hit_country', 0)*100:.0f}% | {tr.get('hit_10km', 0)*100:.0f}% | "
                f"{tr.get('hit_1km', 0)*100:.0f}% | {t_lbl} | {a['mean_latency_s']:.2f}s | {a['mean_tokens_reasoning']} | ${a['total_cost_usd']:.4f} |"
            )
        md_lines.append("")

    (out_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")

    interactive = _build_interactive_export(conn, set_row, official_a, official_b)
    (out_dir / "interactive.json").write_text(
        json.dumps(interactive, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    return out_dir


_ISO2_COUNTRY_NAMES: dict[str, str] = {
    "AD": "Andorra", "AL": "Albania", "AT": "Austria", "BA": "Bosnia & Herzegovina",
    "BE": "Belgium", "BG": "Bulgaria", "BY": "Belarus", "CH": "Switzerland",
    "CY": "Cyprus", "CZ": "Czechia", "DE": "Germany", "DK": "Denmark",
    "EE": "Estonia", "ES": "Spain", "FI": "Finland", "FR": "France",
    "GB": "United Kingdom", "GR": "Greece", "HR": "Croatia", "HU": "Hungary",
    "IE": "Ireland", "IS": "Iceland", "IT": "Italy", "LI": "Liechtenstein",
    "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MC": "Monaco",
    "MD": "Moldova", "ME": "Montenegro", "MK": "North Macedonia", "MT": "Malta",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RO": "Romania", "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia",
    "SK": "Slovakia", "SM": "San Marino", "TR": "Türkiye", "UA": "Ukraine",
    "VA": "Vatican City", "XK": "Kosovo",
}


def _human_model_name(openrouter_id: str) -> str:
    slug = openrouter_id.removeprefix("openrouter/").split("/")[-1]
    special = {
        "gpt": "GPT",
        "hy4": "HY4",
        "pro": "Pro",
        "flash": "Flash",
        "mini": "Mini",
        "nano": "Nano",
        "sonnet": "Sonnet",
        "haiku": "Haiku",
        "opus": "Opus",
        "gemini": "Gemini",
        "claude": "Claude",
        "grok": "Grok",
        "terra": "Terra",
        "hydrogen": "Hydrogen",
        "luna": "Luna",
    }
    parts: list[str] = []
    for tok in slug.split("-"):
        low = tok.lower()
        if low in special:
            parts.append(special[low])
        elif tok and tok[0].isdigit():
            if parts and parts[-1] == "GPT":
                parts[-1] = f"GPT-{tok}"
            else:
                parts.append(tok)
        else:
            parts.append(tok.capitalize())
    return " ".join(parts)


def _bootstrap_ci(
    values: list[float],
    stat_fn: Any,
    n_boot: int = 2000,
    seed: int = 42,
) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    draws: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        draws.append(float(stat_fn(sample)))
    draws.sort()
    lo = draws[int(0.025 * (n_boot - 1))]
    hi = draws[int(0.975 * (n_boot - 1))]
    return [round(lo, 4), round(hi, 4)]


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    idx = (len(sorted_vals) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _build_interactive_export(
    conn: sqlite3.Connection,
    set_row: sqlite3.Row,
    official_a: list[dict[str, Any]],
    official_b: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build leak-free interactive JSON for public blog visualizations.

    Contains ZERO ground-truth coordinates, ZERO photo UUIDs, ZERO filenames,
    ZERO place labels, and ZERO commitment salt. Per-sample geometry stores only
    relative (distance_km, bearing_deg) from an unknown (0, 0) origin; geographic
    aggregates are strictly country-level ISO-2 summaries.
    """
    set_id = str(set_row["id"])
    photo_rows = [
        dict(r)
        for r in conn.execute(
            """
            SELECT p.id, p.lat, p.lon, p.country_iso2, p.continent,
                   p.scene_tag, p.has_landmark, p.has_text, p.near_border
            FROM set_members sm
            JOIN photos p ON p.id = sm.photo_id
            WHERE sm.set_id = ?
            ORDER BY sm.position ASC
            """,
            (set_id,),
        ).fetchall()
    ]
    ordered_items = official_a + official_b

    # Fetch all official results for this set keyed by (track, model_config_id, photo_id)
    res_rows = [
        dict(r)
        for r in conn.execute(
            """
            SELECT r.track, r.model_config_id, res.photo_id, res.parse_status,
                   res.pred_lat, res.pred_lon, res.distance_km, res.tier_points,
                   res.hit_continent, res.hit_country, res.hit_100km, res.hit_10km,
                   res.hit_1km, res.hit_100m, res.hit_10m, res.smooth_score,
                   res.tokens_in, res.tokens_out, res.tokens_reasoning,
                   res.latency_ms, res.cost_usd
            FROM results res
            JOIN runs r ON r.id = res.run_id
            WHERE r.set_id = ? AND res.is_official = 1
            ORDER BY r.started_at ASC
            """,
            (set_id,),
        ).fetchall()
    ]

    by_model_photo: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in res_rows:
        mkey = (str(row["track"]), str(row["model_config_id"]))
        by_model_photo.setdefault(mkey, {})[str(row["photo_id"])] = row

    models_out: list[dict[str, Any]] = []
    model_keys: list[tuple[str, tuple[str, str]]] = []
    for item in ordered_items:
        r, agg = item["run"], item["agg"]
        slug_part = str(r["openrouter_id"]).removeprefix("openrouter/").split("/")[-1]
        eff = r.get("reasoning_effort") or "default"
        key = f"{slug_part}-{r['track']}-{eff}"
        mkey = (str(r["track"]), str(r["model_config_id"]))
        model_keys.append((key, mkey))

        m_rows = list(by_model_photo.get(mkey, {}).values())
        pts_vals = [float(x["tier_points"]) for x in m_rows]
        dist_vals = sorted(float(x["distance_km"]) for x in m_rows if x["distance_km"] is not None)
        fmt_errs = sum(1 for x in m_rows if x["parse_status"] == "format_error")

        strict_ci = _bootstrap_ci(pts_vals, statistics.mean, seed=42)
        median_ci = _bootstrap_ci(dist_vals, statistics.median, seed=42) if dist_vals else None
        p90 = _percentile(dist_vals, 0.90)

        models_out.append(
            {
                "key": key,
                "display_name": _human_model_name(str(r["openrouter_id"])),
                "model_id": r["openrouter_id"],
                "model_label": r["model_label"],
                "vendor": r["vendor"],
                "reasoning_effort": r.get("reasoning_effort"),
                "track": r["track"],
                "started_at": r["started_at"],
                "n": int(agg["count"]),
                "strict_mean_points": round(float(agg["strict_mean_points"]), 4),
                "strict_ci95": [round(x, 3) for x in strict_ci] if strict_ci else None,
                "adjusted_mean_points": round(float(agg["adjusted_mean_points"]), 4),
                "median_distance_km": round(float(agg["median_distance_km"]), 4)
                if agg["median_distance_km"] is not None
                else None,
                "median_ci95": [round(x, 4) for x in median_ci] if median_ci else None,
                "mean_distance_km": round(statistics.mean(dist_vals), 3) if dist_vals else None,
                "p90_distance_km": round(p90, 3) if p90 is not None else None,
                "max_distance_km": round(dist_vals[-1], 2) if dist_vals else None,
                "mean_smooth_score": round(float(agg["mean_smooth_score"]), 2),
                "refusal_rate": round(float(agg["refusal_rate"]), 4),
                "format_errors": fmt_errs,
                "tier_rates": {
                    k.removeprefix("hit_"): round(float(v), 4)
                    for k, v in agg["tier_rates"].items()
                },
                "im2gps_rates": {k: round(float(v), 4) for k, v in agg["im2gps_rates"].items()},
                "total_cost_usd": round(float(agg["total_cost_usd"]), 4),
                "mean_latency_s": round(float(agg["mean_latency_s"]), 2),
                "total_time_s": round(float(agg["total_time_s"]), 1),
                "mean_tokens_in": int(agg["mean_tokens_in"]),
                "mean_tokens_out": int(agg["mean_tokens_out"]),
                "mean_tokens_reasoning": int(agg["mean_tokens_reasoning"]),
            }
        )

    # Sort anonymized photos hardest -> easiest by cross-model mean score (zero IDs/coords/country)
    photo_difficulty: list[tuple[float, float, dict[str, Any]]] = []
    for p in photo_rows:
        pid = p["id"]
        scores = [
            float(by_model_photo[mkey][pid]["tier_points"])
            for _, mkey in model_keys
            if pid in by_model_photo.get(mkey, {})
        ]
        dists = [
            float(by_model_photo[mkey][pid]["distance_km"])
            for _, mkey in model_keys
            if pid in by_model_photo.get(mkey, {}) and by_model_photo[mkey][pid]["distance_km"] is not None
        ]
        mean_pts = statistics.mean(scores) if scores else 0.0
        med_d = statistics.median(dists) if dists else 99999.0
        photo_difficulty.append((mean_pts, -med_d, p))
    photo_difficulty.sort(key=lambda t: (t[0], t[1]))
    sorted_photos = [t[2] for t in photo_difficulty]

    photos_out: list[dict[str, Any]] = []
    for idx, p in enumerate(sorted_photos, 1):
        pid = p["id"]
        scores = [
            float(by_model_photo[mkey][pid]["tier_points"])
            for _, mkey in model_keys
            if pid in by_model_photo.get(mkey, {})
        ]
        photos_out.append(
            {
                "i": idx,
                "tag": p["scene_tag"],
                "lm": int(p["has_landmark"]),
                "txt": int(p["has_text"]),
                "mean": round(statistics.mean(scores), 2) if scores else 0.0,
            }
        )

    # Compact per-model relative samples aligned to sorted_photos: [dist_km, bearing_deg, tier_points]
    samples_out: dict[str, list[list[float | int | None]]] = {}
    for key, mkey in model_keys:
        arr: list[list[float | int | None]] = []
        pmap = by_model_photo.get(mkey, {})
        for p in sorted_photos:
            row = pmap.get(p["id"])
            if not row:
                arr.append([None, None, 0])
                continue
            dist = row["distance_km"]
            bearing: int | None = None
            if dist is not None and row["pred_lat"] is not None and row["pred_lon"] is not None:
                gt_lat = float(p["lat"])
                gt_lon = float(p["lon"])
                dx = (float(row["pred_lon"]) - gt_lon) * 111.32 * max(0.2, math.cos(math.radians(gt_lat)))
                dy = (float(row["pred_lat"]) - gt_lat) * 111.32
                # Compass bearing clockwise from North (0=N, 90=E, 180=S, 270=W)
                deg = math.degrees(math.atan2(dx, dy)) % 360.0
                bearing = round(deg) % 360
            arr.append(
                [
                    round(float(dist), 4) if dist is not None else None,
                    bearing,
                    int(row["tier_points"]),
                ]
            )
        samples_out[key] = arr

    # Country-level aggregates (ISO-2 only, ZERO photo coordinates)
    by_iso: dict[str, list[str]] = {}
    for p in photo_rows:
        iso = str(p["country_iso2"]).upper()
        by_iso.setdefault(iso, []).append(str(p["id"]))

    countries_out: list[dict[str, Any]] = []
    for iso, pids in sorted(by_iso.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        all_pts: list[float] = []
        all_dists: list[float] = []
        all_ctry = 0
        all_1km = 0
        all_100m = 0
        total_evals = 0
        per_model: dict[str, dict[str, float | None]] = {}

        for key, mkey in model_keys:
            pmap = by_model_photo.get(mkey, {})
            m_pts: list[float] = []
            m_dists: list[float] = []
            m_ctry = m_10km = m_1km = m_100m = 0
            for pid in pids:
                row = pmap.get(pid)
                if not row:
                    continue
                m_pts.append(float(row["tier_points"]))
                if row["distance_km"] is not None:
                    m_dists.append(float(row["distance_km"]))
                m_ctry += int(row["hit_country"])
                m_10km += int(row["hit_10km"])
                m_1km += int(row["hit_1km"])
                m_100m += int(row["hit_100m"])
            cnt = len(m_pts)
            if cnt:
                per_model[key] = {
                    "pts": round(statistics.mean(m_pts), 2),
                    "med_km": round(statistics.median(m_dists), 4) if m_dists else None,
                    "ctry": round(m_ctry / cnt, 4),
                    "km10": round(m_10km / cnt, 4),
                    "km1": round(m_1km / cnt, 4),
                    "m100": round(m_100m / cnt, 4),
                }
                all_pts.extend(m_pts)
                all_dists.extend(m_dists)
                all_ctry += m_ctry
                all_1km += m_1km
                all_100m += m_100m
                total_evals += cnt

        countries_out.append(
            {
                "iso2": iso,
                "name": _ISO2_COUNTRY_NAMES.get(iso, iso),
                "n": len(pids),
                "mean_points": round(statistics.mean(all_pts), 2) if all_pts else 0.0,
                "median_km": round(statistics.median(all_dists), 4) if all_dists else None,
                "hit_country": round(all_ctry / total_evals, 4) if total_evals else 0.0,
                "hit_1km": round(all_1km / total_evals, 4) if total_evals else 0.0,
                "hit_100m": round(all_100m / total_evals, 4) if total_evals else 0.0,
                "by_model": per_model,
            }
        )

    # Scene-category aggregates
    scene_buckets: list[tuple[str, str, list[str]]] = [
        (
            "urban_landmark",
            "Urban (landmark)",
            [p["id"] for p in photo_rows if p["scene_tag"] == "urban" and int(p["has_landmark"]) == 1],
        ),
        (
            "urban_street",
            "Urban (non-landmark)",
            [p["id"] for p in photo_rows if p["scene_tag"] == "urban" and int(p["has_landmark"]) == 0],
        ),
        (
            "rural_nature",
            "Rural & nature",
            [p["id"] for p in photo_rows if p["scene_tag"] in ("rural", "nature")],
        ),
        (
            "other",
            "Landmark / minimal clue",
            [p["id"] for p in photo_rows if p["scene_tag"] not in ("urban", "rural", "nature")],
        ),
    ]
    scenes_out: list[dict[str, Any]] = []
    for skey, slabel, pids in scene_buckets:
        if not pids:
            continue
        all_pts_s: list[float] = []
        all_dists_s: list[float] = []
        per_model_s: dict[str, dict[str, float | None]] = {}
        for key, mkey in model_keys:
            pmap = by_model_photo.get(mkey, {})
            m_pts = [float(pmap[pid]["tier_points"]) for pid in pids if pid in pmap]
            m_dists = [
                float(pmap[pid]["distance_km"])
                for pid in pids
                if pid in pmap and pmap[pid]["distance_km"] is not None
            ]
            if m_pts:
                m_100m = sum(int(pmap[pid]["hit_100m"]) for pid in pids if pid in pmap)
                m_1km = sum(int(pmap[pid]["hit_1km"]) for pid in pids if pid in pmap)
                per_model_s[key] = {
                    "pts": round(statistics.mean(m_pts), 2),
                    "med_km": round(statistics.median(m_dists), 4) if m_dists else None,
                    "km1": round(m_1km / len(m_pts), 4),
                    "m100": round(m_100m / len(m_pts), 4),
                }
                all_pts_s.extend(m_pts)
                all_dists_s.extend(m_dists)
        scenes_out.append(
            {
                "key": skey,
                "label": slabel,
                "n": len(pids),
                "mean_points": round(statistics.mean(all_pts_s), 2) if all_pts_s else 0.0,
                "median_km": round(statistics.median(all_dists_s), 4) if all_dists_s else None,
                "by_model": per_model_s,
            }
        )

    return {
        "schema": "pinpoint-interactive/1",
        "meta": {
            "set_name": set_row["name"],
            "version": int(set_row["version"]),
            "set_slug": f"{set_row['name']}-v{set_row['version']}".replace(" ", "-").lower(),
            "description": set_row["description"],
            "photo_count": len(photo_rows),
            "country_count": len(by_iso),
            "frozen_at": set_row["frozen_at"],
            "commitment_hash": set_row["commitment_hash"],
            "redaction_mode": "option_1_bullseye_and_country_agg",
        },
        "models": models_out,
        "countries": countries_out,
        "scenes": scenes_out,
        "photos": photos_out,
        "samples": samples_out,
    }


if __name__ == "__main__":
    job_arg = sys.argv[1]
    data_arg = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    cfg_arg = PinpointConfig(data_dir=data_arg) if data_arg else get_config()
    execute_single_job(cfg_arg, job_arg)
