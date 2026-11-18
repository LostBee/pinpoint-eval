"""Cost estimation, sequential job queue worker, Inspect execution, and exports."""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
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
    return out_dir


if __name__ == "__main__":
    job_arg = sys.argv[1]
    data_arg = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    cfg_arg = PinpointConfig(data_dir=data_arg) if data_arg else get_config()
    execute_single_job(cfg_arg, job_arg)
