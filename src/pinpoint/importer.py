"""Import Inspect AI .eval logs into SQLite runs and results tables."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, read_eval_log

from pinpoint.config import format_model_label, normalize_reasoning_effort
from pinpoint.db import utc_now
from pinpoint.geo import aggregate_run_metrics, parse_model_response, score_prediction

logger = logging.getLogger(__name__)


def _count_tool_calls(sample: Any) -> int:
    count = 0
    for msg in getattr(sample, "messages", None) or []:
        tcalls = getattr(msg, "tool_calls", None)
        if tcalls:
            count += len(tcalls)
    return count


def _provider_reported_cost(sample: Any) -> float | None:
    """Read exact upstream USD cost returned in OpenRouter response.usage.cost if present."""
    reported = 0.0
    found = False
    for ev in getattr(sample, "events", None) or []:
        if getattr(ev, "event", None) != "model":
            continue
        call = getattr(ev, "call", None)
        resp = getattr(call, "response", None) if call else None
        usage = resp.get("usage") if isinstance(resp, dict) else None
        if isinstance(usage, dict) and usage.get("cost") is not None:
            try:
                reported += float(usage["cost"])
                found = True
            except (TypeError, ValueError):
                pass
    return reported if found else None


def _extract_tokens_and_cost(
    sample: Any,
    params_json: str,
    tool_calls_count: int,
    track: str,
) -> tuple[int, int, int, int, float]:
    """Return (tokens_in_total, tokens_out, tokens_reasoning, tokens_cached, cost_usd)."""
    tokens_uncached = 0
    tokens_cached = 0
    tokens_out = 0
    tokens_reasoning = 0
    inspect_cost = 0.0
    model_usage = getattr(sample, "model_usage", None) or {}
    for usage in model_usage.values():
        tokens_uncached += int(getattr(usage, "input_tokens", 0) or 0)
        tokens_cached += int(getattr(usage, "input_tokens_cache_read", 0) or 0)
        tokens_out += int(getattr(usage, "output_tokens", 0) or 0)
        tokens_reasoning += int(getattr(usage, "reasoning_tokens", 0) or 0)
        u_cost = getattr(usage, "total_cost", None)
        if u_cost is not None:
            inspect_cost += float(u_cost)

    tokens_in = tokens_uncached + tokens_cached
    provider_cost = _provider_reported_cost(sample)
    if provider_cost is not None:
        total_cost = provider_cost
    elif inspect_cost > 0.0:
        total_cost = inspect_cost
    else:
        params = json.loads(params_json or "{}")
        p_in = float(params.get("est_usd_per_1m_in", 0.5))
        p_out = float(params.get("est_usd_per_1m_out", 2.0))
        total_cost = (tokens_in / 1_000_000.0) * p_in + (tokens_out / 1_000_000.0) * p_out

    if track.upper() == "B":
        # Tavily runs locally via Inspect; pay-as-you-go approx $0.008/credit beyond free tier
        total_cost += tool_calls_count * 0.008

    return tokens_in, tokens_out, tokens_reasoning, tokens_cached, round(total_cost, 6)


def import_eval_log(
    conn: sqlite3.Connection,
    log_path: Path | str,
    run_id: str,
    set_id: str,
    track: str,
    model_config_id: str,
    prompt_version_id: str,
    git_commit: str | None = None,
    override_reason: str | None = None,
    skip_sample_errors: bool = False,
    run_status_override: str | None = None,
) -> int:
    """Import samples from an Inspect .eval log into runs & results (idempotent per run_id)."""
    log: EvalLog = read_eval_log(str(log_path))
    model_row = conn.execute(
        "SELECT params_json FROM model_configs WHERE id = ?", (model_config_id,)
    ).fetchone()
    params_json = model_row["params_json"] if model_row else "{}"

    started_at = getattr(log.stats, "started_at", None) or utc_now()
    finished_at = getattr(log.stats, "completed_at", None) or utc_now()
    run_status = run_status_override or ("completed" if log.status == "success" else "failed")

    conn.execute(
        """
        INSERT OR REPLACE INTO runs (
            id, set_id, track, model_config_id, prompt_version_id,
            inspect_log_path, epochs, status, cost_usd, git_commit,
            started_at, finished_at, override_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            set_id,
            track.upper(),
            model_config_id,
            prompt_version_id,
            str(log_path),
            int(getattr(log.eval.config, "epochs", None) or 1),
            run_status,
            0.0,
            git_commit,
            str(started_at),
            str(finished_at),
            override_reason,
        ),
    )
    conn.execute("DELETE FROM results WHERE run_id = ?", (run_id,))

    samples = log.samples or []
    run_cost = 0.0
    inserted = 0

    for sample in samples:
        photo_id = str(sample.id)
        photo_row = conn.execute(
            "SELECT lat, lon, country_iso2, continent FROM photos WHERE id = ?",
            (photo_id,),
        ).fetchone()
        if not photo_row:
            continue

        raw_text = sample.output.completion if getattr(sample, "output", None) else ""
        if getattr(sample, "error", None) is not None and not raw_text:
            if skip_sample_errors:
                continue
            parse_status = "api_error"
            pred = None
            breakdown = score_prediction(0, 0, "", "", parse_model_response(""))
        else:
            pred = parse_model_response(raw_text)
            parse_status = pred.parse_status
            breakdown = score_prediction(
                gt_lat=float(photo_row["lat"]),
                gt_lon=float(photo_row["lon"]),
                gt_country_iso2=photo_row["country_iso2"],
                gt_continent=photo_row["continent"],
                pred=pred,
            )

        tool_calls = _count_tool_calls(sample)
        tokens_in, tokens_out, tokens_reasoning, tokens_cached, cost_usd = _extract_tokens_and_cost(
            sample, params_json, tool_calls, track
        )
        run_cost += cost_usd
        latency_ms = int((getattr(sample, "total_time", 0.0) or 0.0) * 1000)
        epoch = int(getattr(sample, "epoch", 1) or 1)

        # Official result rule (Q15): first recorded result for (set, track, model, photo, epoch=1)
        existing_official = conn.execute(
            """
            SELECT res.id
            FROM results res
            JOIN runs r ON r.id = res.run_id
            WHERE r.set_id = ?
              AND r.track = ?
              AND r.model_config_id = ?
              AND res.photo_id = ?
              AND res.is_official = 1
              AND res.run_id != ?
            LIMIT 1
            """,
            (set_id, track.upper(), model_config_id, photo_id, run_id),
        ).fetchone()
        is_official = 1 if (epoch == 1 and existing_official is None) else 0

        conn.execute(
            """
            INSERT INTO results (
                id, run_id, photo_id, epoch, attempt, is_official,
                raw_response, parse_status, pred_lat, pred_lon, pred_country_iso2,
                pred_confidence, reasoning_text, tool_calls_count,
                tokens_in, tokens_out, tokens_reasoning, tokens_cached,
                latency_ms, cost_usd, distance_km,
                hit_continent, hit_country, hit_100km, hit_10km, hit_1km,
                hit_100m, hit_10m, tier_points, smooth_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                run_id,
                photo_id,
                epoch,
                1,
                is_official,
                raw_text,
                parse_status,
                pred.lat if pred else None,
                pred.lon if pred else None,
                pred.country_iso2 if pred else None,
                pred.confidence if pred else None,
                pred.reasoning if pred else None,
                tool_calls,
                tokens_in,
                tokens_out,
                tokens_reasoning,
                tokens_cached,
                latency_ms,
                cost_usd,
                breakdown.distance_km,
                1 if breakdown.hit_continent else 0,
                1 if breakdown.hit_country else 0,
                1 if breakdown.hit_100km else 0,
                1 if breakdown.hit_10km else 0,
                1 if breakdown.hit_1km else 0,
                1 if breakdown.hit_100m else 0,
                1 if breakdown.hit_10m else 0,
                breakdown.tier_points,
                breakdown.smooth_score,
            ),
        )
        inserted += 1

    conn.execute("UPDATE runs SET cost_usd = ? WHERE id = ?", (round(run_cost, 6), run_id))
    conn.commit()
    return inserted


def backfill_runs_from_eval_logs(conn: sqlite3.Connection) -> int:
    """Re-import all runs whose .eval log file still exists on disk (e.g. after schema upgrades)."""
    runs = conn.execute(
        "SELECT id, set_id, track, model_config_id, prompt_version_id, inspect_log_path, git_commit, override_reason "
        "FROM runs WHERE inspect_log_path IS NOT NULL ORDER BY started_at ASC"
    ).fetchall()
    updated = 0
    for r in runs:
        lpath = Path(r["inspect_log_path"])
        if not lpath.exists():
            continue
        import_eval_log(
            conn=conn,
            log_path=lpath,
            run_id=r["id"],
            set_id=r["set_id"],
            track=r["track"],
            model_config_id=r["model_config_id"],
            prompt_version_id=r["prompt_version_id"],
            git_commit=r["git_commit"],
            override_reason=r["override_reason"],
        )
        updated += 1
    return updated


def recover_partial_eval_logs(
    conn: sqlite3.Connection,
    logs_dir: Path,
    set_id: str,
    track: str,
    model_config_id: str,
    openrouter_id: str,
    reasoning_effort: str | None,
    prompt_version_id: str,
    git_commit: str | None = None,
) -> int:
    """Import completed samples from any un-imported partial/cancelled .eval logs matching this run."""
    if not logs_dir.exists():
        return 0
    imported_paths = {
        str(Path(r["inspect_log_path"]).resolve())
        for r in conn.execute("SELECT inspect_log_path FROM runs WHERE inspect_log_path IS NOT NULL").fetchall()
    }
    expected_dataset = f"pinpoint_{set_id}_{track.upper()}"
    target_model = openrouter_id.strip()
    if target_model and not target_model.startswith("openrouter/"):
        target_model = f"openrouter/{target_model}"
    target_effort = normalize_reasoning_effort(reasoning_effort)

    recovered_samples = 0
    for lpath in sorted(logs_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime):
        if str(lpath.resolve()) in imported_paths:
            continue
        try:
            hdr = read_eval_log(str(lpath), header_only=True)
        except (OSError, ValueError, RuntimeError):
            logger.warning("Skipping unreadable eval log %s", lpath, exc_info=True)
            continue
        ev = getattr(hdr, "eval", None)
        ds = getattr(ev, "dataset", None) if ev else None
        if not ev or not ds or ds.name != expected_dataset:
            continue
        if getattr(ev, "model", None) != target_model:
            continue
        log_effort = normalize_reasoning_effort(
            getattr(getattr(ev, "model_generate_config", None), "reasoning_effort", None)
        )
        if log_effort != target_effort:
            continue

        full_log = read_eval_log(str(lpath), header_only=False)
        valid_count = sum(
            1
            for s in (full_log.samples or [])
            if getattr(s, "error", None) is None or (getattr(s, "output", None) and s.output.completion)
        )
        if valid_count == 0:
            continue

        is_complete = full_log.status == "success"
        recovered_samples += import_eval_log(
            conn=conn,
            log_path=lpath,
            run_id=str(uuid.uuid4()),
            set_id=set_id,
            track=track,
            model_config_id=model_config_id,
            prompt_version_id=prompt_version_id,
            git_commit=git_commit,
            skip_sample_errors=not is_complete,
            run_status_override=None if is_complete else "incomplete",
        )
    return recovered_samples


def collect_official_leaderboard(
    conn: sqlite3.Connection,
    set_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate Official (is_official=1) metrics per (set_id, track, model_config_id),
    seamlessly merging samples across resumed partial runs."""
    where_clause = "WHERE res.is_official = 1"
    params: tuple = ()
    if set_id and set_id != "all":
        where_clause += " AND r.set_id = ?"
        params = (set_id,)

    rows = conn.execute(
        f"""
        SELECT r.set_id, r.track, r.model_config_id, r.id AS run_id, r.started_at,
               s.name AS set_name, s.version AS set_version,
               mc.vendor, mc.family, mc.openrouter_id, mc.reasoning_effort,
               res.*
        FROM results res
        JOIN runs r ON r.id = res.run_id
        JOIN image_sets s ON s.id = r.set_id
        JOIN model_configs mc ON mc.id = r.model_config_id
        {where_clause}
        ORDER BY r.started_at ASC
        """,
        params,
    ).fetchall()

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        d = dict(row)
        key = (d["set_id"], d["track"], d["model_config_id"])
        bucket = grouped.get(key)
        if bucket is None:
            bucket = {
                "run": {
                    "id": d["run_id"],
                    "set_id": d["set_id"],
                    "set_name": d["set_name"],
                    "set_version": d["set_version"],
                    "track": d["track"],
                    "model_config_id": d["model_config_id"],
                    "vendor": d["vendor"],
                    "family": d["family"],
                    "openrouter_id": d["openrouter_id"],
                    "reasoning_effort": d["reasoning_effort"],
                    "model_label": format_model_label(d["openrouter_id"], d.get("reasoning_effort")),
                    "started_at": d["started_at"],
                },
                "rows": [],
            }
            grouped[key] = bucket
        bucket["run"]["id"] = d["run_id"]
        bucket["rows"].append(d)

    official_a: list[dict[str, Any]] = []
    official_b: list[dict[str, Any]] = []
    for bucket in grouped.values():
        agg = aggregate_run_metrics(bucket["rows"])
        agg["total_searches"] = sum(int(x.get("tool_calls_count") or 0) for x in bucket["rows"])
        item = {"run": bucket["run"], "agg": agg}
        if bucket["run"]["track"] == "A":
            official_a.append(item)
        else:
            official_b.append(item)

    sort_key = lambda x: (
        -x["agg"]["strict_mean_points"],
        x["agg"]["median_distance_km"] if x["agg"]["median_distance_km"] is not None else 999999.0,
    )
    official_a.sort(key=sort_key)
    official_b.sort(key=sort_key)
    return official_a, official_b
