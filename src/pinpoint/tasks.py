"""Inspect AI tasks, solvers, and scorer for Pinpoint (Track A & Track B)."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver, use_tools
from inspect_ai.tool import web_search

from pinpoint.config import PinpointConfig, get_config
from pinpoint.db import connect_db, init_db
from pinpoint.geo import parse_model_response, score_prediction
from pinpoint.prompts import build_prompt


def load_set_dataset(cfg: PinpointConfig, set_id: str, track: str) -> MemoryDataset:
    """Load sanitized images and ground truth from SQLite for an Inspect task."""
    conn = connect_db(cfg.db_path)
    init_db(conn)
    try:
        rows = conn.execute(
            """
            SELECT p.id, p.lat, p.lon, p.country_iso2, p.continent, p.scene_tag, p.near_border
            FROM set_members sm
            JOIN photos p ON p.id = sm.photo_id
            WHERE sm.set_id = ?
            ORDER BY sm.position ASC, p.id ASC
            """,
            (set_id,),
        ).fetchall()
    finally:
        conn.close()

    prompt_text = build_prompt(track)
    samples: list[Sample] = []
    for r in rows:
        img_path = cfg.images_dir / f"{r['id']}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing sanitized image for photo {r['id']}: {img_path}")
        samples.append(
            Sample(
                id=r["id"],
                input=[
                    ChatMessageUser(
                        content=[
                            ContentText(text=prompt_text),
                            # Only the sanitized JPEG path is passed; filename is a random UUID.
                            ContentImage(image=str(img_path)),
                        ]
                    )
                ],
                target=json.dumps(
                    {
                        "lat": float(r["lat"]),
                        "lon": float(r["lon"]),
                        "country_iso2": r["country_iso2"],
                        "continent": r["continent"],
                    }
                ),
                metadata={
                    "photo_id": r["id"],
                    "scene_tag": r["scene_tag"],
                    "near_border": bool(r["near_border"]),
                },
            )
        )
    return MemoryDataset(samples=samples, name=f"pinpoint_{set_id}_{track}")


@solver
def pinpoint_json_solver() -> Solver:
    """Generate an answer and perform at most one logged retry on malformed JSON."""

    async def solve(state: TaskState, gen: Generate) -> TaskState:
        state = await gen(state)
        raw = state.output.completion if state.output else ""
        parsed = parse_model_response(raw)
        if parsed.parse_status == "format_error":
            state.messages.append(
                ChatMessageUser(
                    content=(
                        "Your previous reply was not valid JSON matching the required schema. "
                        "Return ONLY the raw JSON object with fields status, country, latitude, "
                        "longitude, confidence, and reasoning."
                    )
                )
            )
            state = await gen(state)
        return state

    return solve


_progress_done = 0


def _record_job_progress(cfg: PinpointConfig) -> None:
    global _progress_done
    job_id = os.getenv("PINPOINT_JOB_ID")
    if not job_id:
        return
    _progress_done += 1
    total = os.getenv("PINPOINT_JOB_TOTAL") or "?"
    try:
        conn = connect_db(cfg.db_path)
        try:
            conn.execute(
                "UPDATE jobs SET progress_text = ? WHERE id = ?",
                (f"{_progress_done} / {total} photos", job_id),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error:  # pragma: no cover — progress updates are best-effort
        logging.getLogger(__name__).debug("Failed to update job progress", exc_info=True)


@scorer(metrics=[mean()])
def pinpoint_scorer(data_dir: str | None = None):
    """Score parsed geolocation predictions using Pinpoint's rising-weight tier rubric."""
    cfg = PinpointConfig(data_dir=Path(data_dir).resolve()) if data_dir else get_config()

    async def score(state: TaskState, target: Target) -> Score:
        gt = json.loads(target.text)
        raw = state.output.completion if state.output else ""
        parsed = parse_model_response(raw)
        breakdown = score_prediction(
            gt_lat=float(gt["lat"]),
            gt_lon=float(gt["lon"]),
            gt_country_iso2=str(gt["country_iso2"]),
            gt_continent=str(gt["continent"]),
            pred=parsed,
        )
        _record_job_progress(cfg)
        return Score(
            value=float(breakdown.tier_points),
            answer=raw,
            explanation=(
                f"status={parsed.parse_status}, "
                f"dist_km={breakdown.distance_km}, "
                f"pred_iso={parsed.country_iso2}"
            ),
            metadata={
                "parse_status": parsed.parse_status,
                "pred_lat": parsed.lat,
                "pred_lon": parsed.lon,
                "pred_country_iso2": parsed.country_iso2,
                "pred_confidence": parsed.confidence,
                "reasoning_text": parsed.reasoning,
                "distance_km": breakdown.distance_km,
                "hit_continent": breakdown.hit_continent,
                "hit_country": breakdown.hit_country,
                "hit_100km": breakdown.hit_100km,
                "hit_10km": breakdown.hit_10km,
                "hit_1km": breakdown.hit_1km,
                "hit_100m": breakdown.hit_100m,
                "hit_10m": breakdown.hit_10m,
                "tier_points": breakdown.tier_points,
                "smooth_score": breakdown.smooth_score,
            },
        )

    return score


def build_pinpoint_task(set_id: str, track: str, data_dir: str | None = None) -> Task:
    cfg = PinpointConfig(data_dir=Path(data_dir).resolve()) if data_dir else get_config()
    cfg.ensure_dirs()
    dataset = load_set_dataset(cfg, set_id=set_id, track=track)

    if track.upper() == "A":
        solvers = [pinpoint_json_solver()]
        msg_limit = 4
    elif track.upper() == "B":
        solvers = [
            use_tools(web_search({"tavily": {"max_results": cfg.max_search_calls}})),
            pinpoint_json_solver(),
        ]
        msg_limit = 14
    else:
        raise ValueError(f"Unsupported track: {track!r}")

    return Task(
        dataset=dataset,
        solver=solvers,
        scorer=pinpoint_scorer(data_dir=str(cfg.data_dir)),
        message_limit=msg_limit,
    )


@task
def pinpoint_a(set_id: str, data_dir: str | None = None) -> Task:
    """Track A: Pixels-only geolocation (no web search or external tools)."""
    return build_pinpoint_task(set_id=set_id, track="A", data_dir=data_dir)


@task
def pinpoint_b(set_id: str, data_dir: str | None = None) -> Task:
    """Track B: Visual geolocation with uniform Tavily web search enabled."""
    return build_pinpoint_task(set_id=set_id, track="B", data_dir=data_dir)
