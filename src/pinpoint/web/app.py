"""Local FastAPI web UI for Pinpoint."""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from pinpoint.charts import (
    build_bullseye_error_svg,
    build_guess_map_svg,
    build_ranking_bars_svg,
)
from pinpoint.config import (
    VALID_REASONING_EFFORTS,
    PinpointConfig,
    format_model_label,
    get_config,
    normalize_reasoning_effort,
)
from pinpoint.db import connect_db, init_db
from pinpoint.geo import (
    aggregate_run_metrics,
    continent_for_location,
    european_country_choices,
    normalize_country,
)
from pinpoint.ingest import SUPPORTED_EXTENSIONS, ingest_photo, list_inbox_files, read_exif_metadata
from pinpoint.runner import (
    SequentialJobWorker,
    enqueue_run_job,
    estimate_run_cost,
    export_public_release,
    export_summary_files,
    normalize_openrouter_id,
)
from pinpoint.sets import (
    add_photos_to_set,
    check_vendor_rotation_guard,
    create_image_set,
    freeze_set,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(cfg: PinpointConfig | None = None, start_worker: bool = True) -> FastAPI:
    config = cfg or get_config()
    config.ensure_dirs()
    conn = connect_db(config.db_path)
    init_db(conn)
    conn.close()

    worker = SequentialJobWorker(config) if start_worker else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if worker:
            worker.start()
        try:
            yield
        finally:
            if worker:
                worker.stop()

    app = FastAPI(title="Pinpoint (pinpoint-eval)", lifespan=lifespan)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def _db():
        return connect_db(config.db_path)

    @app.get("/", response_class=HTMLResponse)
    def inbox_page(request: Request, msg: str = "", err: str = ""):
        files = list_inbox_files(config)
        items = []
        for f in files:
            meta = read_exif_metadata(f)
            items.append({"filename": f.name, "meta": meta})
        conn = _db()
        try:
            pool_count = conn.execute(
                "SELECT COUNT(*) AS c FROM photos WHERE status IN ('pool', 'in_set')"
            ).fetchone()["c"]
        finally:
            conn.close()
        return templates.TemplateResponse(
            request,
            "inbox.html",
            {
                "items": items,
                "pool_count": pool_count,
                "inbox_dir": str(config.inbox_dir),
                "country_choices": european_country_choices(),
                "msg": msg,
                "err": err,
            },
        )

    @app.post("/inbox/upload")
    async def inbox_upload(files: Annotated[list[UploadFile], File(...)]):
        saved = 0
        for uf in files:
            if not uf.filename:
                continue
            name = Path(uf.filename).name
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            dest = config.inbox_dir / name
            with dest.open("wb") as out:
                shutil.copyfileobj(uf.file, out)
            saved += 1
        return RedirectResponse(url=f"/?msg=Uploaded+{saved}+file(s)+to+inbox", status_code=303)

    @app.get("/media/inbox/{filename}")
    def serve_inbox_image(filename: str):
        path = (config.inbox_dir / Path(filename).name).resolve()
        if not path.exists() or path.parent != config.inbox_dir.resolve():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(path)

    @app.get("/media/sanitized/{photo_id}")
    def serve_sanitized_image(photo_id: str):
        safe_id = Path(photo_id).name
        path = (config.images_dir / f"{safe_id}.jpg").resolve()
        if not path.exists() or path.parent != config.images_dir.resolve():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(path)

    @app.post("/inbox/ingest")
    def inbox_ingest(
        filename: str = Form(...),
        lat: float = Form(...),
        lon: float = Form(...),
        country: str = Form(...),
        place_label: str = Form(""),
        scene_tag: str = Form("urban"),
        near_border: str | None = Form(None),
        has_text: str | None = Form(None),
        has_landmark: str | None = Form(None),
    ):
        src = (config.inbox_dir / Path(filename).name).resolve()
        if not src.exists() or src.parent != config.inbox_dir.resolve():
            return RedirectResponse(url="/?err=File+not+found+in+inbox", status_code=303)
        conn = _db()
        try:
            pid = ingest_photo(
                conn,
                config,
                src,
                lat=lat,
                lon=lon,
                country_input=country,
                place_label=place_label,
                scene_tag=scene_tag,
                near_border=near_border is not None,
                has_text=has_text is not None,
                has_landmark=has_landmark is not None,
            )
            return RedirectResponse(url=f"/?msg=Ingested+{pid[:8]}", status_code=303)
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
            return RedirectResponse(url=f"/?err={str(exc).replace(' ', '+')}", status_code=303)
        finally:
            conn.close()

    @app.post("/inbox/exclude")
    def inbox_exclude(filename: str = Form(...)):
        src = (config.inbox_dir / Path(filename).name).resolve()
        if src.exists() and src.parent == config.inbox_dir.resolve():
            excluded_dir = config.data_dir / "excluded"
            excluded_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(excluded_dir / src.name))
        return RedirectResponse(url="/?msg=Moved+photo+to+excluded/", status_code=303)

    @app.get("/pool", response_class=HTMLResponse)
    def pool_page(request: Request, country: str = "", msg: str = "", err: str = ""):
        conn = _db()
        try:
            where = "WHERE p.country_iso2 = ?" if country.strip() else ""
            params: tuple = (country.strip().upper(),) if country.strip() else ()
            photos = [
                dict(r)
                for r in conn.execute(
                    f"""
                    SELECT p.*,
                           EXISTS(
                               SELECT 1 FROM set_members sm
                               JOIN image_sets s ON s.id = sm.set_id
                               WHERE sm.photo_id = p.id AND s.frozen_at IS NOT NULL
                           ) AS is_frozen
                    FROM photos p
                    {where}
                    ORDER BY p.added_at DESC
                    """,
                    params,
                ).fetchall()
            ]
            countries = [
                r["country_iso2"]
                for r in conn.execute(
                    "SELECT DISTINCT country_iso2 FROM photos ORDER BY country_iso2"
                ).fetchall()
            ]
            country_counts = [
                dict(r)
                for r in conn.execute(
                    "SELECT country_iso2, COUNT(*) AS cnt FROM photos GROUP BY country_iso2 ORDER BY cnt DESC, country_iso2 ASC"
                ).fetchall()
            ]
            scene_counts = [
                dict(r)
                for r in conn.execute(
                    "SELECT scene_tag, COUNT(*) AS cnt FROM photos GROUP BY scene_tag ORDER BY cnt DESC"
                ).fetchall()
            ]
        finally:
            conn.close()
        return templates.TemplateResponse(
            request,
            "pool.html",
            {
                "photos": photos,
                "countries": countries,
                "country_counts": country_counts,
                "scene_counts": scene_counts,
                "selected_country": country.strip().upper(),
                "country_choices": european_country_choices(),
                "msg": msg,
                "err": err,
            },
        )

    @app.post("/pool/update")
    def pool_update(
        photo_id: str = Form(...),
        lat: float = Form(...),
        lon: float = Form(...),
        country: str = Form(...),
        place_label: str = Form(""),
        scene_tag: str = Form("urban"),
        near_border: str | None = Form(None),
        has_text: str | None = Form(None),
        has_landmark: str | None = Form(None),
    ):
        conn = _db()
        try:
            frozen = conn.execute(
                """
                SELECT 1 FROM set_members sm
                JOIN image_sets s ON s.id = sm.set_id
                WHERE sm.photo_id = ? AND s.frozen_at IS NOT NULL
                LIMIT 1
                """,
                (photo_id,),
            ).fetchone()
            if frozen:
                return RedirectResponse(
                    url="/pool?err=Cannot+edit+a+photo+in+a+frozen+set", status_code=303
                )
            iso2 = normalize_country(country)
            if not iso2:
                return RedirectResponse(
                    url=f"/pool?err=Unrecognized+country+{country}", status_code=303
                )
            continent = continent_for_location(iso2, lat, lon) or "OTHER"
            conn.execute(
                """
                UPDATE photos
                SET lat = ?, lon = ?, country_iso2 = ?, continent = ?,
                    place_label = ?, scene_tag = ?, near_border = ?,
                    has_text = ?, has_landmark = ?
                WHERE id = ?
                """,
                (
                    lat,
                    lon,
                    iso2,
                    continent,
                    place_label.strip() or None,
                    scene_tag,
                    1 if near_border is not None else 0,
                    1 if has_text is not None else 0,
                    1 if has_landmark is not None else 0,
                    photo_id,
                ),
            )
            conn.commit()
            return RedirectResponse(url=f"/pool?msg=Updated+{photo_id[:8]}", status_code=303)
        finally:
            conn.close()

    @app.get("/sets", response_class=HTMLResponse)
    def sets_page(request: Request, msg: str = "", err: str = ""):
        conn = _db()
        try:
            sets = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT s.*, COUNT(sm.photo_id) AS member_count
                    FROM image_sets s
                    LEFT JOIN set_members sm ON sm.set_id = s.id
                    GROUP BY s.id
                    ORDER BY s.created_at DESC
                    """
                ).fetchall()
            ]
            pool_photos = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM photos WHERE status IN ('pool', 'in_set') ORDER BY added_at DESC"
                ).fetchall()
            ]
            country_counts = [
                dict(r)
                for r in conn.execute(
                    "SELECT country_iso2, COUNT(*) AS cnt FROM photos WHERE status IN ('pool', 'in_set') GROUP BY country_iso2 ORDER BY cnt DESC, country_iso2 ASC"
                ).fetchall()
            ]
        finally:
            conn.close()
        return templates.TemplateResponse(
            request,
            "sets.html",
            {
                "sets": sets,
                "pool_photos": pool_photos,
                "country_counts": country_counts,
                "msg": msg,
                "err": err,
            },
        )

    @app.post("/sets/create")
    def sets_create(
        name: str = Form(...),
        version: int = Form(1),
        description: str = Form(""),
    ):
        conn = _db()
        try:
            create_image_set(conn, name=name, version=version, description=description)
            return RedirectResponse(url="/sets?msg=Created+set", status_code=303)
        except (ValueError, sqlite3.Error) as exc:
            return RedirectResponse(url=f"/sets?err={str(exc).replace(' ', '+')}", status_code=303)
        finally:
            conn.close()

    @app.post("/sets/add-photos")
    def sets_add_photos(
        set_id: str = Form(...),
        photo_ids: Annotated[list[str], Form()] = (),
    ):
        conn = _db()
        try:
            added = add_photos_to_set(conn, set_id, list(photo_ids))
            return RedirectResponse(url=f"/sets?msg=Added+{added}+photo(s)", status_code=303)
        except (ValueError, RuntimeError, sqlite3.Error) as exc:
            return RedirectResponse(url=f"/sets?err={str(exc).replace(' ', '+')}", status_code=303)
        finally:
            conn.close()

    @app.post("/sets/freeze")
    def sets_freeze(set_id: str = Form(...)):
        conn = _db()
        try:
            chash = freeze_set(conn, set_id)
            return RedirectResponse(url=f"/sets?msg=Frozen.+Commitment+{chash[:12]}", status_code=303)
        except (ValueError, RuntimeError, sqlite3.Error) as exc:
            return RedirectResponse(url=f"/sets?err={str(exc).replace(' ', '+')}", status_code=303)
        finally:
            conn.close()

    @app.get("/runs", response_class=HTMLResponse)
    def runs_page(request: Request, msg: str = "", err: str = "", set_id: str = "", model_id: str = "", track: str = "A"):
        conn = _db()
        try:
            sets = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT s.*, COUNT(sm.photo_id) AS member_count
                    FROM image_sets s
                    LEFT JOIN set_members sm ON sm.set_id = s.id
                    GROUP BY s.id
                    ORDER BY s.created_at DESC
                    """
                ).fetchall()
            ]
            models = [
                dict(r)
                for r in conn.execute("SELECT * FROM model_configs ORDER BY vendor, id").fetchall()
            ]
            run_counts = {
                r["model_config_id"]: r["cnt"]
                for r in conn.execute(
                    "SELECT model_config_id, COUNT(*) AS cnt FROM runs GROUP BY model_config_id"
                ).fetchall()
            }
            for m in models:
                m["run_count"] = run_counts.get(m["id"], 0)
            jobs = []
            for r in conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 30").fetchall():
                d = dict(r)
                d["payload"] = json.loads(d["payload_json"])
                jobs.append(d)
            has_active_jobs = any(j["state"] in ("pending", "running") for j in jobs)
            selected_set = set_id or (sets[0]["id"] if sets else "")
            selected_model = model_id or (models[0]["id"] if models else "")
            guard = None
            estimate = None
            if selected_set and selected_model:
                guard = check_vendor_rotation_guard(conn, selected_set, selected_model)
                estimate = estimate_run_cost(conn, selected_set, selected_model, track)
        finally:
            conn.close()
        return templates.TemplateResponse(
            request,
            "runs.html",
            {
                "sets": sets,
                "models": models,
                "reasoning_efforts": VALID_REASONING_EFFORTS,
                "jobs": jobs,
                "has_active_jobs": has_active_jobs,
                "selected_set": selected_set,
                "selected_model": selected_model,
                "selected_track": track,
                "guard": guard,
                "estimate": estimate,
                "msg": msg,
                "err": err,
            },
        )

    @app.post("/models/add")
    def models_add(
        model_id: str = Form(...),
        vendor: str = Form(...),
        family: str = Form(...),
        tier: str = Form("fast"),
        openrouter_id: str = Form(...),
        reasoning_effort: str = Form("minimal"),
    ):
        conn = _db()
        try:
            effort = normalize_reasoning_effort(reasoning_effort)
            conn.execute(
                """
                INSERT INTO model_configs (id, vendor, family, tier, openrouter_id, reasoning_effort, params_json)
                VALUES (?, ?, ?, ?, ?, ?, '{}')
                """,
                (
                    model_id.strip(),
                    vendor.strip().lower(),
                    family.strip(),
                    tier.strip(),
                    normalize_openrouter_id(openrouter_id),
                    effort,
                ),
            )
            conn.commit()
            return RedirectResponse(url="/runs?msg=Added+model+config", status_code=303)
        except (ValueError, sqlite3.Error) as exc:
            return RedirectResponse(url=f"/runs?err={str(exc).replace(' ', '+')}", status_code=303)
        finally:
            conn.close()

    @app.post("/models/update")
    def models_update(
        model_id: str = Form(...),
        openrouter_id: str = Form(...),
        reasoning_effort: str = Form(""),
    ):
        conn = _db()
        try:
            effort = normalize_reasoning_effort(reasoning_effort)
            conn.execute(
                "UPDATE model_configs SET openrouter_id = ?, reasoning_effort = ? WHERE id = ?",
                (normalize_openrouter_id(openrouter_id), effort, model_id),
            )
            conn.commit()
            return RedirectResponse(url="/runs?msg=Updated+model+config", status_code=303)
        except (ValueError, sqlite3.Error) as exc:
            return RedirectResponse(url=f"/runs?err={str(exc).replace(' ', '+')}", status_code=303)
        finally:
            conn.close()

    @app.post("/models/delete")
    def models_delete(model_id: str = Form(...)):
        conn = _db()
        try:
            run_cnt = conn.execute(
                "SELECT COUNT(*) AS c FROM runs WHERE model_config_id = ?", (model_id,)
            ).fetchone()["c"]
            if run_cnt > 0:
                return RedirectResponse(
                    url=f"/runs?err=Cannot+delete+{model_id}:+{run_cnt}+recorded+run(s)+reference+it",
                    status_code=303,
                )
            active_jobs = conn.execute(
                "SELECT payload_json FROM jobs WHERE state IN ('pending', 'running')"
            ).fetchall()
            if any(json.loads(j["payload_json"]).get("model_config_id") == model_id for j in active_jobs):
                return RedirectResponse(
                    url=f"/runs?err=Cannot+delete+{model_id}:+pending+or+running+job+references+it",
                    status_code=303,
                )
            conn.execute("DELETE FROM model_configs WHERE id = ?", (model_id,))
            conn.commit()
            return RedirectResponse(url=f"/runs?msg=Deleted+model+{model_id}", status_code=303)
        except sqlite3.Error as exc:
            return RedirectResponse(url=f"/runs?err={str(exc).replace(' ', '+')}", status_code=303)
        finally:
            conn.close()

    @app.post("/runs/enqueue")
    def runs_enqueue(
        set_id: str = Form(...),
        track: str = Form("A"),
        model_config_id: str = Form(...),
        override_reason: str = Form(""),
    ):
        conn = _db()
        try:
            job_id = enqueue_run_job(
                conn,
                set_id=set_id,
                track=track,
                model_config_id=model_config_id,
                override_reason=override_reason or None,
            )
            return RedirectResponse(url=f"/runs?msg=Queued+job+{job_id[:8]}", status_code=303)
        except (ValueError, RuntimeError, sqlite3.Error) as exc:
            return RedirectResponse(url=f"/runs?err={str(exc).replace(' ', '+')}", status_code=303)
        finally:
            conn.close()

    @app.post("/runs/rerun-photo")
    def runs_rerun_photo(
        set_id: str = Form(...),
        track: str = Form(...),
        model_config_id: str = Form(...),
        photo_id: str = Form(...),
    ):
        conn = _db()
        try:
            job_id = enqueue_run_job(
                conn,
                set_id=set_id,
                track=track,
                model_config_id=model_config_id,
                photo_id=photo_id,
                override_reason="single-photo-rerun",
            )
            return RedirectResponse(url=f"/runs?msg=Queued+photo+rerun+{job_id[:8]}", status_code=303)
        except (ValueError, RuntimeError, sqlite3.Error) as exc:
            return RedirectResponse(url=f"/runs?err={str(exc).replace(' ', '+')}", status_code=303)
        finally:
            conn.close()

    @app.post("/jobs/cancel")
    def jobs_cancel(job_id: str = Form(...)):
        conn = _db()
        try:
            conn.execute(
                "UPDATE jobs SET state = 'cancelled' WHERE id = ? AND state = 'pending'",
                (job_id,),
            )
            conn.commit()
            return RedirectResponse(url="/runs?msg=Cancelled+pending+job", status_code=303)
        finally:
            conn.close()

    @app.post("/jobs/retry")
    def jobs_retry(job_id: str = Form(...)):
        conn = _db()
        try:
            conn.execute(
                "UPDATE jobs SET state = 'pending', error = NULL WHERE id = ? AND state = 'failed'",
                (job_id,),
            )
            conn.commit()
            return RedirectResponse(url="/runs?msg=Re-queued+failed+job", status_code=303)
        finally:
            conn.close()

    @app.get("/results", response_class=HTMLResponse)
    def results_page(request: Request, set_id: str = "", msg: str = "", err: str = ""):
        conn = _db()
        try:
            sets = [
                dict(s)
                for s in conn.execute(
                    "SELECT id, name, version, frozen_at FROM image_sets ORDER BY created_at DESC"
                ).fetchall()
            ]
            selected_set = set_id
            if not selected_set and sets:
                frozen = next((s["id"] for s in sets if s["frozen_at"] is not None), None)
                selected_set = frozen or sets[0]["id"]

            where_set = "WHERE r.set_id = ?" if (selected_set and selected_set != "all") else ""
            set_params: tuple = (selected_set,) if where_set else ()

            runs = conn.execute(
                f"""
                SELECT r.*, s.name AS set_name, s.version AS set_version,
                       mc.vendor, mc.openrouter_id, mc.reasoning_effort
                FROM runs r
                JOIN image_sets s ON s.id = r.set_id
                JOIN model_configs mc ON mc.id = r.model_config_id
                {where_set}
                ORDER BY r.started_at ASC
                """,
                set_params,
            ).fetchall()

            official_a: list[dict] = []
            official_b: list[dict] = []
            history_runs: list[dict] = []

            for r in runs:
                r_dict = dict(r)
                r_dict["model_label"] = format_model_label(
                    r_dict["openrouter_id"], r_dict.get("reasoning_effort")
                )
                all_res = [
                    dict(x)
                    for x in conn.execute(
                        "SELECT * FROM results WHERE run_id = ?", (r["id"],)
                    ).fetchall()
                ]
                official_res = [x for x in all_res if x["is_official"] == 1]
                agg_all = aggregate_run_metrics(all_res)
                agg_all["total_searches"] = sum(int(x.get("tool_calls_count") or 0) for x in all_res)

                is_official_run = len(official_res) > 0
                history_runs.append(
                    {
                        "run": r_dict,
                        "agg": agg_all,
                        "is_official_run": is_official_run,
                        "official_count": len(official_res),
                        "total_count": len(all_res),
                    }
                )

                if is_official_run:
                    agg_off = aggregate_run_metrics(official_res)
                    agg_off["total_searches"] = sum(
                        int(x.get("tool_calls_count") or 0) for x in official_res
                    )
                    item = {"run": r_dict, "agg": agg_off}
                    if r["track"] == "A":
                        official_a.append(item)
                    else:
                        official_b.append(item)

            sort_key = lambda x: (
                -x["agg"]["strict_mean_points"],
                x["agg"]["median_distance_km"] if x["agg"]["median_distance_km"] is not None else 999999.0,
            )
            official_a.sort(key=sort_key)
            official_b.sort(key=sort_key)
            history_runs.reverse()

            map_where = "WHERE res.is_official = 1" + (" AND r.set_id = ?" if where_set else "")
            map_rows = [
                dict(x)
                for x in conn.execute(
                    f"""
                    SELECT res.photo_id, res.pred_lat, res.pred_lon, res.pred_country_iso2,
                           res.distance_km, r.track, mc.vendor, mc.openrouter_id, mc.reasoning_effort,
                           p.lat AS gt_lat, p.lon AS gt_lon, p.country_iso2 AS gt_country, p.place_label
                    FROM results res
                    JOIN runs r ON r.id = res.run_id
                    JOIN model_configs mc ON mc.id = r.model_config_id
                    JOIN photos p ON p.id = res.photo_id
                    {map_where}
                    """,
                    set_params,
                ).fetchall()
            ]
            for row in map_rows:
                row["model_label"] = format_model_label(
                    row["openrouter_id"], row.get("reasoning_effort")
                )
            ranking_svg = build_ranking_bars_svg(official_a, official_b)
            map_svg = build_guess_map_svg(map_rows, "Official Results — Ground Truth vs Model Guesses (Local Only)")
            bullseye_svg = build_bullseye_error_svg(map_rows, "Relative Distance Error Bullseye (Safe Public View)")
        finally:
            conn.close()
        return templates.TemplateResponse(
            request,
            "results.html",
            {
                "sets": sets,
                "selected_set": selected_set,
                "track_a": official_a,
                "track_b": official_b,
                "history_runs": history_runs,
                "ranking_svg": ranking_svg,
                "map_svg": map_svg,
                "bullseye_svg": bullseye_svg,
                "msg": msg,
                "err": err,
            },
        )

    @app.get("/results/{run_id}", response_class=HTMLResponse)
    def run_detail_page(request: Request, run_id: str):
        conn = _db()
        try:
            run = conn.execute(
                """
                SELECT r.*, s.name AS set_name, s.version AS set_version,
                       mc.vendor, mc.openrouter_id, mc.reasoning_effort
                FROM runs r
                JOIN image_sets s ON s.id = r.set_id
                JOIN model_configs mc ON mc.id = r.model_config_id
                WHERE r.id = ?
                """,
                (run_id,),
            ).fetchone()
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")
            run_dict = dict(run)
            run_dict["model_label"] = format_model_label(
                run_dict["openrouter_id"], run_dict.get("reasoning_effort")
            )
            rows = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT res.*, p.country_iso2 AS gt_country, p.lat AS gt_lat, p.lon AS gt_lon,
                           p.place_label, p.scene_tag, p.near_border,
                           r_tbl.track, mc.vendor, mc.openrouter_id, mc.reasoning_effort
                    FROM results res
                    JOIN photos p ON p.id = res.photo_id
                    JOIN runs r_tbl ON r_tbl.id = res.run_id
                    JOIN model_configs mc ON mc.id = r_tbl.model_config_id
                    WHERE res.run_id = ?
                    ORDER BY res.photo_id ASC
                    """,
                    (run_id,),
                ).fetchall()
            ]
            for row in rows:
                row["model_label"] = format_model_label(
                    row["openrouter_id"], row.get("reasoning_effort")
                )
            map_svg = build_guess_map_svg(rows, f"Run {run_id[:8]} — Truth vs Guesses")
        finally:
            conn.close()
        return templates.TemplateResponse(
            request, "run_detail.html", {"run": run_dict, "rows": rows, "map_svg": map_svg}
        )

    @app.post("/results/export")
    def results_export():
        conn = _db()
        try:
            jpath, cpath = export_summary_files(conn, config)
            return RedirectResponse(
                url=f"/results?msg=Exported+{jpath.name}+and+{cpath.name}", status_code=303
            )
        finally:
            conn.close()

    @app.post("/results/export-public")
    def results_export_public(set_id: str = Form("")):
        conn = _db()
        try:
            target_set = None if (not set_id or set_id == "all") else set_id
            out_dir = export_public_release(conn, config, set_id=target_set)
            return RedirectResponse(
                url=f"/results?set_id={set_id}&msg=Exported+public+bundle+to+results/{out_dir.name}",
                status_code=303,
            )
        except ValueError as exc:
            return RedirectResponse(
                url=f"/results?err={str(exc).replace(' ', '+')}", status_code=303
            )
        finally:
            conn.close()

    return app
