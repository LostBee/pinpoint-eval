# Phase 1 Runbook (`europe v1`)

> Living operational guide for running **Phase 1** of Pinpoint (`200` European photos across `> 10` countries, fast models swept across OpenRouter `reasoning.effort` levels). Update this document whenever workflow rules, target counts, or export policies change.

---

## Target Scope (Current Defaults)

| Parameter | Current Target | Notes |
|---|---|---|
| Image Set | `europe v1` | Freeze with `SHA-256` commitment before running any model |
| Dataset Size | `200` photos | Target `> 10` European countries (`~20%` urban, `~40%` regional/medium, `~40%` rural/hard) |
| Model Scope | Fast / Flash tier + `reasoning.effort` sweep | OpenRouter values: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| Tracks | **Track A** (Pixels Only) & **Track B** (Tavily Web Search) | Max `5` web searches per photo in Track B |
| Public Export | **Option 1** (`results/europe-v1/`) | Publishes aggregate scores + `bullseye_error.svg`; hides exact coords & salt until retired |

---

## Step 0 — Preflight & Server Startup

1. Confirm `.env` contains valid keys and an appropriate per-job safety cap:
   ```dotenv
   OPENROUTER_API_KEY=...
   TAVILY_API_KEY=...
   PINPOINT_MAX_RUN_COST_USD=5.00
   ```
   *(Raise `PINPOINT_MAX_RUN_COST_USD` before queuing if a `200`-photo run with high reasoning tokens or Track B searches might exceed `$5.00`.)*
2. Start the local Web UI (either double-click `start_ui.bat` or run in PowerShell):
   ```powershell
   .\.venv\Scripts\pinpoint.exe serve
   ```
3. Open `http://127.0.0.1:8000`.

---

## Step 1 — Collect, Verify, and Sanitize `200` Photos (`/`, `/pool`)

1. Drop source photos directly into `data/inbox/` (fastest for large batches) or upload via the form on `/` (`Inbox`).
2. Skip/click **Exclude** on any image containing:
   - Identifiable people/faces or readable vehicle license plates
   - Private homes / interiors you don't want evaluated
   - Any image previously uploaded online (to avoid training/web contamination)
3. For each kept photo on `/`:
   - Verify EXIF GPS using the **Open in OSM** link and correct `Lat` / `Lon` if GPS drifted.
   - Fill `Country` (ISO-2, e.g. `DE`, `FR`, `IT`, `TR`), `Label` (city/area, stored privately), `scene_tag` (`urban`, `rural`, `nature`, `indoor_context`, `no_clue`), and checkboxes (`near_border`, `has_text`, `landmark`).
   - Click **Sanitize & Add to Pool** (moves raw original to `data/vault/` and writes metadata-free `1536 px` baseline JPEG to `data/images/<uuid>.jpg`).
4. Check `/pool` country and scene badges as you go until you reach **`200` photos** across **`> 10` countries** with balanced difficulty.
5. Run the metadata verification check before freezing:
   ```powershell
   .\.venv\Scripts\pinpoint.exe verify-clean
   ```

---

## Step 2 — Assemble & Freeze `europe v1` (`/sets`)

1. On `/sets`, create a set: `name = europe`, `version = 1` (plus short description).
2. Check the pool coverage banner (`photos` and `countries`), select your curated `200` photos, and click **Add Selected Photos**.
3. Verify the final photo count and click **Freeze & Hash**.
   - Freezing permanently locks ground-truth edits for those photos and generates the salted `SHA-256` `commitment_hash`.

---

## Step 3 — Publish Pre-Run Commitment to GitHub (Before Any Eval Runs)

Publish the commitment hash *before* running models so nobody can claim photo selection was tuned after seeing results:

1. On `/results` (select `europe v1`), click **Export Public Bundle (results/)** — or run:
   ```powershell
   .\.venv\Scripts\pinpoint.exe export-public
   ```
2. Verify `results/europe-v1/commitment.json` contains only metadata and `commitment_hash` (zero `commitment_salt`, zero photo filenames, zero coordinates).
3. Commit and push to GitHub:
   ```powershell
   git add results/europe-v1/commitment.json
   git commit -m "Pre-register europe v1 commitment hash"
   git push
   ```

---

## Step 4 — Configure Fast Models & OpenRouter Thinking Levels (`/runs`)

1. On `/runs` under **Model Configs (OpenRouter IDs & Reasoning Effort)**:
   - Verify starter `minimal` configs (`anthropic-fast`, `openai-fast`, `google-fast`, `xai-fast`) use the exact OpenRouter model slugs you want.
2. Add a distinct config row for each model and thinking level you want to evaluate using **Add Model**, matching OpenRouter's `reasoning.effort` values (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`):
   - Example IDs: `openai-terra-pro-medium`, `openai-terra-medium`, `google-flash-low`, `google-flash-high`.
   - You can freely run multiple existing models and thinking levels from the same AI company on `europe v1`—each `model_config_id` gets its own separate `Official` leaderboard entry and is locked against in-place edits once evaluated so previous results are never overwritten.
   - Later in the future, when AI companies release a new generation of models (whose training cutoff could post-date `europe v1`), click **Retire Set** on `/sets` and create `europe v2`.
3. *(Optional)* Smoke-test one photo via single-photo rerun or confirm cost estimates on a fresh config before queuing all `200` photos across multiple effort levels.

---

## Step 5 — Execute Sequential Runs (`/runs`)

1. Select Set **`europe v1`**, Track (**`A`** first, then **`B`**), and Model config, check the estimated USD cost, and click **Queue Sequential Run (1 photo at a time)**.
   - You can queue multiple models/efforts back-to-back; `SequentialJobWorker` executes one queued job at a time in an isolated subprocess, processing `1` photo per request.
2. Monitor progress in **Job Queue** on `/runs` (auto-refreshes every 3s while active).
3. If an API outage/rate limit fails a job, click **Retry** after resolving the issue.
4. Once completed, inspect `/results/{run_id}` for `api_error` or `format_error` samples. Note that the **first** recorded result per `(set, track, model_config_id, photo)` remains `Official` while any single-photo **Rerun** is logged under `Variance`.

---

## Step 6 — Publish Option-1 Results & Back Up Private Data

1. Select **`europe v1`** on `/results` and click **Export Public Bundle (results/)** (or run `.\.venv\Scripts\pinpoint.exe export-public`).
   - Updates `results/europe-v1/` with `summary.json`, `ranking_chart.svg`, `bullseye_error.svg` (Option-1 safe relative error plot), `commitment.json`, and `README.md`.
   - Click **Export Private CSV/JSON** if you also want full local coordinates and per-photo diagnostics saved under gitignored `data/exports/`.
2. Spot-check `results/europe-v1/` (confirm no private coordinates or `map_guesses.svg` were copied), then push results to GitHub:
   ```powershell
   git add results/europe-v1
   git commit -m "Add europe v1 results"
   git push
   ```
3. **Back up private state locally/offline:** back up `data/vault/`, `data/pinpoint.sqlite`, and `data/logs/` (never commit these to the public repo). Do not reveal `commitment_salt` or `map_guesses.svg` until `europe v1` is retired.

---

## Step 7 — Reviewability, Human Baseline & Temporal Controls

1. **Public Representative Sample Split (Section 1 Reviewability):** Maintain a small public calibration split (`25–30` non-active/retired photos with sanitized JPEGs, ground truth, and sample `.eval` logs) so independent reviewers can verify the scorer without NDA access. For full active-set review under NDA, export via:
   ```powershell
   .\.venv\Scripts\pinpoint.exe export-audit --set-id <set-id>
   .\.venv\Scripts\pinpoint.exe verify-commitment <audit-dir>
   ```
2. **Human Completability Baseline (Section 3):** Before locking final comparisons, evaluate `1–2` human evaluators (e.g., an experienced geolocator and a general baseline) on a stratified `50`-photo subset to confirm human solvability and anchor a reference ceiling.
3. **Provenance & Reverse-Image-Search Audit:** Confirm all active photos are original, never-published captures and spot-check a stratified sample with reverse image search so exposure can be documented as `0% tasks public, 0% solutions public`.
4. **Track A vs. Track B Temporal Policy:** Treat **Track A (`Pixels Only`)** as the primary Epoch-auditable benchmark. For **Track B (`Web Search`)**, run all compared models within a tight synchronized date window (and archive `.eval` tool transcripts) to prevent live web-search index drift from invalidating cross-model comparisons.

---

## Revision Log

| Date | Update |
|---|---|
| 2026-11-16 | Initial Phase 1 runbook (`200` photos, `> 10` countries, OpenRouter `reasoning.effort` sweeps, Option-1 public exports). |
| 2026-11-17 | Clarified rotation policy: multiple existing models & thinking levels per vendor allowed on active sets (`europe v1`) without overwriting prior results; retire set & rotate to `v2` only for future model releases. Added latency/token/OpenRouter cost telemetry. |
| 2026-11-24 | Hardened against Epoch AI Rubric v1: `pinpoint_v2` camera-location prompt, enforced Track B 5-call cap, `api_error`/truncation separation, complete-only official leaderboard, NDA `export-audit` + `verify-commitment`, bootstrap 95% CIs, naive baseline floor, and Step 7 reviewability/human-baseline/provenance controls. |
