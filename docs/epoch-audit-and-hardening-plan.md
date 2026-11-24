# Pinpoint (`pinpoint-eval`) — Epoch AI Rubric v1 Audit & Hardening Plan

> **Implementation Status:** **Phases 1–4 Implemented (November 24, 2026)** — Code fixes, `pinpoint_v2` prompts, ingest/set guards, `export-audit` / `verify-commitment` CLIs, bootstrap CIs, multi-epoch support, and documentation updates have been applied and verified with `ruff` and `pytest`.
>
> **Audit Date:** November 24, 2026  
> **Reference Standard:** [Epoch AI Benchmark Reviews Methodology (`Rubric v1`)](./epoch-benchmark-reviews-methodology.md)  
> **Scope:** Full codebase and documentation review (`src/pinpoint/`, `docs/`, `README.md`, `tests/`)

---

## 1. Executive Verdict

If Epoch AI audited `pinpoint-eval` **today** under `Rubric v1`:

1. **As publicly published → `Not enough info` (fails Section 1: Reviewability).**  
   Because active sets keep all images and ground truth private, and public exports (`results/<set>/`) omit key harness/API parameters and lack either a public representative sample or a documented NDA/auditor bundle, Epoch would mark Reviewability **Inadequate** and stop at Section 1.
2. **Under private NDA access today → high risk of `Flawed` (fails Section 2), especially Track B.**  
   While Pinpoint's contamination and immutability architecture is unusually strong, code inspection surfaced **concrete scoring, consistency, and elicitation bugs**—including an unenforced Track B search cap, an `api_error`-on-resume official-score bug, unversioned prompt/scorer mixing on the leaderboard, and uninstructed scoring/ground-truth ambiguities.
3. **Path to `Verified`:**  
   - **Track A (`Pixels Only`)** is close to **`Verified`** once Phase 1–3 fixes land.  
   - **Track B (`Web Search`)** faces a structural Epoch trigger (*dependence on live external state that can drift*) and requires synchronized evaluation cohorts (or a frozen/cached search snapshot) alongside the tool-cap fix.

---

## 2. Rubric v1 Scorecard Summary

| Epoch Rubric Section | Current Status | Key Reason |
|---|---|---|
| **1. Reviewability** | **Inadequate → NEI** | Zero public sample tasks; no documented NDA/auditor bundle (`export-audit`) or `verify-commitment` CLI; public export omits prompts, limits, `temperature`, `max_tokens`, image resize specs, and upstream provider routing. |
| **2a. Scoring** | **At Risk / Flag** | Track B search cap miswired (`max_results` instead of call cap); `api_error` bug poisons resumed official scores; `TR` continent boundary contradicts `README.md`; camera-vs-landmark ground-truth ambiguity; missing/unparseable `country` field penalizes 0 km bullseyes; Track B live web drift. |
| **2b. Benchmark Consistency** | **At Risk / Flag** | `prompt_versions` only hashes Track A and never bumps on edits; leaderboard does not isolate by `prompt_version_id` or `git_commit` and mixes incomplete partial runs with full 200-photo runs. |
| **2c. Elicitation** | **At Risk / Flag** | Unset `max_tokens` / `temperature` / `timeout`; `output_tokens=8192` fallback risks truncating high-reasoning models (silently scored as `format_error = 0`); Track B `message_limit=14` risks involuntary termination; flat `\$5.00` `cost_limit` truncates expensive models mid-set. |
| **2d. Bias in Evaluation Setup** | **At Risk / Flag** | OpenRouter dynamic provider routing/fallbacks unpinned and unlogged; `reasoning_effort` semantics vary by vendor; flat `\$5.00` cost cap affects models unequally; parallel tool callers bypass Track B's intended 5-search cap. |
| **3. Evaluation Quality** | **Mixed** | **Strong:** shared scaffold, contamination resistance. **Missing:** human completability baseline, ≥ 5 runs/model or bootstrap CIs, empirical floor/ceiling (Europe-only prior), provenance/reverse-image-search audit. |

### What Pinpoint Already Gets Right

- **Contamination & tampering controls:** Pixel-only JPEG re-encoding with byte-level EXIF/XMP/MotionPhoto verification (`src/pinpoint/ingest.py:103-149`), pre-run salted `SHA-256` commitments (`src/pinpoint/sets.py:65-123`), and set retirement (`expires_at`) before evaluating future model generations (`src/pinpoint/sets.py:126-190`).
- **Shared scaffold & immutability:** Unified Inspect AI harness across all models (`src/pinpoint/tasks.py:175-197`), locked `model_configs` once evaluated (`src/pinpoint/web/app.py:512-519`), and separation of `Official` vs. `Variance` results.
- **Rich telemetry:** Logs reasoning tokens, cached tokens, latency, refusal rate, strict vs. adjusted means, median km, and IM2GPS thresholds (`src/pinpoint/geo.py:364-446`).

---

## 3. Detailed Audit Findings

### Section 1 — Reviewability Blockers (`Not enough info` Triggers)

1. **No inspectable task sample or NDA auditor workflow:**  
   `export_public_release()` (`src/pinpoint/runner.py:548-675`) withholds all images, ground truth, and `commitment_salt` while a set is active. Epoch's *Partial* reviewability requires at least a **representative sample of tasks + scoring logic** (publicly or privately to reviewers under NDA) **plus full harness/API disclosure**. Currently missing:
   - a small public representative sample/calibration split (e.g., 25–30 non-active/public photos with full ground truth & `.eval` logs),
   - a confidential auditor export command (e.g., `pinpoint export-audit`) bundling sanitized images, ground truth, salt, and `.eval` transcripts for NDA review, and
   - a `pinpoint verify-commitment` CLI command to verify a revealed salt + manifest against `commitment.json`.
2. **Incomplete harness/API disclosure in `results/<set>/summary.json`:**  
   Public exports (`src/pinpoint/runner.py:593-609`) record `openrouter_id` and `reasoning_effort`, but omit `prompt_version_id` / exact prompt text, `git_commit`, `inspect-ai` version, `temperature`, `max_tokens`, `message_limit`, `retry_on_error`, image preprocessing (`max_long_edge=1536`, `jpeg_quality=85`), Tavily tool config, `cost_limit`, and OpenRouter upstream provider routing.

---

### Section 2a — Scoring Defects (`Flawed` Risks)

1. **Track B 5-search cap is NOT enforced (`src/pinpoint/tasks.py:185`):**  
   `src/pinpoint/config.py:64` defines `max_search_calls = 5` and `src/pinpoint/prompts.py:43` instructs the model *"up to 5 searches max"*, but `src/pinpoint/tasks.py:185` passes it to Tavily's `max_results`:
   ```python
   use_tools(web_search({"tavily": {"max_results": cfg.max_search_calls}}))
   ```
   `max_results` caps search hits *per query*, not the number of tool calls. With `message_limit=14` (and parallel tool calls allowed per turn), models can issue more than 5 searches, violating the benchmark spec and introducing uneven search budgets.
2. **`api_error` permanently poisons Official scores on resume (`src/pinpoint/importer.py:154-160, 180-195`):**  
   Samples failing with an API error are stored with `parse_status = "api_error"`, `is_official = 1`, and `tier_points = 0`. When a job resumes, `_official_completed_photo_ids()` (`src/pinpoint/runner.py:190`) excludes `api_error` so the photo is re-run—**but** `import_eval_log()`'s `existing_official` check (`src/pinpoint/importer.py:180-195`) does *not* exclude `api_error`. Consequently, the `api_error` row remains `is_official = 1` (0 pts), the valid retry is demoted to `is_official = 0` (`Variance`), and `aggregate_run_metrics()` (`src/pinpoint/geo.py:395-400`) counts `api_error` as 0 points in `strict_mean_points` without reporting `api_error_rate`.
3. **`TR` continent boundary mismatch & `"OTHER"` fallback (`README.md:31`, `src/pinpoint/geo.py:189-212`):**  
   - `README.md:31` documents `TR` as `EU` for `lon < 29.5°E`, whereas `src/pinpoint/geo.py:196-205` implements `lat >= 40.0 and lon <= 29.15`.  
   - `continent_for_location()` (`src/pinpoint/geo.py:189-212`) maps all non-European countries outside Asian `TR`/`RU` to `"OTHER"`, which breaks continent-level scoring if non-European sets are added.
4. **Uninstructed scoring rules & camera-vs-subject ground-truth ambiguity (`src/pinpoint/prompts.py:23-40`, `src/pinpoint/geo.py:181, 331-334`):**  
   - The prompt never specifies whether coordinates mean **camera/photographer position** or **depicted landmark/subject** (a 1–20 km difference on skylines/bridges/mountains swings `<= 1 km` (+5) and `<= 10 km` (+4) = 9/15 points).  
   - The prompt does not disclose that `"country"` is scored independently of `(lat, lon)`. In `score_prediction()` (`src/pinpoint/geo.py:331-334`), if `normalize_country()` returns `None` (e.g., `"France (Paris)"`), the model loses both `hit_country` (+2) and `hit_continent` (+1) even on a 0 m coordinate bullseye.  
   - `pycountry.countries.search_fuzzy()` (`src/pinpoint/geo.py:181`) risks false-positive substring matches on non-standard country strings.
5. **Fragile JSON extraction & silent truncation (`src/pinpoint/geo.py:223-239`):**  
   `_extract_json_object()` slices from the *first* `{` to the *last* `}`, so stray braces in preamble text or multiple JSON blocks fail parsing. Truncated outputs (`stop_reason == "max_tokens"`) are silently graded as `format_error` (0 pts) with no distinct `truncated` status.
6. **Ingest & ground-truth validation gaps (`src/pinpoint/web/app.py:139-172`, `docs/phase1-runbook.md:45`):**  
   - No automated sanity check verifying `(lat, lon)` falls within `country_iso2` bounds.  
   - No GPS accuracy ceiling/warning for high `gps_accuracy_m` (relevant to `<= 10 m` and `<= 100 m` diagnostics).  
   - The `"no_clue"` scene tag (`docs/phase1-runbook.md:45`) can currently enter scored sets, risking Epoch's *"impossible to correctly answer as written"* flag.
7. **Track B live external state drift:**  
   Epoch's Section 2 explicitly flags benchmarks *"dependent on live external state that can drift"*. Live Tavily web results mutate over time, making Track B scores from different months incomparable unless evaluated in synchronized cohorts or against a cached index.

---

### Section 2b — Benchmark Consistency Defects

1. **Unversioned prompt & scorer changes (`src/pinpoint/db.py:208-215`, `src/pinpoint/importer.py:345-417`):**  
   `init_db()` inserts hardcoded `PROMPT_V1_ID` via `INSERT OR IGNORE` and only hashes Track A (never Track B). `collect_official_leaderboard()` neither filters nor groups by `prompt_version_id` or `git_commit`, so prompt or scorer edits silently mix incompatible runs on one leaderboard.
2. **Incomplete runs ranked alongside complete runs (`src/pinpoint/importer.py:345-417`, `src/pinpoint/runner.py:657`):**  
   `collect_official_leaderboard()` averages whatever `is_official = 1` rows exist without checking `count == set_photo_count`, and the exported Markdown leaderboard (`src/pinpoint/runner.py:657`) omits `N` (photos evaluated).

---

### Section 2c & 2d — Elicitation & Bias in Evaluation Setup Defects

1. **Unspecified generation parameters & unpinned OpenRouter routing (`src/pinpoint/runner.py:288-301`):**  
   `execute_single_job()` omits `temperature`, `max_tokens`, and per-sample `timeout`, and neither pins nor logs the actual OpenRouter upstream provider serving each sample.
2. **Low fallback `output_tokens` cap (`src/pinpoint/runner.py:167-168`):**  
   `ensure_inspect_model_cost()` defaults unknown models to `output_tokens=8192`, risking truncation on `high`/`xhigh`/`max` reasoning sweeps where reasoning tokens share the output budget.
3. **Flat `\$5.00` `cost_limit` & Track B `message_limit=14` (`src/pinpoint/config.py:66`, `src/pinpoint/tasks.py:188`):**  
   A flat `\$5.00` cap with fallback pricing (`0.5/2.0` per 1M tokens) stops expensive models early while letting cheap models complete. On Track B, `message_limit=14` can involuntarily terminate an agent before its final JSON reply.
4. **Sanitized image readability (`1536 px`, JPEG `q=85`):**  
   Small signs/text readable in `data/vault/` originals should be verified on the sanitized JPEG (`data/images/<uuid>.jpg`) during ingest so downscaling does not introduce false-negative unsolvable clues.

---

### Section 3 — Evaluation Quality Gaps (Non-Disqualifying Caveats)

1. **Statistical adequacy (`epochs = 1`):** `src/pinpoint/importer.py:195` marks only `epoch == 1` as `is_official = 1`, with no multi-run averaging (Epoch recommends **≥ 5 runs/model**) and no bootstrap 95% confidence intervals across photos.
2. **Human completability (`Not established`):** No human baseline exists on a representative subset to confirm solvability and anchor a human ceiling.
3. **Score range / Europe-only prior floor:** On `europe v1`, a naive Europe-centroid guess (`~50°N, 10°E`) scores > 1.0/15 automatically; the floor and geographic scope assumption should be documented.
4. **Provenance & reverse-image-search audit:** Document photo provenance (100% original unpublished photos) and spot-check via reverse image search to substantiate `0% tasks public, 0% solutions public`.

---

## 4. Phased Implementation Plan

### Phase 1 — Fix Section 2 Hard Bugs (Scoring & Consistency)

**Goal:** Eliminate every code defect that would trigger an automatic `Flag [stop → Flawed]` under Section 2a and 2b.

1. **Enforce Track B per-sample search cap (`src/pinpoint/tasks.py`, `src/pinpoint/config.py`):**
   - Separate `tavily_max_results: int = 5` (hits per query) from `max_search_calls: int = 5` (tool invocations per photo) in `PinpointConfig`.
   - Wrap `web_search` in a per-sample stateful tool wrapper inside `build_pinpoint_task()` that tracks call count per sample, cleanly returns a limit-reached tool message if `calls > cfg.max_search_calls` (allowing the model to still emit its final JSON), and raises Track B `message_limit` from `14` to `24` (plus a 1-turn format-retry margin) to prevent involuntary termination.
2. **Fix `api_error` official-row poisoning & surface truncation (`src/pinpoint/importer.py`, `src/pinpoint/geo.py`):**
   - Never assign `is_official = 1` to `parse_status == "api_error"` rows (and in `import_eval_log()`, exclude `api_error` from `existing_official` and clean up/demote any legacy `api_error` rows when a valid retry succeeds).
   - Inspect `sample.output.stop_reason` / completion metadata in `importer.py` to classify `max_tokens` cutoffs as `parse_status = "truncated"` instead of `format_error`.
   - Expose `api_error_rate` and `truncated_rate` in `aggregate_run_metrics()` (`src/pinpoint/geo.py`).
3. **Fix continent rules, country normalization, and JSON extraction (`src/pinpoint/geo.py`, `README.md`):**
   - Reconcile the `TR` Europe/Asia boundary across `README.md` and `src/pinpoint/geo.py` (`lat >= 40.0 and lon <= 29.15`, or document exact polygon/threshold consistently).
   - Replace `"OTHER"` in `continent_for_location()` with a full ISO-2 → continent mapping (`EU`, `AS`, `AF`, `NA`, `SA`, `OC`, `AN`), keeping `TR`/`RU` coordinate splits.
   - Harden `_extract_json_object()` using `json.JSONDecoder().raw_decode()` scanning across `{` positions (instead of naive `find("{")..rfind("}")`).
   - Tighten `normalize_country()`: remove or strictly guard `pycountry.countries.search_fuzzy()` to avoid false-positive matches, and consider falling back to `(pred_lat, pred_lon)` only when clearly documented—or explicitly warn/instruct models on exact ISO-2 format.
4. **Fix prompt/scorer versioning & leaderboard completeness (`src/pinpoint/prompts.py`, `src/pinpoint/db.py`, `src/pinpoint/importer.py`, `src/pinpoint/runner.py`):**
   - Compute deterministic hashes for **both** Track A and Track B prompts plus `SCORER_VERSION` (`src/pinpoint/prompts.py`, `src/pinpoint/geo.py`), storing track-specific `prompt_versions` rows in `db.py` rather than ignoring updates under a static `PROMPT_V1_ID`.
   - Update `collect_official_leaderboard()` to check `answered/evaluated count == set_photo_count`: separate incomplete partial runs from complete Official rankings (and display `N / total` plus `prompt_version_id` in UI tables and `results/<set>/README.md`).
   - Add unit & `mockllm` tests in `tests/` covering: >5 Track B search enforcement, `api_error` followed by valid resume (`is_official = 1` on valid retry), `raw_decode` JSON parsing with stray braces, `max_tokens` truncation classification, global continent mapping, and incomplete-run segregation.

---

### Phase 2 — Harden Elicitation, Prompts & Ingest Validation (Section 2)

**Goal:** Eliminate prompt ambiguity, API/provider variance, and unsolvable/misaligned ground truth.

1. **Publish `pinpoint_v2` prompt (`src/pinpoint/prompts.py`):**
   - Explicitly state that target coordinates represent the **camera / photographer position** when taking the photo (not the distant subject/landmark).
   - Explicitly note that both `"country"` (ISO 3166-1 alpha-2) and `(latitude, longitude)` are scored in the rubric so models never omit or decorate `"country"`.
2. **Pin generation parameters & log provider metadata (`src/pinpoint/runner.py`, `src/pinpoint/importer.py`, `src/pinpoint/db.py`):**
   - Pass explicit `temperature` (e.g., `0.0` where supported, or documented per-config in `model_configs.params_json`), generous `max_tokens` (raising the `ensure_inspect_model_cost()` fallback above `8192`, e.g., `32768`, for high-reasoning models), and an explicit per-sample `timeout`.
   - Extract and store OpenRouter upstream `provider` name and `stop_reason` per sample in `results` / `runs` so provider drift or routing differences are auditable.
   - Make `cost_limit` configurable/disabled (`None` or per-tier) for official full-set runs so frontier models are never unevenly truncated mid-set.
3. **Add ingest ground-truth & solvability guards (`src/pinpoint/ingest.py`, `src/pinpoint/web/app.py`, `src/pinpoint/sets.py`):**
   - Add a `coord_basis` field (`camera` default) and a country bounding-box / sanity check on ingest/update to catch swapped lat/lon or wrong ISO-2 entries before freezing.
   - Display a warning badge if `gps_accuracy_m > 25.0` (affecting `<= 10 m` / `<= 100 m` diagnostics) and provide a preview of the **sanitized `1536 px` image** on `/` so human annotators confirm clues remain legible post-resize.
   - Block `scene_tag == "no_clue"` photos from being added to official frozen sets (or require explicit calibration-only exclusion).

---

### Phase 3 — Solve Reviewability Without Leaking Active Sets (Section 1)

**Goal:** Move Section 1 from `Inadequate (Not enough info)` to `Full` / `Partial` without compromising active-set anti-contamination.

1. **Full harness manifest in public exports (`src/pinpoint/runner.py`, `src/pinpoint/sets.py`):**
   - Extend `results/<set>/summary.json`, `commitment.json`, and `README.md` with a `harness_manifest` block disclosing: `prompt_version_id` (Track A & B text + hashes), `scorer_version`, `git_commit`, `inspect_ai` version, `temperature`, `max_tokens`, `message_limit`, `retry_on_error`, `max_long_edge`, `jpeg_quality`, `max_search_calls`, `tavily_max_results`, `cost_limit`, and observed OpenRouter upstream providers.
2. **NDA auditor export & commitment verification CLIs (`src/pinpoint/cli.py`, `src/pinpoint/runner.py`):**
   - Add `pinpoint export-audit --set-id <id>` to produce a confidential, self-contained auditor bundle (sanitized JPEGs, ground-truth manifest, `commitment_salt`, and `.eval` logs) suitable for sharing with independent reviewers like Epoch AI under NDA.
   - Add `pinpoint verify-commitment <bundle_or_dir>` to recompute and assert `SHA-256(salt | payload) == commitment_hash`, enabling both NDA verification today and public post-retirement verification later.
3. **Public representative sample split (`docs/phase1-runbook.md`, `README.md`):**
   - Curate and publish a small, non-active **public sample split** (e.g., 25–30 retired/public photos with sanitized images, ground truth, and sample `.eval` logs) in the repository so any reviewer can inspect representative tasks and run the scorer without NDA access.

---

### Phase 4 — Upgrade Evaluation Quality (Section 3)

**Goal:** Address all non-disqualifying Section 3 caveats so a `Verified` review carries minimal limitations.

1. **Bootstrap confidence intervals & multi-epoch support (`src/pinpoint/geo.py`, `src/pinpoint/importer.py`, `src/pinpoint/runner.py`):**
   - Compute and export paired/non-parametric **bootstrap 95% confidence intervals** across photos for `strict_mean_points` and `median_distance_km` in `aggregate_run_metrics()` and `summary.json`.
   - Support multi-epoch official aggregation (`epochs >= 3–5`, especially for Track B or `temperature > 0`) in `collect_official_leaderboard()`, reporting per-model run count (`runs/model`) and across-run standard error.
2. **Baseline floors & human completability (`docs/phase1-runbook.md`, `src/pinpoint/runner.py`):**
   - Compute and publish the **naive Europe-centroid baseline floor** (and optionally a random-Europe-sample baseline) alongside model scores in `summary.json` and `results/<set>/README.md`.
   - Conduct and document a **human completability baseline** on a stratified 50-photo subset (reporting solve rates, human median km, and any unsolvable items found).
3. **Provenance & Track B temporal scoping (`docs/phase1-runbook.md`, `README.md`):**
   - Document photo provenance and reverse-image-search contamination checks so public/solution exposure can be reported as `0% tasks public, 0% solutions public`.
   - Designate **Track A (`Pixels Only`)** as the primary Epoch-auditable benchmark; for **Track B (`Web Search`)**, enforce synchronized cohort run dates (or freeze/archive Tavily search responses in `.eval` logs) and explicitly caveat temporal web-index drift.

---

## 5. Verification Plan (For Future Implementation Phases)

Every implementation phase must pass:
```powershell
ruff check src tests
pytest -v
```
With dedicated offline tests (`mockllm/model` + synthetic images in `tmp_path`) verifying:
- **Phase 1:** Track B call-cap enforcement (`<= 5` calls regardless of `max_results` or parallel calls); `api_error` superseded by resumed valid sample as `is_official=1`; `max_tokens` truncation tagged `truncated`; JSON `raw_decode` with surrounding braces; global continent codes; incomplete runs segregated from complete Official leaderboard.
- **Phase 2:** `pinpoint_v2` prompt hash registration; explicit `temperature`/`max_tokens`/`timeout` passed to `inspect_eval`; upstream provider & `stop_reason` persisted; ingest country-bounds and `"no_clue"` set-freeze guards.
- **Phase 3:** `harness_manifest` present in `summary.json` without leaking private coords/salt; `pinpoint export-audit` + `pinpoint verify-commitment` round-trip passes.
- **Phase 4:** Bootstrap 95% CI calculation and multi-epoch official aggregation.
