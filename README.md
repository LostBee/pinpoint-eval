# Pinpoint (`pinpoint-eval`)

A contamination-aware visual geolocation benchmark for Vision-Language Models (VLMs), built on [Inspect AI](https://inspect.aisi.org.uk/), OpenRouter, FastAPI, and SQLite.

## Why Pinpoint?

Public geolocation datasets quickly leak into pre-training corpora or web search indexes. Pinpoint keeps ground-truth images and coordinates in a private local vault while publishing verifiable pre-run commitments and sanitized aggregate benchmarks:

- **Zero-Metadata Sanitization:** Originals stay in a private `vault/`. Photos sent to models are decoded, orientation-baked, resized (`<= 1536 px` long edge), and re-encoded as clean baseline sRGB JPEGs with opaque UUID filenames (`<uuid>.jpg`)—stripping all EXIF, GPS tags, XMP, ICC profiles, HDR gain maps, and Motion Photo MP4 trailers.
- **Pre-Run Cryptographic Commitment:** Freezing an image set (`europe v1`) computes a salted `SHA-256` hash over ordered ground truth + sanitized image hashes before any model runs, proving photo lists were not cherry-picked after evaluation.
- **Rotation Guard (Future Model Releases):** Multiple existing models and `reasoning.effort` configurations from the same AI company can be evaluated side-by-side on an active set (each under its own locked `model_config_id` so previous results are never overwritten). When a vendor later releases a newer generation of models whose training cutoff could post-date the evaluation, the set is marked **Retired** on `/sets` and rotated to a new version (`v2`).
- **Safe Public Release Policy & Auditability:** While a benchmark set is active, `pinpoint export-public` publishes `commitment.json` (hash + full `harness_manifest`, salt withheld), `summary.json` (including `naive_baseline` floor and bootstrap 95% CIs), `ranking_chart.svg`, and a center-relative `bullseye_error.svg` (`0 km` origin) so exact photo coordinates cannot be trilaterated from public artifacts or crawled by Track B web search. For confidential independent review (e.g., under NDA), `pinpoint export-audit` packages sanitized images, ground truth, salt, and `.eval` logs, verifiable with `pinpoint verify-commitment`.

## Two Evaluation Tracks

- **Track A (`Pixels Only`):** Primary Epoch-auditable benchmark—pure visual geolocation (`pinpoint_v2_a`, explicit camera-location target) with zero external tools.
- **Track B (`Web Search`):** Uniform `web_search` tool via Tavily strictly capped at `5` tool calls per photo (`pinpoint_v2_b`). Because live web search indexes drift over time, Track B runs should be executed in synchronized date cohorts.

## OpenRouter Reasoning / Thinking Levels

Each model configuration supports OpenRouter's unified `reasoning.effort` parameter forwarded through Inspect AI (`GenerateConfig.reasoning_effort`) alongside explicit generation parameters (`temperature=0.0`, `max_tokens=16384`, `timeout=180s`):

`none` · `minimal` · `low` · `medium` · `high` · `xhigh` · `max` *(or omitted for model default)*

## Scoring Rubric (Max 15 Points / Photo)

Cumulative rising tier weights computed via WGS84 geodesic distance (`geographiclib`) from predicted coordinates to the **camera/photographer position**:

| Tier | Condition | Points |
|---|---|---:|
| Continent | Match (`TR` evaluated as `EU` for European Thrace / Istanbul `lat >= 40.0°N` and `lon <= 29.15°E` else `AS`; `RU` as `EU` for `lon <= 60.0°E` else `AS`) | +1 |
| Country | Exact ISO-3166 alpha-2 match | +2 |
| `<= 100 km` | Geodesic distance `<= 100 km` | +3 |
| `<= 10 km` | Geodesic distance `<= 10 km` | +4 |
| `<= 1 km` | Geodesic distance `<= 1 km` | +5 |

Additional reported diagnostics: `<= 100 m`, `<= 10 m`, IM2GPS thresholds (`25 / 200 / 750 / 2500 km`), Strict vs Adjusted (non-refusal) scores with bootstrap 95% CIs, Refusal / Truncation / API-error rates, Median km error, Naive Europe-centroid floor, and Smooth log-distance score. Multi-epoch runs (`epochs = 1, 3, 5`) are supported for statistical error bars.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
# Add OPENROUTER_API_KEY (and TAVILY_API_KEY for Track B) to .env

pinpoint init
pinpoint verify-clean
pinpoint serve
```

Open `http://127.0.0.1:8000` to:
1. Upload/ingest photos and verify GPS ground truth + scene tags (`/`, `/pool`)
2. Assemble and freeze sets with a pre-run SHA-256 commitment (`/sets`)
3. Configure OpenRouter model slugs & `reasoning.effort` levels and run strictly 1-photo-at-a-time queued jobs (`/runs`)
4. Inspect leaderboards, tier breakdowns, local maps, and export safe Option-1 bundles to `results/` (`/results`, or run `pinpoint export-public`)

See [docs/phase1-runbook.md](docs/phase1-runbook.md) for the step-by-step Phase 1 checklist, [docs/architecture.md](docs/architecture.md) for the system architecture & scoring methodology, and [docs/design-requirements.md](docs/design-requirements.md) for the core design principles.

## Development & Tests

```powershell
ruff check src tests
pytest -v
```
