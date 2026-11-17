# Pinpoint (`pinpoint-eval`)

A contamination-aware visual geolocation benchmark for Vision-Language Models (VLMs), built on [Inspect AI](https://inspect.aisi.org.uk/), OpenRouter, FastAPI, and SQLite.

## Why Pinpoint?

Public geolocation datasets quickly leak into pre-training corpora or web search indexes. Pinpoint keeps ground-truth images and coordinates in a private local vault while publishing verifiable pre-run commitments and sanitized aggregate benchmarks:

- **Zero-Metadata Sanitization:** Originals stay in a private `vault/`. Photos sent to models are decoded, orientation-baked, resized (`<= 1536 px` long edge), and re-encoded as clean baseline sRGB JPEGs with opaque UUID filenames (`<uuid>.jpg`)—stripping all EXIF, GPS tags, XMP, ICC profiles, HDR gain maps, and Motion Photo MP4 trailers.
- **Pre-Run Cryptographic Commitment:** Freezing an image set (`europe v1`) computes a salted `SHA-256` hash over ordered ground truth + sanitized image hashes before any model runs, proving photo lists were not cherry-picked after evaluation.
- **Rotation Guard:** Enforces dataset rotation before testing a second distinct model from the same AI vendor on a frozen set (while allowing thinking-effort sweeps on the same model slug).
- **Safe Public Release Policy:** While a benchmark set is active, `pinpoint export-public` publishes `commitment.json` (hash only, salt withheld), `summary.json`, `ranking_chart.svg`, and a center-relative `bullseye_error.svg` (`0 km` origin) so exact photo coordinates cannot be trilaterated from public artifacts or crawled by Track B web search.

## Two Evaluation Tracks

- **Track A (`Pixels Only`):** Pure visual geolocation with zero external tools.
- **Track B (`Web Search`):** Uniform `web_search` tool via Tavily (capped at `5` searches per photo) alongside visual reasoning.

## OpenRouter Reasoning / Thinking Levels

Each model configuration supports OpenRouter's unified `reasoning.effort` parameter forwarded through Inspect AI (`GenerateConfig.reasoning_effort`):

`none` · `minimal` · `low` · `medium` · `high` · `xhigh` · `max` *(or omitted for model default)*

## Scoring Rubric (Max 15 Points / Photo)

Cumulative rising tier weights computed via WGS84 geodesic distance (`geographiclib`):

| Tier | Condition | Points |
|---|---|---:|
| Continent | Match (`TR` evaluated as `EU` for European Thrace / Istanbul `lon < 29.5°E` else `AS`) | +1 |
| Country | Exact ISO-3166 alpha-2 match | +2 |
| `<= 100 km` | Geodesic distance `<= 100 km` | +3 |
| `<= 10 km` | Geodesic distance `<= 10 km` | +4 |
| `<= 1 km` | Geodesic distance `<= 1 km` | +5 |

Additional reported diagnostics: `<= 100 m`, `<= 10 m`, IM2GPS thresholds (`25 / 200 / 750 / 2500 km`), Strict vs Adjusted (non-refusal) scores, Refusal rate, Median km error, and Smooth log-distance score.

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
