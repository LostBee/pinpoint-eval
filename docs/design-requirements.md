# Pinpoint Design Requirements

Goal: Test different Vision-Language AI models via OpenRouter API to measure how accurately they identify a photo's real-world location strictly from visual evidence (and optionally web search).

## Core Requirements
1. Any image fed to evaluation must be stripped of all metadata and renamed with an opaque UUID by the benchmark tool itself.
2. Location accuracy scoring tracks continent, country, and distance thresholds (`<= 100 km`, `<= 10 km`, `<= 1 km`, plus `<= 100 m` and `<= 10 m` high-precision reporting).
3. Multiple existing models and reasoning-effort levels from the same AI company can be evaluated on an active photo set (each recorded as a distinct `model_config_id` without overwriting prior results); photo sets are retired and rotated (`v2`) before evaluating future model releases whose training cutoffs post-date the current set's evaluation.
4. Phase 1 focuses on curated European locations (`200+` photos across `> 10` countries).
5. Local-first SQLite + FastAPI Web UI for uploading/curating images, annotating ground truth, freezing sets, and reviewing runs.
6. Immutable historical run log separating first-recorded `Official` results from single-photo reruns and repeats (`Variance`).
7. Strictly sequential execution (`1` photo per API call/sample) in a background queue.
