"""Versioned prompts for Pinpoint."""

from __future__ import annotations

import hashlib
import json

PROMPT_V1_ID = "pinpoint_v1"
PROMPT_V2_A_ID = "pinpoint_v2_a"
PROMPT_V2_B_ID = "pinpoint_v2_b"

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["status", "country", "latitude", "longitude", "confidence", "reasoning"],
    "properties": {
        "status": {"type": "string", "enum": ["ok", "unknown"]},
        "country": {"type": "string"},
        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
        "longitude": {"type": "number", "minimum": -180, "maximum": 180},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
    },
}

_BASE_PROMPT = """You are participating in Pinpoint, an academic visual-geolocation benchmark. All images in this benchmark are anonymized public scenes contributed by the creator specifically for evaluation (no private residences or identifiable people).

Identify the exact camera/photographer location where this photo was taken (not a distant landmark or subject being photographed), based only on visual evidence{track_clause}.

Scoring rubric (max 15 points per photo): cumulative points for continent (+1), ISO 3166-1 alpha-2 country (+2), and geodesic distance from your predicted coordinates to the camera location (<=100 km +3, <=10 km +4, <=1 km +5). Both "country" and ("latitude", "longitude") are graded.

Return ONLY a valid JSON object matching this exact schema (no markdown fences, no extra text):
{{
  "status": "ok",
  "country": "Two-letter ISO 3166-1 alpha-2 code (e.g. FR, DE, LU)",
  "latitude": 0.0,
  "longitude": 0.0,
  "confidence": 0.0,
  "reasoning": "Brief visual clues (1-3 sentences)"
}}

Rules:
- "country" must be a valid two-letter ISO 3166-1 alpha-2 country code (or standard country name).
- "confidence" must be a float from 0.0 (pure guess) to 1.0 (certain).
- Always provide your best coordinate and country estimate even if uncertain.
- Only return "status": "unknown" if the image is completely uninformative (e.g., blank or corrupted)."""

_TRACK_B_CLAUSE = (
    " and web searches via the web_search tool (strictly capped at 5 searches max). "
    "Formulate text queries from visible signs, landmarks, or architectural clues"
)


def prompt_version_id_for_track(track: str) -> str:
    track_upper = track.upper()
    if track_upper == "A":
        return PROMPT_V2_A_ID
    if track_upper == "B":
        return PROMPT_V2_B_ID
    raise ValueError(f"Unsupported track: {track!r} (expected 'A' or 'B')")


def build_prompt(track: str) -> str:
    track_upper = track.upper()
    if track_upper == "A":
        clause = ""
    elif track_upper == "B":
        clause = _TRACK_B_CLAUSE
    else:
        raise ValueError(f"Unsupported track: {track!r} (expected 'A' or 'B')")
    return _BASE_PROMPT.format(track_clause=clause)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def schema_json() -> str:
    return json.dumps(OUTPUT_SCHEMA, sort_keys=True)
