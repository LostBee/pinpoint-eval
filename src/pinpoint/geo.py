"""Geodesic distance, country/continent normalization, parsing, and scoring."""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from dataclasses import dataclass
from typing import Any

import pycountry
from geographiclib.geodesic import Geodesic

from pinpoint.config import TIER_WEIGHTS

# European ISO2 codes (geographic Europe; TR & RU handled with coordinate boundary rules).
EUROPEAN_ISO2: frozenset[str] = frozenset(
    {
        "AD", "AL", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ",
        "DE", "DK", "EE", "ES", "FI", "FO", "FR", "GB", "GG", "GI",
        "GR", "HR", "HU", "IE", "IM", "IS", "IT", "JE", "LI", "LT",
        "LU", "LV", "MC", "MD", "ME", "MK", "MT", "NL", "NO", "PL",
        "PT", "RO", "RS", "SE", "SI", "SJ", "SK", "SM", "UA", "VA",
        "XK",
    }
)

# Common English/native/colloquial country aliases -> ISO 3166-1 alpha-2
COUNTRY_ALIASES: dict[str, str] = {
    "uk": "GB",
    "u.k.": "GB",
    "united kingdom": "GB",
    "great britain": "GB",
    "britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "deutschland": "DE",
    "germany": "DE",
    "federal republic of germany": "DE",
    "france": "FR",
    "österreich": "AT",
    "osterreich": "AT",
    "austria": "AT",
    "schweiz": "CH",
    "suisse": "CH",
    "svizzera": "CH",
    "switzerland": "CH",
    "nederland": "NL",
    "holland": "NL",
    "the netherlands": "NL",
    "netherlands": "NL",
    "belgië": "BE",
    "belgie": "BE",
    "belgique": "BE",
    "belgium": "BE",
    "lëtzebuerg": "LU",
    "letzebuerg": "LU",
    "luxembourg": "LU",
    "italia": "IT",
    "italy": "IT",
    "slovensko": "SK",
    "slovakia": "SK",
    "slovak republic": "SK",
    "česko": "CZ",
    "czechia": "CZ",
    "czech republic": "CZ",
    "españa": "ES",
    "espana": "ES",
    "spain": "ES",
    "türkiye": "TR",
    "turkiye": "TR",
    "turkey": "TR",
    "republic of turkey": "TR",
    "kosovo": "XK",
    "vatican": "VA",
    "vatican city": "VA",
    "holy see": "VA",
    "russia": "RU",
    "russian federation": "RU",
    "macedonia": "MK",
    "north macedonia": "MK",
    "bosnia": "BA",
    "bosnia and herzegovina": "BA",
}

_EXTRA_EU_ISO2 = frozenset({"AX"})
_ASIAN_ISO2 = frozenset(
    {
        "AE", "AF", "AM", "AZ", "BD", "BH", "BN", "BT", "CC", "CN", "CX", "GE",
        "HK", "ID", "IL", "IN", "IO", "IQ", "IR", "JO", "JP", "KG", "KH", "KP",
        "KR", "KW", "KZ", "LA", "LB", "LK", "MM", "MN", "MO", "MV", "MY", "NP",
        "OM", "PH", "PK", "PS", "QA", "SA", "SG", "SY", "TH", "TJ", "TL", "TM",
        "TW", "UZ", "VN", "YE",
    }
)
_AFRICAN_ISO2 = frozenset(
    {
        "AO", "BF", "BI", "BJ", "BW", "CD", "CF", "CG", "CI", "CM", "CV", "DJ",
        "DZ", "EG", "EH", "ER", "ET", "GA", "GH", "GM", "GN", "GQ", "GW", "KE",
        "KM", "LR", "LS", "LY", "MA", "MG", "ML", "MR", "MU", "MW", "MZ", "NA",
        "NE", "NG", "RE", "RW", "SC", "SD", "SH", "SL", "SN", "SO", "SS", "ST",
        "SZ", "TD", "TG", "TN", "TZ", "UG", "YT", "ZA", "ZM", "ZW",
    }
)
_NORTH_AMERICAN_ISO2 = frozenset(
    {
        "AG", "AI", "AW", "BB", "BL", "BM", "BQ", "BS", "BZ", "CA", "CR", "CU",
        "CW", "DM", "DO", "GD", "GL", "GP", "GT", "HN", "HT", "JM", "KN", "KY",
        "LC", "MF", "MQ", "MS", "MX", "NI", "PA", "PM", "PR", "SV", "SX", "TC",
        "TT", "US", "VC", "VG", "VI",
    }
)
_SOUTH_AMERICAN_ISO2 = frozenset(
    {"AR", "BO", "BR", "CL", "CO", "EC", "FK", "GF", "GY", "PE", "PY", "SR", "UY", "VE"}
)
_OCEANIAN_ISO2 = frozenset(
    {
        "AS", "AU", "CK", "FJ", "FM", "GU", "KI", "MH", "MP", "NC", "NF", "NR",
        "NU", "NZ", "PF", "PG", "PN", "PW", "SB", "TK", "TO", "TV", "UM", "VU",
        "WF", "WS",
    }
)
_ANTARCTIC_ISO2 = frozenset({"AQ", "BV", "GS", "HM", "TF"})

# Generous (min_lat, max_lat, min_lon, max_lon) sanity bounding boxes for European countries.
EUROPEAN_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "AD": (42.3, 42.8, 1.3, 1.9),
    "AL": (39.5, 42.8, 19.1, 21.2),
    "AT": (46.2, 49.2, 9.4, 17.3),
    "BA": (42.4, 45.4, 15.6, 19.8),
    "BE": (49.4, 51.6, 2.4, 6.5),
    "BG": (41.1, 44.3, 22.2, 28.8),
    "BY": (51.1, 56.3, 23.0, 32.9),
    "CH": (45.7, 47.9, 5.8, 10.6),
    "CY": (34.5, 35.8, 32.1, 34.7),
    "CZ": (48.4, 51.2, 11.9, 19.0),
    "DE": (47.1, 55.2, 5.7, 15.2),
    "DK": (54.4, 57.9, 7.9, 15.3),
    "EE": (57.4, 59.8, 21.6, 28.3),
    "ES": (27.5, 43.9, -18.5, 4.5),
    "FI": (59.7, 70.2, 19.4, 31.7),
    "FO": (61.3, 62.5, -7.8, -6.1),
    "FR": (41.2, 51.2, -5.3, 9.7),
    "GB": (49.8, 61.0, -8.8, 1.9),
    "GG": (49.3, 49.8, -2.8, -2.1),
    "GI": (36.0, 36.2, -5.4, -5.3),
    "GR": (34.7, 41.9, 19.2, 29.8),
    "HR": (42.3, 46.7, 13.3, 19.6),
    "HU": (45.6, 48.7, 16.0, 23.0),
    "IE": (51.3, 55.5, -10.8, -5.8),
    "IM": (54.0, 54.5, -4.9, -4.2),
    "IS": (63.2, 66.7, -24.7, -13.3),
    "IT": (35.4, 47.2, 6.5, 18.7),
    "JE": (49.1, 49.3, -2.3, -1.9),
    "LI": (47.0, 47.3, 9.4, 9.7),
    "LT": (53.8, 56.5, 20.9, 26.9),
    "LU": (49.4, 50.2, 5.7, 6.6),
    "LV": (55.6, 58.2, 20.9, 28.3),
    "MC": (43.7, 43.8, 7.3, 7.5),
    "MD": (45.4, 48.6, 26.5, 30.2),
    "ME": (41.8, 43.6, 18.4, 20.4),
    "MK": (40.8, 42.5, 20.4, 23.1),
    "MT": (35.7, 36.1, 14.1, 14.6),
    "NL": (50.7, 53.7, 3.2, 7.3),
    "NO": (57.8, 71.3, 4.4, 31.3),
    "PL": (48.9, 55.0, 14.0, 24.3),
    "PT": (32.3, 42.2, -31.5, -6.1),
    "RO": (43.5, 48.4, 20.2, 29.9),
    "RS": (42.2, 46.3, 18.8, 23.1),
    "RU": (41.0, 82.0, 19.0, 180.0),
    "SE": (55.2, 69.2, 10.9, 24.3),
    "SI": (45.3, 47.0, 13.3, 16.7),
    "SJ": (74.0, 81.0, 9.0, 34.0),
    "SK": (47.7, 49.7, 16.8, 22.7),
    "SM": (43.8, 44.0, 12.3, 12.6),
    "TR": (35.7, 42.2, 25.6, 44.9),
    "UA": (44.3, 52.5, 22.0, 40.3),
    "VA": (41.8, 42.0, 12.4, 12.5),
    "XK": (41.8, 43.3, 19.9, 21.9),
}


def validate_coord_country(lat: float, lon: float, country_iso2: str) -> tuple[bool, str | None]:
    """Sanity-check that (lat, lon) are valid WGS84 coords and within the country's bounding box."""
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return False, f"Coordinates out of range: lat={lat}, lon={lon}"
    iso = country_iso2.upper()
    bbox = EUROPEAN_BBOXES.get(iso)
    if bbox is None:
        return True, None
    min_lat, max_lat, min_lon, max_lon = bbox
    margin = 0.5
    if not ((min_lat - margin) <= lat <= (max_lat + margin) and (min_lon - margin) <= lon <= (max_lon + margin)):
        return (
            False,
            (
                f"Coordinates ({lat:.4f}, {lon:.4f}) fall outside expected bounding box for {iso} "
                f"[{min_lat}..{max_lat}, {min_lon}..{max_lon}]"
            ),
        )
    return True, None


def european_country_choices() -> list[tuple[str, str]]:
    """Return sorted (ISO2, display_name) pairs for UI autocomplete."""
    items: list[tuple[str, str]] = []
    for code in sorted(EUROPEAN_ISO2):
        if code == "XK":
            items.append(("XK", "Kosovo"))
            continue
        c = pycountry.countries.get(alpha_2=code)
        items.append((code, c.name if c else code))
    items.append(("TR", "Türkiye"))
    items.append(("RU", "Russia (European part)"))
    return sorted(items, key=lambda x: x[1])

_REFUSAL_PATTERNS = (
    "i cannot",
    "i can't",
    "unable to",
    "cannot assist",
    "cannot identify",
    "can't identify",
    "against my policy",
    "privacy",
    "sorry, i",
    "i apologize, but i cannot",
    "i'm unable to",
    "i am unable to",
)


@dataclass(frozen=True)
class ParsedPrediction:
    parse_status: str  # ok | unknown | refusal | format_error
    country_iso2: str | None = None
    lat: float | None = None
    lon: float | None = None
    confidence: float | None = None
    reasoning: str | None = None
    error_detail: str | None = None


@dataclass(frozen=True)
class ScoreBreakdown:
    distance_km: float | None
    hit_continent: bool
    hit_country: bool
    hit_100km: bool
    hit_10km: bool
    hit_1km: bool
    hit_100m: bool
    hit_10m: bool
    tier_points: int
    smooth_score: float


def geodesic_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute WGS-84 geodesic distance in kilometres using Karney's algorithm."""
    res = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    return float(res["s12"]) / 1000.0


def _lookup_country_exact(candidate: str) -> str | None:
    cleaned = candidate.strip()
    if not cleaned:
        return None
    lower = cleaned.lower()
    if lower in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[lower]

    if len(cleaned) == 2 and cleaned.isalpha():
        upper = cleaned.upper()
        if upper == "UK":
            return "GB"
        if upper == "XK":
            return "XK"
        if pycountry.countries.get(alpha_2=upper):
            return upper

    if len(cleaned) == 3 and cleaned.isalpha():
        match = pycountry.countries.get(alpha_3=cleaned.upper())
        if match:
            return match.alpha_2

    match = (
        pycountry.countries.get(name=cleaned)
        or pycountry.countries.get(common_name=cleaned)
        or pycountry.countries.get(official_name=cleaned)
    )
    if match:
        return match.alpha_2
    return None


def normalize_country(raw: str | None) -> str | None:
    """Normalize a country code or exact/aliased name to ISO 3166-1 alpha-2 (uppercase)."""
    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None

    exact = _lookup_country_exact(cleaned)
    if exact:
        return exact

    # Handle decorated strings such as "France (Paris)" or "DE - Germany" without risky fuzzy search.
    paren_stripped = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned).strip()
    if paren_stripped and paren_stripped != cleaned:
        exact = _lookup_country_exact(paren_stripped)
        if exact:
            return exact

    return None


def continent_for_location(
    country_iso2: str | None,
    lat: float | None = None,
    lon: float | None = None,
) -> str | None:
    """Determine continent code ('EU', 'AS', 'AF', 'NA', 'SA', 'OC', 'AN') with TR/RU rules.

    - European Türkiye (Eastern Thrace + European Istanbul): TR with lat >= 40.0 and lon <= 29.15 -> 'EU', else 'AS'.
    - European Russia: RU with lon <= 60.0 -> 'EU', else 'AS'.
    """
    if not country_iso2:
        return None
    iso = country_iso2.upper()
    if iso == "TR":
        if lat is not None and lon is not None and lat >= 40.0 and lon <= 29.15:
            return "EU"
        return "AS"
    if iso == "RU":
        if lon is not None and lon <= 60.0:
            return "EU"
        return "AS"
    if iso in EUROPEAN_ISO2 or iso in _EXTRA_EU_ISO2:
        return "EU"
    if iso in _ASIAN_ISO2:
        return "AS"
    if iso in _AFRICAN_ISO2:
        return "AF"
    if iso in _NORTH_AMERICAN_ISO2:
        return "NA"
    if iso in _SOUTH_AMERICAN_ISO2:
        return "SA"
    if iso in _OCEANIAN_ISO2:
        return "OC"
    if iso in _ANTARCTIC_ISO2:
        return "AN"
    return "OTHER"


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = _strip_code_fences(text)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    schema_keys = {"status", "latitude", "longitude", "country"}
    first_dict: dict[str, Any] | None = None
    pos = candidate.find("{")
    while pos != -1:
        try:
            obj, _ = decoder.raw_decode(candidate, pos)
            if isinstance(obj, dict):
                if schema_keys.intersection(obj.keys()):
                    return obj
                if first_dict is None:
                    first_dict = obj
        except json.JSONDecodeError:
            pass
        pos = candidate.find("{", pos + 1)
    return first_dict


def _is_valid_coords(lat: Any, lon: Any) -> bool:
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return False
    if math.isnan(lat_f) or math.isnan(lon_f) or math.isinf(lat_f) or math.isinf(lon_f):
        return False
    return -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0


_TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "length", "max_output_tokens"})


def parse_model_response(raw_text: str | None, stop_reason: str | None = None) -> ParsedPrediction:
    """Parse raw model output into structured prediction, truncation, refusal, or error status."""
    is_truncated = bool(stop_reason and stop_reason.strip().lower() in _TRUNCATION_STOP_REASONS)
    if not raw_text or not raw_text.strip():
        if is_truncated:
            return ParsedPrediction(parse_status="truncated", error_detail=f"truncated ({stop_reason})")
        return ParsedPrediction(parse_status="format_error", error_detail="empty response")

    obj = _extract_json_object(raw_text)
    if obj is None:
        lower = raw_text.lower()
        if any(pat in lower for pat in _REFUSAL_PATTERNS):
            return ParsedPrediction(
                parse_status="refusal",
                reasoning=raw_text.strip()[:500],
                error_detail="model refused",
            )
        if is_truncated:
            return ParsedPrediction(
                parse_status="truncated",
                reasoning=raw_text.strip()[:500],
                error_detail=f"truncated ({stop_reason})",
            )
        return ParsedPrediction(parse_status="format_error", error_detail="no valid JSON object")

    status = str(obj.get("status", "ok")).strip().lower()
    reasoning = str(obj.get("reasoning", "")).strip() or None
    if status == "unknown":
        return ParsedPrediction(
            parse_status="unknown",
            reasoning=reasoning,
            confidence=_safe_confidence(obj.get("confidence")),
        )

    lat = obj.get("latitude")
    lon = obj.get("longitude")
    if not _is_valid_coords(lat, lon):
        return ParsedPrediction(
            parse_status="format_error",
            reasoning=reasoning,
            error_detail=f"invalid coordinates: lat={lat!r}, lon={lon!r}",
        )

    iso2 = normalize_country(obj.get("country"))
    return ParsedPrediction(
        parse_status="ok",
        country_iso2=iso2,
        lat=float(lat),
        lon=float(lon),
        confidence=_safe_confidence(obj.get("confidence")),
        reasoning=reasoning,
    )


def _safe_confidence(val: Any) -> float | None:
    try:
        c = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(c) or math.isinf(c):
        return None
    return max(0.0, min(1.0, c))


def score_prediction(
    gt_lat: float,
    gt_lon: float,
    gt_country_iso2: str,
    gt_continent: str,
    pred: ParsedPrediction,
) -> ScoreBreakdown:
    """Compute distance, tier booleans, rising-weight tier points (0..15), and smooth score."""
    if pred.parse_status != "ok" or pred.lat is None or pred.lon is None:
        return ScoreBreakdown(
            distance_km=None,
            hit_continent=False,
            hit_country=False,
            hit_100km=False,
            hit_10km=False,
            hit_1km=False,
            hit_100m=False,
            hit_10m=False,
            tier_points=0,
            smooth_score=0.0,
        )

    d_km = geodesic_distance_km(gt_lat, gt_lon, pred.lat, pred.lon)
    pred_continent = continent_for_location(pred.country_iso2, pred.lat, pred.lon)

    hit_continent = pred_continent is not None and pred_continent == gt_continent
    hit_country = pred.country_iso2 is not None and pred.country_iso2.upper() == gt_country_iso2.upper()
    hit_100km = d_km <= 100.0
    hit_10km = d_km <= 10.0
    hit_1km = d_km <= 1.0
    hit_100m = d_km <= 0.1
    hit_10m = d_km <= 0.01

    pts = (
        (TIER_WEIGHTS["continent"] if hit_continent else 0)
        + (TIER_WEIGHTS["country"] if hit_country else 0)
        + (TIER_WEIGHTS["100km"] if hit_100km else 0)
        + (TIER_WEIGHTS["10km"] if hit_10km else 0)
        + (TIER_WEIGHTS["1km"] if hit_1km else 0)
    )
    smooth = 5000.0 * math.exp(-d_km / 1492.7)

    return ScoreBreakdown(
        distance_km=round(d_km, 5),
        hit_continent=hit_continent,
        hit_country=hit_country,
        hit_100km=hit_100km,
        hit_10km=hit_10km,
        hit_1km=hit_1km,
        hit_100m=hit_100m,
        hit_10m=hit_10m,
        tier_points=pts,
        smooth_score=round(smooth, 2),
    )


def bootstrap_ci(
    values: list[float],
    stat: str = "mean",
    n_resamples: int = 1000,
    seed: int = 42,
) -> list[float] | None:
    """Compute deterministic non-parametric bootstrap 95% CI [low, high]."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    fn = statistics.median if stat == "median" else statistics.mean
    samples: list[float] = []
    for _ in range(n_resamples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(float(fn(draw)))
    samples.sort()
    lo_idx = max(0, int(0.025 * (n_resamples - 1)))
    hi_idx = min(n_resamples - 1, int(0.975 * (n_resamples - 1)))
    return [round(samples[lo_idx], 3), round(samples[hi_idx], 3)]


def compute_naive_baseline(
    gt_rows: list[dict[str, Any]],
    ref_lat: float = 50.0,
    ref_lon: float = 10.0,
    ref_country: str = "DE",
    ref_continent: str = "EU",
) -> dict[str, Any]:
    """Compute aggregate floor metrics for a constant Europe-centroid guess (no GT coords exposed)."""
    if not gt_rows:
        return {}
    pred = ParsedPrediction(
        parse_status="ok",
        country_iso2=ref_country,
        lat=ref_lat,
        lon=ref_lon,
        confidence=0.0,
        reasoning="Naive constant baseline",
    )
    scored_rows: list[dict[str, Any]] = []
    for r in gt_rows:
        bd = score_prediction(
            gt_lat=float(r["lat"]),
            gt_lon=float(r["lon"]),
            gt_country_iso2=str(r["country_iso2"]),
            gt_continent=str(r["continent"]),
            pred=pred,
        )
        scored_rows.append(
            {
                "photo_id": r.get("id") or r.get("photo_id"),
                "epoch": 1,
                "parse_status": "ok",
                "distance_km": bd.distance_km,
                "hit_continent": int(bd.hit_continent),
                "hit_country": int(bd.hit_country),
                "hit_100km": int(bd.hit_100km),
                "hit_10km": int(bd.hit_10km),
                "hit_1km": int(bd.hit_1km),
                "hit_100m": int(bd.hit_100m),
                "hit_10m": int(bd.hit_10m),
                "tier_points": bd.tier_points,
                "smooth_score": bd.smooth_score,
            }
        )
    agg = aggregate_run_metrics(scored_rows)
    return {
        "ref_label": f"Constant centroid {ref_country} ({ref_lat}, {ref_lon})",
        "photo_count": agg["count"],
        "strict_mean_points": agg["strict_mean_points"],
        "strict_ci95": agg["strict_ci95"],
        "median_distance_km": agg["median_distance_km"],
        "mean_smooth_score": agg["mean_smooth_score"],
        "tier_rates": agg["tier_rates"],
    }


def aggregate_run_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize Strict vs Adjusted tier points, error/truncation rates, CIs, distances, and cost."""
    n_total = len(rows)
    api_errors = sum(1 for r in rows if r.get("parse_status") == "api_error")
    valid_rows = [r for r in rows if r.get("parse_status") != "api_error"]
    n = len(valid_rows)
    if n == 0:
        return {
            "count": 0,
            "total_rows": n_total,
            "unique_photos": 0,
            "epochs_count": 0,
            "answered_count": 0,
            "refusal_rate": 0.0,
            "unknown_rate": 0.0,
            "format_error_rate": 0.0,
            "truncated_rate": 0.0,
            "api_error_rate": round(api_errors / n_total, 4) if n_total else 0.0,
            "strict_mean_points": 0.0,
            "strict_ci95": None,
            "adjusted_mean_points": 0.0,
            "mean_smooth_score": 0.0,
            "median_distance_km": None,
            "median_distance_ci95": None,
            "tier_rates": {},
            "im2gps_rates": {},
            "total_cost_usd": 0.0,
            "cost_per_photo_usd": 0.0,
            "total_time_s": 0.0,
            "mean_latency_s": 0.0,
            "median_latency_s": 0.0,
            "mean_tokens_in": 0,
            "mean_tokens_cached": 0,
            "mean_tokens_out": 0,
            "mean_tokens_reasoning": 0,
            "max_tokens_reasoning": 0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_tokens_reasoning": 0,
        }

    answered = [r for r in valid_rows if r["parse_status"] == "ok"]
    refusals = sum(1 for r in valid_rows if r["parse_status"] == "refusal")
    unknowns = sum(1 for r in valid_rows if r["parse_status"] == "unknown")
    fmt_errors = sum(1 for r in valid_rows if r["parse_status"] == "format_error")
    truncated = sum(1 for r in valid_rows if r["parse_status"] == "truncated")

    pts_list = [float(r["tier_points"]) for r in valid_rows]
    strict_mean = sum(pts_list) / n
    adjusted_mean = (sum(r["tier_points"] for r in answered) / len(answered)) if answered else 0.0
    mean_smooth = sum(r["smooth_score"] for r in valid_rows) / n
    dists = [float(r["distance_km"]) for r in answered if r["distance_km"] is not None]
    median_km = statistics.median(dists) if dists else None

    unique_photos = len({r["photo_id"] for r in valid_rows if r.get("photo_id")}) or n
    epochs_set = {int(r.get("epoch") or 1) for r in valid_rows}

    tier_keys = ("hit_continent", "hit_country", "hit_100km", "hit_10km", "hit_1km", "hit_100m", "hit_10m")
    tier_rates = {k: sum(1 for r in valid_rows if r[k]) / n for k in tier_keys}
    im2gps_rates = {
        f"{km}km": sum(1 for r in answered if r["distance_km"] is not None and r["distance_km"] <= km) / n
        for km in (25, 200, 750, 2500)
    }

    latencies_s = [float(r.get("latency_ms") or 0) / 1000.0 for r in valid_rows]
    tot_in = sum(int(r.get("tokens_in") or 0) for r in valid_rows)
    tot_cached = sum(int(r.get("tokens_cached") or 0) for r in valid_rows)
    tot_out = sum(int(r.get("tokens_out") or 0) for r in valid_rows)
    reasoning_vals = [int(r.get("tokens_reasoning") or 0) for r in valid_rows]
    tot_reasoning = sum(reasoning_vals)
    tot_cost = sum(float(r.get("cost_usd") or 0.0) for r in rows)

    return {
        "count": n,
        "total_rows": n_total,
        "unique_photos": unique_photos,
        "epochs_count": len(epochs_set),
        "answered_count": len(answered),
        "refusal_rate": refusals / n,
        "unknown_rate": unknowns / n,
        "format_error_rate": fmt_errors / n,
        "truncated_rate": truncated / n,
        "api_error_rate": round(api_errors / n_total, 4) if n_total else 0.0,
        "strict_mean_points": round(strict_mean, 3),
        "strict_ci95": bootstrap_ci(pts_list, stat="mean"),
        "adjusted_mean_points": round(adjusted_mean, 3),
        "mean_smooth_score": round(mean_smooth, 2),
        "median_distance_km": round(median_km, 3) if median_km is not None else None,
        "median_distance_ci95": bootstrap_ci(dists, stat="median"),
        "tier_rates": tier_rates,
        "im2gps_rates": im2gps_rates,
        "total_cost_usd": round(tot_cost, 6),
        "cost_per_photo_usd": round(tot_cost / n, 6),
        "total_time_s": round(sum(latencies_s), 1),
        "mean_latency_s": round(statistics.mean(latencies_s), 2),
        "median_latency_s": round(statistics.median(latencies_s), 2),
        "mean_tokens_in": round(tot_in / n),
        "mean_tokens_cached": round(tot_cached / n),
        "mean_tokens_out": round(tot_out / n),
        "mean_tokens_reasoning": round(tot_reasoning / n),
        "max_tokens_reasoning": max(reasoning_vals),
        "total_tokens_in": tot_in,
        "total_tokens_out": tot_out,
        "total_tokens_reasoning": tot_reasoning,
    }
