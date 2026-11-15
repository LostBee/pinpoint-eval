"""Geodesic distance, country/continent normalization, parsing, and scoring."""

from __future__ import annotations

import json
import math
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


def normalize_country(raw: str | None) -> str | None:
    """Normalize a country code or name to ISO 3166-1 alpha-2 (uppercase)."""
    if not raw:
        return None
    cleaned = raw.strip()
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

    match = pycountry.countries.get(name=cleaned) or pycountry.countries.get(common_name=cleaned)
    if match:
        return match.alpha_2

    try:
        fuzzy = pycountry.countries.search_fuzzy(cleaned)
        if fuzzy:
            return fuzzy[0].alpha_2
    except LookupError:
        pass
    return None


def continent_for_location(
    country_iso2: str | None,
    lat: float | None = None,
    lon: float | None = None,
) -> str | None:
    """Determine continent code ('EU' vs 'OTHER') with Türkiye/Russia boundary rules.

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
    if iso in EUROPEAN_ISO2:
        return "EU"
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

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(candidate[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _is_valid_coords(lat: Any, lon: Any) -> bool:
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return False
    if math.isnan(lat_f) or math.isnan(lon_f) or math.isinf(lat_f) or math.isinf(lon_f):
        return False
    return -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0


def parse_model_response(raw_text: str | None) -> ParsedPrediction:
    """Parse raw model output into structured prediction or refusal/error status."""
    if not raw_text or not raw_text.strip():
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


def aggregate_run_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize Strict vs Adjusted tier points, refusal rate, distances, and cost."""
    n = len(rows)
    if n == 0:
        return {
            "count": 0,
            "answered_count": 0,
            "refusal_rate": 0.0,
            "unknown_rate": 0.0,
            "format_error_rate": 0.0,
            "strict_mean_points": 0.0,
            "adjusted_mean_points": 0.0,
            "mean_smooth_score": 0.0,
            "median_distance_km": None,
            "tier_rates": {},
            "im2gps_rates": {},
            "total_cost_usd": 0.0,
        }

    answered = [r for r in rows if r["parse_status"] == "ok"]
    refusals = sum(1 for r in rows if r["parse_status"] == "refusal")
    unknowns = sum(1 for r in rows if r["parse_status"] == "unknown")
    fmt_errors = sum(1 for r in rows if r["parse_status"] == "format_error")

    strict_mean = sum(r["tier_points"] for r in rows) / n
    adjusted_mean = (sum(r["tier_points"] for r in answered) / len(answered)) if answered else 0.0
    mean_smooth = sum(r["smooth_score"] for r in rows) / n
    dists = [r["distance_km"] for r in answered if r["distance_km"] is not None]
    median_km = statistics.median(dists) if dists else None

    tier_keys = ("hit_continent", "hit_country", "hit_100km", "hit_10km", "hit_1km", "hit_100m", "hit_10m")
    tier_rates = {k: sum(1 for r in rows if r[k]) / n for k in tier_keys}
    im2gps_rates = {
        f"{km}km": sum(1 for r in answered if r["distance_km"] is not None and r["distance_km"] <= km) / n
        for km in (25, 200, 750, 2500)
    }

    return {
        "count": n,
        "answered_count": len(answered),
        "refusal_rate": refusals / n,
        "unknown_rate": unknowns / n,
        "format_error_rate": fmt_errors / n,
        "strict_mean_points": round(strict_mean, 3),
        "adjusted_mean_points": round(adjusted_mean, 3),
        "mean_smooth_score": round(mean_smooth, 2),
        "median_distance_km": round(median_km, 3) if median_km is not None else None,
        "tier_rates": tier_rates,
        "im2gps_rates": im2gps_rates,
        "total_cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 6),
    }
