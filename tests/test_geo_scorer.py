"""Unit tests for geodesic distance, country normalization, parsing, and scoring."""

from pinpoint.geo import (
    aggregate_run_metrics,
    continent_for_location,
    geodesic_distance_km,
    normalize_country,
    parse_model_response,
    score_prediction,
)


def test_country_aliases_and_iso():
    assert normalize_country("Germany") == "DE"
    assert normalize_country("Deutschland") == "DE"
    assert normalize_country("UK") == "GB"
    assert normalize_country("Great Britain") == "GB"
    assert normalize_country("Österreich") == "AT"
    assert normalize_country("Schweiz") == "CH"
    assert normalize_country("The Netherlands") == "NL"
    assert normalize_country("Lëtzebuerg") == "LU"
    assert normalize_country("Türkiye") == "TR"
    assert normalize_country("Czech Republic") == "CZ"
    assert normalize_country("Slovensko") == "SK"
    assert normalize_country("LU") == "LU"


def test_turkiye_europe_vs_asia_rule():
    # European Istanbul (~41.01, 28.97)
    assert continent_for_location("TR", 41.01, 28.97) == "EU"
    # Ankara (~39.93, 32.85)
    assert continent_for_location("TR", 39.93, 32.85) == "AS"
    # Luxembourg
    assert continent_for_location("LU", 49.61, 6.13) == "EU"
    # Japan
    assert continent_for_location("JP", 35.68, 139.76) == "OTHER"


def test_exact_hit_scores_max_15():
    pred = parse_model_response(
        '{"status": "ok", "country": "LU", "latitude": 49.6116, "longitude": 6.1319, "confidence": 0.9, "reasoning": "Adolphe Bridge"}'
    )
    assert pred.parse_status == "ok"
    s = score_prediction(49.6116, 6.1319, "LU", "EU", pred)
    assert s.distance_km is not None and s.distance_km < 0.01
    assert s.hit_continent and s.hit_country and s.hit_100km and s.hit_10km and s.hit_1km
    assert s.hit_100m and s.hit_10m
    assert s.tier_points == 15
    assert s.smooth_score == 5000.0


def test_distance_boundaries_and_cross_border():
    # Ground truth: Luxembourg City (49.6116, 6.1319)
    # Predicted: ~20 km east in Germany (Trier direction, e.g. 49.75, 6.64 ~ 40 km)
    d = geodesic_distance_km(49.6116, 6.1319, 49.75, 6.64)
    assert 35.0 < d < 45.0

    pred = parse_model_response(
        '```json\n{"status": "ok", "country": "DE", "latitude": 49.75, "longitude": 6.64, "confidence": 0.6, "reasoning": "Moselle"}\n```'
    )
    s = score_prediction(49.6116, 6.1319, "LU", "EU", pred)
    assert s.hit_continent is True
    assert s.hit_country is False
    assert s.hit_100km is True
    assert s.hit_10km is False
    assert s.hit_1km is False
    assert s.tier_points == 1 + 3  # continent(1) + 100km(3) = 4


def test_refusal_unknown_and_format_error():
    ref = parse_model_response("I apologize, but I cannot identify private locations for privacy reasons.")
    assert ref.parse_status == "refusal"

    unk = parse_model_response('{"status": "unknown", "country": "", "latitude": 0, "longitude": 0, "confidence": 0.0, "reasoning": "blank"}')
    assert unk.parse_status == "unknown"

    bad = parse_model_response('{"status": "ok", "latitude": 999, "longitude": 10}')
    assert bad.parse_status == "format_error"

    s_ref = score_prediction(48.85, 2.35, "FR", "EU", ref)
    assert s_ref.tier_points == 0
    assert s_ref.distance_km is None


def test_strict_vs_adjusted_aggregation():
    rows = [
        {"parse_status": "ok", "tier_points": 15, "smooth_score": 5000.0, "distance_km": 0.1,
         "hit_continent": 1, "hit_country": 1, "hit_100km": 1, "hit_10km": 1, "hit_1km": 1, "hit_100m": 0, "hit_10m": 0, "cost_usd": 0.001},
        {"parse_status": "refusal", "tier_points": 0, "smooth_score": 0.0, "distance_km": None,
         "hit_continent": 0, "hit_country": 0, "hit_100km": 0, "hit_10km": 0, "hit_1km": 0, "hit_100m": 0, "hit_10m": 0, "cost_usd": 0.001},
    ]
    agg = aggregate_run_metrics(rows)
    assert agg["strict_mean_points"] == 7.5
    assert agg["adjusted_mean_points"] == 15.0
    assert agg["refusal_rate"] == 0.5
