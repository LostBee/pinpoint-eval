"""SVG chart and coordinate-map generators for Pinpoint results."""

from __future__ import annotations

import html
import math
from typing import Any

from pinpoint.config import format_model_label

# Paired (Track A light, Track B dark) hex colors for high contrast on #0f172a
MODEL_TRACK_COLORS: dict[str, tuple[str, str]] = {
    "openai": ("#6ee7b7", "#059669"),     # light mint / deep emerald
    "google": ("#93c5fd", "#2563eb"),     # light sky / deep royal blue
    "anthropic": ("#fcd34d", "#d97706"),  # light gold / deep amber
    "xai": ("#f9a8d4", "#db2777"),        # light pink / deep magenta
}
MODEL_COLORS: dict[str, str] = {k: v[0] for k, v in MODEL_TRACK_COLORS.items()}


def color_for_vendor(vendor: str, track: str = "A") -> str:
    pair = MODEL_TRACK_COLORS.get(vendor.lower(), ("#cbd5e1", "#64748b"))
    return pair[1] if str(track).upper() == "B" else pair[0]


def build_ranking_bars_svg(
    track_a: list[dict[str, Any]],
    track_b: list[dict[str, Any]],
) -> str:
    """Render side-by-side horizontal bar chart comparing Track A and Track B Strict (/15)."""
    by_model: dict[str, dict[str, Any]] = {}
    for track_label, items in (("A", track_a), ("B", track_b)):
        for item in items:
            r = item["run"]
            key = str(r.get("model_config_id") or r["openrouter_id"])
            label = format_model_label(r["openrouter_id"], r.get("reasoning_effort"))
            entry = by_model.setdefault(
                key,
                {"label": label, "vendor": r["vendor"], "A": None, "B": None},
            )
            if entry[track_label] is None:
                entry[track_label] = item["agg"]

    if not by_model:
        return ""

    entries = sorted(
        by_model.values(),
        key=lambda e: -max(
            (e["A"]["strict_mean_points"] if e["A"] else 0.0),
            (e["B"]["strict_mean_points"] if e["B"] else 0.0),
        ),
    )

    width = 800
    label_w = 275
    bar_max_w = 400
    row_h = 54
    pad_top = 34
    height = pad_top + len(entries) * row_h + 24

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;background:#0f172a;border-radius:8px;border:1px solid #334155;">',
        '<text x="16" y="22" fill="#e2e8f0" font-size="13" font-weight="700">Strict Score (/15) — Track A (light) vs Track B Web Search (dark)</text>',
    ]

    # Vertical gridlines at 0, 5, 10, 15
    for tick in (0, 5, 10, 15):
        tx = label_w + (tick / 15.0) * bar_max_w
        parts.append(f'<line x1="{tx:.1f}" y1="{pad_top}" x2="{tx:.1f}" y2="{height - 18}" stroke="#1e293b" stroke-width="1"/>')
        parts.append(f'<text x="{tx:.1f}" y="{height - 5}" fill="#64748b" font-size="11" text-anchor="middle">{tick}</text>')

    for idx, entry in enumerate(entries):
        y0 = pad_top + idx * row_h + 8
        parts.append(
            f'<text x="16" y="{y0 + 18}" fill="#e2e8f0" font-size="12" font-weight="600">{html.escape(entry["label"])}</text>'
        )
        for sub_idx, trk in enumerate(("A", "B")):
            agg = entry[trk]
            by = y0 + sub_idx * 18
            if agg is None:
                parts.append(
                    f'<text x="{label_w + 6}" y="{by + 11}" fill="#475569" font-size="11">Track {trk}: —</text>'
                )
                continue
            pts = float(agg["strict_mean_points"])
            bw = max(2.0, (pts / 15.0) * bar_max_w)
            col = color_for_vendor(entry["vendor"], trk)
            parts.append(
                f'<rect x="{label_w}" y="{by}" width="{bw:.1f}" height="14" rx="3" fill="{col}"/>'
            )
            parts.append(
                f'<text x="{label_w + bw + 8:.1f}" y="{by + 11}" fill="#cbd5e1" font-size="11">Track {trk}: {pts:.2f}/15 (${agg["total_cost_usd"]:.4f})</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def build_guess_map_svg(points: list[dict[str, Any]], title: str = "Truth vs Model Guesses") -> str:
    """Render a lightweight SVG scatter/error-line map covering all ground-truth and guess coords."""
    valid_pts = [
        p for p in points
        if p.get("gt_lat") is not None and p.get("gt_lon") is not None
    ]
    if not valid_pts:
        return ""

    lats = [float(p["gt_lat"]) for p in valid_pts] + [
        float(p["pred_lat"]) for p in valid_pts if p.get("pred_lat") is not None
    ]
    lons = [float(p["gt_lon"]) for p in valid_pts] + [
        float(p["pred_lon"]) for p in valid_pts if p.get("pred_lon") is not None
    ]

    min_lat = min(lats) - 1.2
    max_lat = max(lats) + 1.2
    min_lon = min(lons) - 1.5
    max_lon = max(lons) + 1.5

    width = 820
    height = 420
    pad = 40

    # Equirectangular scaled by cos(mid_lat) for realistic Europe proportions
    mid_lat_rad = math.radians((min_lat + max_lat) / 2.0)
    cos_lat = max(0.3, math.cos(mid_lat_rad))
    lon_span = max(1.0, (max_lon - min_lon) * cos_lat)
    lat_span = max(1.0, max_lat - min_lat)

    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    scale = min(plot_w / lon_span, plot_h / lat_span)
    x_offset = pad + (plot_w - lon_span * scale) / 2.0
    y_offset = pad + (plot_h - lat_span * scale) / 2.0

    def project(lat: float, lon: float) -> tuple[float, float]:
        x = x_offset + ((lon - min_lon) * cos_lat) * scale
        y = y_offset + (max_lat - lat) * scale
        return x, y

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;background:#0f172a;border-radius:8px;border:1px solid #334155;">',
        f'<text x="16" y="22" fill="#e2e8f0" font-size="13" font-weight="700">{html.escape(title)}</text>',
    ]

    # Lat/lon graticule every 5 degrees
    for g_lat in range(int(math.floor(min_lat / 5) * 5), math.ceil(max_lat) + 1, 5):
        if min_lat <= g_lat <= max_lat:
            _, gy = project(float(g_lat), min_lon)
            parts.append(f'<line x1="{pad}" y1="{gy:.1f}" x2="{width - pad}" y2="{gy:.1f}" stroke="#1e293b" stroke-width="1"/>')
            parts.append(f'<text x="6" y="{gy + 3:.1f}" fill="#475569" font-size="10">{g_lat}°N</text>')
    for g_lon in range(int(math.floor(min_lon / 5) * 5), math.ceil(max_lon) + 1, 5):
        if min_lon <= g_lon <= max_lon:
            gx, _ = project(min_lat, float(g_lon))
            lon_lbl = f"{abs(g_lon)}°W" if g_lon < 0 else f"{g_lon}°E"
            parts.append(f'<line x1="{gx:.1f}" y1="{pad}" x2="{gx:.1f}" y2="{height - pad}" stroke="#1e293b" stroke-width="1"/>')
            parts.append(f'<text x="{gx:.1f}" y="{height - 10}" fill="#475569" font-size="10" text-anchor="middle">{lon_lbl}</text>')

    # Layer 1: error lines; collect guess markers and unique truth markers
    guess_markers: list[dict[str, Any]] = []
    truth_markers: dict[str, dict[str, Any]] = {}
    seen_vendors: set[str] = set()

    for p in valid_pts:
        gx, gy = project(float(p["gt_lat"]), float(p["gt_lon"]))
        vendor = str(p.get("vendor") or "")
        track = str(p.get("track") or "A")
        col = color_for_vendor(vendor, track)
        seen_vendors.add(vendor)

        pid = str(p.get("photo_id") or f"{gx},{gy}")
        if pid not in truth_markers:
            truth_markers[pid] = {
                "x": gx,
                "y": gy,
                "tip": (
                    f"TRUTH: {p.get('gt_country')} ({p.get('place_label') or pid[:8]}) "
                    f"[{float(p['gt_lat']):.4f}, {float(p['gt_lon']):.4f}]"
                ),
            }

        if p.get("pred_lat") is not None and p.get("pred_lon") is not None:
            px, py = project(float(p["pred_lat"]), float(p["pred_lon"]))
            dist = p.get("distance_km")
            dist_lbl = f"{dist:.2f} km" if dist is not None else "?"
            tip = (
                f"{p.get('model_label', vendor)} · Track {p.get('track', '')} · "
                f"Truth {p.get('gt_country')} vs Guess {p.get('pred_country_iso2') or '?'} ({dist_lbl})"
            )
            parts.append(
                f'<line x1="{gx:.1f}" y1="{gy:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
                f'stroke="{col}" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="3,2"/>'
            )
            guess_markers.append({"x": px, "y": py, "col": col, "tip": tip})

    # Layer 2: spread overlapping guess markers in pixel space so all remain visible
    parts.extend(_render_spread_circles(guess_markers))

    # Layer 3: Ground truth diamonds drawn last so truth is never hidden under guesses
    for tm in truth_markers.values():
        tx, ty = tm["x"], tm["y"]
        parts.append(
            f'<polygon points="{tx:.1f},{ty - 5.5:.1f} {tx + 5.5:.1f},{ty:.1f} {tx:.1f},{ty + 5.5:.1f} {tx - 5.5:.1f},{ty:.1f}" '
            f'fill="#f8fafc" stroke="#0f172a" stroke-width="1.5">'
            f'<title>{html.escape(tm["tip"])}</title></polygon>'
        )

    parts.extend(_render_vendor_legend(seen_vendors, width, truth_label="Ground Truth"))
    parts.append("</svg>")
    return "\n".join(parts)


def _render_spread_circles(
    guess_markers: list[dict[str, Any]],
    threshold_px: float = 8.0,
    radius_px: float = 4.5,
) -> list[str]:
    """Offset overlapping markers radially in pixel space so closely clustered guesses stay visible."""
    out: list[str] = []
    placed: list[tuple[float, float]] = []
    for gm in guess_markers:
        mx, my = gm["x"], gm["y"]
        overlaps = sum(1 for ox, oy in placed if math.hypot(mx - ox, my - oy) < threshold_px)
        if overlaps > 0:
            angle = (overlaps - 1) * (2.0 * math.pi / 6.0) + 0.4
            spread = 6.5 + 2.5 * (overlaps // 6)
            mx += math.cos(angle) * spread
            my += math.sin(angle) * spread
        placed.append((mx, my))
        out.append(
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="{radius_px}" fill="{gm["col"]}" stroke="#0f172a" stroke-width="1">'
            f'<title>{html.escape(gm["tip"])}</title></circle>'
        )
    return out


def _render_vendor_legend(seen_vendors: set[str], width: int, truth_label: str) -> list[str]:
    """Render Ground Truth diamond + Track A/B vendor color swatches across top of SVG."""
    out = [
        '<polygon points="20,30 25,35 20,40 15,35" fill="#f8fafc" stroke="#0f172a" stroke-width="1.2"/>',
        f'<text x="30" y="39" fill="#f8fafc" font-size="11">{html.escape(truth_label)}</text>',
    ]
    x_cursor = 140
    for v in sorted(seen_vendors):
        if not v:
            continue
        col_a = color_for_vendor(v, "A")
        col_b = color_for_vendor(v, "B")
        out.append(f'<circle cx="{x_cursor}" cy="35" r="4.5" fill="{col_a}"/>')
        out.append(f'<circle cx="{x_cursor + 11}" cy="35" r="4.5" fill="{col_b}"/>')
        out.append(f'<text x="{x_cursor + 20}" y="39" fill="#cbd5e1" font-size="11">{html.escape(v)}</text>')
        x_cursor += 96
    out.append(
        f'<text x="{width - 16}" y="22" fill="#94a3b8" font-size="11" text-anchor="end">Light = Track A · Dark = Track B</text>'
    )
    return out


def build_bullseye_error_svg(
    points: list[dict[str, Any]],
    title: str = "Relative Distance Error Bullseye (Truth = 0 km)",
) -> str:
    """Render an origin-centered log-radial bullseye error plot safe for public active sets.

    Reveals zero ground-truth coordinates, place names, country codes, or photo IDs;
    each marker encodes only bearing and distance error relative to (0, 0).
    """
    valid_pts = [
        p for p in points
        if (
            p.get("gt_lat") is not None
            and p.get("gt_lon") is not None
            and p.get("pred_lat") is not None
            and p.get("pred_lon") is not None
        )
    ]
    if not valid_pts:
        return ""

    width = 620
    height = 460
    cx = width / 2.0
    cy = height / 2.0 + 20.0
    r_max = 175.0
    scale_denom = math.log10(1.0 + 1500.0 * 20.0)

    def _km_to_px(km: float) -> float:
        clamped = max(0.0, min(km, 1500.0))
        return r_max * (math.log10(1.0 + clamped * 20.0) / scale_denom)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;background:#0f172a;border-radius:8px;border:1px solid #334155;">',
        f'<text x="16" y="22" fill="#e2e8f0" font-size="13" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{cx - r_max - 12:.1f}" y1="{cy:.1f}" x2="{cx + r_max + 12:.1f}" y2="{cy:.1f}" stroke="#1e293b" stroke-width="1"/>',
        f'<line x1="{cx:.1f}" y1="{cy - r_max - 12:.1f}" x2="{cx:.1f}" y2="{cy + r_max + 12:.1f}" stroke="#1e293b" stroke-width="1"/>',
    ]

    for ring_km, ring_lbl in ((0.1, "100m"), (1.0, "1km"), (10.0, "10km"), (100.0, "100km"), (1000.0, "1000km")):
        rr = _km_to_px(ring_km)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rr:.1f}" fill="none" stroke="#1e293b" stroke-width="1" stroke-dasharray="3,3"/>'
        )
        parts.append(
            f'<text x="{cx + 4:.1f}" y="{cy - rr + 10:.1f}" fill="#64748b" font-size="9">{ring_lbl}</text>'
        )

    for lbl, tx, ty, anc in (
        ("N", cx, cy - r_max - 16, "middle"),
        ("S", cx, cy + r_max + 22, "middle"),
        ("W", cx - r_max - 18, cy + 4, "end"),
        ("E", cx + r_max + 18, cy + 4, "start"),
    ):
        parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" fill="#64748b" font-size="10" text-anchor="{anc}">{lbl}</text>')

    guess_markers: list[dict[str, Any]] = []
    seen_vendors: set[str] = set()
    for p in valid_pts:
        gt_lat = float(p["gt_lat"])
        gt_lon = float(p["gt_lon"])
        pred_lat = float(p["pred_lat"])
        pred_lon = float(p["pred_lon"])
        dx_km = (pred_lon - gt_lon) * 111.32 * max(0.2, math.cos(math.radians(gt_lat)))
        dy_km = (pred_lat - gt_lat) * 111.32
        bearing_rad = math.atan2(dy_km, dx_km)
        dist_km = float(p["distance_km"]) if p.get("distance_km") is not None else math.hypot(dx_km, dy_km)
        r_px = _km_to_px(dist_km)
        mx = cx + math.cos(bearing_rad) * r_px
        my = cy - math.sin(bearing_rad) * r_px

        vendor = str(p.get("vendor") or "")
        track = str(p.get("track") or "A")
        col = color_for_vendor(vendor, track)
        seen_vendors.add(vendor)
        lbl = format_model_label(str(p.get("model_label") or vendor), p.get("reasoning_effort"))
        tip = f"{lbl} · Track {track} · Error {dist_km:.2f} km"
        guess_markers.append({"x": mx, "y": my, "col": col, "tip": tip})

    parts.extend(_render_spread_circles(guess_markers, threshold_px=6.0, radius_px=3.8))
    parts.append(
        f'<polygon points="{cx:.1f},{cy - 5:.1f} {cx + 5:.1f},{cy:.1f} {cx:.1f},{cy + 5:.1f} {cx - 5:.1f},{cy:.1f}" '
        f'fill="#f8fafc" stroke="#0f172a" stroke-width="1.5"><title>Truth Origin (0 km error)</title></polygon>'
    )
    parts.extend(_render_vendor_legend(seen_vendors, width, truth_label="Truth (0 km)"))
    parts.append("</svg>")
    return "\n".join(parts)
