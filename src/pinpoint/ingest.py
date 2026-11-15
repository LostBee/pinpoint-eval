"""Photo EXIF extraction, Pixel MotionPhoto/metadata stripping, and ingest."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import ExifTags, Image, ImageOps

from pinpoint.config import PinpointConfig
from pinpoint.db import utc_now
from pinpoint.geo import continent_for_location, normalize_country

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ExifMetadata:
    lat: float | None
    lon: float | None
    gps_accuracy_m: float | None
    taken_date: str | None
    camera_model: str | None


class SanitizationError(RuntimeError):
    """Raised when a sanitized image still contains metadata or trailing payload."""


def _dms_to_decimal(dms: tuple, ref: str) -> float | None:
    try:
        deg, mins, secs = (float(x) for x in dms)
    except (TypeError, ValueError):
        return None
    val = deg + mins / 60.0 + secs / 3600.0
    if ref in ("S", "W"):
        val = -val
    return round(val, 7)


def read_exif_metadata(image_path: Path) -> ExifMetadata:
    """Extract GPS coordinates, capture timestamp, and camera model from a photo."""
    with Image.open(image_path) as img:
        exif = img.getexif()
        if not exif:
            return ExifMetadata(None, None, None, None, None)

        make = str(exif.get(ExifTags.Base.Make, "")).strip()
        model = str(exif.get(ExifTags.Base.Model, "")).strip()
        camera_model = f"{make} {model}".strip() or None

        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        taken_date = (
            exif_ifd.get(ExifTags.Base.DateTimeOriginal)
            or exif.get(ExifTags.Base.DateTime)
            or None
        )
        if taken_date is not None:
            taken_date = str(taken_date).strip()

        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        lat = lon = acc = None
        if gps_ifd:
            lat_dms = gps_ifd.get(ExifTags.GPS.GPSLatitude)
            lat_ref = gps_ifd.get(ExifTags.GPS.GPSLatitudeRef, "N")
            lon_dms = gps_ifd.get(ExifTags.GPS.GPSLongitude)
            lon_ref = gps_ifd.get(ExifTags.GPS.GPSLongitudeRef, "E")
            if lat_dms and lon_dms:
                lat = _dms_to_decimal(lat_dms, str(lat_ref))
                lon = _dms_to_decimal(lon_dms, str(lon_ref))
            h_pos_err = gps_ifd.get(ExifTags.GPS.GPSHPositioningError)
            if h_pos_err is not None:
                try:
                    acc = float(h_pos_err)
                except (TypeError, ValueError):
                    acc = None

        return ExifMetadata(
            lat=lat,
            lon=lon,
            gps_accuracy_m=acc,
            taken_date=taken_date,
            camera_model=camera_model,
        )


def compute_dhash(img: Image.Image) -> str:
    """Compute a 64-bit difference hash for near-duplicate burst detection."""
    gray = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.get_flattened_data())
    bits = 0
    for row in range(8):
        row_start = row * 9
        for col in range(8):
            bits = (bits << 1) | (1 if pixels[row_start + col] > pixels[row_start + col + 1] else 0)
    return f"{bits:016x}"


def sanitize_image(
    src_path: Path,
    dst_path: Path,
    max_long_edge: int = 1536,
    quality: int = 85,
) -> tuple[str, str]:
    """Re-encode pixels to a clean baseline sRGB JPEG, returning (sha256, phash).

    Applies EXIF orientation before stripping so portrait Pixel photos stay upright,
    and discards EXIF, XMP, ICC profiles, HDR gain maps, and MotionPhoto MP4 trailers.
    """
    with Image.open(src_path) as raw:
        transposed = ImageOps.exif_transpose(raw)
        rgb = transposed.convert("RGB")
        phash = compute_dhash(rgb)
        rgb.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)

        # Construct a fresh RGB image from raw pixel bytes so Pillow carries zero ancillary info.
        clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        clean.save(dst_path, format="JPEG", quality=quality, optimize=True)

    verify_sanitized_image(dst_path)
    sha256 = hashlib.sha256(dst_path.read_bytes()).hexdigest()
    return sha256, phash


def verify_sanitized_image(image_path: Path) -> None:
    """Fail loudly if any metadata, XMP, ICC profile, or trailing MP4 data remains."""
    with Image.open(image_path) as img:
        if len(img.getexif()) > 0:
            raise SanitizationError(f"EXIF tags still present in {image_path.name}")
        forbidden_info = {"exif", "icc_profile", "xmp", "photoshop", "XML:com.adobe.xmp"}
        found = forbidden_info.intersection(img.info.keys())
        if found:
            raise SanitizationError(f"Forbidden metadata keys {found} in {image_path.name}")

    raw_bytes = image_path.read_bytes()
    for marker in (b"Exif\x00\x00", b"http://ns.adobe.com/xap/", b"MotionPhoto", b"ftyp"):
        if marker in raw_bytes:
            raise SanitizationError(f"Forbidden byte marker {marker!r} found in {image_path.name}")

    eoi_idx = raw_bytes.rfind(b"\xff\xd9")
    if eoi_idx == -1 or eoi_idx != len(raw_bytes) - 2:
        raise SanitizationError(
            f"Trailing bytes after JPEG EOI in {image_path.name} (eoi={eoi_idx}, len={len(raw_bytes)})"
        )


def ingest_photo(
    conn: sqlite3.Connection,
    cfg: PinpointConfig,
    inbox_file: Path,
    lat: float,
    lon: float,
    country_input: str,
    place_label: str = "",
    scene_tag: str = "urban",
    near_border: bool = False,
    has_text: bool = False,
    has_landmark: bool = False,
    status: str = "pool",
    exclude_reason: str | None = None,
) -> str:
    """Sanitize an inbox photo, move original to private vault, and insert SQLite row."""
    iso2 = normalize_country(country_input)
    if not iso2:
        raise ValueError(f"Unrecognized country: {country_input!r}")
    continent = continent_for_location(iso2, lat, lon) or "OTHER"

    meta = read_exif_metadata(inbox_file)
    photo_id = str(uuid.uuid4())
    dst_sanitized = cfg.images_dir / f"{photo_id}.jpg"
    vault_name = f"{photo_id}{inbox_file.suffix.lower()}"
    dst_vault = cfg.vault_dir / vault_name

    sha256, phash = sanitize_image(
        inbox_file,
        dst_sanitized,
        max_long_edge=cfg.max_long_edge,
        quality=cfg.jpeg_quality,
    )

    existing = conn.execute(
        "SELECT id FROM photos WHERE sha256_sanitized = ?", (sha256,)
    ).fetchone()
    if existing:
        dst_sanitized.unlink(missing_ok=True)
        raise ValueError(f"Duplicate sanitized image already ingested as {existing['id']}")

    shutil.move(str(inbox_file), str(dst_vault))

    conn.execute(
        """
        INSERT INTO photos (
            id, sha256_sanitized, phash, lat, lon, gps_accuracy_m,
            country_iso2, continent, place_label, scene_tag,
            near_border, has_text, has_landmark, status, exclude_reason,
            taken_date, camera_model, vault_filename, added_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            photo_id,
            sha256,
            phash,
            lat,
            lon,
            meta.gps_accuracy_m,
            iso2,
            continent,
            place_label.strip() or None,
            scene_tag,
            1 if near_border else 0,
            1 if has_text else 0,
            1 if has_landmark else 0,
            status,
            exclude_reason,
            meta.taken_date,
            meta.camera_model,
            vault_name,
            utc_now(),
        ),
    )
    conn.commit()
    return photo_id


def list_inbox_files(cfg: PinpointConfig) -> list[Path]:
    return sorted(
        p for p in cfg.inbox_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
