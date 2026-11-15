"""Unit tests for Pixel EXIF extraction, MotionPhoto MP4 stripping, and ingest."""

from pathlib import Path

import pytest
from PIL import Image

from pinpoint.config import PinpointConfig
from pinpoint.db import connect_db, init_db
from pinpoint.ingest import SanitizationError, ingest_photo, sanitize_image, verify_sanitized_image


def _make_dirty_pixel_like_jpeg(path: Path) -> None:
    """Create a JPEG with EXIF tags, orientation, and an appended MP4 MotionPhoto trailer."""
    img = Image.new("RGB", (2400, 1200), color=(40, 120, 200))
    exif = Image.Exif()
    exif[0x010F] = "Google"        # Make
    exif[0x0110] = "Pixel 9"       # Model
    exif[0x0112] = 6               # Orientation: rotate 90 CW
    exif[0x9003] = "2026:05:10 14:22:00"
    img.save(path, format="JPEG", exif=exif, quality=90)

    # Append a fake XMP + MP4 ftyp trailer just like Google Pixel Top Shot / Motion Photos
    with path.open("ab") as f:
        f.write(b"http://ns.adobe.com/xap/1.0/ MotionPhoto ftypmp42\x00\x00\x00\x00mp42isom")


def test_sanitize_strips_exif_orientation_and_motionphoto(tmp_path: Path):
    src = tmp_path / "PXL_20260510_142200123.MP.jpg"
    dst = tmp_path / "clean.jpg"
    _make_dirty_pixel_like_jpeg(src)

    with pytest.raises(SanitizationError):
        verify_sanitized_image(src)

    sha, phash = sanitize_image(src, dst, max_long_edge=1536, quality=85)
    assert len(sha) == 64
    assert len(phash) == 16
    verify_sanitized_image(dst)

    with Image.open(dst) as clean_img:
        # Orientation 6 rotated 2400x1200 -> 1200x2400, then resized to long edge 1536 -> 768x1536
        assert clean_img.size == (768, 1536)


def test_full_ingest_moves_to_vault_and_stores_uuid(tmp_path: Path):
    cfg = PinpointConfig(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    conn = connect_db(cfg.db_path)
    init_db(conn)

    src = cfg.inbox_dir / "PXL_20260510_142200123.jpg"
    _make_dirty_pixel_like_jpeg(src)

    photo_id = ingest_photo(
        conn,
        cfg,
        src,
        lat=49.6116,
        lon=6.1319,
        country_input="Luxembourg",
        place_label="Grund",
        scene_tag="urban",
        near_border=False,
        has_text=True,
        has_landmark=False,
    )

    assert not src.exists()
    assert (cfg.images_dir / f"{photo_id}.jpg").exists()
    assert (cfg.vault_dir / f"{photo_id}.jpg").exists()
    row = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
    assert row["country_iso2"] == "LU"
    assert row["continent"] == "EU"
    assert row["camera_model"] == "Google Pixel 9"
    conn.close()
