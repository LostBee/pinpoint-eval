"""Image set management, commitment hashing, and vendor rotation guard."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import uuid

from pinpoint.db import utc_now


def create_image_set(
    conn: sqlite3.Connection,
    name: str,
    version: int = 1,
    description: str = "",
) -> str:
    set_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO image_sets (id, name, version, description, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (set_id, name.strip(), version, description.strip() or None, utc_now()),
    )
    conn.commit()
    return set_id


def add_photos_to_set(
    conn: sqlite3.Connection,
    set_id: str,
    photo_ids: list[str],
) -> int:
    row = conn.execute("SELECT frozen_at FROM image_sets WHERE id = ?", (set_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown set_id: {set_id}")
    if row["frozen_at"] is not None:
        raise RuntimeError("Cannot modify a frozen image set; create a new version instead.")

    max_pos_row = conn.execute(
        "SELECT COALESCE(MAX(position), 0) AS m FROM set_members WHERE set_id = ?",
        (set_id,),
    ).fetchone()
    pos = int(max_pos_row["m"])
    added = 0
    for pid in photo_ids:
        pos += 1
        cur = conn.execute(
            "INSERT OR IGNORE INTO set_members (set_id, photo_id, position) VALUES (?, ?, ?)",
            (set_id, pid, pos),
        )
        if cur.rowcount:
            conn.execute(
                "UPDATE photos SET status = 'in_set' WHERE id = ? AND status = 'pool'",
                (pid,),
            )
            added += 1
    conn.commit()
    return added


def compute_commitment_hash(
    conn: sqlite3.Connection,
    set_id: str,
    salt: str,
) -> str:
    """Compute SHA-256 over salt + ordered ground-truth rows + sanitized image hashes."""
    rows = conn.execute(
        """
        SELECT p.id, p.sha256_sanitized, p.lat, p.lon, p.country_iso2, p.continent
        FROM set_members sm
        JOIN photos p ON p.id = sm.photo_id
        WHERE sm.set_id = ?
        ORDER BY sm.position ASC, p.id ASC
        """,
        (set_id,),
    ).fetchall()
    payload = [
        {
            "id": r["id"],
            "sha256": r["sha256_sanitized"],
            "lat": round(float(r["lat"]), 7),
            "lon": round(float(r["lon"]), 7),
            "country_iso2": r["country_iso2"],
            "continent": r["continent"],
        }
        for r in rows
    ]
    blob = f"{salt}|{json.dumps(payload, separators=(',', ':'), sort_keys=True)}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def freeze_set(conn: sqlite3.Connection, set_id: str) -> str:
    """Freeze a set and generate its pre-run commitment hash."""
    row = conn.execute(
        "SELECT frozen_at, commitment_hash FROM image_sets WHERE id = ?", (set_id,)
    ).fetchone()
    if not row:
        raise ValueError(f"Unknown set_id: {set_id}")
    if row["frozen_at"] is not None:
        return str(row["commitment_hash"])

    count = conn.execute(
        "SELECT COUNT(*) AS c FROM set_members WHERE set_id = ?", (set_id,)
    ).fetchone()["c"]
    if count == 0:
        raise ValueError("Cannot freeze an empty image set.")

    salt = secrets.token_hex(16)
    commit_hash = compute_commitment_hash(conn, set_id, salt)
    conn.execute(
        """
        UPDATE image_sets
        SET frozen_at = ?, commitment_hash = ?, commitment_salt = ?
        WHERE id = ?
        """,
        (utc_now(), commit_hash, salt, set_id),
    )
    conn.commit()
    return commit_hash


def check_vendor_rotation_guard(
    conn: sqlite3.Connection,
    set_id: str,
    model_config_id: str,
) -> dict[str, str | bool | None]:
    """Check Requirement 3: warn/block testing a 2nd distinct model from the same vendor on the same set.

    Sweeping reasoning_effort levels on the same OpenRouter model slug is permitted;
    testing a different model slug from the same vendor requires a refreshed image set
    (or an explicit override reason).
    """
    target = conn.execute(
        "SELECT vendor, openrouter_id FROM model_configs WHERE id = ?",
        (model_config_id,),
    ).fetchone()
    if not target:
        raise ValueError(f"Unknown model_config_id: {model_config_id}")

    target_slug = str(target["openrouter_id"]).removeprefix("openrouter/").strip()
    prior_rows = conn.execute(
        """
        SELECT r.id AS run_id, r.model_config_id, mc.openrouter_id
        FROM runs r
        JOIN model_configs mc ON mc.id = r.model_config_id
        WHERE r.set_id = ?
          AND mc.vendor = ?
          AND r.model_config_id != ?
          AND r.status IN ('running', 'completed')
        """,
        (set_id, target["vendor"], model_config_id),
    ).fetchall()

    prior = next(
        (
            row
            for row in prior_rows
            if str(row["openrouter_id"]).removeprefix("openrouter/").strip() != target_slug
        ),
        None,
    )

    if prior:
        return {
            "blocked": True,
            "vendor": target["vendor"],
            "prior_model_id": prior["model_config_id"],
            "message": (
                f"Vendor '{target['vendor']}' already tested on this set with "
                f"'{prior['model_config_id']}' ({prior['openrouter_id']}). Refresh the photo set before testing a 2nd model "
                f"from the same company (or provide an explicit override reason)."
            ),
        }
    return {"blocked": False, "vendor": target["vendor"], "prior_model_id": None, "message": None}
