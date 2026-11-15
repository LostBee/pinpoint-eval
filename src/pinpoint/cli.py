"""CLI entry point for Pinpoint."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from pinpoint.config import get_config
from pinpoint.db import connect_db, init_db
from pinpoint.ingest import verify_sanitized_image
from pinpoint.runner import export_public_release, export_summary_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pinpoint", description="Pinpoint visual geolocation benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize data directories and SQLite database")

    serve_p = sub.add_parser("serve", help="Start the local FastAPI web UI")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)

    sub.add_parser("verify-clean", help="Verify all sanitized images have zero metadata/MP4 trailers")
    sub.add_parser("export", help="Export private summary.json and results.csv to data/exports/")

    pub_p = sub.add_parser(
        "export-public",
        help="Export leak-free Option-1 results bundle (commitment, summary, bullseye) to results/",
    )
    pub_p.add_argument("--set-id", default=None, help="Optional image_set UUID (defaults to latest frozen set)")

    args = parser.parse_args(argv)
    cfg = get_config()
    conn = connect_db(cfg.db_path)
    init_db(conn)

    if args.command == "init":
        print(f"Initialized Pinpoint data directory at: {cfg.data_dir}")
        conn.close()
        return 0

    if args.command == "verify-clean":
        images = sorted(cfg.images_dir.glob("*.jpg"))
        for img in images:
            verify_sanitized_image(img)
        print(f"Verified {len(images)} sanitized image(s) clean.")
        conn.close()
        return 0

    if args.command == "export":
        jpath, cpath = export_summary_files(conn, cfg)
        print(f"Exported:\n  {jpath}\n  {cpath}")
        conn.close()
        return 0

    if args.command == "export-public":
        out_dir = export_public_release(conn, cfg, set_id=args.set_id)
        print(f"Exported safe public bundle to:\n  {out_dir}")
        conn.close()
        return 0

    if args.command == "serve":
        conn.close()
        uvicorn.run("pinpoint.web.app:create_app", factory=True, host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
