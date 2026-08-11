"""M1 sensor-only 資料 listing、下載、scan 與 integrity CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallsense.data.download import (  # noqa: E402
    OFFICIAL_SENSOR_FOLDER_ID,
    GoogleDriveFileBackend,
    create_manifest,
    download_manifest,
    load_manifest,
    records_for_download,
    save_manifest,
    scan_raw,
    verify_integrity,
)
from fallsense.data.drive import GoogleDriveFolderLister  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="只列官方 folder 並建立 sensor-only manifest")
    list_parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    list_parser.add_argument("--workers", type=int, default=8)

    download_parser = subparsers.add_parser("download", help="依 manifest 下載/resume")
    download_parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    download_parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    download_parser.add_argument("--limit", type=int)
    download_parser.add_argument("--workers", type=int, default=2)

    scan_parser = subparsers.add_parser("scan", help="接納手動放入的檔案")
    scan_parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    scan_parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))

    verify_parser = subparsers.add_parser("verify", help="輸出完整性報告")
    verify_parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    verify_parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        listing = GoogleDriveFolderLister(workers=args.workers).list(OFFICIAL_SENSOR_FOLDER_ID)
        manifest = create_manifest(listing)
        records_for_download(manifest)
        save_manifest(manifest, args.manifest)
        print(
            json.dumps(
                {
                    "folders_visited": listing.folders_visited,
                    "remote_files": len(listing.entries),
                    "download_records": len(manifest.records),
                    "known_missing": manifest.known_missing,
                    "known_unusable": manifest.known_unusable,
                    "excluded_camera": manifest.excluded_camera_count,
                },
                ensure_ascii=False,
            )
        )
        return 0

    manifest = load_manifest(args.manifest)
    if args.command == "download":
        completed, skipped = download_manifest(
            manifest,
            args.raw_dir,
            GoogleDriveFileBackend(),
            limit=args.limit,
            workers=args.workers,
        )
        save_manifest(manifest, args.manifest)
        print(json.dumps({"completed": completed, "skipped": skipped}, ensure_ascii=False))
        return (
            0 if all(record.status != "ERROR" for record in manifest.records[: args.limit]) else 1
        )
    if args.command == "scan":
        scan_report = scan_raw(manifest, args.raw_dir)
        save_manifest(manifest, args.manifest)
        print(json.dumps(asdict(scan_report), ensure_ascii=False))
        return 0 if not scan_report.rejected else 1
    integrity_report = verify_integrity(manifest, args.raw_dir)
    print(
        json.dumps(
            asdict(integrity_report) | {"ok": integrity_report.ok},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if integrity_report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
