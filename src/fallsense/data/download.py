"""Sensor-only download plan、manifest、resume、scan 與 integrity report。"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Protocol, TypedDict, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fallsense.data.drive import DriveEntry, DriveListing
from fallsense.data.synthetic import sha256_file, trial_filename

OFFICIAL_SENSOR_FOLDER_ID = "1AItqj3Ue-iv7NSdR7Qta1Ez4spRjCo58"
TIMESTAMPS_NAME = "Tagged_TimeStamps.zip"
TRIAL_PATTERN = re.compile(r"^Subject(\d+)Activity(\d+)Trial(\d+)\.csv$")


@dataclass
class ManifestRecord:
    remote_path: str
    file_id: str
    kind: str
    local_path: str
    sha256: str | None = None
    size_bytes: int | None = None
    status: str = "PLANNED"


@dataclass
class DatasetManifest:
    schema_version: int
    source_folder_id: str
    generated_at: str
    expected_sensor_trials: int
    known_missing: list[str]
    known_unusable: list[str]
    folders_visited: int
    remote_file_count: int
    excluded_camera_count: int
    records: list[ManifestRecord]


@dataclass(frozen=True)
class IntegrityReport:
    expected: int
    verified: int
    known_missing: tuple[str, ...]
    known_unusable: tuple[str, ...]
    missing: tuple[str, ...]
    checksum_mismatch: tuple[str, ...]
    unverified: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.checksum_mismatch or self.unverified)


@dataclass(frozen=True)
class ScanReport:
    adopted: tuple[str, ...]
    rejected: tuple[str, ...]


class _RecordPayload(TypedDict):
    remote_path: str
    file_id: str
    kind: str
    local_path: str
    sha256: str | None
    size_bytes: int | None
    status: str


class _ManifestPayload(TypedDict):
    schema_version: int
    source_folder_id: str
    generated_at: str
    expected_sensor_trials: int
    known_missing: list[str]
    known_unusable: list[str]
    folders_visited: int
    remote_file_count: int
    excluded_camera_count: int
    records: list[_RecordPayload]


class DownloadBackend(Protocol):
    def download(self, file_id: str, destination: Path) -> bool: ...


class GoogleDriveFileBackend:
    """公開 file endpoint downloader，支援 timeout、partial resume 與 atomic rename。"""

    def __init__(
        self,
        *,
        retries: int = 3,
        retry_delay_seconds: float = 2.0,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self.timeout_seconds = timeout_seconds

    def download(self, file_id: str, destination: Path) -> bool:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        url = (
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        )
        for attempt in range(self.retries):
            try:
                existing = partial.stat().st_size if partial.exists() else 0
                headers = {"User-Agent": "Mozilla/5.0"}
                if existing:
                    headers["Range"] = f"bytes={existing}-"
                request = Request(url, headers=headers)
                with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" in content_type.casefold():
                        raise RuntimeError("Drive 回傳 HTML 而非檔案")
                    append = existing > 0 and response.status == 206
                    with partial.open("ab" if append else "wb") as handle:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            handle.write(block)
                os.replace(partial, destination)
                return destination.is_file()
            except (HTTPError, URLError, TimeoutError, OSError, RuntimeError):
                if attempt + 1 < self.retries:
                    time.sleep(self.retry_delay_seconds * (attempt + 1))
            if attempt + 1 < self.retries:
                continue
        return False


def expected_trial_names() -> set[str]:
    return {
        trial_filename(subject, activity, trial)
        for subject in range(1, 18)
        for activity in range(1, 12)
        for trial in range(1, 4)
    }


def _validated_sensor_identity(entry: DriveEntry) -> tuple[int, int, int] | None:
    name = PurePosixPath(entry.remote_path).name
    match = TRIAL_PATTERN.fullmatch(name)
    if match is None:
        return None
    subject, activity, trial = (int(value) for value in match.groups())
    if not (1 <= subject <= 17 and 1 <= activity <= 11 and 1 <= trial <= 3):
        return None
    expected_parent = f"Subject{subject}/Activity{activity}/Trial{trial}"
    if str(PurePosixPath(entry.remote_path).parent) != expected_parent:
        return None
    return subject, activity, trial


def is_camera_entry(entry: DriveEntry) -> bool:
    return "camera" in entry.remote_path.casefold()


def is_allowed_entry(entry: DriveEntry) -> bool:
    """唯一下載 allowlist；任何 Camera 路徑先行 hard reject。"""

    if is_camera_entry(entry):
        return False
    if entry.remote_path == TIMESTAMPS_NAME:
        return True
    return _validated_sensor_identity(entry) is not None


def create_manifest(listing: DriveListing) -> DatasetManifest:
    """完整 listing 轉成只含 sensor/timestamps 的 manifest。"""

    allowed = [entry for entry in listing.entries if is_allowed_entry(entry)]
    sensor_entries = [entry for entry in allowed if entry.remote_path != TIMESTAMPS_NAME]
    timestamps = [entry for entry in allowed if entry.remote_path == TIMESTAMPS_NAME]
    if len(timestamps) != 1:
        raise ValueError(f"預期恰好一份 {TIMESTAMPS_NAME}，實際為 {len(timestamps)}")
    names = [PurePosixPath(entry.remote_path).name for entry in sensor_entries]
    if len(names) != len(set(names)):
        raise ValueError("官方 listing 出現重複 sensor trial filename")
    expected = expected_trial_names()
    unknown = set(names) - expected
    if unknown:
        raise ValueError(f"listing 含超出 17×11×3 的 trial：{sorted(unknown)}")

    records = [
        ManifestRecord(
            remote_path=entry.remote_path,
            file_id=entry.file_id,
            kind="timestamps" if entry.remote_path == TIMESTAMPS_NAME else "sensor",
            local_path=entry.remote_path,
        )
        for entry in allowed
    ]
    return DatasetManifest(
        schema_version=2,
        source_folder_id=OFFICIAL_SENSOR_FOLDER_ID,
        generated_at=datetime.now(timezone.utc).isoformat(),
        expected_sensor_trials=len(expected),
        known_missing=sorted(expected - set(names)),
        known_unusable=[],
        folders_visited=listing.folders_visited,
        remote_file_count=len(listing.entries),
        excluded_camera_count=sum(is_camera_entry(entry) for entry in listing.entries),
        records=records,
    )


def save_manifest(manifest: DatasetManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_manifest(path: Path) -> DatasetManifest:
    payload = cast(_ManifestPayload, json.loads(path.read_text(encoding="utf-8")))
    records = [ManifestRecord(**record) for record in payload["records"]]
    return DatasetManifest(
        schema_version=payload["schema_version"],
        source_folder_id=payload["source_folder_id"],
        generated_at=payload["generated_at"],
        expected_sensor_trials=payload["expected_sensor_trials"],
        known_missing=payload["known_missing"],
        known_unusable=payload.get("known_unusable", []),
        folders_visited=payload["folders_visited"],
        remote_file_count=payload["remote_file_count"],
        excluded_camera_count=payload["excluded_camera_count"],
        records=records,
    )


def _valid_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            following_rows = list(islice(reader, 2))
        valid_widths = {len(header), len(header) + 1}
        return len(header) >= 4 and any(
            bool(row) and row[0] != "" and len(row) in valid_widths for row in following_rows
        )
    except (OSError, UnicodeError, StopIteration, csv.Error):
        return False


def _is_official_header_only(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        return (
            len(rows) == 2
            and bool(rows[0])
            and rows[0][0] == "TimeStamps"
            and bool(rows[1])
            and rows[1][0] == ""
        )
    except (OSError, UnicodeError, csv.Error):
        return False


def download_manifest(
    manifest: DatasetManifest,
    raw_dir: Path,
    backend: DownloadBackend,
    *,
    limit: int | None = None,
    workers: int = 1,
) -> tuple[int, int]:
    """下載 manifest；已通過 checksum 的檔案直接跳過。"""

    if workers < 1:
        raise ValueError("workers 必須至少為 1")
    selected = manifest.records if limit is None else manifest.records[:limit]

    def process(record: ManifestRecord) -> str:
        destination = raw_dir / PurePosixPath(record.local_path)
        if destination.is_file():
            valid = record.kind != "sensor" or _valid_csv(destination)
            current_sha = sha256_file(destination)
            if record.sha256 is not None and current_sha == record.sha256 and valid:
                record.status = "VERIFIED"
                return "skipped"
            if record.sha256 is None and valid:
                record.sha256 = current_sha
                record.size_bytes = destination.stat().st_size
                record.status = "VERIFIED"
                return "completed"
            if record.kind == "sensor" and _is_official_header_only(destination):
                record.sha256 = current_sha
                record.size_bytes = destination.stat().st_size
                record.status = "KNOWN_UNUSABLE"
                return "known_unusable"
            record.status = "INVALID"
            return "invalid"
        if not backend.download(record.file_id, destination):
            record.status = "ERROR"
            return "error"
        if record.kind == "sensor" and not _valid_csv(destination):
            record.status = "INVALID"
            return "invalid"
        record.sha256 = sha256_file(destination)
        record.size_bytes = destination.stat().st_size
        record.status = "VERIFIED"
        return "completed"

    if workers == 1:
        outcomes = [process(record) for record in selected]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outcomes = list(executor.map(process, selected))
    manifest.known_unusable = sorted(
        PurePosixPath(record.local_path).name
        for record in manifest.records
        if record.status == "KNOWN_UNUSABLE"
    )
    return outcomes.count("completed"), outcomes.count("skipped")


def scan_raw(manifest: DatasetManifest, raw_dir: Path) -> ScanReport:
    """接納手動放入的合法檔，並指名拒絕重複、壞檔或未知 trial。"""

    by_name = {PurePosixPath(record.local_path).name: record for record in manifest.records}
    candidates: dict[str, list[Path]] = {}
    for path in raw_dir.rglob("*") if raw_dir.exists() else ():
        if path.is_file():
            candidates.setdefault(path.name, []).append(path)

    adopted: list[str] = []
    rejected: list[str] = []
    for name, paths in sorted(candidates.items()):
        record = by_name.get(name)
        if record is None:
            rejected.extend(str(path.relative_to(raw_dir)) for path in paths)
            continue
        if len(paths) != 1:
            rejected.extend(str(path.relative_to(raw_dir)) for path in paths)
            continue
        path = paths[0]
        if record.kind == "sensor" and not _valid_csv(path):
            rejected.append(str(path.relative_to(raw_dir)))
            continue
        record.local_path = path.relative_to(raw_dir).as_posix()
        record.sha256 = sha256_file(path)
        record.size_bytes = path.stat().st_size
        record.status = "VERIFIED"
        adopted.append(record.local_path)
    return ScanReport(tuple(adopted), tuple(rejected))


def verify_integrity(manifest: DatasetManifest, raw_dir: Path) -> IntegrityReport:
    expected = expected_trial_names()
    known_missing = set(manifest.known_missing)
    known_unusable = set(manifest.known_unusable)
    records = {
        PurePosixPath(record.local_path).name: record
        for record in manifest.records
        if record.kind == "sensor"
    }
    missing: list[str] = []
    mismatch: list[str] = []
    unverified: list[str] = []
    verified = 0
    for name in sorted(expected - known_missing - known_unusable):
        record = records.get(name)
        if record is None:
            missing.append(name)
            continue
        path = raw_dir / PurePosixPath(record.local_path)
        if not path.is_file():
            missing.append(name)
        elif record.sha256 is None:
            unverified.append(name)
        elif sha256_file(path) != record.sha256:
            mismatch.append(name)
        else:
            verified += 1
    for record in (record for record in manifest.records if record.kind != "sensor"):
        name = PurePosixPath(record.local_path).name
        path = raw_dir / PurePosixPath(record.local_path)
        if not path.is_file():
            missing.append(name)
        elif record.sha256 is None:
            unverified.append(name)
        elif sha256_file(path) != record.sha256:
            mismatch.append(name)
        else:
            verified += 1
    auxiliary_count = sum(record.kind != "sensor" for record in manifest.records)
    return IntegrityReport(
        expected=len(expected) - len(known_missing) - len(known_unusable) + auxiliary_count,
        verified=verified,
        known_missing=tuple(sorted(known_missing)),
        known_unusable=tuple(sorted(known_unusable)),
        missing=tuple(missing),
        checksum_mismatch=tuple(mismatch),
        unverified=tuple(unverified),
    )


def records_for_download(manifest: DatasetManifest) -> Sequence[ManifestRecord]:
    """供測試與 UI 顯示；再次斷言 camera 永不出現在 download records。"""

    if any("camera" in record.remote_path.casefold() for record in manifest.records):
        raise RuntimeError("安全違規：manifest 含 camera entry")
    return manifest.records
