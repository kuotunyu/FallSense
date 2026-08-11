from __future__ import annotations

from pathlib import Path

from fallsense.data.download import (
    TIMESTAMPS_NAME,
    DatasetManifest,
    create_manifest,
    download_manifest,
    expected_trial_names,
    is_allowed_entry,
    records_for_download,
    scan_raw,
    verify_integrity,
)
from fallsense.data.drive import DriveEntry, DriveListing


def _sensor_entry(subject: int, activity: int, trial: int) -> DriveEntry:
    name = f"Subject{subject}Activity{activity}Trial{trial}.csv"
    path = f"Subject{subject}/Activity{activity}/Trial{trial}/{name}"
    return DriveEntry(path, f"id-{subject}-{activity}-{trial}", "file/csv")


def _timestamps_entry() -> DriveEntry:
    return DriveEntry(TIMESTAMPS_NAME, "timestamps-id", "application/zip")


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def download(self, file_id: str, destination: Path) -> bool:
        self.calls.append(file_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix == ".csv":
            destination.write_text("a,b,c,d\n1,2,3,4\n", encoding="utf-8", newline="\n")
        else:
            destination.write_bytes(b"synthetic timestamps fixture")
        return True


def _full_manifest() -> DatasetManifest:
    entries = [
        _sensor_entry(subject, activity, trial)
        for subject in range(1, 18)
        for activity in range(1, 12)
        for trial in range(1, 4)
    ]
    entries.append(_timestamps_entry())
    return create_manifest(DriveListing(tuple(entries), folders_visited=766))


def test_camera_entries_never_reach_download_records(tmp_path: Path) -> None:
    camera1 = DriveEntry(
        "Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera1.zip",
        "camera-1",
        "file/zip",
    )
    camera2 = DriveEntry(
        "Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera2.zip",
        "camera-2",
        "file/zip",
    )
    listing = DriveListing((_sensor_entry(1, 1, 1), camera1, camera2, _timestamps_entry()), 4)
    manifest = create_manifest(listing)
    assert not is_allowed_entry(camera1)
    assert not is_allowed_entry(camera2)
    assert manifest.excluded_camera_count == 2

    backend = FakeBackend()
    download_manifest(manifest, tmp_path, backend)
    assert set(backend.calls) == {"id-1-1-1", "timestamps-id"}
    assert all(
        "camera" not in record.remote_path.casefold() for record in records_for_download(manifest)
    )


def test_resume_skips_checksum_verified_files(tmp_path: Path) -> None:
    manifest = create_manifest(
        DriveListing((_sensor_entry(1, 1, 1), _timestamps_entry()), folders_visited=4)
    )
    first_backend = FakeBackend()
    assert download_manifest(manifest, tmp_path, first_backend) == (2, 0)

    second_backend = FakeBackend()
    assert download_manifest(manifest, tmp_path, second_backend) == (0, 2)
    assert second_backend.calls == []


def test_integrity_report_precisely_lists_ten_deleted_files(tmp_path: Path) -> None:
    manifest = _full_manifest()
    backend = FakeBackend()
    completed, skipped = download_manifest(manifest, tmp_path, backend)
    assert (completed, skipped) == (562, 0)
    deleted = sorted(expected_trial_names())[:10]
    records = {Path(record.local_path).name: record for record in manifest.records}
    for name in deleted:
        (tmp_path / records[name].local_path).unlink()

    report = verify_integrity(manifest, tmp_path)
    assert report.missing == tuple(deleted)
    assert report.checksum_mismatch == ()
    assert report.unverified == ()
    assert not report.ok


def test_scan_adopts_valid_manual_file_and_names_corruption(tmp_path: Path) -> None:
    manifest = create_manifest(
        DriveListing(
            (_sensor_entry(1, 1, 1), _sensor_entry(1, 1, 2), _timestamps_entry()),
            folders_visited=5,
        )
    )
    valid = tmp_path / "manual" / "Subject1Activity1Trial1.csv"
    valid.parent.mkdir()
    valid.write_text("a,b,c,d\n1,2,3,4\n", encoding="utf-8")
    broken = tmp_path / "Subject1Activity1Trial2.csv"
    broken.write_bytes(b"not,a,valid,row\nshort")

    report = scan_raw(manifest, tmp_path)
    assert report.adopted == ("manual/Subject1Activity1Trial1.csv",)
    assert report.rejected == ("Subject1Activity1Trial2.csv",)


def test_official_header_only_file_is_known_unusable(tmp_path: Path) -> None:
    manifest = create_manifest(
        DriveListing((_sensor_entry(1, 1, 1), _timestamps_entry()), folders_visited=4)
    )
    record = next(record for record in manifest.records if record.kind == "sensor")
    path = tmp_path / record.local_path
    path.parent.mkdir(parents=True)
    path.write_text("TimeStamps,a,b,c\n,x,y,z\n", encoding="utf-8")
    backend = FakeBackend()
    download_manifest(manifest, tmp_path, backend)
    assert record.status == "KNOWN_UNUSABLE"
    assert manifest.known_unusable == ["Subject1Activity1Trial1.csv"]
    assert backend.calls == ["timestamps-id"]
