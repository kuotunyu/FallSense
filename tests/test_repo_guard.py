import subprocess
from pathlib import Path

from scripts.check_repo_guard import collect_violations, tracked_paths


def test_guard_accepts_small_source_file(tmp_path: Path) -> None:
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("pass\n", encoding="utf-8")
    assert collect_violations(tmp_path, ["src/module.py"]) == []


def test_guard_rejects_private_research_state_secret_and_large_files(tmp_path: Path) -> None:
    data_file = tmp_path / "data" / "raw" / "participant.csv"
    data_file.parent.mkdir(parents=True)
    data_file.write_text("sensitive\n", encoding="utf-8")
    secret = tmp_path / ".env"
    secret.write_text("TOKEN=secret\n", encoding="utf-8")
    large = tmp_path / "model.bin"
    large.write_bytes(b"12345")
    private_model = tmp_path / "artifacts" / "models" / "model.onnx"
    private_model.parent.mkdir(parents=True)
    private_model.write_bytes(b"onnx")
    internal_plan = tmp_path / "PLAN.md"
    internal_plan.write_text("private session plan\n", encoding="utf-8")

    violations = collect_violations(
        tmp_path,
        [
            "data/raw/participant.csv",
            ".env",
            "model.bin",
            "artifacts/models/model.onnx",
            "PLAN.md",
        ],
        max_bytes=4,
    )
    reasons = {(item.path, item.reason) for item in violations}
    assert any(path == "data/raw/participant.csv" and "data/" in reason for path, reason in reasons)
    assert any(path == ".env" and "機密" in reason for path, reason in reasons)
    assert any(path == "model.bin" and "超過" in reason for path, reason in reasons)
    assert any(
        path == "artifacts/models/model.onnx" and "模型" in reason for path, reason in reasons
    )
    assert any(path == "PLAN.md" and "內部" in reason for path, reason in reasons)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_canary_commit_is_rejected_then_reverted(tmp_path: Path) -> None:
    """以隔離 temp repo 實際驗證 commit guard，不污染 FallSense history。"""

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "ci@example.invalid")
    _git(tmp_path, "config", "user.name", "FallSense CI")
    readme = tmp_path / "README.md"
    readme.write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-m", "base")

    canary = tmp_path / "data" / "raw" / "canary.csv"
    canary.parent.mkdir(parents=True)
    canary.write_text("synthetic canary\n", encoding="utf-8")
    large = tmp_path / "oversized.canary"
    large.write_bytes(b"0" * (5 * 1024 * 1024 + 1))
    _git(tmp_path, "add", "-f", "data/raw/canary.csv", "oversized.canary")
    _git(tmp_path, "commit", "-m", "canary: must be rejected")

    violations = collect_violations(tmp_path, tracked_paths(tmp_path))
    assert {item.path for item in violations} == {"data/raw/canary.csv", "oversized.canary"}

    _git(tmp_path, "revert", "--no-edit", "HEAD")
    assert collect_violations(tmp_path, tracked_paths(tmp_path)) == []
