"""CI guard：拒絕追蹤資料、衍生模型、內部狀態、秘密檔與超限大檔。"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

MAX_TRACKED_BYTES = 5 * 1024 * 1024
FORBIDDEN_EXACT = frozenset(
    {
        ".env",
        "CLAUDE.md",
        "PLAN.md",
        "PROGRESS.md",
        "client_secrets.json",
        "settings.yaml",
    }
)
FORBIDDEN_PREFIXES = {
    "data/": "data/ 不得被 git 追蹤",
    "artifacts/models/": "由真實資料衍生的模型不得被 git 追蹤",
    ".agents/": "內部 agent 狀態不得被 git 追蹤",
    ".claude/": "內部 agent 狀態不得被 git 追蹤",
}


@dataclass(frozen=True)
class Violation:
    path: str
    reason: str


def tracked_paths(root: Path) -> list[str]:
    """讀取 git index 中所有 tracked paths。"""

    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]


def collect_violations(
    root: Path,
    paths: Iterable[str],
    *,
    max_bytes: int = MAX_TRACKED_BYTES,
) -> list[Violation]:
    """檢查已追蹤路徑；synthetic fixtures 是唯一允許的 fixture 區。"""

    violations: list[Violation] = []
    for path_text in paths:
        normalized = path_text.replace("\\", "/")
        for prefix, reason in FORBIDDEN_PREFIXES.items():
            if normalized == prefix.rstrip("/") or normalized.startswith(prefix):
                violations.append(Violation(normalized, reason))
        if normalized in FORBIDDEN_EXACT:
            category = "內部 session 文件" if normalized.endswith(".md") else "機密設定檔"
            violations.append(Violation(normalized, f"{category}不得被 git 追蹤"))
        elif normalized.endswith("/.env"):
            violations.append(Violation(normalized, "機密設定檔不得被 git 追蹤"))
        absolute = root / Path(normalized)
        if absolute.is_file() and absolute.stat().st_size > max_bytes:
            violations.append(Violation(normalized, f"檔案超過 {max_bytes} bytes"))
    return violations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-bytes", type=int, default=MAX_TRACKED_BYTES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    violations = collect_violations(root, tracked_paths(root), max_bytes=args.max_bytes)
    if violations:
        for violation in violations:
            print(f"ERROR: {violation.path}: {violation.reason}")
        return 1
    print("Repository guard passed: 無資料、衍生模型、內部狀態、機密檔或超限大檔被追蹤。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
