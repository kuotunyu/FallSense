"""輸出官方 sensor CSV 的實測 schema/rate/tag/missing-channel profile。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallsense.data.download import load_manifest  # noqa: E402
from fallsense.data.profile import profile_dataset, save_profile  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/profile.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    args = parser.parse_args(argv)
    paths = sorted(args.raw_dir.rglob("Subject*Activity*Trial*.csv"))
    profile = profile_dataset(paths)
    manifest = load_manifest(args.manifest)
    known_unusable = set(manifest.known_unusable)
    failures = cast(dict[str, str], profile["failures"])
    profile["known_unusable"] = sorted(known_unusable)
    profile["unexpected_failures"] = sorted(set(failures) - known_unusable)
    save_profile(profile, args.output)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0 if not profile["unexpected_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
