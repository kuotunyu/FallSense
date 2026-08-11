"""從 per-trial CSV 產生 window 與 subject-grouped split manifest。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallsense.data.download import load_manifest  # noqa: E402
from fallsense.data.windowing import WindowConfig, window_files  # noqa: E402
from fallsense.evaluation.splits import (  # noqa: E402
    make_group_kfold,
    make_loso,
    save_split_manifest,
    split_manifest_payload,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--split", choices=("loso", "group-kfold"), default="loso")
    parser.add_argument("--n-splits", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = WindowConfig.from_yaml(args.config)
    paths = sorted(args.raw_dir.rglob("Subject*Activity*Trial*.csv"))
    if args.manifest.exists():
        known_unusable = set(load_manifest(args.manifest).known_unusable)
        paths = [path for path in paths if path.name not in known_unusable]
    batch = window_files(paths, config, root=args.raw_dir)
    if len(batch.labels) == 0:
        raise SystemExit("沒有可用 windows")
    folds = (
        make_loso(batch.groups)
        if args.split == "loso"
        else make_group_kfold(batch.groups, n_splits=args.n_splits)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "windows.npz",
        values=batch.values,
        labels=batch.labels,
        groups=batch.groups,
        channel_mask=batch.channel_mask,
    )
    payload = split_manifest_payload(
        folds,
        tuple(item.window_id for item in batch.metadata),
        config_checksum=config.checksum,
    )
    save_split_manifest(payload, args.output_dir / "split_manifest.json")
    print(
        json.dumps(
            {
                "trials": len(paths),
                "windows": len(batch.labels),
                "subjects": len(np.unique(batch.groups)),
                "split_checksum": payload["checksum"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
