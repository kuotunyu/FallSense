"""繪製單一 trial 的 sensor / Tag 對齊圖。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallsense.data.io import read_trial_csv  # noqa: E402
from fallsense.data.visualize import plot_trial_alignment  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    args = parser.parse_args(argv)
    frame = read_trial_csv(args.input)
    plot_trial_alignment(frame, args.output, title=args.title or args.input.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
