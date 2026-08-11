"""從 source checkout 直接產生 synthetic fixture。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fallsense.data.synthetic import main

if __name__ == "__main__":
    raise SystemExit(main())
