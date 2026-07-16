#!/usr/bin/env python3
"""Fail the build early if the selected OpenCV wheel lacks ML support."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from histoanalyzer.engine import opencv_ml_diagnostics  # noqa: E402


def main() -> int:
    report = opencv_ml_diagnostics()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("ok"):
        print("OpenCV ML verification failed.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
