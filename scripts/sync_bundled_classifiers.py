#!/usr/bin/env python3
"""Synchronize repository classifier resources into the installable package."""
from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "classifiers"
DESTINATION = ROOT / "src" / "histoanalyzer" / "resources" / "classifiers"
NAMES = (
    "TissueClassifierANNFullJuly06.json",
    "AnthraJuly06.json",
    "DABCNNThreshold0.17DAB.json",
)


def main() -> int:
    missing = [SOURCE / name for name in NAMES if not (SOURCE / name).is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Bundled classifier source files are missing:\n{formatted}")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        source = SOURCE / name
        destination = DESTINATION / name
        if not destination.is_file() or not filecmp.cmp(source, destination, shallow=False):
            shutil.copy2(source, destination)
            print(f"Synchronized classifier: {destination}")
        else:
            print(f"Classifier already synchronized: {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
