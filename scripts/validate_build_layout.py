#!/usr/bin/env python3
"""Fail early when a release build is launched from the wrong directory."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    ROOT / "pyproject.toml",
    ROOT / "run_histoanalyzer.py",
    ROOT / "build" / "HistoAnalyzer.spec",
    ROOT / "src" / "histoanalyzer" / "__main__.py",
    ROOT / "src" / "histoanalyzer" / "gui" / "main_window.py",
)

missing = [path for path in REQUIRED if not path.is_file()]
if missing:
    formatted = "\n".join(f"  - {path}" for path in missing)
    raise SystemExit(f"Build layout validation failed. Missing:\n{formatted}")

sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.find_spec("histoanalyzer.__main__")
if spec is None or spec.origin is None:
    raise SystemExit("Unable to import histoanalyzer.__main__ from src/")

print(f"Build layout OK: {ROOT}")
print(f"Python entry module: {spec.origin}")
