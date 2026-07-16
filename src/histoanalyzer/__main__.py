from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path


def _restore_worker_streams() -> None:
    """Restore redirected QProcess pipes in PyInstaller windowed builds.

    PyInstaller may set sys.stdout/sys.stderr to None for a windowed executable.
    When the GUI relaunches itself through QProcess, file descriptors 1 and 2 are
    pipes; duplicating them restores line-oriented worker logging without a
    console window.
    """
    for name, fd in (("stdout", 1), ("stderr", 2)):
        if getattr(sys, name) is None:
            try:
                setattr(sys, name, os.fdopen(os.dup(fd), "w", buffering=1, encoding="utf-8", errors="replace"))
            except Exception:
                setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def main() -> int:
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", default=None)
    parser.add_argument("--self-test-opencv", action="store_true")
    parser.add_argument("--self-test-output", default=None)
    known, _ = parser.parse_known_args()
    if known.self_test_opencv:
        _restore_worker_streams()
        from .engine import opencv_ml_diagnostics

        report = opencv_ml_diagnostics()
        payload = json.dumps(report, indent=2, sort_keys=True)
        print(payload, flush=True)
        if known.self_test_output:
            output = Path(known.self_test_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload + "\n", encoding="utf-8")
        return 0 if report.get("ok") else 2
    if known.worker:
        _restore_worker_streams()
        from .worker import run_job_file
        return run_job_file(known.worker)
    try:
        from .gui.main_window import launch
    except ImportError as exc:
        print("HistoAnalyzer GUI requires PySide6. Install with: pip install PySide6", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
