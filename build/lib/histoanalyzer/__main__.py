from __future__ import annotations

import argparse
import multiprocessing
import os
import sys


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
    known, _ = parser.parse_known_args()
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
