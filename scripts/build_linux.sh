#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python scripts/validate_build_layout.py
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
python -m pip install --force-reinstall --no-deps --no-cache-dir opencv-python-headless==4.10.0.84
python scripts/verify_opencv_ml.py
python scripts/make_icons.py
rm -rf dist build/.pyinstaller release/HistoAnalyzer-Linux-x64.tar.gz
python -m PyInstaller --noconfirm --clean --distpath dist --workpath build/.pyinstaller build/HistoAnalyzer.spec
SELF_TEST="build/opencv_ml_frozen_self_test.json"
rm -f "$SELF_TEST"
dist/HistoAnalyzer/HistoAnalyzer --self-test-opencv --self-test-output "$SELF_TEST"
python - "$SELF_TEST" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Frozen OpenCV self-test output missing: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if not report.get("ok"):
    raise SystemExit(f"Frozen OpenCV ML self-test failed: {report}")
print(f"Frozen OpenCV ML self-test passed using {report.get('loader_mode')}.")
PY
mkdir -p release
tar -C dist -czf release/HistoAnalyzer-Linux-x64.tar.gz HistoAnalyzer
echo "Build: release/HistoAnalyzer-Linux-x64.tar.gz"
