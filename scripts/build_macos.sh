#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
python scripts/make_icons.py
rm -rf dist build/.pyinstaller release/HistoAnalyzer-macOS.zip
python -m PyInstaller --noconfirm --clean --distpath dist --workpath build/.pyinstaller build/HistoAnalyzer.spec
mkdir -p release
if [[ -d dist/HistoAnalyzer.app ]]; then
  ditto -c -k --sequesterRsrc --keepParent dist/HistoAnalyzer.app release/HistoAnalyzer-macOS.zip
else
  ditto -c -k --sequesterRsrc --keepParent dist/HistoAnalyzer release/HistoAnalyzer-macOS.zip
fi
echo "Build: release/HistoAnalyzer-macOS.zip"
