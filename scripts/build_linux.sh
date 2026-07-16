#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
python scripts/make_icons.py
rm -rf dist build/.pyinstaller release/HistoAnalyzer-Linux-x64.tar.gz
python -m PyInstaller --noconfirm --clean --distpath dist --workpath build/.pyinstaller build/HistoAnalyzer.spec
mkdir -p release
tar -C dist -czf release/HistoAnalyzer-Linux-x64.tar.gz HistoAnalyzer
echo "Build: release/HistoAnalyzer-Linux-x64.tar.gz"
