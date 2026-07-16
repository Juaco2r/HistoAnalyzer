#!/usr/bin/env python3
from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
PNG = ASSETS / "icon.png"


def main() -> None:
    image = Image.open(PNG).convert("RGBA")
    image.save(ASSETS / "icon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    if platform.system() == "Darwin" and shutil.which("iconutil"):
        iconset = ASSETS / "HistoAnalyzer.iconset"
        iconset.mkdir(exist_ok=True)
        sizes = [16, 32, 128, 256, 512]
        for size in sizes:
            image.resize((size, size), Image.Resampling.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
            image.resize((size * 2, size * 2), Image.Resampling.LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(ASSETS / "icon.icns")], check=True)
        shutil.rmtree(iconset, ignore_errors=True)


if __name__ == "__main__":
    main()
