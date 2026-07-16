# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import platform

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH).parent.parent
SRC = ROOT / "src"
ASSETS = ROOT / "assets"
ENTRY = SRC / "histoanalyzer" / "__main__.py"

packages = [
    "PySide6", "numpy", "cv2", "tifffile", "zarr", "PIL", "scipy", "skimage",
    "sklearn", "joblib", "shapely", "torch", "instanseg", "einops", "fastremap",
    "matplotlib", "requests", "tqdm",
]

datas = [
    (str(ASSETS / "icon.png"), "assets"),
    (str(ASSETS / "icon.ico"), "assets"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "THIRD_PARTY_NOTICES.md"), "."),
]
binaries = []
hiddenimports = [
    "histoanalyzer.worker", "histoanalyzer.engine", "histoanalyzer.gui.main_window",
    "sklearn.ensemble._forest", "sklearn.tree._tree", "sklearn.utils._cython_blas",
    "torch._C", "torchvision", "zarr.storage", "numcodecs",
]

for package in packages:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception:
        pass

for package in ["histoanalyzer", "instanseg-torch", "PySide6", "torch", "scikit-learn"]:
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

icon = ASSETS / ("icon.ico" if platform.system() == "Windows" else "icon.icns" if (ASSETS / "icon.icns").exists() else "icon.png")

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "IPython", "jupyter", "notebook", "seaborn"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HistoAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon),
    version=str(ROOT / "build" / "version_info.txt") if platform.system() == "Windows" else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HistoAnalyzer",
)

if platform.system() == "Darwin":
    app = BUNDLE(
        coll,
        name="HistoAnalyzer.app",
        icon=str(icon),
        bundle_identifier="org.histoanalyzer.app",
        info_plist={
            "CFBundleName": "HistoAnalyzer",
            "CFBundleDisplayName": "HistoAnalyzer",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1.0.0",
            "NSHighResolutionCapable": True,
        },
    )
