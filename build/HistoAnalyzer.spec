# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import platform

from PyInstaller.utils.hooks import collect_all, copy_metadata

# PyInstaller defines SPECPATH as the directory containing this spec file.
# GitHub Actions checks repositories out into .../<repo>/<repo>, so moving up
# two directories points outside the checkout. Resolve the project root from
# the spec directory itself and verify it before Analysis starts.
_spec_path = Path(SPECPATH).resolve()
SPEC_DIR = _spec_path if _spec_path.is_dir() else _spec_path.parent
ROOT = SPEC_DIR.parent
SRC = ROOT / "src"
ASSETS = ROOT / "assets"
ENTRY = ROOT / "run_histoanalyzer.py"

for required in (ROOT / "pyproject.toml", ENTRY, SRC / "histoanalyzer" / "__main__.py"):
    if not required.is_file():
        raise FileNotFoundError(
            f"Required build input not found: {required}\n"
            f"SPECPATH={SPECPATH!r}\nResolved project root={ROOT}"
        )

print(f"HistoAnalyzer build root: {ROOT}")
print(f"HistoAnalyzer entry point: {ENTRY}")

packages = [
    # Let PyInstaller's dedicated OpenCV hook collect cv2. Manually collecting the cv2 package
    # can interfere with the wheel's generated loader and optional namespaces.
    "PySide6", "numpy", "tifffile", "zarr", "PIL", "scipy", "skimage",
    "sklearn", "joblib", "shapely", "torch", "instanseg", "einops", "fastremap",
    "matplotlib", "requests", "tqdm",
]

datas = [
    (str(ASSETS / "icon.png"), "assets"),
    (str(ASSETS / "icon.ico"), "assets"),
    (str(SRC / "histoanalyzer" / "resources" / "classifiers"), "histoanalyzer/resources/classifiers"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "THIRD_PARTY_NOTICES.md"), "."),
]
binaries = []
hiddenimports = [
    "cv2",
    "histoanalyzer", "histoanalyzer.__main__", "histoanalyzer.worker",
    "histoanalyzer.engine", "histoanalyzer.gui.main_window",
    "sklearn.ensemble._forest", "sklearn.tree._tree", "sklearn.utils._cython_blas",
    "torch._C", "torchvision", "zarr.storage", "numcodecs",
]

for package in packages:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception as exc:
        print(f"Optional PyInstaller collection skipped for {package}: {exc}")

for package in ["HistoAnalyzer", "instanseg-torch", "PySide6", "torch", "scikit-learn"]:
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

icon = ASSETS / (
    "icon.ico"
    if platform.system() == "Windows"
    else "icon.icns"
    if (ASSETS / "icon.icns").exists()
    else "icon.png"
)

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC), str(ROOT)],
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
            "CFBundleShortVersionString": "1.0.5",
            "CFBundleVersion": "1.0.5",
            "NSHighResolutionCapable": True,
        },
    )
