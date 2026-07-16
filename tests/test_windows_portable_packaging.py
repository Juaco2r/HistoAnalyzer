from pathlib import Path


def test_windows_portable_readme_has_runtime_fallback():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")
    assert "$PortableReadmeSource" in script
    assert "if (Test-Path $PortableReadmeSource)" in script
    assert "generating README_FIRST.txt from the build script" in script
    assert "Set-Content -Path $PortableReadmeDestination" in script


def test_documented_windows_portable_readme_is_present():
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "WINDOWS_PORTABLE_README.txt").is_file()
