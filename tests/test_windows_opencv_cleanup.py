from pathlib import Path


def test_windows_build_uninstalls_only_installed_opencv_variants():
    script = (Path(__file__).parents[1] / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")
    assert "pip list --format=json" in script
    assert "$InstalledNames.Count -gt 0" in script
    assert "@($PipListJson | ConvertFrom-Json)" in script
    assert "pip uninstall -y opencv-python opencv-python-headless" not in script
    assert "opencv-python-headless==4.10.0.84" in script
