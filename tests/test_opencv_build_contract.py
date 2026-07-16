from pathlib import Path


def test_only_one_opencv_distribution_is_declared() -> None:
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "opencv-python-headless==4.10.0.84" in requirements
    active = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip().lower().startswith((
            "opencv-python",
            "opencv-contrib-python",
        ))
    ]
    assert active == ["opencv-python-headless==4.10.0.84"]


def test_spec_uses_dedicated_opencv_hook() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = (root / "build" / "HistoAnalyzer.spec").read_text(encoding="utf-8")
    assert 'collect_all("cv2")' not in spec.replace('# can interfere with the wheel\'s generated loader and optional namespaces.', '')
    assert '"cv2"' in spec


def test_frozen_runtime_self_test_is_required_on_all_platforms() -> None:
    root = Path(__file__).resolve().parents[1]
    main = (root / "src" / "histoanalyzer" / "__main__.py").read_text(encoding="utf-8")
    assert "--self-test-opencv" in main
    assert "opencv_ml_diagnostics" in main
    for script in (
        root / "scripts" / "build_windows.ps1",
        root / "scripts" / "build_macos.sh",
        root / "scripts" / "build_linux.sh",
    ):
        text = script.read_text(encoding="utf-8")
        assert "--self-test-opencv" in text
        assert "verify_opencv_ml.py" in text
