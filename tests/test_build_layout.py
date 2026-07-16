from pathlib import Path


def test_release_layout_contains_entry_points():
    root = Path(__file__).resolve().parents[1]
    assert (root / "run_histoanalyzer.py").is_file()
    assert (root / "src" / "histoanalyzer" / "__main__.py").is_file()
    assert (root / "build" / "HistoAnalyzer.spec").is_file()


def test_spec_directory_resolves_project_root():
    root = Path(__file__).resolve().parents[1]
    spec_dir = root / "build"
    assert spec_dir.parent == root
    assert (spec_dir.parent / "run_histoanalyzer.py").is_file()
