from pathlib import Path

from histoanalyzer.job import JobConfig
from histoanalyzer.resources import bundled_classifier_paths


def test_bundled_classifiers_exist():
    paths = bundled_classifier_paths()
    assert paths.tissue.name == "TissueClassifierANNFullJuly06.json"
    assert paths.anthra.name == "AnthraJuly06.json"
    assert paths.dab.name == "DABCNNThreshold0.17DAB.json"
    assert all(path.is_file() and path.stat().st_size > 100 for path in paths.as_dict().values())


def test_job_uses_bundled_defaults(tmp_path: Path):
    image = tmp_path / "image.png"
    image.write_bytes(b"not-analyzed-in-this-test")
    job = JobConfig(images=[str(image)])
    job.apply_bundled_classifier_defaults()
    assert Path(job.tissue_classifier).is_file()
    assert Path(job.anthra_classifier).is_file()
    assert Path(job.dab_classifier).is_file()


def test_repository_and_package_classifier_copies_match():
    root = Path(__file__).resolve().parents[1]
    names = (
        "TissueClassifierANNFullJuly06.json",
        "AnthraJuly06.json",
        "DABCNNThreshold0.17DAB.json",
    )
    for name in names:
        repository_copy = root / "classifiers" / name
        package_copy = root / "src" / "histoanalyzer" / "resources" / "classifiers" / name
        assert repository_copy.read_bytes() == package_copy.read_bytes()


def test_pyinstaller_spec_has_root_classifier_fallback():
    root = Path(__file__).resolve().parents[1]
    spec = (root / "build" / "HistoAnalyzer.spec").read_text(encoding="utf-8")
    assert "PACKAGE_CLASSIFIERS" in spec
    assert "ROOT_CLASSIFIERS" in spec
    assert "CLASSIFIER_SOURCE" in spec
