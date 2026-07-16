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
