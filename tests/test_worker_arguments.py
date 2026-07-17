from histoanalyzer.job import JobConfig
from histoanalyzer.worker import nuclei_cli, nucleus_classification_cli


def test_preferred_instanseg_defaults() -> None:
    args = nuclei_cli(JobConfig())
    assert "brightfield_nuclei" in args
    assert "rgb" in args
    assert "0.5" in args
    assert "1" in args


def test_nucleus_classification_defaults() -> None:
    args = nucleus_classification_cli(JobConfig())
    assert "--nucleus-classification-tile-size" in args
    assert "1024" in args
    assert "--nucleus-graph-k" in args
    assert "6" in args
    assert "--no-nucleus-classification" not in args


def test_nucleus_classification_can_be_disabled() -> None:
    args = nucleus_classification_cli(JobConfig(enable_nucleus_classification=False))
    assert "--no-nucleus-classification" in args
