from histoanalyzer.job import JobConfig
from histoanalyzer.worker import nuclei_cli


def test_preferred_instanseg_defaults() -> None:
    args = nuclei_cli(JobConfig())
    assert "brightfield_nuclei" in args
    assert "rgb" in args
    assert "0.5" in args
    assert "1" in args
