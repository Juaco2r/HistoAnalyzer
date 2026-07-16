from pathlib import Path

import pytest

from histoanalyzer.job import JobConfig, discover_images, safe_stem


def test_safe_stem_ome_tiff() -> None:
    assert safe_stem(Path("sample.ome.tiff")) == "sample"
    assert safe_stem(Path("sample.tif")) == "sample"


def test_discover_images(tmp_path: Path) -> None:
    (tmp_path / "a.tif").write_bytes(b"")
    (tmp_path / "b.txt").write_text("x")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.svs").write_bytes(b"")
    found = discover_images(tmp_path, recursive=True)
    assert {Path(path).name for path in found} == {"a.tif", "c.svs"}


def test_annotation_pairing(tmp_path: Path) -> None:
    image = tmp_path / "case.ome.tif"
    image.write_bytes(b"")
    annotations = tmp_path / "case_compartments.geojson"
    annotations.write_text("{}")
    config = JobConfig(images=[str(image)], annotation_folder=str(tmp_path))
    assert config.annotation_paths_for(str(image)) == [str(annotations)]


def test_validate_requires_images() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        JobConfig().validate()
