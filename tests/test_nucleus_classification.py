from pathlib import Path

import numpy as np

from histoanalyzer.nucleus_classification import (
    ALL_NUCLEUS_CLASSES,
    NUCLEUS_CLASS_COLORS,
    classify_records,
    compute_spatial_features,
    infer_tissue_regions,
    probability_column,
    write_all_outputs,
)


def _record(i: int, x: float, y: float, *, area=40.0, diameter=7.0, major=8.0, minor=6.5,
            eccentricity=0.3, solidity=0.95, circularity=0.86, h=0.3, entropy=0.25):
    return {
        "nucleus_id": i,
        "centroid_x_px": x * 2,
        "centroid_y_px": y * 2,
        "centroid_x_um": x,
        "centroid_y_um": y,
        "area_px": area * 4,
        "area_um2": area,
        "equivalent_diameter_um": diameter,
        "major_axis_um": major,
        "minor_axis_um": minor,
        "aspect_ratio": major / minor,
        "eccentricity": eccentricity,
        "solidity": solidity,
        "circularity": circularity,
        "extent": 0.8,
        "h_mean": h,
        "h_std": 0.04,
        "h_entropy": entropy,
        "h_p10": h - 0.05,
        "h_p25": h - 0.02,
        "h_median": h,
        "h_p75": h + 0.02,
        "h_p90": h + 0.05,
        "gradient_mean": 0.1,
        "gradient_std": 0.03,
        "laplacian_std": entropy,
        "orientation_rad": 0.0,
        "contour": [[x * 2 - 2, y * 2 - 2], [x * 2 + 2, y * 2 - 2], [x * 2 + 2, y * 2 + 2], [x * 2 - 2, y * 2 + 2], [x * 2 - 2, y * 2 - 2]],
    }


def test_probability_vector_and_uncertainty_sum_to_one():
    records = [_record(i + 1, i * 8.0, 0.0) for i in range(8)]
    compute_spatial_features(records, graph_k=3, radius_um=25.0)
    info = classify_records(records)
    assert info["nuclei"] == len(records)
    for record in records:
        probabilities = [record[probability_column(name)] for name in ALL_NUCLEUS_CLASSES]
        assert np.isclose(sum(probabilities), 1.0)
        assert record["predicted_class"] in ALL_NUCLEUS_CLASSES
        assert 0.0 <= record["uncertainty_entropy"] <= 1.0
        assert 0.0 <= record["uncertainty_margin"] <= 1.0
        assert 0.0 <= record["confidence"] <= 1.0


def test_fibroblast_candidate_for_elongated_nucleus():
    records = [
        _record(1, 0, 0, area=45, diameter=7.5, major=13, minor=3.2,
                eccentricity=0.96, solidity=0.92, circularity=0.42, h=0.25, entropy=0.3),
        _record(2, 8, 0, area=47, diameter=7.7, major=13.5, minor=3.4,
                eccentricity=0.95, solidity=0.91, circularity=0.44, h=0.26, entropy=0.3),
        _record(3, 16, 0, area=44, diameter=7.4, major=12.5, minor=3.1,
                eccentricity=0.95, solidity=0.93, circularity=0.43, h=0.24, entropy=0.28),
    ]
    compute_spatial_features(records, graph_k=2, radius_um=25)
    classify_records(records)
    assert all(record["candidate_class"] in {"Fibroblast/myofibroblast", "Endothelial cell"} for record in records)


def test_graph_regions_and_outputs(tmp_path: Path):
    records = [_record(i + 1, (i % 4) * 8.0, (i // 4) * 8.0) for i in range(12)]
    edges = compute_spatial_features(records, graph_k=3, radius_um=25.0)
    model_info = classify_records(records)
    regions, region_px = infer_tissue_regions(records, 256, 192, 0.5, region_size_um=60)
    assert edges
    assert regions
    thumbnail = np.full((192, 256, 3), 230, dtype=np.uint8)
    summary = write_all_outputs(
        tmp_path, records, edges, regions, thumbnail, 256, 192,
        "synthetic.tif", model_info, region_px, 0.5, {"watershed": 1},
    )
    assert summary["nuclei_count"] == 12
    expected = [
        "nuclei_classification.csv", "nuclei_classification.geojson",
        "nuclei_class_overlay.png", "nuclei_class_uncertainty_overlay.png",
        "nuclei_graph.graphml", "nuclei_graph_overlay.png",
        "tissue_region_features.csv", "tissue_regions.geojson",
        "tissue_region_overlay.png", "nuclei_class_legend.png",
    ]
    assert all((tmp_path / name).exists() for name in expected)


def test_every_class_has_a_distinct_color():
    assert set(NUCLEUS_CLASS_COLORS) == set(ALL_NUCLEUS_CLASSES)
    assert len(set(NUCLEUS_CLASS_COLORS.values())) == len(ALL_NUCLEUS_CLASSES)
