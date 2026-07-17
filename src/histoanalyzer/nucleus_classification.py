"""Per-nucleus morphology classification and graph-based tissue-region inference.

The built-in classifier is deliberately transparent: it converts the morphology,
hematoxylin texture, and local-neighbour clues supplied by the user into soft
compatibility scores.  The resulting values are *research-oriented morphology
probabilities*, not clinically calibrated diagnostic probabilities.  A trained
scikit-learn ``predict_proba`` model can be supplied later without changing the
export format.
"""

from __future__ import annotations

import csv
import json
import math
import xml.sax.saxutils as xmlutils
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None


NUCLEUS_CLASSES: Tuple[str, ...] = (
    "Small lymphocyte",
    "Plasma cell",
    "Neutrophil",
    "Macrophage",
    "Fibroblast/myofibroblast",
    "Endothelial cell",
    "Normal pneumocyte/bronchial epithelial cell",
    "Tumour epithelial cell",
)
UNCERTAIN_CLASS = "Uncertain"
ALL_NUCLEUS_CLASSES: Tuple[str, ...] = NUCLEUS_CLASSES + (UNCERTAIN_CLASS,)

NUCLEUS_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Small lymphocyte": (40, 105, 220),
    "Plasma cell": (146, 78, 185),
    "Neutrophil": (0, 180, 210),
    "Macrophage": (241, 132, 36),
    "Fibroblast/myofibroblast": (46, 160, 67),
    "Endothelial cell": (238, 196, 45),
    "Normal pneumocyte/bronchial epithelial cell": (225, 105, 165),
    "Tumour epithelial cell": (215, 45, 45),
    "Uncertain": (145, 145, 145),
}

REGION_CLASSES: Tuple[str, ...] = (
    "Tumour-rich",
    "Stroma-rich",
    "Immune-rich",
    "Vascular-rich",
    "Mixed",
    "Low-nuclei/other",
)
REGION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Tumour-rich": (215, 45, 45),
    "Stroma-rich": (46, 160, 67),
    "Immune-rich": (40, 105, 220),
    "Vascular-rich": (238, 196, 45),
    "Mixed": (150, 90, 180),
    "Low-nuclei/other": (145, 145, 145),
}

# Scalar columns accepted by an optional trained model package.
NUCLEUS_FEATURE_NAMES: Tuple[str, ...] = (
    "area_um2",
    "equivalent_diameter_um",
    "major_axis_um",
    "minor_axis_um",
    "aspect_ratio",
    "eccentricity",
    "solidity",
    "circularity",
    "extent",
    "h_mean",
    "h_std",
    "h_entropy",
    "h_p10",
    "h_median",
    "h_p90",
    "gradient_mean",
    "laplacian_std",
    "darkness_rank",
    "texture_rank",
    "nearest_neighbor_um",
    "neighbors_within_radius",
    "local_density_per_1000um2",
    "local_density_rank",
    "local_orientation_coherence",
    "local_linearity",
    "local_area_cv",
    "local_area_cv_rank",
    "local_h_cv",
    "local_spacing_cv",
)


def probability_column(class_name: str) -> str:
    text = class_name.lower().replace("/", "_").replace("-", "_")
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return "p_" + cleaned.strip("_")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-value))


def _range_score(value: float, low: float, high: float, softness: float) -> float:
    value = _safe_float(value)
    softness = max(float(softness), 1e-6)
    if low <= value <= high:
        centre = (low + high) / 2.0
        half = max((high - low) / 2.0, softness)
        return 0.82 + 0.18 * math.exp(-0.5 * ((value - centre) / half) ** 2)
    distance = low - value if value < low else value - high
    return max(1e-4, math.exp(-0.5 * (distance / softness) ** 2))


def _gaussian_score(value: float, mean: float, sd: float) -> float:
    sd = max(float(sd), 1e-6)
    return max(1e-4, math.exp(-0.5 * ((_safe_float(value) - mean) / sd) ** 2))


def _high_score(value: float, centre: float, scale: float) -> float:
    return max(1e-4, _sigmoid((_safe_float(value) - centre) / max(scale, 1e-6)))


def _low_score(value: float, centre: float, scale: float) -> float:
    return max(1e-4, _sigmoid((centre - _safe_float(value)) / max(scale, 1e-6)))


def _weighted_geometric(terms: Sequence[Tuple[float, float]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for score, weight in terms:
        weight = max(0.0, float(weight))
        numerator += weight * math.log(max(float(score), 1e-6))
        denominator += weight
    return math.exp(numerator / max(denominator, 1e-6))


def _entropy(values: np.ndarray, bins: int = 16) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return 0.0
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(low, high))
    p = hist.astype(np.float64)
    p /= max(float(p.sum()), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / math.log2(bins))


def _simplified_contour(prop: Any, global_x: int, global_y: int, max_points: int = 48) -> List[List[float]]:
    mask = np.asarray(prop.image, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(0.5, 0.008 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if approx.shape[0] > max_points:
        indices = np.linspace(0, approx.shape[0] - 1, max_points).round().astype(int)
        approx = approx[indices]
    minr, minc, _, _ = prop.bbox
    points = [[float(global_x + minc + x), float(global_y + minr + y)] for x, y in approx]
    if points and points[0] != points[-1]:
        points.append(points[0])
    return points


def extract_records_from_tile(
    labels: np.ndarray,
    props: Sequence[Any],
    h_channel: np.ndarray,
    rgb: np.ndarray,
    origin_x: int,
    origin_y: int,
    core_bounds_local: Tuple[int, int, int, int],
    pixel_size_um: float,
    start_id: int,
    tile_id: int,
) -> List[Dict[str, Any]]:
    """Extract scalar and contour features for nuclei whose centroid is in the core."""
    x0, y0, x1, y1 = core_bounds_local
    mpp = max(float(pixel_size_um), 1e-6)
    records: List[Dict[str, Any]] = []
    next_id = int(start_id)
    h_channel = np.asarray(h_channel, dtype=np.float32)
    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(h_channel, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(h_channel, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)

    for prop in props:
        cy, cx = prop.centroid
        if not (x0 <= cx < x1 and y0 <= cy < y1):
            continue
        coords = np.asarray(prop.coords)
        if coords.size == 0:
            continue
        values = h_channel[coords[:, 0], coords[:, 1]].astype(np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            values = np.asarray([0.0], dtype=np.float64)
        grad_values = gradient[coords[:, 0], coords[:, 1]].astype(np.float64)
        lap_values = laplacian[coords[:, 0], coords[:, 1]].astype(np.float64)
        area_px = float(prop.area)
        perimeter_px = float(prop.perimeter) if float(prop.perimeter) > 0 else 0.0
        circularity = float(np.clip(4.0 * math.pi * area_px / max(perimeter_px * perimeter_px, 1e-6), 0.0, 1.0))
        major_px = max(float(prop.major_axis_length), 1e-6)
        minor_px = max(float(prop.minor_axis_length), 1e-6)
        next_id += 1
        record: Dict[str, Any] = {
            "nucleus_id": next_id,
            "tile_id": int(tile_id),
            "centroid_x_px": float(origin_x + cx),
            "centroid_y_px": float(origin_y + cy),
            "centroid_x_um": float((origin_x + cx) * mpp),
            "centroid_y_um": float((origin_y + cy) * mpp),
            "area_px": area_px,
            "area_um2": area_px * mpp * mpp,
            "equivalent_diameter_px": float(prop.equivalent_diameter_area),
            "equivalent_diameter_um": float(prop.equivalent_diameter_area) * mpp,
            "major_axis_px": major_px,
            "major_axis_um": major_px * mpp,
            "minor_axis_px": minor_px,
            "minor_axis_um": minor_px * mpp,
            "aspect_ratio": major_px / minor_px,
            "eccentricity": float(prop.eccentricity),
            "solidity": float(prop.solidity),
            "extent": float(prop.extent),
            "perimeter_px": perimeter_px,
            "circularity": circularity,
            "orientation_rad": float(prop.orientation),
            "orientation_deg": float(math.degrees(float(prop.orientation))),
            "h_mean": float(np.mean(values)),
            "h_std": float(np.std(values)),
            "h_p10": float(np.percentile(values, 10)),
            "h_p25": float(np.percentile(values, 25)),
            "h_median": float(np.median(values)),
            "h_p75": float(np.percentile(values, 75)),
            "h_p90": float(np.percentile(values, 90)),
            "h_entropy": _entropy(values),
            "gradient_mean": float(np.mean(grad_values)) if grad_values.size else 0.0,
            "gradient_std": float(np.std(grad_values)) if grad_values.size else 0.0,
            "laplacian_std": float(np.std(lap_values)) if lap_values.size else 0.0,
            "contour": _simplified_contour(prop, origin_x, origin_y),
        }
        records.append(record)
    return records


def _rank01(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full(arr.shape, 0.5, dtype=np.float64)
    valid = arr[finite]
    order = np.argsort(valid, kind="mergesort")
    ranks = np.empty(valid.size, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, valid.size) if valid.size > 1 else 0.5
    out = np.full(arr.shape, 0.5, dtype=np.float64)
    out[finite] = ranks
    return out


def compute_spatial_features(
    records: List[Dict[str, Any]],
    graph_k: int = 6,
    radius_um: float = 25.0,
) -> List[Tuple[int, int, float]]:
    """Add neighbour features and return unique k-nearest-neighbour edges."""
    if not records:
        return []
    if cKDTree is None:
        raise RuntimeError("scipy is required for nucleus graph construction")
    coords = np.asarray([[r["centroid_x_um"], r["centroid_y_um"]] for r in records], dtype=np.float64)
    areas = np.asarray([r["area_um2"] for r in records], dtype=np.float64)
    hmeans = np.asarray([r["h_mean"] for r in records], dtype=np.float64)
    orientations = np.asarray([r["orientation_rad"] for r in records], dtype=np.float64)
    tree = cKDTree(coords)
    k = min(max(1, int(graph_k)), max(1, len(records) - 1))
    query_k = min(len(records), k + 1)
    distances, indices = tree.query(coords, k=query_k)
    if query_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    radius = max(float(radius_um), 1.0)
    edges: Dict[Tuple[int, int], float] = {}

    for i, record in enumerate(records):
        nearest = [float(d) for d, j in zip(np.atleast_1d(distances[i]), np.atleast_1d(indices[i])) if int(j) != i and math.isfinite(float(d))]
        neighbours = [int(j) for j in tree.query_ball_point(coords[i], r=radius) if int(j) != i]
        local = neighbours if neighbours else [int(j) for j in np.atleast_1d(indices[i]) if int(j) != i]
        local = list(dict.fromkeys(local))
        local_areas = areas[local] if local else np.asarray([], dtype=np.float64)
        local_h = hmeans[local] if local else np.asarray([], dtype=np.float64)
        local_dist = np.linalg.norm(coords[local] - coords[i], axis=1) if local else np.asarray([], dtype=np.float64)

        orientation_set = np.concatenate([[orientations[i]], orientations[local]]) if local else np.asarray([orientations[i]])
        coherence = abs(np.mean(np.exp(2j * orientation_set))) if orientation_set.size else 0.0
        if len(local) >= 2:
            centred = coords[local] - np.mean(coords[local], axis=0)
            covariance = np.cov(centred.T)
            eigen = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
            linearity = float((eigen[0] - eigen[1]) / max(eigen[0] + eigen[1], 1e-9))
        else:
            linearity = 0.0

        record.update({
            "nearest_neighbor_um": float(min(nearest)) if nearest else 0.0,
            "mean_knn_distance_um": float(np.mean(nearest[:k])) if nearest else 0.0,
            "neighbors_within_radius": int(len(neighbours)),
            "local_density_per_1000um2": float(len(neighbours) / (math.pi * radius * radius) * 1000.0),
            "local_orientation_coherence": float(np.clip(coherence, 0.0, 1.0)),
            "local_linearity": float(np.clip(linearity, 0.0, 1.0)),
            "local_area_cv": float(np.std(local_areas) / max(np.mean(local_areas), 1e-9)) if local_areas.size > 1 else 0.0,
            "local_h_cv": float(np.std(local_h) / max(abs(np.mean(local_h)), 1e-9)) if local_h.size > 1 else 0.0,
            "local_spacing_cv": float(np.std(local_dist) / max(np.mean(local_dist), 1e-9)) if local_dist.size > 1 else 0.0,
        })
        for d, j in zip(np.atleast_1d(distances[i]), np.atleast_1d(indices[i])):
            j = int(j)
            d = float(d)
            if j == i or not math.isfinite(d):
                continue
            a, b = sorted((i, j))
            edges[(a, b)] = min(edges.get((a, b), float("inf")), d)

    darkness = _rank01([r["h_mean"] for r in records])
    texture = _rank01([0.55 * r["h_entropy"] + 0.25 * r["h_std"] + 0.20 * r["laplacian_std"] for r in records])
    density = _rank01([r["local_density_per_1000um2"] for r in records])
    area_cv = _rank01([r["local_area_cv"] for r in records])
    for i, record in enumerate(records):
        record["darkness_rank"] = float(darkness[i])
        record["texture_rank"] = float(texture[i])
        record["local_density_rank"] = float(density[i])
        record["local_area_cv_rank"] = float(area_cv[i])
    return [(a, b, d) for (a, b), d in sorted(edges.items())]


def _heuristic_compatibilities(record: Mapping[str, Any]) -> Dict[str, float]:
    d = _safe_float(record.get("equivalent_diameter_um"))
    area = _safe_float(record.get("area_um2"))
    major = _safe_float(record.get("major_axis_um"))
    minor = _safe_float(record.get("minor_axis_um"))
    aspect = _safe_float(record.get("aspect_ratio"), 1.0)
    ecc = _safe_float(record.get("eccentricity"))
    solidity = _safe_float(record.get("solidity"), 1.0)
    circularity = _safe_float(record.get("circularity"), 1.0)
    dark = _safe_float(record.get("darkness_rank"), 0.5)
    texture = _safe_float(record.get("texture_rank"), 0.5)
    density = _safe_float(record.get("local_density_rank"), 0.5)
    orient = _safe_float(record.get("local_orientation_coherence"), 0.0)
    linearity = _safe_float(record.get("local_linearity"), 0.0)
    area_cv = _safe_float(record.get("local_area_cv_rank"), 0.5)
    spacing_cv = _safe_float(record.get("local_spacing_cv"), 0.5)
    irregularity = float(np.clip(0.52 * (1.0 - circularity) + 0.48 * (1.0 - solidity), 0.0, 1.0))
    uniform = 1.0 - texture
    regular_spacing = 1.0 - float(np.clip(spacing_cv / 1.1, 0.0, 1.0))

    return {
        "Small lymphocyte": _weighted_geometric([
            (_range_score(d, 5.0, 8.0, 1.5), 1.5),
            (_range_score(area, 20.0, 50.0, 18.0), 1.2),
            (_high_score(dark, 0.66, 0.13), 1.4),
            (_high_score(circularity, 0.74, 0.10), 1.1),
            (_high_score(solidity, 0.88, 0.07), 1.0),
            (_low_score(aspect, 1.45, 0.25), 0.9),
            (_high_score(density, 0.55, 0.20), 0.7),
        ]),
        "Plasma cell": _weighted_geometric([
            (_range_score(d, 7.0, 10.0, 1.8), 1.3),
            (_range_score(area, 35.0, 90.0, 25.0), 1.0),
            (_high_score(dark, 0.58, 0.16), 1.0),
            (_high_score(texture, 0.58, 0.16), 1.4),
            (_high_score(circularity, 0.65, 0.13), 0.9),
            (_high_score(solidity, 0.82, 0.10), 0.8),
            (_high_score(density, 0.48, 0.22), 0.6),
        ]),
        "Neutrophil": _weighted_geometric([
            (_range_score(d, 8.0, 12.0, 2.2), 1.2),
            (_range_score(area, 35.0, 125.0, 35.0), 0.8),
            (_high_score(dark, 0.62, 0.15), 1.0),
            (_high_score(texture, 0.60, 0.16), 0.9),
            (_high_score(irregularity, 0.30, 0.12), 1.5),
            (_low_score(solidity, 0.86, 0.10), 1.1),
        ]),
        "Macrophage": _weighted_geometric([
            (_range_score(d, 8.0, 15.0, 2.8), 1.3),
            (_range_score(area, 50.0, 185.0, 45.0), 1.1),
            (_low_score(dark, 0.56, 0.18), 1.3),
            (_range_score(circularity, 0.42, 0.88, 0.18), 0.7),
            (_range_score(ecc, 0.15, 0.82, 0.22), 0.6),
            (_high_score(texture, 0.42, 0.20), 0.6),
            (_low_score(density, 0.72, 0.22), 0.5),
        ]),
        "Fibroblast/myofibroblast": _weighted_geometric([
            (_range_score(major, 7.0, 15.0, 3.0), 1.3),
            (_range_score(minor, 2.0, 5.0, 1.4), 1.3),
            (_high_score(aspect, 1.9, 0.45), 1.6),
            (_high_score(ecc, 0.78, 0.10), 1.4),
            (_range_score(dark, 0.30, 0.82, 0.20), 0.6),
            (_high_score(orient, 0.48, 0.20), 0.9),
            (_high_score(linearity, 0.35, 0.20), 0.6),
            (_range_score(area, 18.0, 90.0, 30.0), 0.6),
        ]),
        "Endothelial cell": _weighted_geometric([
            (_range_score(major, 6.0, 12.0, 2.4), 1.2),
            (_range_score(minor, 2.0, 4.0, 1.2), 1.4),
            (_high_score(aspect, 2.0, 0.50), 1.4),
            (_high_score(ecc, 0.80, 0.09), 1.3),
            (_range_score(dark, 0.24, 0.72, 0.20), 0.7),
            (_high_score(linearity, 0.58, 0.16), 1.2),
            (_high_score(orient, 0.58, 0.16), 1.0),
            (_low_score(texture, 0.62, 0.18), 0.7),
            (_range_score(area, 12.0, 65.0, 22.0), 0.7),
        ]),
        "Normal pneumocyte/bronchial epithelial cell": _weighted_geometric([
            (_range_score(d, 6.0, 10.0, 2.0), 1.3),
            (_range_score(area, 28.0, 100.0, 30.0), 0.9),
            (_range_score(dark, 0.25, 0.72, 0.20), 0.7),
            (_high_score(uniform, 0.54, 0.18), 1.3),
            (_high_score(circularity, 0.64, 0.14), 1.0),
            (_high_score(solidity, 0.84, 0.10), 0.9),
            (_high_score(regular_spacing, 0.50, 0.20), 0.8),
            (_low_score(area_cv, 0.58, 0.20), 1.0),
        ]),
        "Tumour epithelial cell": _weighted_geometric([
            (_range_score(d, 8.0, 18.0, 3.5), 1.3),
            (_range_score(area, 50.0, 250.0, 65.0), 1.2),
            (_high_score(texture, 0.48, 0.18), 1.0),
            (_high_score(irregularity, 0.20, 0.14), 1.0),
            (_high_score(area_cv, 0.48, 0.20), 1.1),
            (_high_score(density, 0.48, 0.22), 0.8),
            (_range_score(circularity, 0.35, 0.88, 0.20), 0.5),
        ]),
    }


def _softmax(values: np.ndarray, temperature: float = 0.45) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-6)
    values -= float(np.max(values))
    exp = np.exp(np.clip(values, -80.0, 80.0))
    return exp / max(float(exp.sum()), 1e-12)


def _apply_uncertainty(base_probs: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    order = np.argsort(base_probs)[::-1]
    top1 = float(base_probs[order[0]])
    top2 = float(base_probs[order[1]]) if base_probs.size > 1 else 0.0
    margin = max(0.0, top1 - top2)
    entropy = float(-(base_probs * np.log(np.clip(base_probs, 1e-12, 1.0))).sum() / math.log(len(base_probs)))
    # High entropy, low winning probability, and a small top-two margin jointly
    # increase the explicit Uncertain probability.
    uncertain = _sigmoid(4.5 * (entropy - 0.88) + 4.0 * (0.22 - top1) + 4.0 * (0.04 - margin))
    uncertain = float(np.clip(uncertain, 0.02, 0.82))
    final = np.concatenate([base_probs * (1.0 - uncertain), np.asarray([uncertain])])
    final /= max(float(final.sum()), 1e-12)
    return final, entropy, 1.0 - margin, uncertain


def classify_records(
    records: List[Dict[str, Any]],
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify nuclei and append probabilities/uncertainty to each record."""
    if not records:
        return {"model_type": "none", "classes": list(ALL_NUCLEUS_CLASSES), "nuclei": 0}

    model_package = None
    model = None
    model_classes: List[str] = []
    feature_names = list(NUCLEUS_FEATURE_NAMES)
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if joblib is None:
            raise RuntimeError("joblib is required to load a trained nucleus classifier")
        model_package = joblib.load(path)
        model = model_package.get("model") if isinstance(model_package, dict) else model_package
        feature_names = list(model_package.get("feature_names", feature_names)) if isinstance(model_package, dict) else feature_names
        fallback_classes = model_package.get("classes", []) if isinstance(model_package, dict) else []
        model_classes = [str(x) for x in getattr(model, "classes_", fallback_classes)]
        unsupported = [name for name in model_classes if name not in NUCLEUS_CLASSES]
        if unsupported:
            raise ValueError(f"Unsupported nucleus classes in trained model: {unsupported}")

    for record in records:
        if model is None:
            compatibility = _heuristic_compatibilities(record)
            logits = np.log(np.asarray([max(compatibility[name], 1e-8) for name in NUCLEUS_CLASSES]))
            base = _softmax(logits)
            model_type = "built-in morphology compatibility v1"
        else:
            sample = np.asarray([[_safe_float(record.get(name)) for name in feature_names]], dtype=np.float64)
            predicted = np.asarray(model.predict_proba(sample)[0], dtype=np.float64)
            base = np.full(len(NUCLEUS_CLASSES), 1e-8, dtype=np.float64)
            for class_name, value in zip(model_classes, predicted):
                base[NUCLEUS_CLASSES.index(class_name)] = max(float(value), 1e-8)
            base /= base.sum()
            model_type = "trained predict_proba model"

        final, entropy, margin_uncertainty, uncertain_probability = _apply_uncertainty(base)
        best_index = int(np.argmax(final))
        base_order = np.argsort(base)[::-1]
        candidate = NUCLEUS_CLASSES[int(base_order[0])]
        candidate2 = NUCLEUS_CLASSES[int(base_order[1])] if len(base_order) > 1 else candidate
        record.update({
            "candidate_class": candidate,
            "candidate_probability": float(base[base_order[0]]),
            "second_candidate_class": candidate2,
            "second_candidate_probability": float(base[base_order[1]]) if len(base_order) > 1 else 0.0,
            "predicted_class": ALL_NUCLEUS_CLASSES[best_index],
            "probability_top1": float(final[best_index]),
            "uncertainty_entropy": float(entropy),
            "uncertainty_margin": float(margin_uncertainty),
            "uncertain_probability": float(uncertain_probability),
            "confidence": float(1.0 - 0.55 * entropy - 0.45 * margin_uncertainty),
            "classification_model": model_type,
        })
        record["confidence"] = float(np.clip(record["confidence"], 0.0, 1.0))
        for class_name, probability in zip(ALL_NUCLEUS_CLASSES, final):
            record[probability_column(class_name)] = float(probability)

    return {
        "model_type": "trained predict_proba model" if model is not None else "built-in morphology compatibility v1",
        "model_path": str(model_path or ""),
        "classes": list(ALL_NUCLEUS_CLASSES),
        "feature_names": feature_names,
        "nuclei": len(records),
        "probability_note": (
            "Built-in values are morphology compatibility probabilities derived from the supplied size, "
            "hematoxylin, geometry and spatial clues; they are not clinically calibrated probabilities."
            if model is None else
            "Probabilities originate from the supplied trained model predict_proba output, with an explicit uncertainty class."
        ),
    }


def infer_tissue_regions(
    records: Sequence[Mapping[str, Any]],
    width_px: int,
    height_px: int,
    pixel_size_um: float,
    region_size_um: float = 120.0,
    fallback_region_px: int = 240,
) -> Tuple[List[Dict[str, Any]], int]:
    mpp = max(float(pixel_size_um), 1e-6)
    region_px = max(48, int(round(region_size_um / mpp))) if region_size_um > 0 else max(48, int(fallback_region_px))
    bins: Dict[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        gx = int(float(record["centroid_x_px"]) // region_px)
        gy = int(float(record["centroid_y_px"]) // region_px)
        bins[(gx, gy)].append(record)

    regions: List[Dict[str, Any]] = []
    for (gx, gy), members in sorted(bins.items(), key=lambda item: (item[0][1], item[0][0])):
        x = gx * region_px
        y = gy * region_px
        w = min(region_px, width_px - x)
        h = min(region_px, height_px - y)
        sums = {name: float(sum(_safe_float(m.get(probability_column(name))) for m in members)) for name in ALL_NUCLEUS_CLASSES}
        known_total = max(sum(sums[name] for name in NUCLEUS_CLASSES), 1e-9)
        fractions = {name: sums[name] / known_total for name in NUCLEUS_CLASSES}
        immune = sum(fractions[name] for name in ("Small lymphocyte", "Plasma cell", "Neutrophil", "Macrophage"))
        stroma = fractions["Fibroblast/myofibroblast"] + fractions["Endothelial cell"]
        tumour = fractions["Tumour epithelial cell"]
        vascular = fractions["Endothelial cell"]
        mean_linearity = float(np.mean([_safe_float(m.get("local_linearity")) for m in members]))
        mean_confidence = float(np.mean([_safe_float(m.get("confidence")) for m in members]))

        uncertain_fraction = sums[UNCERTAIN_CLASS] / max(sum(sums.values()), 1e-9)
        if len(members) < 3 or uncertain_fraction >= 0.58 or mean_confidence < 0.18:
            label = "Low-nuclei/other"
        elif vascular >= 0.30 and mean_linearity >= 0.52:
            label = "Vascular-rich"
        elif tumour >= 0.42:
            label = "Tumour-rich"
        elif immune >= 0.48:
            label = "Immune-rich"
        elif stroma >= 0.46:
            label = "Stroma-rich"
        else:
            label = "Mixed"
        region: Dict[str, Any] = {
            "region_id": len(regions) + 1,
            "grid_x": gx,
            "grid_y": gy,
            "x_px": x,
            "y_px": y,
            "width_px": w,
            "height_px": h,
            "region_size_px": region_px,
            "predicted_region": label,
            "nuclei_count": len(members),
            "mean_confidence": mean_confidence,
            "mean_local_linearity": mean_linearity,
            "immune_fraction": immune,
            "stroma_fraction": stroma,
            "tumour_fraction": tumour,
            "vascular_fraction": vascular,
            "uncertain_fraction": uncertain_fraction,
        }
        for name in NUCLEUS_CLASSES:
            region["fraction_" + probability_column(name)[2:]] = fractions[name]
        regions.append(region)
    return regions, region_px


def _scalar_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if key != "contour" and not isinstance(value, (list, dict, tuple))}


def write_nuclei_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_scalar_record(record) for record in records]
    if not rows:
        path.write_text("nucleus_id,predicted_class\n", encoding="utf-8")
        return
    preferred = [
        "nucleus_id", "centroid_x_px", "centroid_y_px", "centroid_x_um", "centroid_y_um",
        "predicted_class", "probability_top1", "candidate_class", "candidate_probability",
        "second_candidate_class", "second_candidate_probability", "uncertainty_entropy",
        "uncertainty_margin", "uncertain_probability", "confidence", "classification_model",
    ]
    fieldnames = preferred + sorted({key for row in rows for key in row} - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_class_summary(path: Path, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    counts = Counter(str(record.get("predicted_class", UNCERTAIN_CLASS)) for record in records)
    total = max(len(records), 1)
    rows: List[Dict[str, Any]] = []
    for name in ALL_NUCLEUS_CLASSES:
        selected = [record for record in records if record.get("predicted_class") == name]
        rows.append({
            "class": name,
            "red": NUCLEUS_CLASS_COLORS[name][0],
            "green": NUCLEUS_CLASS_COLORS[name][1],
            "blue": NUCLEUS_CLASS_COLORS[name][2],
            "count": counts.get(name, 0),
            "fraction": counts.get(name, 0) / total,
            "mean_probability_top1": float(np.mean([_safe_float(r.get("probability_top1")) for r in selected])) if selected else 0.0,
            "mean_uncertainty_entropy": float(np.mean([_safe_float(r.get("uncertainty_entropy")) for r in selected])) if selected else 0.0,
            "mean_confidence": float(np.mean([_safe_float(r.get("confidence")) for r in selected])) if selected else 0.0,
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_palette(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "red", "green", "blue", "hex"])
        writer.writeheader()
        for name in ALL_NUCLEUS_CLASSES:
            r, g, b = NUCLEUS_CLASS_COLORS[name]
            writer.writerow({"class": name, "red": r, "green": g, "blue": b, "hex": f"#{r:02X}{g:02X}{b:02X}"})


def write_nuclei_geojson(path: Path, records: Sequence[Mapping[str, Any]], image_path: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write('{"type":"FeatureCollection","name":"Nucleus classifications","features":[\n')
        first = True
        for record in records:
            contour = record.get("contour") or []
            if len(contour) >= 4:
                geometry = {"type": "Polygon", "coordinates": [contour]}
            else:
                geometry = {"type": "Point", "coordinates": [record["centroid_x_px"], record["centroid_y_px"]]}
            properties = _scalar_record(record)
            properties["image"] = image_path
            properties["color_rgb"] = list(NUCLEUS_CLASS_COLORS.get(str(record.get("predicted_class")), NUCLEUS_CLASS_COLORS[UNCERTAIN_CLASS]))
            feature = {"type": "Feature", "geometry": geometry, "properties": properties}
            if not first:
                handle.write(",\n")
            handle.write(json.dumps(feature, ensure_ascii=False, separators=(",", ":")))
            first = False
        handle.write("\n]}\n")


def write_regions_csv(path: Path, regions: Sequence[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in regions]
    if not rows:
        path.write_text("region_id,predicted_region,nuclei_count\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_regions_geojson(path: Path, regions: Sequence[Mapping[str, Any]], image_path: str = "") -> None:
    features = []
    for region in regions:
        x = float(region["x_px"])
        y = float(region["y_px"])
        x2 = x + float(region["width_px"])
        y2 = y + float(region["height_px"])
        label = str(region["predicted_region"])
        properties = dict(region)
        properties["image"] = image_path
        properties["color_rgb"] = list(REGION_COLORS[label])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[x, y], [x2, y], [x2, y2], [x, y2], [x, y]]]},
            "properties": properties,
        })
    path.write_text(json.dumps({"type": "FeatureCollection", "name": "Graph-derived tissue regions", "features": features}, indent=2, ensure_ascii=False), encoding="utf-8")


def write_graphml(path: Path, records: Sequence[Mapping[str, Any]], edges: Sequence[Tuple[int, int, float]]) -> None:
    keys = [
        ("class", "string"), ("candidate_class", "string"), ("x_um", "double"), ("y_um", "double"),
        ("probability", "double"), ("entropy", "double"), ("confidence", "double"),
        ("area_um2", "double"), ("diameter_um", "double"), ("distance_um", "double"),
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        handle.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        for key, kind in keys:
            scope = "edge" if key == "distance_um" else "node"
            handle.write(f'  <key id="{key}" for="{scope}" attr.name="{key}" attr.type="{kind}"/>\n')
        handle.write('  <graph id="nuclei" edgedefault="undirected">\n')
        for i, record in enumerate(records):
            values = {
                "class": record.get("predicted_class", UNCERTAIN_CLASS),
                "candidate_class": record.get("candidate_class", ""),
                "x_um": record.get("centroid_x_um", 0.0),
                "y_um": record.get("centroid_y_um", 0.0),
                "probability": record.get("probability_top1", 0.0),
                "entropy": record.get("uncertainty_entropy", 0.0),
                "confidence": record.get("confidence", 0.0),
                "area_um2": record.get("area_um2", 0.0),
                "diameter_um": record.get("equivalent_diameter_um", 0.0),
            }
            handle.write(f'    <node id="n{int(record.get("nucleus_id", i + 1))}">\n')
            for key, value in values.items():
                handle.write(f'      <data key="{key}">{xmlutils.escape(str(value))}</data>\n')
            handle.write("    </node>\n")
        for edge_id, (a, b, distance) in enumerate(edges, 1):
            source_id = int(records[a].get("nucleus_id", a + 1))
            target_id = int(records[b].get("nucleus_id", b + 1))
            handle.write(f'    <edge id="e{edge_id}" source="n{source_id}" target="n{target_id}"><data key="distance_um">{distance:.6g}</data></edge>\n')
        handle.write("  </graph>\n</graphml>\n")


def _draw_nucleus_shape(canvas: np.ndarray, record: Mapping[str, Any], scale_x: float, scale_y: float, color: Tuple[int, int, int], alpha: float) -> None:
    contour = record.get("contour") or []
    layer = canvas.copy()
    if len(contour) >= 4:
        points = np.asarray([[round(float(x) * scale_x), round(float(y) * scale_y)] for x, y in contour], dtype=np.int32)
        if cv2.contourArea(points) >= 1.0:
            cv2.fillPoly(layer, [points], color)
            cv2.polylines(layer, [points], True, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            centre = (round(float(record["centroid_x_px"]) * scale_x), round(float(record["centroid_y_px"]) * scale_y))
            cv2.circle(layer, centre, 2, color, -1, cv2.LINE_AA)
    else:
        centre = (round(float(record["centroid_x_px"]) * scale_x), round(float(record["centroid_y_px"]) * scale_y))
        cv2.circle(layer, centre, 2, color, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, float(alpha), canvas, 1.0 - float(alpha), 0.0, dst=canvas)


def create_visualizations(
    output_dir: Path,
    thumbnail_rgb: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    edges: Sequence[Tuple[int, int, float]],
    regions: Sequence[Mapping[str, Any]],
    image_width: int,
    image_height: int,
) -> None:
    if Image is None:
        return
    base = np.asarray(thumbnail_rgb, dtype=np.uint8)
    h, w = base.shape[:2]
    sx = w / max(float(image_width), 1.0)
    sy = h / max(float(image_height), 1.0)

    class_overlay = base.copy()
    uncertainty_overlay = base.copy()
    for record in records:
        name = str(record.get("predicted_class", UNCERTAIN_CLASS))
        color = NUCLEUS_CLASS_COLORS.get(name, NUCLEUS_CLASS_COLORS[UNCERTAIN_CLASS])
        _draw_nucleus_shape(class_overlay, record, sx, sy, color, 0.68)
        confidence = float(np.clip(_safe_float(record.get("confidence")), 0.0, 1.0))
        _draw_nucleus_shape(uncertainty_overlay, record, sx, sy, color, 0.12 + 0.78 * confidence)

    graph_overlay = base.copy()
    max_edges = min(len(edges), 200000)
    for a, b, _distance in edges[:max_edges]:
        p1 = (round(float(records[a]["centroid_x_px"]) * sx), round(float(records[a]["centroid_y_px"]) * sy))
        p2 = (round(float(records[b]["centroid_x_px"]) * sx), round(float(records[b]["centroid_y_px"]) * sy))
        cv2.line(graph_overlay, p1, p2, (90, 90, 90), 1, cv2.LINE_AA)
    for record in records:
        centre = (round(float(record["centroid_x_px"]) * sx), round(float(record["centroid_y_px"]) * sy))
        color = NUCLEUS_CLASS_COLORS.get(str(record.get("predicted_class")), NUCLEUS_CLASS_COLORS[UNCERTAIN_CLASS])
        cv2.circle(graph_overlay, centre, 2, color, -1, cv2.LINE_AA)

    region_overlay = base.copy().astype(np.float32)
    layer = region_overlay.copy()
    for region in regions:
        x1 = round(float(region["x_px"]) * sx)
        y1 = round(float(region["y_px"]) * sy)
        x2 = round((float(region["x_px"]) + float(region["width_px"])) * sx)
        y2 = round((float(region["y_px"]) + float(region["height_px"])) * sy)
        color = REGION_COLORS[str(region["predicted_region"])]
        cv2.rectangle(layer, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(layer, (x1, y1), (x2, y2), (255, 255, 255), 1)
    region_overlay = np.clip(0.58 * region_overlay + 0.42 * layer, 0, 255).astype(np.uint8)

    Image.fromarray(class_overlay).save(output_dir / "nuclei_class_overlay.png")
    Image.fromarray(uncertainty_overlay).save(output_dir / "nuclei_class_uncertainty_overlay.png")
    Image.fromarray(graph_overlay).save(output_dir / "nuclei_graph_overlay.png")
    Image.fromarray(region_overlay).save(output_dir / "tissue_region_overlay.png")

    legend_width = 820
    line_height = 42
    legend = np.full((line_height * (len(ALL_NUCLEUS_CLASSES) + 1), legend_width, 3), 255, dtype=np.uint8)
    cv2.putText(legend, "Nucleus classes: color, count and mean confidence", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    counts = Counter(str(r.get("predicted_class", UNCERTAIN_CLASS)) for r in records)
    for row, name in enumerate(ALL_NUCLEUS_CLASSES, 1):
        y = row * line_height
        cv2.rectangle(legend, (15, y - 28), (45, y + 2), NUCLEUS_CLASS_COLORS[name], -1)
        selected = [r for r in records if r.get("predicted_class") == name]
        mean_conf = float(np.mean([_safe_float(r.get("confidence")) for r in selected])) if selected else 0.0
        text = f"{name}: n={counts.get(name, 0):,}, mean confidence={mean_conf:.3f}"
        cv2.putText(legend, text, (60, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (25, 25, 25), 1, cv2.LINE_AA)
    Image.fromarray(legend).save(output_dir / "nuclei_class_legend.png")


def write_all_outputs(
    output_dir: Path,
    records: List[Dict[str, Any]],
    edges: Sequence[Tuple[int, int, float]],
    regions: Sequence[Mapping[str, Any]],
    thumbnail_rgb: np.ndarray,
    image_width: int,
    image_height: int,
    image_path: str,
    model_info: Mapping[str, Any],
    region_size_px: int,
    pixel_size_um: float,
    backend_counts: Mapping[str, int],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_nuclei_csv(output_dir / "nuclei_classification.csv", records)
    summary_rows = write_class_summary(output_dir / "nuclei_class_summary.csv", records)
    write_palette(output_dir / "nuclei_class_palette.csv")
    write_nuclei_geojson(output_dir / "nuclei_classification.geojson", records, image_path=image_path)
    write_graphml(output_dir / "nuclei_graph.graphml", records, edges)
    write_regions_csv(output_dir / "tissue_region_features.csv", regions)
    write_regions_geojson(output_dir / "tissue_regions.geojson", regions, image_path=image_path)
    create_visualizations(output_dir, thumbnail_rgb, records, edges, regions, image_width, image_height)
    summary = {
        "saved": True,
        "nuclei_count": len(records),
        "graph_nodes": len(records),
        "graph_edges": len(edges),
        "region_count": len(regions),
        "region_size_px": int(region_size_px),
        "pixel_size_um": float(pixel_size_um),
        "backend_counts": dict(backend_counts),
        "model": dict(model_info),
        "class_summary": summary_rows,
        "outputs": {
            "csv": "nuclei_classification.csv",
            "geojson": "nuclei_classification.geojson",
            "summary": "nuclei_class_summary.csv",
            "palette": "nuclei_class_palette.csv",
            "class_overlay": "nuclei_class_overlay.png",
            "uncertainty_overlay": "nuclei_class_uncertainty_overlay.png",
            "graph": "nuclei_graph.graphml",
            "graph_overlay": "nuclei_graph_overlay.png",
            "region_csv": "tissue_region_features.csv",
            "region_geojson": "tissue_regions.geojson",
            "region_overlay": "tissue_region_overlay.png",
        },
    }
    (output_dir / "nuclei_classification_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
