#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone sequential QuPath pixel-classifier pipeline (no QuPath required).

Pipeline
--------
1. TissueClassifierANNFullJuly06.json -> Tissue
2. AnthraJuly06.json inside Tissue -> Anthracosis/ink
3. Dilate anthracosis and subtract it -> CleanTissue
4. InstanSeg brightfield nuclei segmentation (preferred) or classical watershed fallback
5. H-channel nuclear/regional features -> Tumor/Stroma/Other classifier
6. DABCNNThreshold0.17DAB.json inside Tumor and Stroma separately

Outputs
-------
- Training mode from Tumor/Stroma/Other GeoJSON annotations.
- Prediction mode with compartment GeoJSON, CSV measurements, and overlays.
- Baseline mode preserving the original whole-tissue DAB workflow.

Important
---------
This script reconstructs the OpenCV ANN_MLP and RTrees models serialized inside
QuPath JSON files, using OpenCV's own model loader. It reproduces the operations
used by the three supplied classifiers: RGB Gaussian features (sigma=1), ANN/RTrees
prediction, and H-DAB stain-2 thresholding.

Research-use software. Validate representative regions against the original
QuPath classifier before using results for a complete study.

Version 2.4.0 adds the public InstanSeg brightfield_nuclei model as the
preferred nuclei backend, with RGB input, automatic CPU/CUDA selection,
pixel-size-aware inference, and watershed fallback. It retains unique per-run
temporary folders to avoid Windows/OneDrive WinError 5 folder-lock failures.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tifffile

try:
    import zarr  # type: ignore
except Exception:
    zarr = None

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except Exception:
    Image = None


APP_NAME = "Standalone H-DAB Nuclei Compartment Pipeline"
APP_VERSION = "2.4.0"
BUILD_ID = "2026-07-16-INSTANSEG-QUPATH-PARITY-C"

DEFAULT_BASE = Path(
    r"C:\Users\juaco\OneDrive\Escritorio\HospitalClinic\SegmentationTask\pixel_classifiers"
)
DEFAULT_IMAGE = DEFAULT_BASE / "ID_2041_1.tif"
DEFAULT_TISSUE = DEFAULT_BASE / "TissueClassifierANNFullJuly06.json"
DEFAULT_ANTHRA = DEFAULT_BASE / "AnthraJuly06.json"
DEFAULT_DAB = DEFAULT_BASE / "DABCNNThreshold0.17DAB.json"
DEFAULT_COMPARTMENT_MODEL = DEFAULT_BASE / "TumorStromaRF_InstanSeg.joblib"
DEFAULT_TRAINING_ANNOTATIONS = DEFAULT_BASE / "ID_2041_1_compartments.geojson"

# Spyder-friendly defaults. Running the file with no command uses this mode.
# - "auto": predict if the model exists; otherwise train then predict if the
#   default annotation GeoJSON exists; otherwise show an actionable error.
# - "baseline", "train", or "predict": force that mode.
SPYDER_DEFAULT_MODE = "auto"
DEFAULT_NUCLEI_BACKEND = "instanseg"
DEFAULT_INSTANSEG_MODEL = "brightfield_nuclei"
DEFAULT_INSTANSEG_INPUT = "rgb"
DEFAULT_INSTANSEG_DEVICE = "auto"
DEFAULT_INSTANSEG_TILE_SIZE = 512
DEFAULT_INSTANSEG_BATCH_SIZE = 1
DEFAULT_PIXEL_SIZE_FALLBACK_UM = 0.50

SUPPORTED_TIFF = (".tif", ".tiff", ".ome.tif", ".ome.tiff")
SUPPORTED_RASTER = (".png", ".jpg", ".jpeg", ".bmp")
SUPPORTED_OPENSLIDE = (".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif")


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def log(message: str) -> None:
    print(message, flush=True)


def create_unique_work_dir(parent: Path, prefix: str) -> Path:
    """Create a per-run temporary folder inside the requested output directory.

    A unique folder prevents OneDrive, antivirus software, Explorer previews, or a
    previous interrupted Spyder run from blocking the next analysis because a
    fixed `_work_masks` directory is still locked.
    """
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(parent)))


def close_memmap_dict(masks: Dict[str, np.memmap]) -> None:
    """Flush and explicitly close NumPy memmaps on Windows."""
    for mm in list(masks.values()):
        try:
            mm.flush()
        except Exception:
            pass
        try:
            mmap_obj = getattr(mm, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        except Exception:
            pass
    masks.clear()
    gc.collect()


def safe_remove_tree(path: Path, attempts: int = 5, delay_seconds: float = 0.35) -> bool:
    """Best-effort removal that never aborts a completed analysis.

    Windows and OneDrive may hold a directory briefly after memmap files are
    closed. Retrying avoids most transient WinError 5 failures. If a lock
    remains, the directory is left in place and a warning is printed; the next
    run uses a different unique directory and is therefore unaffected.
    """
    path = Path(path)
    if not path.exists():
        return True
    last_error: Optional[BaseException] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            gc.collect()
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except (PermissionError, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(delay_seconds * attempt)
    log(f"WARNING: Could not remove temporary folder '{path}': {last_error}")
    log("The results are complete; this stale temporary folder can be deleted later.")
    return False


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def percent(num: float, den: float) -> float:
    return 100.0 * safe_div(num, den)


def safe_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for ext in sorted(SUPPORTED_TIFF + SUPPORTED_RASTER + SUPPORTED_OPENSLIDE, key=len, reverse=True):
        if lower.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def find_key_recursive(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key_recursive(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key_recursive(value, key)
            if found is not None:
                return found
    return None


def find_ops_by_type(obj: Any, op_type: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if obj.get("type") == op_type:
            out.append(obj)
        for value in obj.values():
            out.extend(find_ops_by_type(value, op_type))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(find_ops_by_type(value, op_type))
    return out


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("pixel_classifier_type") != "OpenCVPixelClassifier":
        raise ValueError(f"Unsupported classifier type in {path}: {data.get('pixel_classifier_type')}")
    return data


def normalize_classification_labels(labels: Dict[str, Any]) -> Dict[int, str]:
    result: Dict[int, str] = {}
    for key, value in labels.items():
        idx = int(key)
        result[idx] = str(value.get("name")) if isinstance(value, dict) else str(value)
    return result


def validate_resolution_one_px(data: Dict[str, Any], path: Path) -> None:
    res = data.get("metadata", {}).get("inputResolution", {})
    pw = res.get("pixelWidth", {})
    ph = res.get("pixelHeight", {})
    if str(pw.get("unit", "px")).lower() != "px" or str(ph.get("unit", "px")).lower() != "px":
        raise ValueError(f"{path.name}: only classifiers defined in pixels are supported.")
    if abs(float(pw.get("value", 1.0)) - 1.0) > 1e-9 or abs(float(ph.get("value", 1.0)) - 1.0) > 1e-9:
        raise ValueError(
            f"{path.name}: classifier resolution is not 1 px/pixel. "
            "This script currently requires 1.0 px in X and Y."
        )


def get_rgb_gaussian_sigma(data: Dict[str, Any], path: Path) -> Tuple[float, float]:
    transforms = data.get("op", {}).get("colorTransforms", [])
    names = [str(x.get("channelName", "")) for x in transforms]
    if names != ["Red", "Green", "Blue"]:
        raise ValueError(f"{path.name}: expected RGB transforms, found {names}")

    multiscale_ops = find_ops_by_type(data, "op.filters.multiscale")
    if len(multiscale_ops) != 1:
        raise ValueError(f"{path.name}: expected one multiscale feature operation.")
    op = multiscale_ops[0]
    if op.get("features") != ["GAUSSIAN"]:
        raise ValueError(f"{path.name}: only GAUSSIAN features are supported, found {op.get('features')}")

    preprocessors = find_ops_by_type(data, "op.ml.feature-preprocessor")
    if preprocessors:
        normalizer = preprocessors[0].get("preprocessor", {}).get("normalizer", {})
        offsets = np.asarray(normalizer.get("offsets", [0, 0, 0]), dtype=float)
        scales = np.asarray(normalizer.get("scales", [1, 1, 1]), dtype=float)
        if not np.allclose(offsets, 0) or not np.allclose(scales, 1):
            raise ValueError(
                f"{path.name}: this build supports the identity feature normalizer used by the supplied models."
            )
    return float(op.get("sigmaX", 0.0)), float(op.get("sigmaY", 0.0))


def get_classifier_tile_size(data: Dict[str, Any]) -> int:
    width = int(data.get("metadata", {}).get("inputWidth", 512) or 512)
    height = int(data.get("metadata", {}).get("inputHeight", width) or width)
    return max(64, min(width, height))


# -----------------------------------------------------------------------------
# OpenCV model reconstruction from QuPath JSON
# -----------------------------------------------------------------------------


def _fs_write_value(fs: cv2.FileStorage, name: str, value: Any) -> None:
    if isinstance(value, dict):
        fs.startWriteStruct(name, cv2.FileNode_MAP)
        for key, child in value.items():
            _fs_write_value(fs, str(key), child)
        fs.endWriteStruct()
    elif isinstance(value, list):
        fs.startWriteStruct(name, cv2.FileNode_SEQ)
        for child in value:
            _fs_write_value(fs, "", child)
        fs.endWriteStruct()
    elif isinstance(value, bool):
        fs.write(name, int(value))
    elif isinstance(value, int):
        fs.write(name, int(value))
    elif isinstance(value, float):
        fs.write(name, float(value))
    elif value is None:
        fs.write(name, "")
    else:
        fs.write(name, str(value))


def write_opencv_yaml(root_name: str, model_data: Dict[str, Any], output_path: Path) -> None:
    fs = cv2.FileStorage(
        str(output_path), cv2.FileStorage_WRITE | cv2.FileStorage_FORMAT_YAML
    )
    if not fs.isOpened():
        raise RuntimeError(f"Could not create temporary OpenCV model file: {output_path}")
    try:
        _fs_write_value(fs, root_name, model_data)
    finally:
        fs.release()


@dataclass
class ReconstructedModels:
    tissue_ann: Any
    anthra_rtrees: Any
    tissue_sigma: Tuple[float, float]
    anthra_sigma: Tuple[float, float]
    tissue_labels: Dict[int, str]
    anthra_labels: Dict[int, str]
    tile_size: int
    tempdir: tempfile.TemporaryDirectory

    def close(self) -> None:
        self.tempdir.cleanup()


def reconstruct_models(tissue_json: Path, anthra_json: Path) -> ReconstructedModels:
    tissue_data = load_json(tissue_json)
    anthra_data = load_json(anthra_json)
    validate_resolution_one_px(tissue_data, tissue_json)
    validate_resolution_one_px(anthra_data, anthra_json)

    tissue_model_data = find_key_recursive(tissue_data, "opencv_ml_ann_mlp")
    anthra_model_data = find_key_recursive(anthra_data, "opencv_ml_rtrees")
    if tissue_model_data is None:
        raise ValueError(f"{tissue_json.name}: ANN_MLP model data not found.")
    if anthra_model_data is None:
        raise ValueError(f"{anthra_json.name}: RTrees model data not found.")

    tempdir = tempfile.TemporaryDirectory(prefix="qupath_json_models_")
    temp_path = Path(tempdir.name)
    ann_path = temp_path / "tissue_ann.yml"
    rt_path = temp_path / "anthra_rtrees.yml"
    write_opencv_yaml("opencv_ml_ann_mlp", tissue_model_data, ann_path)
    write_opencv_yaml("opencv_ml_rtrees", anthra_model_data, rt_path)

    tissue_ann = cv2.ml.ANN_MLP_load(str(ann_path))
    anthra_rtrees = cv2.ml.RTrees_load(str(rt_path))
    if tissue_ann is None or tissue_ann.empty():
        tempdir.cleanup()
        raise RuntimeError("OpenCV failed to reconstruct the Tissue ANN model.")
    if anthra_rtrees is None or anthra_rtrees.empty():
        tempdir.cleanup()
        raise RuntimeError("OpenCV failed to reconstruct the Anthracosis RTrees model.")

    tissue_labels = normalize_classification_labels(
        tissue_data.get("metadata", {}).get("classificationLabels", {})
    )
    anthra_labels = normalize_classification_labels(
        anthra_data.get("metadata", {}).get("classificationLabels", {})
    )
    if tissue_labels.get(1, "").lower() != "tissue":
        tempdir.cleanup()
        raise ValueError(f"Expected Tissue as class 1, found {tissue_labels}")
    if "anthracosis" not in anthra_labels.get(1, "").lower() and "marker" not in anthra_labels.get(1, "").lower():
        tempdir.cleanup()
        raise ValueError(f"Expected anthracosis/marker as class 1, found {anthra_labels}")

    tile_size = min(get_classifier_tile_size(tissue_data), get_classifier_tile_size(anthra_data))
    return ReconstructedModels(
        tissue_ann=tissue_ann,
        anthra_rtrees=anthra_rtrees,
        tissue_sigma=get_rgb_gaussian_sigma(tissue_data, tissue_json),
        anthra_sigma=get_rgb_gaussian_sigma(anthra_data, anthra_json),
        tissue_labels=tissue_labels,
        anthra_labels=anthra_labels,
        tile_size=tile_size,
        tempdir=tempdir,
    )


# -----------------------------------------------------------------------------
# DAB threshold classifier
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DABThresholdClassifier:
    threshold: float
    stain_matrix: np.ndarray
    inverse_matrix: np.ndarray
    max_rgb: Tuple[float, float, float]
    labels: Dict[int, str]

    @classmethod
    def from_json(cls, path: Path) -> "DABThresholdClassifier":
        data = load_json(path)
        validate_resolution_one_px(data, path)
        transforms = data.get("op", {}).get("colorTransforms", [])
        if len(transforms) != 1 or int(transforms[0].get("stainNumber", -1)) != 2:
            raise ValueError(f"{path.name}: expected stainNumber=2 (DAB).")
        stains = transforms[0].get("stains", {})
        vectors = []
        for key in ("stain1", "stain2", "stain3"):
            s = stains.get(key, {})
            vectors.append([float(s["r"]), float(s["g"]), float(s["b"])])
        # QuPath constructs a 3x3 matrix whose rows are stain vectors, then inverts it.
        matrix = np.asarray(vectors, dtype=np.float64)
        inverse = np.linalg.inv(matrix)

        threshold_ops = find_ops_by_type(data, "op.threshold.constant")
        if len(threshold_ops) != 1:
            raise ValueError(f"{path.name}: expected one constant threshold operation.")
        thresholds = threshold_ops[0].get("thresholds", [])
        if len(thresholds) != 1:
            raise ValueError(f"{path.name}: expected one threshold value.")
        labels = normalize_classification_labels(
            data.get("metadata", {}).get("classificationLabels", {})
        )
        return cls(
            threshold=float(thresholds[0]),
            stain_matrix=matrix,
            inverse_matrix=inverse,
            max_rgb=(
                float(stains.get("maxRed", 255.0)),
                float(stains.get("maxGreen", 255.0)),
                float(stains.get("maxBlue", 255.0)),
            ),
            labels=labels,
        )

    def dab_values(self, rgb: np.ndarray) -> np.ndarray:
        rgb8 = np.asarray(rgb, dtype=np.uint8)
        # QuPath: max(0, -log10(max(value, 1) / channel_max)).
        r = np.maximum(rgb8[..., 0].astype(np.float64), 1.0)
        g = np.maximum(rgb8[..., 1].astype(np.float64), 1.0)
        b = np.maximum(rgb8[..., 2].astype(np.float64), 1.0)
        od_r = np.maximum(0.0, -np.log10(r / self.max_rgb[0]))
        od_g = np.maximum(0.0, -np.log10(g / self.max_rgb[1]))
        od_b = np.maximum(0.0, -np.log10(b / self.max_rgb[2]))
        # QuPath deconvolution: OD row vector multiplied by inverse matrix.
        dab = (
            od_r * self.inverse_matrix[0, 1]
            + od_g * self.inverse_matrix[1, 1]
            + od_b * self.inverse_matrix[2, 1]
        )
        return dab.astype(np.float32)

    def predict_positive(self, rgb: np.ndarray) -> np.ndarray:
        return self.dab_values(rgb) >= self.threshold


# -----------------------------------------------------------------------------
# Image reading
# -----------------------------------------------------------------------------


def _convert_to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape after slicing: {arr.shape}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 2:
        arr = np.repeat(arr[..., :1], 3, axis=-1)
    elif arr.shape[-1] >= 3:
        arr = arr[..., :3]

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = np.clip(arr.astype(np.float32) / float(info.max), 0.0, 1.0) * 255.0
    else:
        arr = arr.astype(np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros(arr.shape, dtype=np.uint8)
        vmax = float(np.max(finite))
        if vmax <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
    return np.ascontiguousarray(np.rint(arr).astype(np.uint8))


def _extract_ome_mpp(ome_xml: Optional[str]) -> Optional[Tuple[float, float]]:
    if not ome_xml:
        return None
    try:
        root = ET.fromstring(ome_xml)
        pixels = next((e for e in root.iter() if e.tag.endswith("Pixels")), None)
        if pixels is None:
            return None
        x = pixels.attrib.get("PhysicalSizeX")
        y = pixels.attrib.get("PhysicalSizeY")
        if x is None or y is None:
            return None

        def to_um(value: str, unit: str) -> float:
            val = float(value)
            u = unit.lower().replace("µ", "u")
            factors = {"um": 1.0, "nm": 0.001, "mm": 1000.0, "cm": 10000.0, "m": 1e6}
            return val * factors.get(u, 1.0)

        return (
            to_um(x, pixels.attrib.get("PhysicalSizeXUnit", "um")),
            to_um(y, pixels.attrib.get("PhysicalSizeYUnit", "um")),
        )
    except Exception:
        return None


def _tag_float(tag: Any) -> Optional[float]:
    if tag is None:
        return None
    value = tag.value
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except Exception:
        return None


class ImageSource:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.width = 0
        self.height = 0
        self.mpp: Optional[Tuple[float, float]] = None
        self.reader = ""
        self.axes = ""
        self._tif: Optional[tifffile.TiffFile] = None
        self._series: Any = None
        self._zarr: Any = None
        self._full_arr: Optional[np.ndarray] = None
        self._slide: Any = None
        self._pil: Any = None
        self._open()

    def _open(self) -> None:
        lower = self.path.name.lower()
        if lower.endswith(SUPPORTED_TIFF):
            try:
                self._open_tiff()
                return
            except Exception as exc:
                tiff_error = exc
                try:
                    self._open_openslide()
                    return
                except Exception:
                    raise RuntimeError(
                        f"Could not open TIFF with tifffile or OpenSlide. tifffile error: {tiff_error}"
                    ) from exc
        if lower.endswith(SUPPORTED_OPENSLIDE):
            self._open_openslide()
            return
        if lower.endswith(SUPPORTED_RASTER):
            self._open_pil()
            return
        raise ValueError(f"Unsupported image extension: {self.path}")

    def _open_tiff(self) -> None:
        self._tif = tifffile.TiffFile(str(self.path))
        if not self._tif.series:
            raise ValueError("No TIFF image series found.")
        self._series = self._tif.series[0]
        self.axes = str(getattr(self._series, "axes", ""))
        shape = tuple(int(v) for v in self._series.shape)
        if "Y" in self.axes and "X" in self.axes:
            self.height = shape[self.axes.index("Y")]
            self.width = shape[self.axes.index("X")]
        else:
            self.height, self.width = shape[:2]

        self.mpp = _extract_ome_mpp(self._tif.ome_metadata)
        if self.mpp is None:
            page = self._series.pages[0] if getattr(self._series, "pages", None) else self._tif.pages[0]
            xres = _tag_float(page.tags.get("XResolution"))
            yres = _tag_float(page.tags.get("YResolution"))
            unit = page.tags.get("ResolutionUnit")
            unit_value = int(unit.value) if unit is not None else 1
            if xres and yres and unit_value in (2, 3):
                length_um = 25400.0 if unit_value == 2 else 10000.0
                self.mpp = (length_um / xres, length_um / yres)

        if zarr is not None:
            try:
                self._zarr = zarr.open(self._series.aszarr(), mode="r")
            except Exception:
                self._zarr = None
        if self._zarr is None:
            # memmap may work for uncompressed TIFF; otherwise full read is the fallback.
            try:
                self._full_arr = tifffile.memmap(str(self.path), series=0)
            except Exception:
                self._full_arr = self._series.asarray()
        self.reader = "tifffile-zarr" if self._zarr is not None else "tifffile-array"

    def _open_openslide(self) -> None:
        try:
            import openslide  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "OpenSlide is required for this image type. Install openslide-python and openslide-bin."
            ) from exc
        self._slide = openslide.OpenSlide(str(self.path))
        self.width, self.height = map(int, self._slide.dimensions)
        props = dict(self._slide.properties or {})
        try:
            mx = float(props.get("openslide.mpp-x"))
            my = float(props.get("openslide.mpp-y"))
            self.mpp = (mx, my)
        except Exception:
            self.mpp = None
        self.reader = "openslide"

    def _open_pil(self) -> None:
        if Image is None:
            raise RuntimeError("Pillow is required for raster images.")
        self._pil = Image.open(str(self.path)).convert("RGB")
        self.width, self.height = map(int, self._pil.size)
        self.reader = "pillow"

    def close(self) -> None:
        if self._tif is not None:
            self._tif.close()
            self._tif = None
        if self._slide is not None:
            self._slide.close()
            self._slide = None
        if self._pil is not None:
            self._pil.close()
            self._pil = None

    def __enter__(self) -> "ImageSource":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _spatial_slice(self, x: slice, y: slice) -> Tuple[Any, ...]:
        ndim = len(self._series.shape)
        axes = self.axes
        slicer: List[Any] = []
        if axes and len(axes) == ndim and "X" in axes and "Y" in axes:
            for axis in axes:
                if axis == "Y":
                    slicer.append(y)
                elif axis == "X":
                    slicer.append(x)
                elif axis in ("C", "S"):
                    slicer.append(slice(None))
                else:
                    slicer.append(0)
        else:
            for i in range(ndim):
                if i == 0:
                    slicer.append(y)
                elif i == 1:
                    slicer.append(x)
                else:
                    slicer.append(slice(None))
        return tuple(slicer)

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        x = max(0, min(int(x), self.width - 1))
        y = max(0, min(int(y), self.height - 1))
        width = max(1, min(int(width), self.width - x))
        height = max(1, min(int(height), self.height - y))
        if self.reader.startswith("tifffile"):
            arr_source = self._zarr if self._zarr is not None else self._full_arr
            arr = arr_source[self._spatial_slice(slice(x, x + width), slice(y, y + height))]
            return _convert_to_rgb_uint8(np.asarray(arr))
        if self.reader == "openslide":
            return np.asarray(
                self._slide.read_region((x, y), 0, (width, height)).convert("RGB"),
                dtype=np.uint8,
            )
        if self.reader == "pillow":
            return np.asarray(self._pil.crop((x, y, x + width, y + height)), dtype=np.uint8)
        raise RuntimeError(f"Unknown reader: {self.reader}")

    def thumbnail(self, max_side: int) -> np.ndarray:
        scale = min(1.0, float(max_side) / float(max(self.width, self.height)))
        tw = max(1, int(round(self.width * scale)))
        th = max(1, int(round(self.height * scale)))
        if self.reader == "openslide":
            return np.asarray(self._slide.get_thumbnail((tw, th)).convert("RGB"), dtype=np.uint8)
        if self.reader == "pillow":
            img = self._pil.copy()
            img.thumbnail((tw, th), Image.Resampling.LANCZOS)
            return np.asarray(img, dtype=np.uint8)
        # TIFF: sample the level-0 array with a regular stride, then resize exactly.
        step = max(1, int(math.floor(max(self.width / tw, self.height / th))))
        arr_source = self._zarr if self._zarr is not None else self._full_arr
        arr = arr_source[self._spatial_slice(slice(0, self.width, step), slice(0, self.height, step))]
        rgb = _convert_to_rgb_uint8(np.asarray(arr))
        if rgb.shape[1] != tw or rgb.shape[0] != th:
            rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        return rgb


# -----------------------------------------------------------------------------
# Tile processing and disk-backed masks
# -----------------------------------------------------------------------------


MASK_NAMES = (
    "tissue",
    "anthra_raw",
    "anthra_dilated",
    "clean_tissue",
    "positive",
    "negative",
)


class MaskWorkspace:
    def __init__(self, root: Path, shape: Tuple[int, int]):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.shape = shape
        self.masks: Dict[str, np.memmap] = {}
        for name in MASK_NAMES:
            path = self.root / f"{name}.uint8.dat"
            mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
            mm[:] = 0
            self.masks[name] = mm

    def flush(self) -> None:
        for mm in self.masks.values():
            mm.flush()

    def close(self) -> None:
        close_memmap_dict(self.masks)


def iter_tiles(width: int, height: int, tile_size: int) -> Iterator[Tuple[int, int, int, int]]:
    for y in range(0, height, tile_size):
        h = min(tile_size, height - y)
        for x in range(0, width, tile_size):
            w = min(tile_size, width - x)
            yield x, y, w, h


def read_with_halo(
    source: ImageSource, x: int, y: int, w: int, h: int, halo: int
) -> Tuple[np.ndarray, Tuple[slice, slice]]:
    rx0 = max(0, x - halo)
    ry0 = max(0, y - halo)
    rx1 = min(source.width, x + w + halo)
    ry1 = min(source.height, y + h + halo)
    rgb = source.read_region(rx0, ry0, rx1 - rx0, ry1 - ry0)
    core_x0 = x - rx0
    core_y0 = y - ry0
    return rgb, (slice(core_y0, core_y0 + h), slice(core_x0, core_x0 + w))


def gaussian_rgb_features(rgb: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    if sigma_x <= 0 and sigma_y <= 0:
        blurred = rgb
    else:
        blurred = cv2.GaussianBlur(
            rgb,
            ksize=(0, 0),
            sigmaX=max(float(sigma_x), 1e-12),
            sigmaY=max(float(sigma_y), 1e-12),
            borderType=cv2.BORDER_REFLECT_101,
        )
    return blurred.astype(np.float32, copy=False)


def predict_tissue(ann: Any, features_rgb: np.ndarray) -> np.ndarray:
    samples = np.ascontiguousarray(features_rgb.reshape(-1, 3), dtype=np.float32)
    _, outputs = ann.predict(samples)
    outputs = np.asarray(outputs)
    if outputs.ndim == 2 and outputs.shape[1] >= 2:
        labels = np.argmax(outputs, axis=1).astype(np.uint8)
    else:
        labels = (outputs.reshape(-1) > 0).astype(np.uint8)
    return labels.reshape(features_rgb.shape[:2]).astype(bool)


def predict_anthra(rtrees: Any, features_rgb: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    result = np.zeros(tissue_mask.shape, dtype=bool)
    flat_tissue = tissue_mask.reshape(-1)
    indices = np.flatnonzero(flat_tissue)
    if indices.size == 0:
        return result
    samples_all = np.ascontiguousarray(features_rgb.reshape(-1, 3), dtype=np.float32)
    # Predict in batches to limit temporary allocations for large tiles.
    out_flat = result.reshape(-1)
    batch = 250_000
    for start in range(0, indices.size, batch):
        idx = indices[start : start + batch]
        _, pred = rtrees.predict(samples_all[idx])
        out_flat[idx] = np.asarray(pred).reshape(-1).astype(np.int32) == 1
    return result


def first_pass_classification(
    source: ImageSource,
    models: ReconstructedModels,
    workspace: MaskWorkspace,
    tile_size: int,
) -> None:
    tissue_mm = workspace.masks["tissue"]
    anthra_mm = workspace.masks["anthra_raw"]
    sigma_x = max(models.tissue_sigma[0], models.anthra_sigma[0])
    sigma_y = max(models.tissue_sigma[1], models.anthra_sigma[1])
    halo = max(2, int(math.ceil(4.0 * max(sigma_x, sigma_y))))
    tiles = list(iter_tiles(source.width, source.height, tile_size))
    total = len(tiles)
    started = time.time()

    for i, (x, y, w, h) in enumerate(tiles, start=1):
        rgb_halo, core_slice = read_with_halo(source, x, y, w, h, halo)
        tissue_features = gaussian_rgb_features(rgb_halo, *models.tissue_sigma)[core_slice]
        tissue = predict_tissue(models.tissue_ann, tissue_features)

        if models.anthra_sigma == models.tissue_sigma:
            anthra_features = tissue_features
        else:
            anthra_features = gaussian_rgb_features(rgb_halo, *models.anthra_sigma)[core_slice]
        anthra = predict_anthra(models.anthra_rtrees, anthra_features, tissue)

        tissue_mm[y : y + h, x : x + w] = tissue.astype(np.uint8)
        anthra_mm[y : y + h, x : x + w] = (anthra & tissue).astype(np.uint8)

        if i == 1 or i == total or i % max(1, total // 20) == 0:
            elapsed = time.time() - started
            log(f"  Tissue/anthracosis tiles: {i}/{total} ({100*i/total:.1f}%) | {elapsed:.1f}s")
    workspace.flush()


def dilate_anthracosis_and_make_clean_tissue(
    workspace: MaskWorkspace,
    width: int,
    height: int,
    radius: int,
    stripe_height: int = 2048,
) -> None:
    tissue = workspace.masks["tissue"]
    raw = workspace.masks["anthra_raw"]
    dilated = workspace.masks["anthra_dilated"]
    clean = workspace.masks["clean_tissue"]
    radius = max(0, int(radius))
    if radius == 0:
        dilated[:] = raw[:]
        clean[:] = np.logical_and(tissue, np.logical_not(raw)).astype(np.uint8)
        workspace.flush()
        return

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    total = math.ceil(height / stripe_height)
    for idx, y0 in enumerate(range(0, height, stripe_height), start=1):
        y1 = min(height, y0 + stripe_height)
        ry0 = max(0, y0 - radius)
        ry1 = min(height, y1 + radius)
        src = np.asarray(raw[ry0:ry1, :], dtype=np.uint8)
        expanded = cv2.dilate(src, kernel, iterations=1)
        core = expanded[y0 - ry0 : y1 - ry0, :]
        tissue_core = np.asarray(tissue[y0:y1, :], dtype=np.uint8)
        core = np.logical_and(core > 0, tissue_core > 0)
        dilated[y0:y1, :] = core.astype(np.uint8)
        clean[y0:y1, :] = np.logical_and(tissue_core > 0, ~core).astype(np.uint8)
        log(f"  Ink dilation stripes: {idx}/{total}")
    workspace.flush()


def dab_pass(
    source: ImageSource,
    dab_classifier: DABThresholdClassifier,
    workspace: MaskWorkspace,
    tile_size: int,
) -> None:
    clean = workspace.masks["clean_tissue"]
    positive = workspace.masks["positive"]
    negative = workspace.masks["negative"]
    tiles = list(iter_tiles(source.width, source.height, tile_size))
    total = len(tiles)
    started = time.time()
    for i, (x, y, w, h) in enumerate(tiles, start=1):
        clean_tile = np.asarray(clean[y : y + h, x : x + w], dtype=np.uint8) > 0
        if not np.any(clean_tile):
            continue
        rgb = source.read_region(x, y, w, h)
        pos = np.logical_and(dab_classifier.predict_positive(rgb), clean_tile)
        neg = np.logical_and(clean_tile, ~pos)
        positive[y : y + h, x : x + w] = pos.astype(np.uint8)
        negative[y : y + h, x : x + w] = neg.astype(np.uint8)
        if i == 1 or i == total or i % max(1, total // 20) == 0:
            elapsed = time.time() - started
            log(f"  DAB tiles: {i}/{total} ({100*i/total:.1f}%) | {elapsed:.1f}s")
    workspace.flush()


# -----------------------------------------------------------------------------
# Measurements
# -----------------------------------------------------------------------------


def count_mask(mask: np.memmap, stripe_height: int = 4096) -> int:
    total = 0
    for y in range(0, mask.shape[0], stripe_height):
        total += int(np.count_nonzero(mask[y : y + stripe_height, :]))
    return total


def compute_measurements(
    image_path: Path,
    source: ImageSource,
    workspace: MaskWorkspace,
    ink_dilation: int,
    dab_threshold: float,
) -> Dict[str, Any]:
    counts = {name: count_mask(workspace.masks[name]) for name in MASK_NAMES}
    total_pixels = int(source.width) * int(source.height)
    pixel_area_um2 = None
    if source.mpp is not None:
        pixel_area_um2 = float(source.mpp[0]) * float(source.mpp[1])

    result: Dict[str, Any] = {
        "image": str(image_path),
        "reader": source.reader,
        "width_px": source.width,
        "height_px": source.height,
        "total_pixels": total_pixels,
        "mpp_x_um": source.mpp[0] if source.mpp else "",
        "mpp_y_um": source.mpp[1] if source.mpp else "",
        "ink_dilation_radius_px": int(ink_dilation),
        "dab_threshold": float(dab_threshold),
        "tissue_pixels": counts["tissue"],
        "anthracosis_raw_pixels": counts["anthra_raw"],
        "anthracosis_dilated_pixels": counts["anthra_dilated"],
        "clean_tissue_pixels": counts["clean_tissue"],
        "positive_pixels": counts["positive"],
        "negative_pixels": counts["negative"],
        "tissue_percent_image": percent(counts["tissue"], total_pixels),
        "anthracosis_raw_percent_tissue": percent(counts["anthra_raw"], counts["tissue"]),
        "anthracosis_dilated_percent_tissue": percent(counts["anthra_dilated"], counts["tissue"]),
        "clean_tissue_percent_tissue": percent(counts["clean_tissue"], counts["tissue"]),
        "positive_percent_clean_tissue": percent(counts["positive"], counts["clean_tissue"]),
        "negative_percent_clean_tissue": percent(counts["negative"], counts["clean_tissue"]),
    }
    if pixel_area_um2 is not None:
        for name, count in counts.items():
            key = {
                "tissue": "tissue_area_mm2",
                "anthra_raw": "anthracosis_raw_area_mm2",
                "anthra_dilated": "anthracosis_dilated_area_mm2",
                "clean_tissue": "clean_tissue_area_mm2",
                "positive": "positive_area_mm2",
                "negative": "negative_area_mm2",
            }[name]
            result[key] = count * pixel_area_um2 / 1_000_000.0
    return result


def save_measurements_csv(path: Path, measurements: Dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(measurements.keys()))
        writer.writeheader()
        writer.writerow(measurements)


# -----------------------------------------------------------------------------
# GeoJSON streaming vectorization
# -----------------------------------------------------------------------------


CLASS_COLORS = {
    "Tissue": [83, 47, 31],
    "Anthracosis": [255, 205, 0],
    "Anthracosis dilated": [255, 160, 0],
    "CleanTissue": [0, 200, 200],
    "Positive": [220, 40, 40],
    "Negative": [60, 100, 230],
}


class GeoJSONStreamWriter:
    def __init__(self, path: Path, metadata: Dict[str, Any]):
        self.path = path
        self.file = path.open("w", encoding="utf-8")
        self.first = True
        header = {
            "type": "FeatureCollection",
            "name": path.stem,
            "properties": metadata,
        }
        # Write all except features, then begin feature array.
        self.file.write("{\n")
        self.file.write('  "type": "FeatureCollection",\n')
        self.file.write(f'  "name": {json.dumps(header["name"])},\n')
        self.file.write(f'  "properties": {json.dumps(metadata, ensure_ascii=False)},\n')
        self.file.write('  "features": [\n')

    def add(self, feature: Dict[str, Any]) -> None:
        if not self.first:
            self.file.write(",\n")
        self.file.write("    " + json.dumps(feature, ensure_ascii=False, separators=(",", ":")))
        self.first = False

    def close(self) -> None:
        self.file.write("\n  ]\n}\n")
        self.file.close()

    def __enter__(self) -> "GeoJSONStreamWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _contour_ring(contour: np.ndarray, x_offset: int, y_offset: int, epsilon: float) -> Optional[List[List[float]]]:
    if epsilon > 0:
        contour = cv2.approxPolyDP(contour, epsilon, True)
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return None
    ring = [[float(x + x_offset), float(y + y_offset)] for x, y in pts]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring if len(ring) >= 4 else None


def add_mask_features(
    writer: GeoJSONStreamWriter,
    mask: np.memmap,
    class_name: str,
    stage: str,
    tile_size: int,
    min_area: float,
    simplify: float,
) -> int:
    height, width = mask.shape
    count = 0
    hierarchy_note = "Objects crossing vectorization tile boundaries may be represented by touching polygon fragments."
    for x, y, w, h in iter_tiles(width, height, tile_size):
        tile = np.asarray(mask[y : y + h, x : x + w], dtype=np.uint8)
        if not np.any(tile):
            continue
        contours, hierarchy = cv2.findContours(tile, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        for i, contour in enumerate(contours):
            if hierarchy[i][3] != -1:
                continue
            outer_area = float(cv2.contourArea(contour))
            if outer_area < min_area:
                continue
            outer = _contour_ring(contour, x, y, simplify)
            if outer is None:
                continue
            rings: List[List[List[float]]] = [outer]
            child = hierarchy[i][2]
            hole_area = 0.0
            while child != -1:
                hole = _contour_ring(contours[child], x, y, simplify)
                if hole is not None:
                    rings.append(hole)
                    hole_area += float(cv2.contourArea(contours[child]))
                child = hierarchy[child][0]
            count += 1
            writer.add(
                {
                    "type": "Feature",
                    "properties": {
                        "id": count,
                        "stage": stage,
                        "class_name": class_name,
                        "classification": {
                            "name": class_name,
                            "color": CLASS_COLORS.get(class_name, [255, 255, 255]),
                        },
                        "area_px2_contour": max(0.0, outer_area - hole_area),
                        "tile_origin_x": x,
                        "tile_origin_y": y,
                        "vectorization_note": hierarchy_note,
                    },
                    "geometry": {"type": "Polygon", "coordinates": rings},
                }
            )
    return count


def save_single_mask_geojson(
    path: Path,
    mask: np.memmap,
    class_name: str,
    stage: str,
    source: ImageSource,
    image_path: Path,
    tile_size: int,
    min_area: float,
    simplify: float,
) -> int:
    metadata = {
        "source_file": str(image_path),
        "coordinate_system": "full_resolution_pixel_coordinates",
        "width": source.width,
        "height": source.height,
        "created_by": f"{APP_NAME} v{APP_VERSION}",
        "stage": stage,
    }
    with GeoJSONStreamWriter(path, metadata) as writer:
        return add_mask_features(writer, mask, class_name, stage, tile_size, min_area, simplify)


def save_combined_geojson(
    path: Path,
    masks_and_classes: Sequence[Tuple[np.memmap, str, str]],
    source: ImageSource,
    image_path: Path,
    tile_size: int,
    min_area: float,
    simplify: float,
) -> Dict[str, int]:
    metadata = {
        "source_file": str(image_path),
        "coordinate_system": "full_resolution_pixel_coordinates",
        "width": source.width,
        "height": source.height,
        "created_by": f"{APP_NAME} v{APP_VERSION}",
        "stage": "Positive versus Negative",
    }
    counts: Dict[str, int] = {}
    with GeoJSONStreamWriter(path, metadata) as writer:
        for mask, class_name, stage in masks_and_classes:
            counts[class_name] = add_mask_features(
                writer, mask, class_name, stage, tile_size, min_area, simplify
            )
    return counts


# -----------------------------------------------------------------------------
# Preview overlays
# -----------------------------------------------------------------------------


def mask_thumbnail(mask: np.memmap, target_width: int, target_height: int) -> np.ndarray:
    ys = np.linspace(0, mask.shape[0] - 1, target_height).astype(np.int64)
    xs = np.linspace(0, mask.shape[1] - 1, target_width).astype(np.int64)
    out = np.empty((target_height, target_width), dtype=bool)
    for row, yy in enumerate(ys):
        out[row, :] = np.asarray(mask[yy, xs]) > 0
    return out


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: Sequence[int], alpha: float = 0.5) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    if np.any(mask):
        c = np.asarray(color, dtype=np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def label_panel(rgb: np.ndarray, text: str) -> np.ndarray:
    out = rgb.copy()
    scale = max(0.55, min(out.shape[0], out.shape[1]) / 1200.0)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(out, (8, 8), (18 + tw, 22 + th + baseline), (255, 255, 255), -1)
    cv2.putText(out, text, (13, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return out


def save_previews(
    source: ImageSource,
    workspace: MaskWorkspace,
    output_dir: Path,
    max_side: int,
    alpha: float = 0.5,
) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required to save preview PNGs.")
    base = source.thumbnail(max_side)
    th, tw = base.shape[:2]
    thumbs = {name: mask_thumbnail(mask, tw, th) for name, mask in workspace.masks.items()}

    tissue_panel = label_panel(
        overlay_mask(base, thumbs["tissue"], CLASS_COLORS["Tissue"], alpha),
        "1. Tissue",
    )
    anthra_panel = label_panel(
        overlay_mask(base, thumbs["anthra_dilated"], CLASS_COLORS["Anthracosis dilated"], alpha),
        "2. Dilated anthracosis / ink",
    )
    clean_panel = label_panel(
        overlay_mask(base, thumbs["clean_tissue"], CLASS_COLORS["CleanTissue"], alpha),
        "3. CleanTissue",
    )
    final_panel = overlay_mask(base, thumbs["negative"], CLASS_COLORS["Negative"], alpha)
    final_panel = overlay_mask(final_panel, thumbs["positive"], CLASS_COLORS["Positive"], alpha)
    final_panel = label_panel(final_panel, "4. DAB: Positive (red) / Negative (blue)")

    montage = np.concatenate(
        [np.concatenate([tissue_panel, anthra_panel], axis=1), np.concatenate([clean_panel, final_panel], axis=1)],
        axis=0,
    )
    Image.fromarray(montage).save(output_dir / "pipeline_stages_50pct.png")

    final_with_ink = overlay_mask(base, thumbs["anthra_dilated"], CLASS_COLORS["Anthracosis dilated"], alpha)
    final_with_ink = overlay_mask(final_with_ink, thumbs["negative"], CLASS_COLORS["Negative"], alpha)
    final_with_ink = overlay_mask(final_with_ink, thumbs["positive"], CLASS_COLORS["Positive"], alpha)
    final_with_ink = label_panel(final_with_ink, "Ink removed (yellow), Positive (red), Negative (blue)")
    Image.fromarray(final_with_ink).save(output_dir / "pipeline_overlay_50pct.png")


# -----------------------------------------------------------------------------
# Optional binary mask TIFF export
# -----------------------------------------------------------------------------


def save_mask_tiff(path: Path, mask: np.memmap, description: str, tile_size: int = 512) -> None:
    # tifffile can stream tiles using an iterator, avoiding a second full in-memory mask.
    h, w = mask.shape
    tile_h = tile_w = tile_size

    def tile_iter() -> Iterator[np.ndarray]:
        for y in range(0, h, tile_h):
            for x in range(0, w, tile_w):
                tile = np.zeros((tile_h, tile_w), dtype=np.uint8)
                hh = min(tile_h, h - y)
                ww = min(tile_w, w - x)
                tile[:hh, :ww] = np.asarray(mask[y : y + hh, x : x + ww], dtype=np.uint8) * 255
                yield tile

    tifffile.imwrite(
        str(path),
        data=tile_iter(),
        shape=(h, w),
        dtype=np.uint8,
        bigtiff=True,
        tile=(tile_h, tile_w),
        compression="deflate",
        photometric="minisblack",
        description=description,
        metadata=None,
    )


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> Path:
    image_path = Path(args.image)
    tissue_path = Path(args.tissue_classifier)
    anthra_path = Path(args.anthra_classifier)
    dab_path = Path(args.dab_classifier)
    for path in (image_path, tissue_path, anthra_path, dab_path):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = Path(args.output) if args.output else image_path.parent / f"{safe_stem(image_path)}_standalone_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = create_unique_work_dir(output_dir, "_work_masks_run_")
    log(f"Temporary workspace: {work_dir.name}")

    log(f"{APP_NAME} v{APP_VERSION}")
    log(f"Image: {image_path}")
    log(f"Output: {output_dir}")
    log("Reconstructing QuPath OpenCV models...")
    models = reconstruct_models(tissue_path, anthra_path)
    dab_classifier = DABThresholdClassifier.from_json(dab_path)

    try:
        with ImageSource(image_path) as source:
            log(f"Reader: {source.reader} | Size: {source.width} x {source.height} | MPP: {source.mpp}")
            tile_size = int(args.tile_size or models.tile_size)
            tile_size = max(64, tile_size)
            workspace = MaskWorkspace(work_dir, (source.height, source.width))
            try:
                log("Stage 1-2: Tissue ANN and anthracosis RTrees...")
                first_pass_classification(source, models, workspace, tile_size)

                log(f"Stage 3: Dilating anthracosis by {args.ink_dilation} px and creating CleanTissue...")
                dilate_anthracosis_and_make_clean_tissue(
                    workspace, source.width, source.height, int(args.ink_dilation)
                )

                log(f"Stage 4: DAB threshold classification at {dab_classifier.threshold}...")
                dab_pass(source, dab_classifier, workspace, tile_size)

                log("Computing measurements...")
                measurements = compute_measurements(
                    image_path, source, workspace, int(args.ink_dilation), dab_classifier.threshold
                )
                save_measurements_csv(output_dir / "pipeline_measurements.csv", measurements)

                vector_tile = int(args.geojson_tile_size)
                min_area = float(args.geojson_min_area)
                simplify = float(args.geojson_simplify)
                log("Exporting GeoJSON stages...")
                geojson_counts = {}
                specs = [
                    ("01_Tissue.geojson", "tissue", "Tissue", "Tissue"),
                    ("02_Anthracosis_raw.geojson", "anthra_raw", "Anthracosis", "Anthracosis raw"),
                    ("02b_Anthracosis_dilated.geojson", "anthra_dilated", "Anthracosis dilated", "Anthracosis dilated"),
                    ("03_CleanTissue.geojson", "clean_tissue", "CleanTissue", "CleanTissue"),
                    ("04a_Positive.geojson", "positive", "Positive", "DAB Positive"),
                    ("04b_Negative.geojson", "negative", "Negative", "DAB Negative"),
                ]
                for filename, mask_name, class_name, stage in specs:
                    count = save_single_mask_geojson(
                        output_dir / filename,
                        workspace.masks[mask_name],
                        class_name,
                        stage,
                        source,
                        image_path,
                        vector_tile,
                        min_area,
                        simplify,
                    )
                    geojson_counts[filename] = count
                    log(f"  {filename}: {count} polygon fragments")

                combined_counts = save_combined_geojson(
                    output_dir / "04_Positive_vs_Negative.geojson",
                    [
                        (workspace.masks["positive"], "Positive", "DAB Positive"),
                        (workspace.masks["negative"], "Negative", "DAB Negative"),
                    ],
                    source,
                    image_path,
                    vector_tile,
                    min_area,
                    simplify,
                )
                geojson_counts["04_Positive_vs_Negative.geojson"] = combined_counts

                log("Saving 50%-opacity preview PNGs...")
                save_previews(source, workspace, output_dir, int(args.preview_max_side), alpha=0.5)

                nuclei_preview_summary: Dict[str, Any] = {"saved": False, "reason": "Skipped"}
                if not bool(getattr(args, "skip_nuclei_preview", False)):
                    log(
                        f"Stage 5 preview: {getattr(args, 'nuclei_backend', DEFAULT_NUCLEI_BACKEND)} "
                        f"nuclei segmentation ({getattr(args, 'instanseg_input', DEFAULT_INSTANSEG_INPUT)} input)..."
                    )
                    backend_config = NucleiBackendConfig(
                        backend=str(getattr(args, "nuclei_backend", DEFAULT_NUCLEI_BACKEND)),
                        instanseg_model=str(getattr(args, "instanseg_model", DEFAULT_INSTANSEG_MODEL)),
                        instanseg_input=str(getattr(args, "instanseg_input", DEFAULT_INSTANSEG_INPUT)),
                        device=str(getattr(args, "instanseg_device", DEFAULT_INSTANSEG_DEVICE)),
                        tile_size=int(getattr(args, "instanseg_tile_size", DEFAULT_INSTANSEG_TILE_SIZE)),
                        batch_size=int(getattr(args, "instanseg_batch_size", DEFAULT_INSTANSEG_BATCH_SIZE)),
                        pixel_size_um=getattr(args, "pixel_size_um", None),
                        pixel_size_fallback_um=float(getattr(args, "pixel_size_fallback_um", DEFAULT_PIXEL_SIZE_FALLBACK_UM)),
                        small_max_side=int(getattr(args, "instanseg_small_max_side", 1500)),
                        fallback_watershed=bool(getattr(args, "instanseg_fallback_watershed", True)),
                    )
                    nuclei_segmenter = NucleiSegmenter(backend_config)
                    try:
                        nuclei_preview_summary = save_nuclei_validation_tile(
                            source=source,
                            clean_mask=workspace.masks["clean_tissue"],
                            stain_classifier=dab_classifier,
                            nuclei_segmenter=nuclei_segmenter,
                            config=CompartmentConfig(),
                            output_dir=output_dir,
                            tile_side=int(getattr(args, "nuclei_preview_size", 2048)),
                        )
                        if nuclei_preview_summary.get("saved"):
                            log(
                                f"  Nuclei preview saved: {nuclei_preview_summary.get('nuclei_count', 0)} "
                                f"instances using {nuclei_preview_summary.get('backend', backend_config.backend)}."
                            )
                        else:
                            log(f"  Nuclei preview was not saved: {nuclei_preview_summary.get('reason', 'unknown reason')}")
                    except Exception as exc:
                        nuclei_preview_summary = {"saved": False, "reason": str(exc)}
                        log(f"WARNING: Nuclei preview failed, but baseline outputs are complete. Reason: {exc}")
                    finally:
                        nuclei_segmenter.close()

                if args.save_mask_tiffs:
                    log("Saving binary mask TIFFs...")
                    for name in MASK_NAMES:
                        save_mask_tiff(
                            output_dir / f"mask_{name}.tif",
                            workspace.masks[name],
                            f"{name} mask generated by {APP_NAME} v{APP_VERSION}",
                        )

                manifest = {
                    "application": APP_NAME,
                    "version": APP_VERSION,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "image": str(image_path),
                    "classifiers": {
                        "tissue": str(tissue_path),
                        "anthracosis": str(anthra_path),
                        "dab": str(dab_path),
                    },
                    "models": {
                        "tissue": "OpenCV ANN_MLP reconstructed from QuPath JSON",
                        "anthracosis": "OpenCV RTrees reconstructed from QuPath JSON",
                        "dab": "QuPath H-DAB stain-2 color deconvolution + constant threshold",
                    },
                    "parameters": {
                        "tile_size": tile_size,
                        "ink_dilation_radius_px": int(args.ink_dilation),
                        "dab_threshold_from_json": dab_classifier.threshold,
                        "geojson_tile_size": vector_tile,
                        "geojson_min_area_px2": min_area,
                        "geojson_simplify_px": simplify,
                        "preview_opacity": 0.5,
                    },
                    "image_metadata": {
                        "reader": source.reader,
                        "width": source.width,
                        "height": source.height,
                        "mpp": source.mpp,
                    },
                    "measurements": measurements,
                    "geojson_feature_counts": geojson_counts,
                    "nuclei_validation_preview": nuclei_preview_summary,
                    "notes": [
                        "Classifier resolution is 1 px/pixel for all supplied models.",
                        "Gaussian feature tiles use an image halo to minimize tile-boundary artifacts.",
                        "GeoJSON is vectorized in tiles; objects crossing a tile boundary may be split into touching polygon fragments.",
                        "Pixel counts in pipeline_measurements.csv are exact for the binary masks and do not depend on contour simplification.",
                    ],
                }
                with (output_dir / "pipeline_manifest.json").open("w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)

                log("Pipeline complete.")
                log(f"Positive area: {measurements['positive_percent_clean_tissue']:.4f}% of CleanTissue")
                log(f"Results: {output_dir}")
                if not bool(getattr(args, "no_inline_preview", False)):
                    display_saved_images_inline(
                        [
                            output_dir / "pipeline_stages_50pct.png",
                            output_dir / "nuclei_validation_montage.png",
                        ]
                    )
            finally:
                workspace.close()
    finally:
        models.close()

    if not args.keep_work_masks:
        safe_remove_tree(work_dir)
    return output_dir


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------


def create_self_test_image(path: Path) -> None:
    h, w = 384, 512
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.ellipse(img, (260, 195), (205, 130), 0, 0, 360, (190, 145, 115), -1)
    cv2.circle(img, (190, 170), 32, (35, 35, 30), -1)  # ink-like region
    cv2.rectangle(img, (285, 125), (415, 270), (125, 82, 45), -1)  # brown/DAB-rich region
    noise = np.random.default_rng(123).normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    tifffile.imwrite(str(path), img, photometric="rgb")


def run_self_test(args: argparse.Namespace) -> int:
    root = Path(tempfile.mkdtemp(prefix="standalone_classifier_selftest_"))
    try:
        image = root / "synthetic.tif"
        create_self_test_image(image)
        args.image = str(image)
        args.output = str(root / "output")
        args.preview_max_side = 800
        args.geojson_tile_size = 256
        args.tile_size = 256
        output = run_pipeline(args)
        expected = [
            "01_Tissue.geojson",
            "02_Anthracosis_raw.geojson",
            "02b_Anthracosis_dilated.geojson",
            "03_CleanTissue.geojson",
            "04_Positive_vs_Negative.geojson",
            "pipeline_measurements.csv",
            "pipeline_stages_50pct.png",
            "pipeline_manifest.json",
        ]
        missing = [name for name in expected if not (output / name).exists()]
        if missing:
            raise RuntimeError(f"Self-test missing outputs: {missing}")
        log(f"SELF-TEST PASSED. Temporary output: {output}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


# -----------------------------------------------------------------------------
# Classical H-channel nuclei segmentation and compartment classification
# -----------------------------------------------------------------------------

try:
    import joblib  # type: ignore
    from scipy import ndimage as ndi  # type: ignore
    from scipy.spatial import cKDTree  # type: ignore
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.metrics import classification_report, confusion_matrix  # type: ignore
    from skimage import feature as skfeature  # type: ignore
    from skimage import measure as skmeasure  # type: ignore
    from skimage import morphology as skmorph  # type: ignore
    from skimage import segmentation as skseg  # type: ignore
    from skimage.filters import threshold_otsu  # type: ignore
except Exception as _ml_import_error:  # pragma: no cover - checked at runtime
    joblib = None
    ndi = None
    cKDTree = None
    RandomForestClassifier = None
    classification_report = None
    confusion_matrix = None
    skfeature = None
    skmeasure = None
    skmorph = None
    skseg = None
    threshold_otsu = None
    _COMPARTMENT_IMPORT_ERROR = _ml_import_error
else:
    _COMPARTMENT_IMPORT_ERROR = None


COMPARTMENT_CLASSES = ("Tumor", "Stroma", "Other")
COMPARTMENT_CLASS_TO_ID = {name: i + 1 for i, name in enumerate(COMPARTMENT_CLASSES)}
COMPARTMENT_ID_TO_CLASS = {v: k for k, v in COMPARTMENT_CLASS_TO_ID.items()}

CLASS_COLORS.update(
    {
        "Tumor": [220, 45, 45],
        "Stroma": [40, 180, 80],
        "Other": [155, 155, 155],
        "Tumor Positive": [255, 0, 0],
        "Tumor Negative": [255, 150, 150],
        "Stroma Positive": [0, 135, 255],
        "Stroma Negative": [150, 220, 255],
    }
)

COMPARTMENT_MASK_NAMES = (
    "tumor",
    "stroma",
    "other",
    "tumor_positive",
    "tumor_negative",
    "stroma_positive",
    "stroma_negative",
)

COMPARTMENT_FEATURE_NAMES = (
    "clean_fraction",
    "h_mean",
    "h_std",
    "h_p25",
    "h_median",
    "h_p75",
    "h_entropy",
    "gradient_mean",
    "gradient_std",
    "structure_coherence_mean",
    "nuclei_count",
    "nuclei_density_per_mm2",
    "nuclear_area_fraction",
    "nucleus_area_mean_um2",
    "nucleus_area_std_um2",
    "nucleus_eccentricity_mean",
    "nucleus_solidity_mean",
    "nucleus_aspect_ratio_mean",
    "nucleus_h_mean",
    "nucleus_h_std",
    "nucleus_orientation_coherence",
    "nearest_neighbor_median_um",
)


def require_compartment_dependencies() -> None:
    if _COMPARTMENT_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The nuclei/compartment workflow requires scipy, scikit-image, "
            "scikit-learn and joblib. Install them with:\n"
            "  py -m pip install scipy scikit-image scikit-learn joblib\n"
            f"Original import error: {_COMPARTMENT_IMPORT_ERROR}"
        )


@dataclass(frozen=True)
class NucleiBackendConfig:
    backend: str = DEFAULT_NUCLEI_BACKEND
    instanseg_model: str = DEFAULT_INSTANSEG_MODEL
    instanseg_input: str = DEFAULT_INSTANSEG_INPUT
    device: str = DEFAULT_INSTANSEG_DEVICE
    tile_size: int = DEFAULT_INSTANSEG_TILE_SIZE
    batch_size: int = DEFAULT_INSTANSEG_BATCH_SIZE
    pixel_size_um: Optional[float] = None
    pixel_size_fallback_um: float = DEFAULT_PIXEL_SIZE_FALLBACK_UM
    small_max_side: int = 1500
    fallback_watershed: bool = True
    # InstanSeg already performs instance post-processing. Keep additional
    # filtering deliberately permissive to preserve small nuclei and better
    # match the QuPath InstanSeg extension output.
    instanseg_min_area_px: int = 1
    instanseg_max_area_px: int = 100000
    instanseg_min_solidity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "NucleiBackendConfig":
        known = {name: values[name] for name in cls.__dataclass_fields__ if name in values}
        return cls(**known)


def require_instanseg() -> Any:
    try:
        from instanseg import InstanSeg  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "InstanSeg is selected but is not installed in this Python environment.\n"
            "Install it in the same environment used by Spyder with:\n"
            "  py -m pip install instanseg-torch\n"
            "For NVIDIA CUDA, install a CUDA-enabled PyTorch build first if needed.\n"
            f"Original import error: {exc}"
        ) from exc
    return InstanSeg


def _relabel_and_filter_instances(
    labels: np.ndarray,
    h_channel: np.ndarray,
    valid_mask: np.ndarray,
    min_area_px: int,
    max_area_px: int,
    min_solidity: float = 0.0,
    minimum_valid_fraction: float = 0.0,
) -> Tuple[np.ndarray, List[Any]]:
    """Relabel model instances without cutting their outlines by CleanTissue.

    QuPath/InstanSeg detects objects first. For this pipeline, objects are then
    retained when their centroid lies within CleanTissue. Pixelwise clipping
    can fragment or erase small nuclei, especially along tissue/ink borders.
    """
    require_compartment_dependencies()
    raw = np.asarray(labels).squeeze()
    if raw.ndim != 2:
        raise RuntimeError(f"Unexpected nuclei-label shape: {raw.shape}")
    raw = np.rint(raw).astype(np.int64, copy=False)
    valid = np.asarray(valid_mask, dtype=bool)
    if raw.shape != valid.shape:
        raw = cv2.resize(
            raw.astype(np.float32),
            (valid.shape[1], valid.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        raw = np.rint(raw).astype(np.int64)
    if not np.any(raw > 0):
        return np.zeros(valid.shape, dtype=np.int32), []

    kept = np.zeros(valid.shape, dtype=np.int32)
    next_id = 0
    min_fraction = float(np.clip(minimum_valid_fraction, 0.0, 1.0))
    for prop in skmeasure.regionprops(
        raw.astype(np.int32), intensity_image=np.asarray(h_channel, dtype=np.float32)
    ):
        if prop.area < int(min_area_px) or prop.area > int(max_area_px):
            continue
        if float(prop.solidity) < float(min_solidity):
            continue
        cy, cx = prop.centroid
        iy = min(max(int(np.floor(cy)), 0), valid.shape[0] - 1)
        ix = min(max(int(np.floor(cx)), 0), valid.shape[1] - 1)
        keep = bool(valid[iy, ix])
        if not keep and min_fraction > 0.0:
            minr, minc, maxr, maxc = prop.bbox
            obj = raw[minr:maxr, minc:maxc] == prop.label
            overlap = obj & valid[minr:maxr, minc:maxc]
            keep = float(np.count_nonzero(overlap)) / max(1, int(np.count_nonzero(obj))) >= min_fraction
        if not keep:
            continue
        next_id += 1
        kept[raw == prop.label] = next_id

    props = list(skmeasure.regionprops(
        kept, intensity_image=np.asarray(h_channel, dtype=np.float32)
    )) if next_id else []
    return kept, props


class NucleiSegmenter:
    """Reusable nuclei backend so the InstanSeg model loads only once per run."""

    def __init__(self, config: NucleiBackendConfig):
        self.config = config
        self._model: Any = None
        self._warned_pixel_size = False
        self._warned_fallback = False
        self._reported_instanseg_settings = False
        self.last_raw_labels: Optional[np.ndarray] = None

    def _load_model(self) -> Any:
        if self._model is None:
            InstanSeg = require_instanseg()
            device = None if str(self.config.device).lower() in ("", "auto", "none") else self.config.device
            log(
                f"Loading InstanSeg model '{self.config.instanseg_model}' "
                f"(device={self.config.device}, input={self.config.instanseg_input})..."
            )
            self._model = InstanSeg(
                self.config.instanseg_model,
                device=device,
                image_reader="auto",
                verbosity=1,
            )
        return self._model

    def close(self) -> None:
        self._model = None
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def _effective_pixel_size(self, source_mpp: Optional[float]) -> float:
        value = self.config.pixel_size_um
        if value is None or float(value) <= 0:
            value = source_mpp
        if value is None or float(value) <= 0:
            value = float(self.config.pixel_size_fallback_um)
            if not self._warned_pixel_size:
                log(
                    "WARNING: Image pixel size was not found. "
                    f"Using fallback {value:.4f} um/px for InstanSeg. "
                    "Set --pixel-size-um to the true value for quantitative work."
                )
                self._warned_pixel_size = True
        return float(value)

    @staticmethod
    def _h_to_rgb(h_channel: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        h = np.asarray(h_channel, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(h)
        values = h[valid]
        if values.size:
            low, high = np.percentile(values, [1.0, 99.5])
        else:
            low, high = 0.0, 1.0
        if high <= low:
            high = low + 1e-6
        # Dark nuclei on a light background, matching brightfield appearance.
        gray = 255.0 * (1.0 - np.clip((h - low) / (high - low), 0.0, 1.0))
        gray = np.nan_to_num(gray, nan=255.0, posinf=0.0, neginf=255.0).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)

    def _segment_instanseg(
        self,
        rgb: np.ndarray,
        h_channel: np.ndarray,
        valid_mask: np.ndarray,
        source_mpp: Optional[float],
        min_area_px: int,
        max_area_px: int,
    ) -> Tuple[np.ndarray, List[Any]]:
        model = self._load_model()
        mode = str(self.config.instanseg_input).lower()
        if mode == "rgb":
            image = np.asarray(rgb, dtype=np.uint8)[..., :3]
        elif mode in ("h", "hematoxylin", "haematoxylin"):
            image = self._h_to_rgb(h_channel, valid_mask)
        else:
            raise ValueError(f"Unsupported InstanSeg input mode: {self.config.instanseg_input}")

        pixel_size = self._effective_pixel_size(source_mpp)
        kwargs = dict(
            image=image,
            pixel_size=pixel_size,
            normalise=True,
            return_image_tensor=False,
            target="nuclei",
            rescale_output=True,
        )
        if max(image.shape[:2]) <= int(self.config.small_max_side):
            output = model.eval_small_image(**kwargs)
        else:
            output = model.eval_medium_image(
                **kwargs,
                tile_size=max(256, int(self.config.tile_size)),
                batch_size=max(1, int(self.config.batch_size)),
            )
        if isinstance(output, (tuple, list)):
            output = output[0]
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        labels = np.asarray(output).squeeze()
        if labels.ndim != 2:
            labels = np.asarray(labels).squeeze()
        if labels.shape != valid_mask.shape:
            labels = cv2.resize(
                labels.astype(np.float32),
                (valid_mask.shape[1], valid_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        labels = np.rint(labels).astype(np.int32, copy=False)
        self.last_raw_labels = labels.copy()
        # Do not reuse the stricter classical-watershed area thresholds here.
        # The public InstanSeg model has already segmented instances; applying
        # a 20 px / solidity 0.40 cleanup removed many valid small nuclei.
        inst_min = max(1, int(self.config.instanseg_min_area_px))
        inst_max = max(inst_min + 1, int(self.config.instanseg_max_area_px))
        inst_solidity = float(np.clip(self.config.instanseg_min_solidity, 0.0, 1.0))
        if not self._reported_instanseg_settings:
            log(
                f"InstanSeg inference pixel size: {pixel_size:.4f} um/px | "
                f"model pixel size: {float(getattr(model.instanseg, 'pixel_size', float('nan'))):.4f} um/px | "
                f"post-filter: area {inst_min}-{inst_max} px, minimum solidity {inst_solidity:.2f} | "
                "CleanTissue selection by centroid or >=5% overlap (no pixelwise clipping)"
            )
            self._reported_instanseg_settings = True
        return _relabel_and_filter_instances(
            labels, h_channel, valid_mask, inst_min, inst_max, inst_solidity,
            minimum_valid_fraction=0.05,
        )

    def segment(
        self,
        rgb: np.ndarray,
        h_channel: np.ndarray,
        valid_mask: np.ndarray,
        threshold: float,
        min_area_px: int,
        max_area_px: int,
        min_distance_px: int,
        source_mpp: Optional[float],
    ) -> Tuple[np.ndarray, List[Any]]:
        backend = str(self.config.backend).lower()
        if backend == "watershed":
            return segment_nuclei_h_channel(
                h_channel, valid_mask, threshold, min_area_px, max_area_px, min_distance_px
            )
        if backend != "instanseg":
            raise ValueError(f"Unsupported nuclei backend: {self.config.backend}")
        try:
            return self._segment_instanseg(
                rgb, h_channel, valid_mask, source_mpp, min_area_px, max_area_px
            )
        except Exception as exc:
            if not self.config.fallback_watershed:
                raise
            if not self._warned_fallback:
                log(f"WARNING: InstanSeg failed; using watershed fallback. Reason: {exc}")
                self._warned_fallback = True
            return segment_nuclei_h_channel(
                h_channel, valid_mask, threshold, min_area_px, max_area_px, min_distance_px
            )


def _hematoxylin_values(classifier: DABThresholdClassifier, rgb: np.ndarray) -> np.ndarray:
    rgb8 = np.asarray(rgb, dtype=np.uint8)
    r = np.maximum(rgb8[..., 0].astype(np.float64), 1.0)
    g = np.maximum(rgb8[..., 1].astype(np.float64), 1.0)
    b = np.maximum(rgb8[..., 2].astype(np.float64), 1.0)
    od_r = np.maximum(0.0, -np.log10(r / classifier.max_rgb[0]))
    od_g = np.maximum(0.0, -np.log10(g / classifier.max_rgb[1]))
    od_b = np.maximum(0.0, -np.log10(b / classifier.max_rgb[2]))
    h = (
        od_r * classifier.inverse_matrix[0, 0]
        + od_g * classifier.inverse_matrix[1, 0]
        + od_b * classifier.inverse_matrix[2, 0]
    )
    return h.astype(np.float32)


@dataclass(frozen=True)
class CompartmentConfig:
    region_size_um: float = 160.0
    region_size_px: int = 320
    min_clean_fraction: float = 0.20
    nucleus_h_threshold: float = 0.0
    min_nucleus_area_um2: float = 12.0
    max_nucleus_area_um2: float = 350.0
    min_nucleus_area_px: int = 20
    max_nucleus_area_px: int = 1400
    nucleus_min_distance_um: float = 3.0
    nucleus_min_distance_px: int = 5
    nucleus_halo_um: float = 12.0
    nucleus_halo_px: int = 24
    min_confidence: float = 0.52
    smoothing_radius_regions: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompartmentConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: data[k] for k in allowed if k in data})


class CompartmentWorkspace:
    def __init__(self, root: Path, shape: Tuple[int, int]):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.shape = shape
        self.masks: Dict[str, np.memmap] = {}
        for name in COMPARTMENT_MASK_NAMES:
            path = self.root / f"{name}.uint8.dat"
            mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
            mm[:] = 0
            self.masks[name] = mm

    def flush(self) -> None:
        for mm in self.masks.values():
            mm.flush()

    def close(self) -> None:
        close_memmap_dict(self.masks)


def _mean_mpp(source: ImageSource) -> Optional[float]:
    if source.mpp is None:
        return None
    return 0.5 * (float(source.mpp[0]) + float(source.mpp[1]))


def resolve_region_geometry(source: ImageSource, config: CompartmentConfig) -> Dict[str, Any]:
    mpp = _mean_mpp(source)
    if mpp is not None and mpp > 0 and config.region_size_um > 0:
        region_px = max(64, int(round(config.region_size_um / mpp)))
        min_area_px = max(4, int(round(config.min_nucleus_area_um2 / (mpp * mpp))))
        max_area_px = max(min_area_px + 1, int(round(config.max_nucleus_area_um2 / (mpp * mpp))))
        min_distance_px = max(1, int(round(config.nucleus_min_distance_um / mpp)))
        halo_px = max(4, int(round(config.nucleus_halo_um / mpp)))
    else:
        region_px = max(64, int(config.region_size_px))
        min_area_px = max(4, int(config.min_nucleus_area_px))
        max_area_px = max(min_area_px + 1, int(config.max_nucleus_area_px))
        min_distance_px = max(1, int(config.nucleus_min_distance_px))
        halo_px = max(4, int(config.nucleus_halo_px))
    return {
        "mpp": mpp,
        "region_px": region_px,
        "min_area_px": min_area_px,
        "max_area_px": max_area_px,
        "min_distance_px": min_distance_px,
        "halo_px": halo_px,
    }


def estimate_nucleus_h_threshold(
    source: ImageSource,
    clean_mask: np.memmap,
    stain_classifier: DABThresholdClassifier,
    max_side: int = 3500,
) -> float:
    require_compartment_dependencies()
    thumb = source.thumbnail(max_side)
    h = _hematoxylin_values(stain_classifier, thumb)
    clean = mask_thumbnail(clean_mask, thumb.shape[1], thumb.shape[0])
    values = h[clean & np.isfinite(h)]
    values = values[(values > 0.01) & (values < 2.5)]
    if values.size < 100:
        return 0.10
    if values.size > 1_000_000:
        rng = np.random.default_rng(4381)
        values = rng.choice(values, size=1_000_000, replace=False)
    threshold = float(threshold_otsu(values))
    # Keep auto-thresholds inside a biologically plausible OD range.
    return float(np.clip(threshold, 0.06, 0.45))


def _read_mask_with_halo(
    mask: np.memmap, x: int, y: int, w: int, h: int, halo: int
) -> Tuple[np.ndarray, Tuple[slice, slice], Tuple[int, int]]:
    height, width = mask.shape
    rx0 = max(0, x - halo)
    ry0 = max(0, y - halo)
    rx1 = min(width, x + w + halo)
    ry1 = min(height, y + h + halo)
    arr = np.asarray(mask[ry0:ry1, rx0:rx1], dtype=np.uint8) > 0
    core_x0 = x - rx0
    core_y0 = y - ry0
    return arr, (slice(core_y0, core_y0 + h), slice(core_x0, core_x0 + w)), (rx0, ry0)


def segment_nuclei_h_channel(
    h_channel: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
    min_area_px: int,
    max_area_px: int,
    min_distance_px: int,
) -> Tuple[np.ndarray, List[Any]]:
    require_compartment_dependencies()
    h = np.asarray(h_channel, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    binary = (h >= float(threshold)) & valid
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=np.int32), []

    minimum_keep = max(3, min_area_px // 3)
    try:
        # scikit-image >= 0.26: remove objects whose size is <= max_size.
        binary = skmorph.remove_small_objects(binary, max_size=minimum_keep - 1)
    except TypeError:  # scikit-image <= 0.25
        binary = skmorph.remove_small_objects(binary, min_size=minimum_keep)
    binary = skmorph.closing(binary, skmorph.disk(1))
    binary = skmorph.opening(binary, skmorph.disk(1))
    distance = ndi.distance_transform_edt(binary)
    coords = skfeature.peak_local_max(
        distance,
        labels=binary,
        min_distance=max(1, int(min_distance_px)),
        exclude_border=False,
    )
    markers = np.zeros(binary.shape, dtype=np.int32)
    if coords.size:
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1, dtype=np.int32)
    else:
        markers, _ = ndi.label(binary)
    labels = skseg.watershed(-distance, markers, mask=binary, watershed_line=False)

    kept = np.zeros(labels.shape, dtype=np.int32)
    kept_id = 0
    props_out: List[Any] = []
    for prop in skmeasure.regionprops(labels, intensity_image=h):
        if prop.area < min_area_px or prop.area > max_area_px:
            continue
        # Reject extreme artifacts but retain elongated stromal nuclei.
        if prop.solidity < 0.45:
            continue
        kept_id += 1
        kept[labels == prop.label] = kept_id
    if kept_id:
        props_out = list(skmeasure.regionprops(kept, intensity_image=h))
    return kept, props_out


def _orientation_coherence(angles: Sequence[float]) -> float:
    if not angles:
        return 0.0
    a = np.asarray(angles, dtype=np.float64)
    return float(np.abs(np.mean(np.exp(2j * a))))


def _entropy_fixed(values: np.ndarray, bins: int = 24, upper: float = 1.5) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(np.clip(values, 0.0, upper), bins=bins, range=(0.0, upper))
    p = hist.astype(np.float64)
    p /= max(1.0, p.sum())
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0


def extract_region_features(
    source: ImageSource,
    clean_mask: np.memmap,
    stain_classifier: DABThresholdClassifier,
    nuclei_segmenter: NucleiSegmenter,
    x: int,
    y: int,
    w: int,
    h: int,
    nucleus_threshold: float,
    geometry: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    halo = int(geometry["halo_px"])
    rgb_halo, core_slice = read_with_halo(source, x, y, w, h, halo)
    clean_halo, mask_core_slice, _ = _read_mask_with_halo(clean_mask, x, y, w, h, halo)
    # Border clipping can differ by one pixel only if source/mask dimensions disagree.
    hh = min(rgb_halo.shape[0], clean_halo.shape[0])
    ww = min(rgb_halo.shape[1], clean_halo.shape[1])
    rgb_halo = rgb_halo[:hh, :ww]
    clean_halo = clean_halo[:hh, :ww]
    h_halo = _hematoxylin_values(stain_classifier, rgb_halo)

    core_h = h_halo[core_slice]
    core_clean = clean_halo[mask_core_slice]
    ch = min(core_h.shape[0], core_clean.shape[0])
    cw = min(core_h.shape[1], core_clean.shape[1])
    core_h = core_h[:ch, :cw]
    core_clean = core_clean[:ch, :cw]
    clean_fraction = float(np.mean(core_clean)) if core_clean.size else 0.0
    details: Dict[str, Any] = {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "clean_fraction": clean_fraction,
        "nuclei_count": 0,
    }
    if clean_fraction <= 0:
        return None, details

    labels, props = nuclei_segmenter.segment(
        rgb_halo,
        h_halo,
        clean_halo,
        nucleus_threshold,
        int(geometry["min_area_px"]),
        int(geometry["max_area_px"]),
        int(geometry["min_distance_px"]),
        geometry.get("mpp"),
    )

    y0 = core_slice[0].start or 0
    x0 = core_slice[1].start or 0
    y1 = y0 + ch
    x1 = x0 + cw
    selected = []
    for prop in props:
        cy, cx = prop.centroid
        if y0 <= cy < y1 and x0 <= cx < x1:
            selected.append(prop)

    values = core_h[core_clean & np.isfinite(core_h)]
    if values.size == 0:
        return None, details
    gx = cv2.Sobel(core_h, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(core_h, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_values = grad[core_clean]
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-8)
    coherence_values = coherence[core_clean]

    mpp = geometry.get("mpp")
    pixel_area_um2 = float(mpp * mpp) if mpp else 1.0
    region_clean_px = max(1, int(np.count_nonzero(core_clean)))
    region_area_mm2 = region_clean_px * pixel_area_um2 / 1_000_000.0

    areas_px = np.asarray([float(p.area) for p in selected], dtype=np.float64)
    areas_um2 = areas_px * pixel_area_um2
    eccentricities = np.asarray([float(p.eccentricity) for p in selected], dtype=np.float64)
    solidities = np.asarray([float(p.solidity) for p in selected], dtype=np.float64)
    aspect_values: List[float] = []
    intensity_values: List[float] = []
    for prop in selected:
        major = float(prop.axis_major_length) if hasattr(prop, "axis_major_length") else float(prop.major_axis_length)
        minor = float(prop.axis_minor_length) if hasattr(prop, "axis_minor_length") else float(prop.minor_axis_length)
        intensity = float(prop.intensity_mean) if hasattr(prop, "intensity_mean") else float(prop.mean_intensity)
        aspect_values.append(major / max(minor, 1e-6))
        intensity_values.append(intensity)
    aspects = np.asarray(aspect_values, dtype=np.float64)
    nuc_h = np.asarray(intensity_values, dtype=np.float64)
    orientations = [float(p.orientation) for p in selected]
    centroids = np.asarray(
        [[float(p.centroid[1] - x0), float(p.centroid[0] - y0)] for p in selected],
        dtype=np.float64,
    )
    nearest_um = 0.0
    if len(centroids) >= 2:
        tree = cKDTree(centroids)
        distances, _ = tree.query(centroids, k=2)
        nearest_px = distances[:, 1]
        nearest_um = float(np.median(nearest_px) * (mpp if mpp else 1.0))

    nuclei_count = len(selected)
    features = np.asarray(
        [
            clean_fraction,
            float(np.mean(values)),
            float(np.std(values)),
            float(np.percentile(values, 25)),
            float(np.median(values)),
            float(np.percentile(values, 75)),
            _entropy_fixed(values),
            float(np.mean(grad_values)) if grad_values.size else 0.0,
            float(np.std(grad_values)) if grad_values.size else 0.0,
            float(np.mean(coherence_values)) if coherence_values.size else 0.0,
            float(nuclei_count),
            safe_div(float(nuclei_count), region_area_mm2),
            safe_div(float(np.sum(areas_px)), float(region_clean_px)),
            float(np.mean(areas_um2)) if areas_um2.size else 0.0,
            float(np.std(areas_um2)) if areas_um2.size else 0.0,
            float(np.mean(eccentricities)) if eccentricities.size else 0.0,
            float(np.mean(solidities)) if solidities.size else 0.0,
            float(np.mean(aspects)) if aspects.size else 0.0,
            float(np.mean(nuc_h)) if nuc_h.size else 0.0,
            float(np.std(nuc_h)) if nuc_h.size else 0.0,
            _orientation_coherence(orientations),
            nearest_um,
        ],
        dtype=np.float32,
    )
    details["nuclei_count"] = nuclei_count
    details["nucleus_threshold"] = float(nucleus_threshold)
    details["feature_values"] = {name: float(value) for name, value in zip(COMPARTMENT_FEATURE_NAMES, features)}
    return features, details


def normalize_compartment_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if "tumor" in text or "tumour" in text or "epithelial" in text:
        return "Tumor"
    if "stroma" in text or "stromal" in text:
        return "Stroma"
    if any(token in text for token in ("other", "ignore", "necrosis", "artifact", "artefact", "benign")):
        return "Other"
    return None


def _feature_class_name(feature_obj: Dict[str, Any]) -> Optional[str]:
    props = feature_obj.get("properties") or {}
    candidates = [
        (props.get("classification") or {}).get("name") if isinstance(props.get("classification"), dict) else None,
        props.get("class_name"),
        props.get("name"),
        props.get("pathClass"),
        props.get("class"),
    ]
    for value in candidates:
        normalized = normalize_compartment_label(value)
        if normalized:
            return normalized
    return None


class AnnotationIndex:
    def __init__(self, geojson_paths: Sequence[Path]):
        require_compartment_dependencies()
        try:
            from shapely.geometry import Point, shape  # type: ignore
            from shapely.strtree import STRtree  # type: ignore
        except Exception as exc:
            raise RuntimeError("Training from GeoJSON requires shapely: py -m pip install shapely") from exc
        self.Point = Point
        self.geometries: List[Any] = []
        self.labels: List[str] = []
        for path in geojson_paths:
            with Path(path).open("r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("features", []) if isinstance(data, dict) else []
            for feature_obj in features:
                label = _feature_class_name(feature_obj)
                geometry = feature_obj.get("geometry")
                if label is None or geometry is None:
                    continue
                try:
                    geom = shape(geometry)
                    if geom.is_empty:
                        continue
                    if not geom.is_valid:
                        geom = geom.buffer(0)
                    if geom.is_empty:
                        continue
                    self.geometries.append(geom)
                    self.labels.append(label)
                except Exception:
                    continue
        if not self.geometries:
            raise ValueError(
                "No Tumor/Stroma/Other polygons were found. Use QuPath class names containing "
                "Tumor, Stroma, or Other/Ignore/Necrosis."
            )
        self.tree = STRtree(self.geometries)

    def label_at(self, x: float, y: float) -> Optional[str]:
        point = self.Point(float(x), float(y))
        candidates = self.tree.query(point)
        for item in candidates:
            # Shapely 2 returns integer indices; older versions may return geometry objects.
            if isinstance(item, (int, np.integer)):
                idx = int(item)
            else:
                try:
                    idx = self.geometries.index(item)
                except ValueError:
                    continue
            geom = self.geometries[idx]
            if geom.covers(point):
                return self.labels[idx]
        return None


def iter_region_grid(width: int, height: int, region_px: int) -> Iterator[Tuple[int, int, int, int, int, int]]:
    row = 0
    for y in range(0, height, region_px):
        h = min(region_px, height - y)
        col = 0
        for x in range(0, width, region_px):
            w = min(region_px, width - x)
            yield row, col, x, y, w, h
            col += 1
        row += 1


def extract_training_case(
    image_path: Path,
    annotation_paths: Sequence[Path],
    tissue_path: Path,
    anthra_path: Path,
    dab_path: Path,
    config: CompartmentConfig,
    args: argparse.Namespace,
    models: ReconstructedModels,
    nuclei_segmenter: NucleiSegmenter,
    work_root: Path,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    index = AnnotationIndex(annotation_paths)
    dab_classifier = DABThresholdClassifier.from_json(dab_path)
    case_root = work_root / safe_stem(image_path)
    with ImageSource(image_path) as source:
        workspace = MaskWorkspace(case_root / "base", (source.height, source.width))
        try:
            tile_size = max(64, int(args.tile_size or models.tile_size))
            log(f"Preparing clean tissue for training case: {image_path.name}")
            first_pass_classification(source, models, workspace, tile_size)
            dilate_anthracosis_and_make_clean_tissue(
                workspace, source.width, source.height, int(args.ink_dilation)
            )
            geometry = resolve_region_geometry(source, config)
            threshold = (
                float(config.nucleus_h_threshold)
                if config.nucleus_h_threshold > 0
                else estimate_nucleus_h_threshold(source, workspace.masks["clean_tissue"], dab_classifier)
            )
            log(
                f"  Region size: {geometry['region_px']} px | H threshold: {threshold:.4f} | "
                f"MPP: {source.mpp}"
            )
            rows: List[np.ndarray] = []
            labels: List[str] = []
            records: List[Dict[str, Any]] = []
            grid = list(iter_region_grid(source.width, source.height, int(geometry["region_px"])))
            total = len(grid)
            for i, (r, c, x, y, w, h) in enumerate(grid, 1):
                label = index.label_at(x + 0.5 * w, y + 0.5 * h)
                if label is None:
                    continue
                clean_fraction = float(
                    np.mean(np.asarray(workspace.masks["clean_tissue"][y : y + h, x : x + w]) > 0)
                )
                if clean_fraction < config.min_clean_fraction:
                    continue
                features, details = extract_region_features(
                    source,
                    workspace.masks["clean_tissue"],
                    dab_classifier,
                    nuclei_segmenter,
                    x,
                    y,
                    w,
                    h,
                    threshold,
                    geometry,
                )
                if features is None:
                    continue
                rows.append(features)
                labels.append(label)
                record = {
                    "image": str(image_path),
                    "row": r,
                    "column": c,
                    "label": label,
                    **{name: float(value) for name, value in zip(COMPARTMENT_FEATURE_NAMES, features)},
                }
                records.append(record)
                if len(rows) == 1 or len(rows) % 100 == 0:
                    log(f"  Training regions accepted: {len(rows)} | grid checked: {i}/{total}")
            if not rows:
                raise RuntimeError(f"No annotated regions yielded features for {image_path}")
            return np.vstack(rows), np.asarray(labels, dtype=object), records
        finally:
            workspace.close()


def _write_records_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        return
    fields: List[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def load_training_cases(args: argparse.Namespace) -> List[Tuple[Path, List[Path]]]:
    cases: List[Tuple[Path, List[Path]]] = []
    if args.training_manifest:
        manifest_path = Path(args.training_manifest)
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                image = Path(str(row.get("image", "")).strip())
                annotation_text = str(row.get("annotations") or row.get("annotation") or "").strip()
                annotations = [Path(x.strip()) for x in annotation_text.split(";") if x.strip()]
                if image and annotations:
                    cases.append((image, annotations))
    else:
        if not args.image or not args.annotations:
            raise ValueError("Training requires --image and --annotations, or --training-manifest.")
        cases.append((Path(args.image), [Path(x) for x in args.annotations]))
    if not cases:
        raise ValueError("No training cases were provided.")
    for image, annotations in cases:
        if not image.exists():
            raise FileNotFoundError(image)
        for annotation in annotations:
            if not annotation.exists():
                raise FileNotFoundError(annotation)
    return cases


def train_compartment_model(args: argparse.Namespace) -> Path:
    require_compartment_dependencies()
    tissue_path = Path(args.tissue_classifier)
    anthra_path = Path(args.anthra_classifier)
    dab_path = Path(args.dab_classifier)
    for path in (tissue_path, anthra_path, dab_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_model = Path(args.model_output)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.output) if args.output else output_model.parent / f"{output_model.stem}_training"
    report_dir.mkdir(parents=True, exist_ok=True)
    work_root = create_unique_work_dir(report_dir, "_training_work_run_")
    log(f"Temporary training workspace: {work_root.name}")

    config = compartment_config_from_args(args)
    nuclei_backend = nuclei_backend_config_from_args(args)
    cases = load_training_cases(args)
    models = reconstruct_models(tissue_path, anthra_path)
    nuclei_segmenter = NucleiSegmenter(nuclei_backend)
    log(f"Nuclei backend: {nuclei_backend.backend} | input: {nuclei_backend.instanseg_input} | device: {nuclei_backend.device}")
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_records: List[Dict[str, Any]] = []
    try:
        for image, annotations in cases:
            x, y, records = extract_training_case(
                image,
                annotations,
                tissue_path,
                anthra_path,
                dab_path,
                config,
                args,
                models,
                nuclei_segmenter,
                work_root,
            )
            all_x.append(x)
            all_y.append(y)
            all_records.extend(records)
    finally:
        nuclei_segmenter.close()
        models.close()
    X = np.vstack(all_x)
    y = np.concatenate(all_y)
    counts = {name: int(np.sum(y == name)) for name in COMPARTMENT_CLASSES}
    missing = [name for name, count in counts.items() if count < int(args.min_training_regions_per_class)]
    if missing:
        raise RuntimeError(
            f"Insufficient annotated regions for {missing}. Counts: {counts}. "
            f"Add annotations or reduce --min-training-regions-per-class."
        )

    classifier = RandomForestClassifier(
        n_estimators=int(args.rf_trees),
        min_samples_leaf=int(args.rf_min_samples_leaf),
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=1729,
        oob_score=True,
    )
    log(f"Training Random Forest on {len(y)} regions: {counts}")
    classifier.fit(X, y)
    train_pred = classifier.predict(X)
    report_text = classification_report(y, train_pred, labels=list(COMPARTMENT_CLASSES), zero_division=0)
    matrix = confusion_matrix(y, train_pred, labels=list(COMPARTMENT_CLASSES))
    package = {
        "model_type": "H-channel nuclei regional RandomForest",
        "version": 2,
        "nuclei_backend": nuclei_backend.to_dict(),
        "classifier": classifier,
        "feature_names": list(COMPARTMENT_FEATURE_NAMES),
        "classes": list(classifier.classes_),
        "config": config.to_dict(),
        "training_counts": counts,
        "training_images": [str(case[0]) for case in cases],
        "oob_score": float(getattr(classifier, "oob_score_", float("nan"))),
    }
    joblib.dump(package, output_model, compress=3)
    _write_records_csv(report_dir / "training_region_features.csv", all_records)
    with (report_dir / "training_report.txt").open("w", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\nConfusion matrix (rows=true, columns=predicted; Tumor, Stroma, Other):\n")
        f.write(np.array2string(matrix))
        f.write(f"\n\nOOB score: {package['oob_score']}\n")
    importance = sorted(
        zip(COMPARTMENT_FEATURE_NAMES, classifier.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    with (report_dir / "feature_importance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        writer.writerows(importance)
    with (report_dir / "training_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in package.items() if k != "classifier"}, f, indent=2)
    if not args.keep_work_masks:
        safe_remove_tree(work_root)
    log(f"Compartment model saved: {output_model}")
    log(f"Training report: {report_dir}")
    return output_model


def smooth_region_labels(labels: np.ndarray, radius: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint8)
    if radius <= 0:
        return labels.copy()
    kernel_size = 2 * int(radius) + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    scores = []
    for class_id in range(1, len(COMPARTMENT_CLASSES) + 1):
        scores.append(ndi.convolve((labels == class_id).astype(np.uint8), kernel, mode="constant", cval=0))
    stack = np.stack(scores, axis=-1)
    out = np.argmax(stack, axis=-1).astype(np.uint8) + 1
    out[labels == 0] = 0
    return out


def predict_regions(
    source: ImageSource,
    clean_mask: np.memmap,
    stain_classifier: DABThresholdClassifier,
    nuclei_segmenter: NucleiSegmenter,
    package: Dict[str, Any],
    config: CompartmentConfig,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    classifier = package["classifier"]
    expected_features = list(package.get("feature_names", []))
    if expected_features != list(COMPARTMENT_FEATURE_NAMES):
        raise ValueError("The compartment model feature list does not match this script version.")
    geometry = resolve_region_geometry(source, config)
    threshold = (
        float(config.nucleus_h_threshold)
        if config.nucleus_h_threshold > 0
        else estimate_nucleus_h_threshold(source, clean_mask, stain_classifier)
    )
    region_px = int(geometry["region_px"])
    nrows = int(math.ceil(source.height / region_px))
    ncols = int(math.ceil(source.width / region_px))
    labels_grid = np.zeros((nrows, ncols), dtype=np.uint8)
    confidence_grid = np.zeros((nrows, ncols), dtype=np.float32)
    records: List[Dict[str, Any]] = []
    grid = list(iter_region_grid(source.width, source.height, region_px))
    total = len(grid)
    for i, (r, c, x, y, w, h) in enumerate(grid, 1):
        clean_fraction = float(np.mean(np.asarray(clean_mask[y : y + h, x : x + w]) > 0))
        if clean_fraction < config.min_clean_fraction:
            continue
        features, details = extract_region_features(
            source,
            clean_mask,
            stain_classifier,
            nuclei_segmenter,
            x,
            y,
            w,
            h,
            threshold,
            geometry,
        )
        if features is None:
            continue
        probs = classifier.predict_proba(features.reshape(1, -1))[0]
        best_idx = int(np.argmax(probs))
        pred_name = str(classifier.classes_[best_idx])
        confidence = float(probs[best_idx])
        if confidence < config.min_confidence:
            pred_name = "Other"
        class_id = COMPARTMENT_CLASS_TO_ID[pred_name]
        labels_grid[r, c] = class_id
        confidence_grid[r, c] = confidence
        record = {
            "row": r,
            "column": c,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "predicted_class": pred_name,
            "confidence": confidence,
            **{f"probability_{str(name).lower()}": float(prob) for name, prob in zip(classifier.classes_, probs)},
            **{name: float(value) for name, value in zip(COMPARTMENT_FEATURE_NAMES, features)},
        }
        records.append(record)
        if i == 1 or i == total or i % max(1, total // 25) == 0:
            log(f"  Compartment regions: {i}/{total} ({100.0*i/total:.1f}%)")
    smoothed = smooth_region_labels(labels_grid, int(config.smoothing_radius_regions))
    _write_records_csv(output_dir / "compartment_region_features.csv", records)
    info = {
        "nuclei_backend": nuclei_segmenter.config.to_dict(),
        "region_size_px": region_px,
        "region_size_um": region_px * geometry["mpp"] if geometry.get("mpp") else None,
        "nucleus_h_threshold": threshold,
        "geometry": geometry,
    }
    return smoothed, confidence_grid, records, info


def rasterize_region_labels(
    labels_grid: np.ndarray,
    clean_mask: np.memmap,
    workspace: CompartmentWorkspace,
    region_px: int,
) -> None:
    height, width = clean_mask.shape
    for r in range(labels_grid.shape[0]):
        y = r * region_px
        h = min(region_px, height - y)
        if h <= 0:
            continue
        for c in range(labels_grid.shape[1]):
            class_id = int(labels_grid[r, c])
            if class_id == 0:
                continue
            x = c * region_px
            w = min(region_px, width - x)
            if w <= 0:
                continue
            clean = np.asarray(clean_mask[y : y + h, x : x + w], dtype=np.uint8) > 0
            name = COMPARTMENT_ID_TO_CLASS[class_id].lower()
            workspace.masks[name][y : y + h, x : x + w] = clean.astype(np.uint8)
    workspace.flush()


def compartment_dab_pass(
    source: ImageSource,
    dab_classifier: DABThresholdClassifier,
    workspace: CompartmentWorkspace,
    tile_size: int,
) -> None:
    tiles = list(iter_tiles(source.width, source.height, tile_size))
    total = len(tiles)
    for i, (x, y, w, h) in enumerate(tiles, 1):
        tumor = np.asarray(workspace.masks["tumor"][y : y + h, x : x + w]) > 0
        stroma = np.asarray(workspace.masks["stroma"][y : y + h, x : x + w]) > 0
        if not (np.any(tumor) or np.any(stroma)):
            continue
        rgb = source.read_region(x, y, w, h)
        pos = dab_classifier.predict_positive(rgb)
        workspace.masks["tumor_positive"][y : y + h, x : x + w] = (tumor & pos).astype(np.uint8)
        workspace.masks["tumor_negative"][y : y + h, x : x + w] = (tumor & ~pos).astype(np.uint8)
        workspace.masks["stroma_positive"][y : y + h, x : x + w] = (stroma & pos).astype(np.uint8)
        workspace.masks["stroma_negative"][y : y + h, x : x + w] = (stroma & ~pos).astype(np.uint8)
        if i == 1 or i == total or i % max(1, total // 20) == 0:
            log(f"  Compartment DAB tiles: {i}/{total}")
    workspace.flush()


def compute_compartment_measurements(
    image_path: Path,
    source: ImageSource,
    base_workspace: MaskWorkspace,
    compartment_workspace: CompartmentWorkspace,
    dab_threshold: float,
    ink_dilation: int,
) -> Dict[str, Any]:
    base_counts = {name: count_mask(base_workspace.masks[name]) for name in ("tissue", "anthra_raw", "anthra_dilated", "clean_tissue")}
    counts = {name: count_mask(compartment_workspace.masks[name]) for name in COMPARTMENT_MASK_NAMES}
    result: Dict[str, Any] = {
        "image": str(image_path),
        "reader": source.reader,
        "width_px": source.width,
        "height_px": source.height,
        "mpp_x_um": source.mpp[0] if source.mpp else "",
        "mpp_y_um": source.mpp[1] if source.mpp else "",
        "ink_dilation_radius_px": int(ink_dilation),
        "dab_threshold": float(dab_threshold),
        **{f"{name}_pixels": count for name, count in base_counts.items()},
        **{f"{name}_pixels": count for name, count in counts.items()},
        "tumor_percent_clean_tissue": percent(counts["tumor"], base_counts["clean_tissue"]),
        "stroma_percent_clean_tissue": percent(counts["stroma"], base_counts["clean_tissue"]),
        "other_percent_clean_tissue": percent(counts["other"], base_counts["clean_tissue"]),
        "tumor_positive_percent": percent(counts["tumor_positive"], counts["tumor"]),
        "tumor_negative_percent": percent(counts["tumor_negative"], counts["tumor"]),
        "stroma_positive_percent": percent(counts["stroma_positive"], counts["stroma"]),
        "stroma_negative_percent": percent(counts["stroma_negative"], counts["stroma"]),
    }
    if source.mpp:
        area_um2 = float(source.mpp[0]) * float(source.mpp[1])
        for name, count in {**base_counts, **counts}.items():
            result[f"{name}_area_mm2"] = count * area_um2 / 1_000_000.0
    return result


def save_compartment_previews(
    source: ImageSource,
    base_workspace: MaskWorkspace,
    compartment_workspace: CompartmentWorkspace,
    output_dir: Path,
    max_side: int,
) -> None:
    if Image is None:
        return
    base = source.thumbnail(max_side)
    th, tw = base.shape[:2]
    b = {name: mask_thumbnail(mask, tw, th) for name, mask in base_workspace.masks.items()}
    c = {name: mask_thumbnail(mask, tw, th) for name, mask in compartment_workspace.masks.items()}

    tissue = label_panel(overlay_mask(base, b["tissue"], CLASS_COLORS["Tissue"], 0.5), "1. Tissue")
    ink = label_panel(overlay_mask(base, b["anthra_dilated"], CLASS_COLORS["Anthracosis dilated"], 0.5), "2. Dilated ink")
    clean = label_panel(overlay_mask(base, b["clean_tissue"], CLASS_COLORS["CleanTissue"], 0.5), "3. Clean tissue")
    comp = overlay_mask(base, c["other"], CLASS_COLORS["Other"], 0.5)
    comp = overlay_mask(comp, c["stroma"], CLASS_COLORS["Stroma"], 0.5)
    comp = overlay_mask(comp, c["tumor"], CLASS_COLORS["Tumor"], 0.5)
    comp = label_panel(comp, "4. Tumor (red), Stroma (green), Other (gray)")
    dab = overlay_mask(base, c["tumor_negative"], CLASS_COLORS["Tumor Negative"], 0.5)
    dab = overlay_mask(dab, c["tumor_positive"], CLASS_COLORS["Tumor Positive"], 0.5)
    dab = overlay_mask(dab, c["stroma_negative"], CLASS_COLORS["Stroma Negative"], 0.5)
    dab = overlay_mask(dab, c["stroma_positive"], CLASS_COLORS["Stroma Positive"], 0.5)
    dab = label_panel(dab, "5. Compartment-specific DAB")
    blank = np.full_like(base, 255)
    blank = label_panel(blank, "Tumor: red | Stroma: blue | light = negative")
    montage = np.concatenate(
        [np.concatenate([tissue, ink], axis=1), np.concatenate([clean, comp], axis=1), np.concatenate([dab, blank], axis=1)],
        axis=0,
    )
    Image.fromarray(montage).save(output_dir / "compartment_pipeline_stages_50pct.png")
    Image.fromarray(comp).save(output_dir / "compartment_overlay_50pct.png")
    Image.fromarray(dab).save(output_dir / "compartment_dab_overlay_50pct.png")


def save_nuclei_validation_tile(
    source: ImageSource,
    clean_mask: np.memmap,
    stain_classifier: DABThresholdClassifier,
    nuclei_segmenter: NucleiSegmenter,
    config: CompartmentConfig,
    output_dir: Path,
    tile_side: int = 2048,
) -> Dict[str, Any]:
    """Save one original-resolution tile with InstanSeg/watershed nuclei boundaries.

    This is intentionally an original-resolution validation crop rather than a
    heavily downsampled whole-slide thumbnail, because nuclei segmentation must
    be inspected at the physical resolution supplied to the model.
    """
    require_compartment_dependencies()
    side = max(256, min(int(tile_side), source.width, source.height))
    best = None
    best_fraction = -1.0
    step = side
    for y in range(0, source.height, step):
        h = min(side, source.height - y)
        for x in range(0, source.width, step):
            w = min(side, source.width - x)
            region = np.asarray(clean_mask[y:y+h, x:x+w]) > 0
            fraction = float(np.mean(region)) if region.size else 0.0
            if fraction > best_fraction:
                best_fraction = fraction
                best = (x, y, w, h)
            if best_fraction >= 0.92:
                break
        if best_fraction >= 0.92:
            break
    if best is None or best_fraction <= 0:
        return {"saved": False, "reason": "No clean tissue found"}

    x, y, w, h = best
    rgb = source.read_region(x, y, w, h)
    valid = np.asarray(clean_mask[y:y+h, x:x+w]) > 0
    hh = min(rgb.shape[0], valid.shape[0])
    ww = min(rgb.shape[1], valid.shape[1])
    rgb = rgb[:hh, :ww]
    valid = valid[:hh, :ww]
    h_channel = _hematoxylin_values(stain_classifier, rgb)
    values = h_channel[valid & np.isfinite(h_channel)]
    if config.nucleus_h_threshold > 0:
        threshold = float(config.nucleus_h_threshold)
    elif values.size >= 100:
        threshold = float(np.clip(threshold_otsu(values), 0.06, 0.45))
    else:
        threshold = 0.10

    mpp = _mean_mpp(source)
    if mpp and mpp > 0:
        min_area_px = max(4, int(round(config.min_nucleus_area_um2 / (mpp*mpp))))
        max_area_px = max(min_area_px + 1, int(round(config.max_nucleus_area_um2 / (mpp*mpp))))
        min_distance_px = max(1, int(round(config.nucleus_min_distance_um / mpp)))
    else:
        min_area_px = max(4, int(config.min_nucleus_area_px))
        max_area_px = max(min_area_px + 1, int(config.max_nucleus_area_px))
        min_distance_px = max(1, int(config.nucleus_min_distance_px))

    labels, props = nuclei_segmenter.segment(
        rgb, h_channel, valid, threshold, min_area_px, max_area_px, min_distance_px, mpp
    )
    raw_labels = nuclei_segmenter.last_raw_labels
    if raw_labels is None or raw_labels.shape != labels.shape:
        raw_labels = labels.copy()

    raw_boundaries = skseg.find_boundaries(raw_labels, mode="outer")
    raw_overlay = rgb.copy()
    raw_overlay[raw_boundaries] = np.asarray([255, 0, 0], dtype=np.uint8)

    boundaries = skseg.find_boundaries(labels, mode="outer")
    overlay = rgb.copy()
    overlay[boundaries] = np.asarray([255, 230, 0], dtype=np.uint8)

    # Deterministic pseudo-colors make adjacent instances easy to inspect.
    instance_rgb = np.zeros_like(rgb, dtype=np.uint8)
    positive = labels > 0
    if np.any(positive):
        ids = labels[positive].astype(np.uint64)
        instance_rgb[..., 0][positive] = ((ids * 73 + 41) % 206 + 35).astype(np.uint8)
        instance_rgb[..., 1][positive] = ((ids * 151 + 67) % 206 + 35).astype(np.uint8)
        instance_rgb[..., 2][positive] = ((ids * 199 + 97) % 206 + 35).astype(np.uint8)
    instance_overlay = rgb.copy().astype(np.float32)
    instance_overlay[positive] = 0.35 * instance_overlay[positive] + 0.65 * instance_rgb[positive].astype(np.float32)
    instance_overlay = np.clip(instance_overlay, 0, 255).astype(np.uint8)
    instance_overlay[boundaries] = np.asarray([255, 255, 255], dtype=np.uint8)

    h_values = h_channel[valid & np.isfinite(h_channel)]
    if h_values.size:
        h_low, h_high = np.percentile(h_values, [1.0, 99.5])
    else:
        h_low, h_high = 0.0, 1.0
    if h_high <= h_low:
        h_high = h_low + 1e-6
    h_gray = 255.0 * (1.0 - np.clip((h_channel - h_low) / (h_high - h_low), 0.0, 1.0))
    h_gray = np.nan_to_num(h_gray, nan=255.0, posinf=0.0, neginf=255.0).astype(np.uint8)
    h_rgb = np.repeat(h_gray[..., None], 3, axis=2)

    if Image is not None:
        raw_overlay_path = output_dir / "nuclei_validation_raw_model_overlay.png"
        overlay_path = output_dir / "nuclei_validation_overlay.png"
        instances_path = output_dir / "nuclei_validation_instances.png"
        montage_path = output_dir / "nuclei_validation_montage.png"
        Image.fromarray(raw_overlay).save(raw_overlay_path)
        Image.fromarray(overlay).save(overlay_path)
        Image.fromarray(instance_overlay).save(instances_path)
        montage = np.concatenate(
            [
                np.concatenate([label_panel(rgb, "1. Original RGB"), label_panel(h_rgb, "2. Hematoxylin channel")], axis=1),
                np.concatenate([
                    label_panel(raw_overlay, "3. Raw InstanSeg output (red)"),
                    label_panel(overlay, "4. Retained by CleanTissue centroid (yellow)"),
                ], axis=1),
            ],
            axis=0,
        )
        Image.fromarray(montage).save(montage_path)
    tifffile.imwrite(
        output_dir / "nuclei_validation_raw_model_labels.tif",
        raw_labels.astype(np.int32),
        metadata={"axes": "YX", "backend": nuclei_segmenter.config.backend, "stage": "raw_model"},
    )
    tifffile.imwrite(
        output_dir / "nuclei_validation_labels.tif",
        labels.astype(np.int32),
        metadata={"axes": "YX", "backend": nuclei_segmenter.config.backend, "stage": "clean_tissue_centroid_filter"},
    )
    areas = np.asarray([float(prop.area) for prop in props], dtype=np.float64)
    pixel_area_um2 = float(mpp*mpp) if mpp else 1.0
    clean_area_mm2 = max(1, int(np.count_nonzero(valid))) * pixel_area_um2 / 1_000_000.0
    summary = {
        "saved": True,
        "x": int(x), "y": int(y), "width": int(ww), "height": int(hh),
        "clean_fraction": float(np.mean(valid)),
        "backend": nuclei_segmenter.config.backend,
        "instanseg_model": nuclei_segmenter.config.instanseg_model,
        "instanseg_input": nuclei_segmenter.config.instanseg_input,
        "instanseg_min_area_px": int(nuclei_segmenter.config.instanseg_min_area_px),
        "instanseg_max_area_px": int(nuclei_segmenter.config.instanseg_max_area_px),
        "instanseg_min_solidity": float(nuclei_segmenter.config.instanseg_min_solidity),
        "raw_model_nuclei_count": int(np.max(raw_labels)) if np.any(raw_labels > 0) else 0,
        "retained_nuclei_count": int(len(props)),
        "nuclei_count": int(len(props)),
        "nuclei_density_per_mm2": safe_div(float(len(props)), clean_area_mm2),
        "mean_nucleus_area_um2": float(np.mean(areas) * pixel_area_um2) if areas.size else 0.0,
        "metadata_pixel_size_um": mpp,
        "inference_pixel_size_um": float(nuclei_segmenter._effective_pixel_size(mpp)),
    }
    with (output_dir / "nuclei_validation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def display_saved_images_inline(paths: Sequence[Path]) -> None:
    """Display saved PNGs in Spyder/IPython while keeping disk outputs authoritative."""
    if Image is None:
        return
    try:
        from IPython import get_ipython  # type: ignore
        from IPython.display import display  # type: ignore
        if get_ipython() is None:
            return
        for path in paths:
            path = Path(path)
            if not path.exists():
                continue
            with Image.open(path) as img:
                display(img.copy())
    except Exception as exc:
        log(f"WARNING: Could not display previews inline: {exc}")


def export_compartment_geojsons(
    source: ImageSource,
    image_path: Path,
    base_workspace: MaskWorkspace,
    compartment_workspace: CompartmentWorkspace,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    vector_tile = int(args.geojson_tile_size)
    min_area = float(args.geojson_min_area)
    simplify = float(args.geojson_simplify)
    specs = [
        ("01_Tissue.geojson", base_workspace.masks["tissue"], "Tissue", "Tissue"),
        ("02_Anthracosis_raw.geojson", base_workspace.masks["anthra_raw"], "Anthracosis", "Anthracosis raw"),
        ("03_Anthracosis_dilated.geojson", base_workspace.masks["anthra_dilated"], "Anthracosis dilated", "Anthracosis dilated"),
        ("04_CleanTissue.geojson", base_workspace.masks["clean_tissue"], "CleanTissue", "Clean tissue"),
        ("05_Tumor.geojson", compartment_workspace.masks["tumor"], "Tumor", "Tumor compartment"),
        ("06_Stroma.geojson", compartment_workspace.masks["stroma"], "Stroma", "Stroma compartment"),
        ("07_Other.geojson", compartment_workspace.masks["other"], "Other", "Other compartment"),
        ("08_Tumor_Positive.geojson", compartment_workspace.masks["tumor_positive"], "Tumor Positive", "Tumor DAB positive"),
        ("09_Tumor_Negative.geojson", compartment_workspace.masks["tumor_negative"], "Tumor Negative", "Tumor DAB negative"),
        ("10_Stroma_Positive.geojson", compartment_workspace.masks["stroma_positive"], "Stroma Positive", "Stroma DAB positive"),
        ("11_Stroma_Negative.geojson", compartment_workspace.masks["stroma_negative"], "Stroma Negative", "Stroma DAB negative"),
    ]
    counts: Dict[str, Any] = {}
    for filename, mask, class_name, stage in specs:
        counts[filename] = save_single_mask_geojson(
            output_dir / filename,
            mask,
            class_name,
            stage,
            source,
            image_path,
            vector_tile,
            min_area,
            simplify,
        )
    counts["12_Compartments.geojson"] = save_combined_geojson(
        output_dir / "12_Compartments.geojson",
        [
            (compartment_workspace.masks["tumor"], "Tumor", "Tumor compartment"),
            (compartment_workspace.masks["stroma"], "Stroma", "Stroma compartment"),
            (compartment_workspace.masks["other"], "Other", "Other compartment"),
        ],
        source,
        image_path,
        vector_tile,
        min_area,
        simplify,
    )
    counts["13_Compartment_DAB.geojson"] = save_combined_geojson(
        output_dir / "13_Compartment_DAB.geojson",
        [
            (compartment_workspace.masks["tumor_positive"], "Tumor Positive", "Tumor DAB positive"),
            (compartment_workspace.masks["tumor_negative"], "Tumor Negative", "Tumor DAB negative"),
            (compartment_workspace.masks["stroma_positive"], "Stroma Positive", "Stroma DAB positive"),
            (compartment_workspace.masks["stroma_negative"], "Stroma Negative", "Stroma DAB negative"),
        ],
        source,
        image_path,
        vector_tile,
        min_area,
        simplify,
    )
    return counts


def run_compartment_prediction(args: argparse.Namespace) -> Path:
    require_compartment_dependencies()
    image_path = Path(args.image)
    tissue_path = Path(args.tissue_classifier)
    anthra_path = Path(args.anthra_classifier)
    dab_path = Path(args.dab_classifier)
    model_path = Path(args.compartment_model)
    for path in (image_path, tissue_path, anthra_path, dab_path, model_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = Path(args.output) if args.output else image_path.parent / f"{safe_stem(image_path)}_nuclei_compartments"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = create_unique_work_dir(output_dir, "_work_masks_run_")
    log(f"Temporary prediction workspace: {work_dir.name}")
    package = joblib.load(model_path)
    base_config = CompartmentConfig.from_dict(package.get("config", {}))
    config = compartment_config_from_args(args, defaults=base_config)
    stored_backend = NucleiBackendConfig.from_dict(package.get("nuclei_backend", {}))
    nuclei_backend = nuclei_backend_config_from_args(args, defaults=stored_backend)
    nuclei_segmenter = NucleiSegmenter(nuclei_backend)
    log(f"Nuclei backend: {nuclei_backend.backend} | input: {nuclei_backend.instanseg_input} | device: {nuclei_backend.device}")
    models = reconstruct_models(tissue_path, anthra_path)
    dab_classifier = DABThresholdClassifier.from_json(dab_path)
    try:
        with ImageSource(image_path) as source:
            log(f"Image: {image_path}")
            log(f"Reader: {source.reader} | Size: {source.width} x {source.height} | MPP: {source.mpp}")
            base_ws = MaskWorkspace(work_dir / "base", (source.height, source.width))
            comp_ws = CompartmentWorkspace(work_dir / "compartments", (source.height, source.width))
            try:
                tile_size = max(64, int(args.tile_size or models.tile_size))
                log("Stage 1-2: Tissue and anthracosis classifiers...")
                first_pass_classification(source, models, base_ws, tile_size)
                log(f"Stage 3: Dilating anthracosis by {args.ink_dilation} px...")
                dilate_anthracosis_and_make_clean_tissue(
                    base_ws, source.width, source.height, int(args.ink_dilation)
                )
                log(f"Stage 4: {nuclei_backend.backend} nuclei segmentation and regional classification...")
                labels_grid, confidence_grid, region_records, region_info = predict_regions(
                    source,
                    base_ws.masks["clean_tissue"],
                    dab_classifier,
                    nuclei_segmenter,
                    package,
                    config,
                    output_dir,
                )
                rasterize_region_labels(
                    labels_grid,
                    base_ws.masks["clean_tissue"],
                    comp_ws,
                    int(region_info["region_size_px"]),
                )
                log("Saving an original-resolution nuclei validation tile...")
                nuclei_validation = save_nuclei_validation_tile(
                    source, base_ws.masks["clean_tissue"], dab_classifier,
                    nuclei_segmenter, config, output_dir,
                    tile_side=min(2048, max(512, int(args.preview_max_side))),
                )
                log(f"Stage 5: DAB threshold {dab_classifier.threshold} inside Tumor and Stroma...")
                compartment_dab_pass(source, dab_classifier, comp_ws, tile_size)
                measurements = compute_compartment_measurements(
                    image_path,
                    source,
                    base_ws,
                    comp_ws,
                    dab_classifier.threshold,
                    int(args.ink_dilation),
                )
                save_measurements_csv(output_dir / "compartment_measurements.csv", measurements)
                log("Exporting GeoJSON stages...")
                geo_counts = export_compartment_geojsons(
                    source, image_path, base_ws, comp_ws, output_dir, args
                )
                save_compartment_previews(
                    source, base_ws, comp_ws, output_dir, int(args.preview_max_side)
                )
                if args.save_mask_tiffs:
                    for name, mask in {**base_ws.masks, **comp_ws.masks}.items():
                        save_mask_tiff(output_dir / f"mask_{name}.tif", mask, name)
                manifest = {
                    "application": APP_NAME,
                    "version": APP_VERSION,
                    "mode": "nuclei_compartment_prediction",
                    "image": str(image_path),
                    "compartment_model": str(model_path),
                    "classifiers": {
                        "tissue": str(tissue_path),
                        "anthracosis": str(anthra_path),
                        "dab": str(dab_path),
                    },
                    "config": config.to_dict(),
                    "nuclei_backend": nuclei_backend.to_dict(),
                    "region_info": region_info,
                    "nuclei_validation": nuclei_validation,
                    "measurements": measurements,
                    "geojson_feature_counts": geo_counts,
                    "notes": [
                        "Compartment predictions are regional and based on H-channel nuclear morphology, density, spacing, alignment and texture.",
                        "DAB intensity is not included in the compartment features.",
                        "Regions below the confidence threshold are assigned to Other.",
                        "Validate compartment predictions against independent pathologist annotations before quantitative study use.",
                    ],
                }
                with (output_dir / "compartment_manifest.json").open("w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)
                log(f"Tumor DAB positive: {measurements['tumor_positive_percent']:.3f}%")
                log(f"Stroma DAB positive: {measurements['stroma_positive_percent']:.3f}%")
                log(f"Results: {output_dir}")
            finally:
                comp_ws.close()
                base_ws.close()
    finally:
        nuclei_segmenter.close()
        models.close()
    if not args.keep_work_masks:
        safe_remove_tree(work_dir)
    return output_dir


def compartment_config_from_args(
    args: argparse.Namespace, defaults: Optional[CompartmentConfig] = None
) -> CompartmentConfig:
    base = defaults or CompartmentConfig()
    values = base.to_dict()
    mapping = {
        "region_size_um": "region_size_um",
        "region_size_px": "region_size_px",
        "min_clean_fraction": "min_clean_fraction",
        "nucleus_h_threshold": "nucleus_h_threshold",
        "min_nucleus_area_um2": "min_nucleus_area_um2",
        "max_nucleus_area_um2": "max_nucleus_area_um2",
        "min_nucleus_area_px": "min_nucleus_area_px",
        "max_nucleus_area_px": "max_nucleus_area_px",
        "nucleus_min_distance_um": "nucleus_min_distance_um",
        "nucleus_min_distance_px": "nucleus_min_distance_px",
        "nucleus_halo_um": "nucleus_halo_um",
        "nucleus_halo_px": "nucleus_halo_px",
        "min_compartment_confidence": "min_confidence",
        "smoothing_radius_regions": "smoothing_radius_regions",
    }
    for arg_name, field_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            values[field_name] = value
    return CompartmentConfig.from_dict(values)


def nuclei_backend_config_from_args(
    args: argparse.Namespace, defaults: Optional[NucleiBackendConfig] = None
) -> NucleiBackendConfig:
    base = defaults or NucleiBackendConfig()
    values = base.to_dict()
    mapping = {
        "nuclei_backend": "backend",
        "instanseg_model": "instanseg_model",
        "instanseg_input": "instanseg_input",
        "instanseg_device": "device",
        "instanseg_tile_size": "tile_size",
        "instanseg_batch_size": "batch_size",
        "pixel_size_um": "pixel_size_um",
        "pixel_size_fallback_um": "pixel_size_fallback_um",
        "instanseg_small_max_side": "small_max_side",
        "instanseg_fallback_watershed": "fallback_watershed",
        "instanseg_min_area_px": "instanseg_min_area_px",
        "instanseg_max_area_px": "instanseg_max_area_px",
        "instanseg_min_solidity": "instanseg_min_solidity",
    }
    for arg_name, field_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            values[field_name] = value
    return NucleiBackendConfig.from_dict(values)


def add_nuclei_backend_arguments(parser: argparse.ArgumentParser, defaults_none: bool = False) -> None:
    def d(value: Any) -> Any:
        return None if defaults_none else value
    parser.add_argument(
        "--nuclei-backend", choices=("instanseg", "watershed"),
        default=d(DEFAULT_NUCLEI_BACKEND),
        help="Nuclei segmentation backend; InstanSeg is preferred."
    )
    parser.add_argument("--instanseg-model", default=d(DEFAULT_INSTANSEG_MODEL))
    parser.add_argument(
        "--instanseg-input", choices=("rgb", "hematoxylin"),
        default=d(DEFAULT_INSTANSEG_INPUT),
        help="RGB is the preferred input for the public brightfield model."
    )
    parser.add_argument(
        "--instanseg-device", default=d(DEFAULT_INSTANSEG_DEVICE),
        help="auto, cpu, cuda, mps, or another PyTorch device."
    )
    parser.add_argument("--instanseg-tile-size", type=int, default=d(DEFAULT_INSTANSEG_TILE_SIZE))
    parser.add_argument("--instanseg-batch-size", type=int, default=d(DEFAULT_INSTANSEG_BATCH_SIZE))
    parser.add_argument(
        "--pixel-size-um", type=float, default=d(None),
        help="Override image resolution in micrometres per pixel. Metadata is preferred."
    )
    parser.add_argument(
        "--pixel-size-fallback-um", type=float, default=d(DEFAULT_PIXEL_SIZE_FALLBACK_UM),
        help="Used only when image metadata and --pixel-size-um are unavailable."
    )
    parser.add_argument("--instanseg-small-max-side", type=int, default=d(1500))
    parser.add_argument(
        "--instanseg-min-area-px", type=int, default=d(1),
        help="Raw-like post-filter after InstanSeg; 1 retains every predicted object."
    )
    parser.add_argument(
        "--instanseg-max-area-px", type=int, default=d(100000),
        help="Maximum InstanSeg object area in input-image pixels."
    )
    parser.add_argument(
        "--instanseg-min-solidity", type=float, default=d(0.0),
        help="Additional solidity filter after InstanSeg; 0 disables it."
    )
    parser.add_argument(
        "--no-instanseg-fallback", dest="instanseg_fallback_watershed",
        action="store_false", default=d(True),
        help="Fail instead of reverting to watershed if InstanSeg cannot run."
    )


def add_common_image_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Input TIFF/WSI image")
    parser.add_argument("--tissue-classifier", default=str(DEFAULT_TISSUE))
    parser.add_argument("--anthra-classifier", default=str(DEFAULT_ANTHRA))
    parser.add_argument("--dab-classifier", default=str(DEFAULT_DAB))
    parser.add_argument("--output", default=None)
    parser.add_argument("--ink-dilation", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--preview-max-side", type=int, default=2500)
    parser.add_argument("--geojson-tile-size", type=int, default=2048)
    parser.add_argument("--geojson-min-area", type=float, default=4.0)
    parser.add_argument("--geojson-simplify", type=float, default=1.0)
    parser.add_argument("--save-mask-tiffs", action="store_true")
    parser.add_argument("--keep-work-masks", action="store_true")
    parser.add_argument(
        "--skip-nuclei-preview", action="store_true",
        help="Skip the automatic InstanSeg/watershed nuclei validation preview."
    )
    parser.add_argument(
        "--nuclei-preview-size", type=int, default=2048,
        help="Side length in pixels of the original-resolution nuclei validation crop."
    )
    parser.add_argument(
        "--no-inline-preview", action="store_true",
        help="Save preview PNGs without displaying them inline in Spyder/IPython."
    )


def add_compartment_feature_arguments(parser: argparse.ArgumentParser, defaults_none: bool = False) -> None:
    def d(value: Any) -> Any:
        return None if defaults_none else value
    parser.add_argument("--region-size-um", type=float, default=d(160.0), help="Regional classification window in micrometres")
    parser.add_argument("--region-size-px", type=int, default=d(320), help="Fallback window when MPP is unavailable")
    parser.add_argument("--min-clean-fraction", type=float, default=d(0.20))
    parser.add_argument("--nucleus-h-threshold", type=float, default=d(0.0), help="0 = automatic Otsu threshold")
    parser.add_argument("--min-nucleus-area-um2", type=float, default=d(12.0))
    parser.add_argument("--max-nucleus-area-um2", type=float, default=d(350.0))
    parser.add_argument("--min-nucleus-area-px", type=int, default=d(20))
    parser.add_argument("--max-nucleus-area-px", type=int, default=d(1400))
    parser.add_argument("--nucleus-min-distance-um", type=float, default=d(3.0))
    parser.add_argument("--nucleus-min-distance-px", type=int, default=d(5))
    parser.add_argument("--nucleus-halo-um", type=float, default=d(12.0))
    parser.add_argument("--nucleus-halo-px", type=int, default=d(24))
    parser.add_argument("--min-compartment-confidence", type=float, default=d(0.52))
    parser.add_argument("--smoothing-radius-regions", type=int, default=d(1))


def build_extended_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone H-DAB pipeline with InstanSeg or watershed nuclei segmentation, Tumor/Stroma/Other training, and compartment-specific DAB quantification."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    baseline = sub.add_parser(
        "baseline",
        help="Run Tissue -> Anthracosis -> CleanTissue -> overall DAB, plus an automatic nuclei preview"
    )
    add_common_image_arguments(baseline)
    add_nuclei_backend_arguments(baseline, defaults_none=False)

    train = sub.add_parser("train", help="Train Tumor/Stroma/Other Random Forest from GeoJSON annotations")
    add_common_image_arguments(train)
    add_compartment_feature_arguments(train, defaults_none=False)
    add_nuclei_backend_arguments(train, defaults_none=False)
    train.add_argument("--annotations", nargs="+", default=None, help="GeoJSON annotation files for the input image")
    train.add_argument("--training-manifest", default=None, help="CSV columns: image, annotations (semicolon-separated)")
    train.add_argument("--model-output", default=str(DEFAULT_COMPARTMENT_MODEL), help="Output .joblib compartment model")
    train.add_argument("--rf-trees", type=int, default=500)
    train.add_argument("--rf-min-samples-leaf", type=int, default=2)
    train.add_argument("--min-training-regions-per-class", type=int, default=20)

    predict = sub.add_parser("predict", help="Predict Tumor/Stroma/Other and quantify DAB by compartment")
    add_common_image_arguments(predict)
    add_compartment_feature_arguments(predict, defaults_none=True)
    add_nuclei_backend_arguments(predict, defaults_none=True)
    predict.add_argument("--compartment-model", default=str(DEFAULT_COMPARTMENT_MODEL), help="Trained .joblib model")

    return parser


def extended_main() -> int:
    log("=" * 72)
    log(f"RUNNING FILE: {Path(__file__).resolve()}")
    log(f"BUILD: {APP_NAME} v{APP_VERSION} | {BUILD_ID}")
    log("This build includes Stage 5 InstanSeg nuclei preview, defaults uncalibrated images to 0.50 um/px, and saves both raw-model and CleanTissue-filtered nuclei outputs.")
    log("IMPORTANT: when TIFF MPP is missing, pass --pixel-size-um with the same calibration used by QuPath.")
    log("=" * 72)
    parser = build_extended_parser()

    # Spyder's %runfile launches the script without command-line arguments.
    # The preferred configuration is InstanSeg brightfield_nuclei + RGB input +
    # automatic device selection. In auto mode, reuse an existing compartment
    # model; otherwise train then predict when the default GeoJSON exists.
    argv = sys.argv[1:]
    spyder_train_then_predict = False
    if not argv:
        preferred = [
            "--nuclei-backend", DEFAULT_NUCLEI_BACKEND,
            "--instanseg-model", DEFAULT_INSTANSEG_MODEL,
            "--instanseg-input", DEFAULT_INSTANSEG_INPUT,
            "--instanseg-device", DEFAULT_INSTANSEG_DEVICE,
        ]
        mode = str(SPYDER_DEFAULT_MODE).lower()
        if mode == "baseline":
            argv = ["baseline"]
        elif mode == "train":
            argv = [
                "train", "--annotations", str(DEFAULT_TRAINING_ANNOTATIONS),
                "--model-output", str(DEFAULT_COMPARTMENT_MODEL), *preferred
            ]
        elif mode == "predict":
            argv = [
                "predict", "--compartment-model", str(DEFAULT_COMPARTMENT_MODEL), *preferred
            ]
        elif mode == "auto":
            if DEFAULT_COMPARTMENT_MODEL.exists():
                log("No command supplied; running preferred InstanSeg compartment prediction.")
                argv = [
                    "predict", "--compartment-model", str(DEFAULT_COMPARTMENT_MODEL), *preferred
                ]
            elif DEFAULT_TRAINING_ANNOTATIONS.exists():
                log("No compartment model found; training with preferred InstanSeg configuration, then predicting.")
                argv = [
                    "train", "--annotations", str(DEFAULT_TRAINING_ANNOTATIONS),
                    "--model-output", str(DEFAULT_COMPARTMENT_MODEL), *preferred
                ]
                spyder_train_then_predict = True
            else:
                log("No compartment model or default Tumor/Stroma/Other annotation GeoJSON was found.")
                log(f"Expected model: {DEFAULT_COMPARTMENT_MODEL}")
                log(f"Expected annotations: {DEFAULT_TRAINING_ANNOTATIONS}")
                log("Running the baseline Tissue/Anthracosis/DAB pipeline plus an InstanSeg nuclei preview. "
                    "After creating the annotation GeoJSON, rerun this file to train and predict Tumor/Stroma/Other.")
                argv = ["baseline"]
        else:
            raise ValueError(f"Unsupported SPYDER_DEFAULT_MODE: {SPYDER_DEFAULT_MODE}")

    args = parser.parse_args(argv)
    try:
        if args.command == "baseline":
            run_pipeline(args)
        elif args.command == "train":
            train_compartment_model(args)
            if spyder_train_then_predict:
                predict_args = parser.parse_args([
                    "predict",
                    "--image", str(DEFAULT_IMAGE),
                    "--tissue-classifier", str(DEFAULT_TISSUE),
                    "--anthra-classifier", str(DEFAULT_ANTHRA),
                    "--dab-classifier", str(DEFAULT_DAB),
                    "--compartment-model", str(DEFAULT_COMPARTMENT_MODEL),
                    "--nuclei-backend", DEFAULT_NUCLEI_BACKEND,
                    "--instanseg-model", DEFAULT_INSTANSEG_MODEL,
                    "--instanseg-input", DEFAULT_INSTANSEG_INPUT,
                    "--instanseg-device", DEFAULT_INSTANSEG_DEVICE,
                ])
                run_compartment_prediction(predict_args)
        elif args.command == "predict":
            run_compartment_prediction(args)
        else:
            parser.error(f"Unknown command: {args.command}")
        return 0
    except KeyboardInterrupt:
        log("Cancelled by user.")
        return 130
    except Exception as exc:
        log(f"ERROR: {exc}")
        traceback.print_exc()
        return 1


# -----------------------------------------------------------------------------
# Extended CLI entry point
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    raise SystemExit(extended_main())
