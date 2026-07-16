from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .resources import bundled_classifier_paths

SUPPORTED_IMAGE_SUFFIXES = (
    ".tif", ".tiff", ".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif",
    ".png", ".jpg", ".jpeg", ".bmp",
)


@dataclass
class JobConfig:
    mode: str = "baseline"  # baseline, predict, train, train_predict
    images: List[str] = field(default_factory=list)
    output_root: str = ""
    tissue_classifier: str = ""
    anthra_classifier: str = ""
    dab_classifier: str = ""
    compartment_model: str = ""
    model_output: str = ""
    annotation_folder: str = ""
    annotations: Dict[str, List[str]] = field(default_factory=dict)

    ink_dilation: int = 5
    tile_size: Optional[int] = None
    preview_max_side: int = 2500
    nuclei_preview_size: int = 2048
    save_mask_tiffs: bool = False
    keep_work_masks: bool = False

    nuclei_backend: str = "instanseg"
    instanseg_model: str = "brightfield_nuclei"
    instanseg_input: str = "rgb"
    instanseg_device: str = "auto"
    instanseg_tile_size: int = 512
    instanseg_batch_size: int = 1
    pixel_size_um: Optional[float] = None
    pixel_size_fallback_um: float = 0.5
    instanseg_small_max_side: int = 1500
    instanseg_min_area_px: int = 1
    instanseg_max_area_px: int = 100000
    instanseg_min_solidity: float = 0.0
    instanseg_fallback_watershed: bool = True

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
    min_compartment_confidence: float = 0.52
    smoothing_radius_regions: int = 1
    rf_trees: int = 500
    rf_min_samples_leaf: int = 2
    min_training_regions_per_class: int = 20

    geojson_tile_size: int = 2048
    geojson_min_area: float = 4.0
    geojson_simplify: float = 1.0

    def apply_bundled_classifier_defaults(self) -> None:
        defaults = bundled_classifier_paths()
        if not self.tissue_classifier:
            self.tissue_classifier = str(defaults.tissue)
        if not self.anthra_classifier:
            self.anthra_classifier = str(defaults.anthra)
        if not self.dab_classifier:
            self.dab_classifier = str(defaults.dab)

    def validate(self) -> None:
        self.apply_bundled_classifier_defaults()
        if self.mode not in {"baseline", "predict", "train", "train_predict"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if not self.images:
            raise ValueError("Add at least one image.")
        for image in self.images:
            if not Path(image).exists():
                raise FileNotFoundError(image)
        for label, value in (
            ("Tissue classifier", self.tissue_classifier),
            ("Anthracosis classifier", self.anthra_classifier),
            ("DAB classifier", self.dab_classifier),
        ):
            if not value or not Path(value).exists():
                raise FileNotFoundError(f"{label}: {value or 'not selected'}")
        if self.mode == "predict" and (not self.compartment_model or not Path(self.compartment_model).exists()):
            raise FileNotFoundError(f"Compartment model: {self.compartment_model or 'not selected'}")
        if self.mode in {"train", "train_predict"}:
            if not self.model_output:
                raise ValueError("Select a model output path.")
            missing = [image for image in self.images if not self.annotation_paths_for(image)]
            if missing:
                raise ValueError(
                    "No Tumor/Stroma/Other GeoJSON found for: " + ", ".join(Path(x).name for x in missing)
                )
        if self.pixel_size_um is not None and self.pixel_size_um <= 0:
            raise ValueError("Pixel size must be positive.")
        if self.pixel_size_fallback_um <= 0:
            raise ValueError("Fallback pixel size must be positive.")

    def annotation_paths_for(self, image: str) -> List[str]:
        image_key = str(Path(image).resolve())
        direct = self.annotations.get(image_key) or self.annotations.get(image)
        if direct:
            return [p for p in direct if Path(p).exists()]
        if not self.annotation_folder:
            return []
        folder = Path(self.annotation_folder)
        stem = safe_stem(Path(image))
        candidates = [
            folder / f"{stem}_compartments.geojson",
            folder / f"{stem}.geojson",
            folder / f"{Path(image).stem}_compartments.geojson",
            folder / f"{Path(image).stem}.geojson",
        ]
        return [str(p) for p in candidates if p.exists()]

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return target

    @classmethod
    def from_json(cls, path: str | Path) -> "JobConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def safe_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def discover_images(folder: str | Path, recursive: bool = True) -> List[str]:
    root = Path(folder)
    iterator = root.rglob("*") if recursive else root.glob("*")
    result = []
    for path in iterator:
        if path.is_file() and any(path.name.lower().endswith(s) for s in SUPPORTED_IMAGE_SUFFIXES):
            result.append(str(path.resolve()))
    return sorted(result)
