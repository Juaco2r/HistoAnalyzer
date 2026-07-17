from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Iterable, List

from .runtime_env import bootstrap_runtime_environment

bootstrap_runtime_environment()

from . import engine
from .job import JobConfig, safe_stem

PROGRESS_PREFIX = "[HISTOANALYZER_PROGRESS]"
RESULT_PREFIX = "[HISTOANALYZER_RESULT]"
DONE_PREFIX = "[HISTOANALYZER_DONE]"


def emit(prefix: str, payload: dict) -> None:
    print(prefix + json.dumps(payload, ensure_ascii=False), flush=True)


def common_cli(config: JobConfig, image: str, output: str) -> List[str]:
    args = [
        "--image", image,
        "--tissue-classifier", config.tissue_classifier,
        "--anthra-classifier", config.anthra_classifier,
        "--dab-classifier", config.dab_classifier,
        "--output", output,
        "--ink-dilation", str(config.ink_dilation),
        "--preview-max-side", str(config.preview_max_side),
        "--nuclei-preview-size", str(config.nuclei_preview_size),
        "--geojson-tile-size", str(config.geojson_tile_size),
        "--geojson-min-area", str(config.geojson_min_area),
        "--geojson-simplify", str(config.geojson_simplify),
        "--no-inline-preview",
    ]
    if config.tile_size:
        args += ["--tile-size", str(config.tile_size)]
    if config.save_mask_tiffs:
        args.append("--save-mask-tiffs")
    if config.keep_work_masks:
        args.append("--keep-work-masks")
    return args


def nuclei_cli(config: JobConfig) -> List[str]:
    args = [
        "--nuclei-backend", config.nuclei_backend,
        "--instanseg-model", config.instanseg_model,
        "--instanseg-input", config.instanseg_input,
        "--instanseg-device", config.instanseg_device,
        "--instanseg-tile-size", str(config.instanseg_tile_size),
        "--instanseg-batch-size", str(config.instanseg_batch_size),
        "--pixel-size-fallback-um", str(config.pixel_size_fallback_um),
        "--instanseg-small-max-side", str(config.instanseg_small_max_side),
        "--instanseg-min-area-px", str(config.instanseg_min_area_px),
        "--instanseg-max-area-px", str(config.instanseg_max_area_px),
        "--instanseg-min-solidity", str(config.instanseg_min_solidity),
    ]
    if config.pixel_size_um is not None:
        args += ["--pixel-size-um", str(config.pixel_size_um)]
    if not config.instanseg_fallback_watershed:
        args.append("--no-instanseg-fallback")
    return args


def compartment_cli(config: JobConfig) -> List[str]:
    return [
        "--region-size-um", str(config.region_size_um),
        "--region-size-px", str(config.region_size_px),
        "--min-clean-fraction", str(config.min_clean_fraction),
        "--nucleus-h-threshold", str(config.nucleus_h_threshold),
        "--min-nucleus-area-um2", str(config.min_nucleus_area_um2),
        "--max-nucleus-area-um2", str(config.max_nucleus_area_um2),
        "--min-nucleus-area-px", str(config.min_nucleus_area_px),
        "--max-nucleus-area-px", str(config.max_nucleus_area_px),
        "--nucleus-min-distance-um", str(config.nucleus_min_distance_um),
        "--nucleus-min-distance-px", str(config.nucleus_min_distance_px),
        "--nucleus-halo-um", str(config.nucleus_halo_um),
        "--nucleus-halo-px", str(config.nucleus_halo_px),
        "--min-compartment-confidence", str(config.min_compartment_confidence),
        "--smoothing-radius-regions", str(config.smoothing_radius_regions),
    ]


def make_training_manifest(config: JobConfig, folder: Path) -> Path:
    path = folder / "training_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "annotations"])
        writer.writeheader()
        for image in config.images:
            writer.writerow({"image": image, "annotations": ";".join(config.annotation_paths_for(image))})
    return path


def run_job(config: JobConfig) -> int:
    config.validate()
    parser = engine.build_extended_parser()
    output_root = Path(config.output_root or Path(config.images[0]).parent / "HistoAnalyzer_results")
    output_root.mkdir(parents=True, exist_ok=True)
    total = len(config.images)

    if config.mode in {"train", "train_predict"}:
        emit(PROGRESS_PREFIX, {"phase": "training", "current": 0, "total": total, "message": "Preparing training manifest"})
        with tempfile.TemporaryDirectory(prefix="histoanalyzer_training_") as temp:
            manifest = make_training_manifest(config, Path(temp))
            training_output = output_root / "training"
            argv = [
                "train", *common_cli(config, config.images[0], str(training_output)),
                *compartment_cli(config), *nuclei_cli(config),
                "--training-manifest", str(manifest),
                "--model-output", config.model_output,
                "--rf-trees", str(config.rf_trees),
                "--rf-min-samples-leaf", str(config.rf_min_samples_leaf),
                "--min-training-regions-per-class", str(config.min_training_regions_per_class),
            ]
            args = parser.parse_args(argv)
            model_path = engine.train_compartment_model(args)
            emit(RESULT_PREFIX, {"kind": "model", "path": str(model_path), "output": str(training_output)})
        if config.mode == "train":
            emit(DONE_PREFIX, {"success": True, "output_root": str(output_root)})
            return 0
        config.compartment_model = config.model_output

    for index, image in enumerate(config.images, 1):
        stem = safe_stem(Path(image))
        phase = "baseline" if config.mode == "baseline" else "prediction"
        suffix = "baseline" if phase == "baseline" else "compartments"
        output = output_root / f"{stem}_{suffix}"
        emit(PROGRESS_PREFIX, {
            "phase": phase, "current": index - 1, "total": total,
            "image": image, "message": f"Starting {Path(image).name}"
        })
        if phase == "baseline":
            argv = ["baseline", *common_cli(config, image, str(output)), *nuclei_cli(config)]
            args = parser.parse_args(argv)
            result = engine.run_pipeline(args)
        else:
            argv = [
                "predict", *common_cli(config, image, str(output)),
                *compartment_cli(config), *nuclei_cli(config),
                "--compartment-model", config.compartment_model,
            ]
            args = parser.parse_args(argv)
            result = engine.run_compartment_prediction(args)
        emit(RESULT_PREFIX, {
            "kind": "image", "image": image, "index": index - 1,
            "output": str(result), "mode": phase,
        })
        emit(PROGRESS_PREFIX, {
            "phase": phase, "current": index, "total": total,
            "image": image, "message": f"Completed {Path(image).name}"
        })

    emit(DONE_PREFIX, {"success": True, "output_root": str(output_root)})
    return 0


def run_job_file(path: str | Path) -> int:
    try:
        return run_job(JobConfig.from_json(path))
    except KeyboardInterrupt:
        emit(DONE_PREFIX, {"success": False, "cancelled": True})
        return 130
    except Exception as exc:
        traceback.print_exc()
        emit(DONE_PREFIX, {"success": False, "error": str(exc)})
        return 1
