## Built-in classifiers

HistoAnalyzer ships with three default QuPath JSON classifiers and selects them automatically:

- Tissue ANN: `TissueClassifierANNFullJuly06.json`
- Anthracosis/ink RTrees: `AnthraJuly06.json`
- H-DAB threshold: `DABCNNThreshold0.17DAB.json` (DAB OD threshold 0.17)

They are visible in the GUI and can be restored with **Use bundled defaults**. Selecting another JSON file overrides the corresponding built-in classifier. The standalone release uses the same defaults without requiring external classifier paths.

# HistoAnalyzer

![HistoAnalyzer icon](assets/icon_512.png)

**HistoAnalyzer** is a cross-platform desktop application for reproducible H-DAB histology analysis. It combines QuPath-exported pixel classifiers, InstanSeg nuclei segmentation, classical regional machine learning, and compartment-specific DAB quantification in one batch-processing interface.

## Main workflow

1. Tissue/background classification from a QuPath ANN JSON.
2. Anthracosis/ink detection from a QuPath RTrees JSON.
3. Configurable dilation and subtraction to obtain CleanTissue.
4. InstanSeg `brightfield_nuclei` segmentation on RGB by default.
5. H-channel nuclear, neighborhood, and regional feature extraction.
6. Random Forest classification of Tumor, Stroma, and Other.
7. DAB threshold quantification independently inside Tumor and Stroma.
8. GeoJSON, CSV, mask TIFF, and quality-control PNG export.

## Desktop features

- Process one image or a batch of images.
- Add individual images, whole folders, or drag and drop.
- Baseline, training, prediction, and train-then-predict workflows.
- Live logs, progress, cancellation, and per-image status.
- Tabs for stage, nuclei, compartment, and DAB previews.
- Persistent settings for classifier and output paths.
- InstanSeg, watershed fallback, CPU, CUDA, and Apple MPS selection.
- TIFF/OME-TIFF and common OpenSlide formats.

## Supported image formats

`TIFF`, `OME-TIFF`, `SVS`, `NDPI`, `MRXS`, `SCN`, `BIF`, `PNG`, `JPEG`, and `BMP`.

TIFF and OME-TIFF are read with `tifffile`/Zarr when possible. Whole-slide formats require the OpenSlide optional dependency.

## Installation from source

Python 3.10–3.12 is supported.

```bash
git clone https://github.com/Juaco2r/HistoAnalyzer.git
cd HistoAnalyzer
python -m venv .venv
```

Activate the environment and install:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[wsi]"
HistoAnalyzer
```

InstanSeg downloads the selected public model on first use. The default is `brightfield_nuclei`. Model weights are not redistributed with HistoAnalyzer.

## Quick start

1. Add one or more H-DAB images.
2. Select Tissue, Anthracosis, and DAB QuPath JSON classifiers.
3. Use **Baseline + nuclei preview** to validate masks and nuclei.
4. Create Tumor/Stroma/Other GeoJSON annotations.
5. Use **Train model, then predict**.
6. Review the Nuclei and Compartments tabs before using measurements.

For images without calibration metadata, HistoAnalyzer defaults to **0.5 µm/px**. Set the actual value whenever known.

## Training annotation names

GeoJSON polygons must use these classification names:

- `Tumor`
- `Stroma`
- `Other`

For batch training, place annotation files in one folder using:

```text
<image_stem>_compartments.geojson
```

## Outputs

Typical baseline outputs:

```text
01_Tissue.geojson
02_Anthracosis_raw.geojson
02b_Anthracosis_dilated.geojson
03_CleanTissue.geojson
04a_Positive.geojson
04b_Negative.geojson
pipeline_measurements.csv
pipeline_stages_50pct.png
nuclei_validation_montage.png
```

Prediction adds:

```text
05_Tumor.geojson
06_Stroma.geojson
07_Other.geojson
08_Tumor_Positive.geojson
09_Tumor_Negative.geojson
10_Stroma_Positive.geojson
11_Stroma_Negative.geojson
compartment_measurements.csv
compartment_overlay_50pct.png
compartment_dab_overlay_50pct.png
```

## Cross-platform releases

GitHub Actions builds a native application on Windows, macOS, and Linux. PyInstaller is not a cross-compiler, so each artifact is generated on its own operating system.

- Windows installer (recommended): `HistoAnalyzer-Windows-x64-Setup.exe`
- Windows portable: `HistoAnalyzer-Windows-x64-PORTABLE-EXTRACT-FIRST.zip`
- macOS: `HistoAnalyzer-macOS.zip`
- Linux: `HistoAnalyzer-Linux-x64.tar.gz`

See [docs/BUILDING.md](docs/BUILDING.md).

## Scientific validation

HistoAnalyzer is research-use software. Validate:

- tissue and anthracosis masks;
- raw and retained InstanSeg nuclei;
- image pixel calibration;
- Tumor/Stroma/Other predictions;
- DAB threshold behavior;
- final area measurements against representative manual review.

Using the same pretrained model does not guarantee identical results across implementations if resolution, tiling, padding, normalization, or postprocessing differ.

## Citation

Use `CITATION.cff` for HistoAnalyzer. If using InstanSeg brightfield nuclei segmentation, also cite the InstanSeg publication listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

HistoAnalyzer source code is released under the MIT License. Third-party packages and pretrained models retain their own licenses.
