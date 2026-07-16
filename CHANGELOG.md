# Changelog

## 1.0.1 — 2026-07-16

- Fixed the PyInstaller project-root resolution used by GitHub Actions.
- Fixed macOS, Windows, and Linux release builds looking one directory above the checkout.
- Added a cross-platform build-layout validation step before PyInstaller runs.
- Switched the frozen entry point to the repository-level `run_histoanalyzer.py` launcher.
- Added clearer build diagnostics showing the resolved root and entry point.

## 1.0.0 — 2026-07-16

- Initial HistoAnalyzer desktop release.
- Single and batch image queue.
- Baseline, training, prediction, and train-then-predict modes.
- QuPath ANN, RTrees, and H-DAB threshold JSON interpretation.
- Anthracosis dilation and CleanTissue generation.
- InstanSeg brightfield nuclei backend with watershed fallback.
- Tumor/Stroma/Other Random Forest training and prediction.
- Compartment-specific DAB quantification.
- GeoJSON, CSV, TIFF mask, and PNG quality-control outputs.
- Windows, macOS, and Linux PyInstaller build workflows.
- Zenodo, Citation File Format, and CodeMeta metadata.
