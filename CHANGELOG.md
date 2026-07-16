# Changelog

## 1.0.7 — 2026-07-16

- Fixed the Windows job failing after a successful PyInstaller build because `docs/WINDOWS_PORTABLE_README.txt` could not be copied.
- Added a self-generating `README_FIRST.txt` fallback to the Windows portable packaging step.
- Added regression tests for both the documented notice and the inline fallback.
- Kept the installer and portable archive validation unchanged.

## 1.0.6 — 2026-07-16

- Fixed Windows, macOS and Linux PyInstaller failures caused by a missing `src/histoanalyzer/resources/classifiers` directory.
- Added the three bundled classifier JSON files to the installable package source tree.
- Added `scripts/sync_bundled_classifiers.py` and run it before every native build.
- Made the PyInstaller spec fall back to the repository-level `classifiers/` directory when needed.
- Added build validation and regression tests for classifier resource synchronization.

## 1.0.5 — 2026-07-16

- Fixed Windows CI failure caused by PowerShell treating pip warnings for absent OpenCV packages as terminating errors.
- The Windows build now queries installed distributions and uninstalls only OpenCV variants that are actually present.
- Added explicit native-command exit-code checks around OpenCV cleanup and reinstall.
- Added a regression test for the Windows OpenCV cleanup contract.

## 1.0.4 — 2026-07-16

- Bundle the validated Tissue, Anthracosis and H-DAB threshold JSON classifiers.
- Use bundled classifiers automatically in the GUI, worker jobs, CLI and frozen applications.
- Add **Use bundled defaults** to restore the included classifier set.
- Include classifier resources in wheels, source distributions and PyInstaller builds.

## 1.0.2 — 2026-07-16

- Makes the Windows Inno Setup installer the preferred release artifact.
- Renames the portable archive to `PORTABLE-EXTRACT-FIRST`.
- Adds a visible `README_FIRST.txt` with extraction instructions.
- Verifies `HistoAnalyzer.exe` and `_internal/python311.dll` before packaging.
- Installs Inno Setup explicitly on the Windows GitHub Actions runner.
- Uploads both the Windows installer and portable archive.


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
