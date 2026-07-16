# Third-party notices

HistoAnalyzer does not redistribute pretrained InstanSeg model weights. InstanSeg downloads the selected model at runtime and the model remains subject to its upstream license and dataset terms.

## InstanSeg

- Project: https://github.com/instanseg/instanseg
- Package: `instanseg-torch`
- Upstream package license: Apache-2.0
- Default model: `brightfield_nuclei`

Suggested citation for brightfield nuclei segmentation:

Goldsborough T. et al. (2024). *InstanSeg: an embedding-based instance segmentation algorithm optimized for accurate, efficient and portable cell segmentation*. arXiv. DOI: 10.48550/arXiv.2408.15954.

## QuPath compatibility

HistoAnalyzer interprets a limited set of QuPath-exported pixel-classifier JSON structures. QuPath is not bundled and is not required at runtime. QuPath remains subject to its own license.

## Python dependencies

The packaged application includes open-source Python dependencies such as PySide6, NumPy, OpenCV, tifffile, scikit-image, scikit-learn, SciPy, PyTorch, and Pillow. Their individual license files should be preserved in release distributions where required.
