# Bundled default classifiers

HistoAnalyzer includes the following user-supplied QuPath pixel classifiers as reproducible defaults:

- `TissueClassifierANNFullJuly06.json` — Tissue vs Ignore (OpenCV ANN_MLP).
- `AnthraJuly06.json` — Anthracosis/ink vs Ignore (OpenCV RTrees).
- `DABCNNThreshold0.17DAB.json` — H-DAB stain-2 positivity threshold at 0.17.

These files are used automatically when no custom classifier is selected. Custom JSON files always override the bundled defaults.

## SHA-256

```text
c92c877aeb80da71fb283c1a5f07da6339b73f99613da4ed04d8c5aec656be15  TissueClassifierANNFullJuly06.json
3584a9b503a2cc47d450c6705b431fdca13c225d02736e5913f2d1c564a1a426  AnthraJuly06.json
2f120b146c6e222fe91cfbbd35dd3ac883831d5eb2635f75b4284c5d9ffaa467  DABCNNThreshold0.17DAB.json
```

The classifiers are analysis assets supplied by the project author and are distributed with HistoAnalyzer for reproducibility. Validate them for each staining/scanner workflow before quantitative use.
