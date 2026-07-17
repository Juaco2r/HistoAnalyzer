# Nucleus classification, uncertainty and tissue graph

HistoAnalyzer 1.1.0 adds a full CleanTissue nucleus stage after InstanSeg or the watershed fallback.

## Nucleus classes and colors

| Class | RGB | Hex |
|---|---:|---:|
| Small lymphocyte | 40, 105, 220 | `#2869DC` |
| Plasma cell | 146, 78, 185 | `#924EB9` |
| Neutrophil | 0, 180, 210 | `#00B4D2` |
| Macrophage | 241, 132, 36 | `#F18424` |
| Fibroblast/myofibroblast | 46, 160, 67 | `#2EA043` |
| Endothelial cell | 238, 196, 45 | `#EEC42D` |
| Normal pneumocyte/bronchial epithelial cell | 225, 105, 165 | `#E169A5` |
| Tumour epithelial cell | 215, 45, 45 | `#D72D2D` |
| Uncertain | 145, 145, 145 | `#919191` |

## Features

Each segmented nucleus receives physical size, area, axes, aspect ratio, eccentricity, solidity, circularity, hematoxylin percentiles, texture, gradient, and local graph-neighbour features. Spatial features include nearest-neighbour distance, density, orientation coherence, local linearity, local size variability and spacing variability.

## Probabilities and uncertainty

The default model converts the published morphology ranges and spatial clues into soft compatibility probabilities. These values are useful for exploratory grouping but are **not clinically calibrated diagnostic probabilities**. Validate them against pathologist-labelled nuclei before quantitative biological interpretation.

Every record includes:

- `predicted_class`
- `candidate_class` and second candidate
- one `p_*` column per class
- `probability_top1`
- normalized entropy uncertainty
- top-two margin uncertainty
- explicit `p_uncertain`
- combined confidence

An optional joblib model with `predict_proba`, `classes_`, and matching feature names may replace the built-in compatibility classifier.

## Graph and tissue regions

Nuclei are graph nodes. Edges connect the configured k nearest neighbours. Local class-probability composition is aggregated into windows and labelled as Tumour-rich, Stroma-rich, Immune-rich, Vascular-rich, Mixed or Low-nuclei/other.

This is a composition-based regional proposal, not a replacement for histopathological annotation.

## Outputs

- `nuclei_classification.csv`
- `nuclei_classification.geojson`
- `nuclei_class_summary.csv`
- `nuclei_class_palette.csv`
- `nuclei_class_overlay.png`
- `nuclei_class_uncertainty_overlay.png`
- `nuclei_class_legend.png`
- `nuclei_graph.graphml`
- `nuclei_graph_overlay.png`
- `tissue_region_features.csv`
- `tissue_regions.geojson`
- `tissue_region_overlay.png`
- `nuclei_classification_manifest.json`

## CLI

Classification is enabled by default in `baseline` and `predict`.

```bash
histoanalyzer baseline \
  --image sample.tif \
  --nucleus-classification-tile-size 1024 \
  --nucleus-classification-halo-px 64 \
  --nucleus-graph-k 6 \
  --nucleus-graph-radius-um 25 \
  --nucleus-tissue-region-size-um 120
```

Disable it with:

```bash
--no-nucleus-classification
```
