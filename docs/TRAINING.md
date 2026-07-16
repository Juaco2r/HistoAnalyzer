# Tumor/Stroma/Other training

## Annotation format

Use GeoJSON polygons classified as `Tumor`, `Stroma`, or `Other`. Include varied morphology, staining intensity, tumor center, invasive edge, dense and loose stroma, necrosis, benign epithelium, vessels, inflammation, and artifacts.

## Batch pairing

For image `ID_2041_1.tif`, use:

```text
ID_2041_1_compartments.geojson
```

Place all annotation files in the selected annotation folder.

## Avoid leakage

Separate patients or specimens between training and independent validation. Do not evaluate only on regions used to train the Random Forest.

## Model inputs

The regional classifier uses hematoxylin-derived nuclear morphology, density, spacing, alignment, and texture. DAB intensity is excluded from the compartment feature vector to reduce circularity.
