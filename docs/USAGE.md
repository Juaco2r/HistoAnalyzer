# Usage guide

## Baseline and nuclei quality control

Choose **Baseline + nuclei preview**. This produces tissue, anthracosis, clean-tissue, overall DAB, and nuclei previews without requiring a compartment model.

Inspect:

- `pipeline_stages_50pct.png`
- `nuclei_validation_raw_model_overlay.png`
- `nuclei_validation_overlay.png`
- `nuclei_validation_montage.png`

The raw-model overlay is the appropriate image for comparing the direct InstanSeg output with another implementation.

## Batch prediction

Choose **Predict Tumor / Stroma / Other**, select a trained `.joblib` model, and add multiple images. Each image is processed sequentially in an isolated worker process.

## Cancellation

The GUI can terminate the active worker process. Completed images remain in the results folder. An interrupted image may retain a temporary work folder that can be deleted later.

## GPU selection

- `auto`: CUDA, MPS, or CPU depending on availability.
- `cuda`: NVIDIA GPU.
- `mps`: Apple Silicon GPU.
- `cpu`: force CPU.

The first InstanSeg run may take longer while model weights are downloaded.
