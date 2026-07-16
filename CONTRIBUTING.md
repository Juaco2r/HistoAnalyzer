# Contributing

1. Create a feature branch from `main`.
2. Install `python -m pip install -e ".[wsi,dev]"`.
3. Add or update tests.
4. Run `pytest`, `ruff check src tests`, and `python -m compileall src`.
5. Open a focused pull request describing scientific and software validation.

Changes affecting segmentation, stain deconvolution, region classification, or measurements should include before/after quality-control evidence and representative test data that can legally be shared.
