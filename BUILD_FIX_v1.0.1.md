# HistoAnalyzer v1.0.1 build fix

GitHub Actions v1.0.0 failed because `build/HistoAnalyzer.spec` resolved the
repository root one directory above the checkout. GitHub checks repositories
out as `<runner-work>/<repository>/<repository>`, and `SPECPATH` already points
to the directory containing the spec file (`build`).

The corrected resolution is:

```python
SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent
ENTRY = ROOT / "run_histoanalyzer.py"
```

v1.0.1 also:

- validates the checkout before installing large build dependencies;
- prints the resolved build root and entry point;
- uses the repository-level launcher for PyInstaller;
- applies the same fix to Windows, macOS, and Linux;
- updates package, installer, CFF, and CodeMeta versions to 1.0.1.

## Rebuild

Commit the corrected files, then either run **Build desktop applications** from
GitHub Actions or create and push a new tag:

```bash
git add .
git commit -m "Fix cross-platform PyInstaller build paths"
git tag v1.0.1
git push origin main --tags
```

Do not reuse the failed `v1.0.0` tag; use `v1.0.1` so GitHub and Zenodo retain a
clear immutable release history.
