# Windows portable README build fix — v1.0.7

The Windows PyInstaller application and frozen OpenCV ML self-test completed successfully, but the CI job failed while copying `docs/WINDOWS_PORTABLE_README.txt` into the portable release staging directory.

v1.0.7 keeps the documentation file in the repository and makes `scripts/build_windows.ps1` robust: it copies the file when present and generates an equivalent `README_FIRST.txt` inline when the source document is missing. A documentation omission can therefore no longer invalidate a completed Windows native build.
