# Windows pip cleanup fix — v1.0.5

The v1.0.4 Windows build failed because `pip uninstall` was called with OpenCV packages that were not installed. pip printed a harmless warning to stderr, and PowerShell converted it into a terminating `NativeCommandError` because `$ErrorActionPreference` was `Stop`.

v1.0.5 queries `pip list --format=json`, identifies which OpenCV variants are present, and uninstalls only those packages. The pinned `opencv-python-headless==4.10.0.84` installation and frozen OpenCV ML self-test remain unchanged.
