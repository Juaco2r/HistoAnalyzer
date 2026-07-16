# Building native applications

PyInstaller must run on the target operating system. The included GitHub Actions matrix builds independently on Windows, macOS, and Linux.

## Local Windows build

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

## Local macOS build

```bash
bash scripts/build_macos.sh
```

## Local Linux build

```bash
bash scripts/build_linux.sh
```

Builds use one-folder mode because PyTorch, Qt, OpenCV, and scientific libraries are large and start more reliably when not extracted from a one-file archive.

InstanSeg pretrained weights are downloaded at runtime and are not embedded in the release artifact.

## Code signing

The default workflows create unsigned artifacts. For public distribution:

- Sign the Windows executable and installer with an Authenticode certificate.
- Sign and notarize the macOS application with an Apple Developer ID.
- Optionally package Linux releases as AppImage, Flatpak, or distribution packages.
