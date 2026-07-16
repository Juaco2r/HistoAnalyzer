# OpenCV ML runtime fix — v1.0.3

HistoAnalyzer requires OpenCV's ANN_MLP, RTrees and FileStorage APIs to reconstruct the supplied QuPath pixel-classifier JSON models.

Version 1.0.3 fixes packaged applications that imported `cv2` but did not expose `cv2.ml`. The release now:

- installs exactly one OpenCV distribution (`opencv-python-headless==4.10.0.84`);
- relies on PyInstaller's dedicated OpenCV hook;
- supports both standard and flattened OpenCV ML Python bindings;
- tests OpenCV ML before packaging; and
- launches the frozen application in self-test mode before publishing any release artifact.

A platform build fails in CI if ANN_MLP, RTrees or FileStorage is unavailable.
