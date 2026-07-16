HistoAnalyzer v1.0.3 — OpenCV ML runtime correction

The Windows v1.0.2 executable could import cv2 but did not expose cv2.ml,
preventing ANN_MLP and RTrees model reconstruction.

Apply this release as a complete repository update, push it, and build a new
v1.0.3 artifact. Do not reuse the v1.0.2 executable.

Key corrections:
- Exactly one OpenCV distribution is installed during native builds:
  opencv-python-headless==4.10.0.84.
- Manual collect_all("cv2") was removed from the PyInstaller spec.
- The engine supports both cv2.ml and flattened top-level OpenCV ML aliases.
- Every build runs a source OpenCV test and then launches the frozen executable
  in self-test mode before any release artifact is uploaded.

Suggested commands:
  git add .
  git commit -m "Fix packaged OpenCV ML runtime"
  git push origin main

After the Actions build succeeds:
  git tag v1.0.3
  git push origin v1.0.3
