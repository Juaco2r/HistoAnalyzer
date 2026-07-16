# Bundled classifier build fix — v1.0.6

The v1.0.5 PyInstaller spec referenced `src/histoanalyzer/resources/classifiers`, but that directory was absent from the repository even though the root-level `classifiers/` folder existed. PyInstaller therefore stopped while adding data files.

v1.0.6 includes identical copies of all three JSON classifiers in the package source tree, synchronizes them before every native build, validates both locations, and lets the PyInstaller spec use the root classifier directory as a fallback.
