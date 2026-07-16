# HistoAnalyzer v1.0.6 classifier-resource build fix

Copy these files over the repository root, preserving the directory structure.

This patch fixes PyInstaller failures reporting that
`src/histoanalyzer/resources/classifiers` could not be found. It includes the
three JSON classifiers in the package source tree, synchronizes them before
all native builds, and adds a root-level fallback in the PyInstaller spec.
