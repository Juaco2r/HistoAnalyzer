# HistoAnalyzer v1.0.11 InstanSeg early-runtime fix

Copy these files over a v1.0.10 repository, preserving the directory structure.
The patch configures a writable runtime home before third-party imports, uses a
PyInstaller runtime hook, downloads public InstanSeg models directly into the
HistoAnalyzer cache, and adds frozen-runtime self-tests to all native builds.
