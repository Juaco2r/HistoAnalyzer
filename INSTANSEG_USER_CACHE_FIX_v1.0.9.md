# InstanSeg user-cache fix — HistoAnalyzer v1.0.9

InstanSeg downloads public model files into `INSTANSEG_BIOIMAGEIO_PATH`. If this variable is not set, InstanSeg defaults to a directory beside its installed Python package. In an installed Windows application this can resolve under `C:\Program Files\HistoAnalyzer`, which standard users cannot modify.

HistoAnalyzer v1.0.9 sets the path before importing InstanSeg:

- Windows: `%LOCALAPPDATA%\HistoAnalyzer\models\instanseg\bioimageio_models`
- macOS: `~/Library/Caches/HistoAnalyzer/models/instanseg/bioimageio_models`
- Linux: `${XDG_CACHE_HOME:-~/.cache}/HistoAnalyzer/models/instanseg/bioimageio_models`

The path can be overridden with `HISTOANALYZER_CACHE_DIR` or with the exact `INSTANSEG_BIOIMAGEIO_PATH` variable.

The nuclei preview and CSV now report the actual backend used (`instanseg` or `watershed`) and include the fallback reason when applicable.
