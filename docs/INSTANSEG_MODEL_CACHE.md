# InstanSeg model cache

HistoAnalyzer sets `INSTANSEG_BIOIMAGEIO_PATH` before loading InstanSeg so model files are downloaded to a writable user directory.

- Windows: `%LOCALAPPDATA%\HistoAnalyzer\models\instanseg\bioimageio_models`
- macOS: `~/Library/Caches/HistoAnalyzer/models/instanseg/bioimageio_models`
- Linux: `${XDG_CACHE_HOME:-~/.cache}/HistoAnalyzer/models/instanseg/bioimageio_models`

Override the base HistoAnalyzer cache with `HISTOANALYZER_CACHE_DIR`, or set the exact InstanSeg location with `INSTANSEG_BIOIMAGEIO_PATH`.
