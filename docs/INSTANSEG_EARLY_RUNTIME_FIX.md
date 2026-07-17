# InstanSeg early-runtime fix

HistoAnalyzer 1.0.11 configures a writable home and model cache through a PyInstaller runtime hook before PyTorch or InstanSeg is imported. Public model archives are downloaded directly into the HistoAnalyzer cache and loaded as TorchScript modules. If fallback occurs, inspect `instanseg_failure_traceback.txt` in the image result folder.
