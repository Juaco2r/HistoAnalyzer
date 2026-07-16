from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


TISSUE_CLASSIFIER_NAME = "TissueClassifierANNFullJuly06.json"
ANTHRA_CLASSIFIER_NAME = "AnthraJuly06.json"
DAB_CLASSIFIER_NAME = "DABCNNThreshold0.17DAB.json"


@dataclass(frozen=True)
class ClassifierPaths:
    tissue: Path
    anthra: Path
    dab: Path

    def as_dict(self) -> Dict[str, Path]:
        return {"tissue": self.tissue, "anthra": self.anthra, "dab": self.dab}


def _candidate_classifier_dirs() -> list[Path]:
    package_root = Path(__file__).resolve().parent
    candidates = [package_root / "resources" / "classifiers"]
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidates = [
            base / "histoanalyzer" / "resources" / "classifiers",
            base / "resources" / "classifiers",
            Path(sys.executable).resolve().parent / "_internal" / "histoanalyzer" / "resources" / "classifiers",
        ] + candidates
    repository_root = package_root.parents[1] if len(package_root.parents) > 1 else package_root
    candidates.append(repository_root / "classifiers")
    return candidates


def bundled_classifier_paths(require_existing: bool = True) -> ClassifierPaths:
    for directory in _candidate_classifier_dirs():
        paths = ClassifierPaths(
            tissue=directory / TISSUE_CLASSIFIER_NAME,
            anthra=directory / ANTHRA_CLASSIFIER_NAME,
            dab=directory / DAB_CLASSIFIER_NAME,
        )
        if all(path.is_file() for path in paths.as_dict().values()):
            return paths
    attempted = "\n".join(str(path) for path in _candidate_classifier_dirs())
    if require_existing:
        raise FileNotFoundError(
            "Bundled HistoAnalyzer classifiers were not found. Checked:\n" + attempted
        )
    directory = _candidate_classifier_dirs()[0]
    return ClassifierPaths(
        directory / TISSUE_CLASSIFIER_NAME,
        directory / ANTHRA_CLASSIFIER_NAME,
        directory / DAB_CLASSIFIER_NAME,
    )
