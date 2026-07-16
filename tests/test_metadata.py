import json
from pathlib import Path


def test_release_json_files_are_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    json.loads((root / "codemeta.json").read_text(encoding="utf-8"))
