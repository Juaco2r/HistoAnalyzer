import os
from pathlib import Path


def test_runtime_bootstrap_survives_missing_home(monkeypatch, tmp_path):
    import histoanalyzer.runtime_env as runtime_env
    for key in ("HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "HOMEDRIVE", "HOMEPATH", "HISTOANALYZER_CACHE_DIR", "INSTANSEG_BIOIMAGEIO_PATH", "TORCH_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(runtime_env.tempfile, "gettempdir", lambda: str(tmp_path))
    result = runtime_env.bootstrap_runtime_environment()
    assert Path.home().is_dir()
    assert Path("~").expanduser().is_dir()
    assert Path(result["instanseg_model_cache"]).is_dir()


def test_spec_uses_early_runtime_hook():
    root = Path(__file__).resolve().parents[1]
    spec = (root / "build" / "HistoAnalyzer.spec").read_text(encoding="utf-8")
    assert "pyi_rth_histoanalyzer_env.py" in spec
    assert (root / "build" / "pyi_rth_histoanalyzer_env.py").is_file()


def test_direct_model_download_mapping():
    from histoanalyzer.engine import INSTANSEG_PUBLIC_MODELS
    assert INSTANSEG_PUBLIC_MODELS["brightfield_nuclei"]["version"] == "0.1.1"
    assert INSTANSEG_PUBLIC_MODELS["brightfield_nuclei"]["url"].endswith("brightfield_nuclei.zip")


def test_native_builds_run_frozen_instanseg_runtime_self_test():
    root = Path(__file__).resolve().parents[1]
    for relative in ("scripts/build_windows.ps1", "scripts/build_macos.sh", "scripts/build_linux.sh"):
        text = (root / relative).read_text(encoding="utf-8")
        assert "--self-test-instanseg-runtime" in text
