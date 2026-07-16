from __future__ import annotations

from pathlib import Path

from histoanalyzer import engine


def test_windows_instanseg_cache_is_user_writable(monkeypatch, tmp_path):
    monkeypatch.delenv("INSTANSEG_BIOIMAGEIO_PATH", raising=False)
    monkeypatch.delenv("HISTOANALYZER_CACHE_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(engine.platform, "system", lambda: "Windows")
    cache = engine.configure_instanseg_model_cache()
    assert cache == tmp_path / "HistoAnalyzer" / "models" / "instanseg" / "bioimageio_models"
    assert cache.is_dir()
    assert engine.os.environ["INSTANSEG_BIOIMAGEIO_PATH"] == str(cache)


def test_explicit_instanseg_cache_takes_precedence(monkeypatch, tmp_path):
    exact = tmp_path / "custom-model-cache"
    monkeypatch.setenv("INSTANSEG_BIOIMAGEIO_PATH", str(exact))
    cache = engine.configure_instanseg_model_cache()
    assert cache == exact
    assert cache.is_dir()


def test_worker_disables_inline_ipython_preview():
    from histoanalyzer.job import JobConfig
    from histoanalyzer.worker import common_cli
    cfg = JobConfig()
    args = common_cli(cfg, "image.tif", "out")
    assert "--no-inline-preview" in args
