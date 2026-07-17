from __future__ import annotations

"""Early runtime environment bootstrap for source and frozen applications.

This module deliberately has no third-party imports.  It is safe to execute as
an early PyInstaller runtime hook before PyTorch, InstanSeg, platformdirs or any
other dependency attempts to resolve the user's home directory.
"""

import os
import platform
import tempfile
from pathlib import Path
from typing import Dict, Optional

_BOOTSTRAPPED = False
_ORIGINAL_PATH_HOME = Path.home
_ORIGINAL_PATH_EXPANDUSER = Path.expanduser
_ORIGINAL_OS_EXPANDUSER = os.path.expanduser
_RUNTIME_HOME: Optional[Path] = None


def _env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    if not value or not value.strip():
        return None
    value = os.path.expandvars(value.strip().strip('"'))
    if value.startswith('~'):
        return None
    return Path(value)


def _windows_known_folder(csidl: int) -> Optional[Path]:
    if platform.system().lower() != 'windows':
        return None
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(32768)
        result = ctypes.windll.shell32.SHGetFolderPathW(None, int(csidl), None, 0, buf)  # type: ignore[attr-defined]
        if result == 0 and buf.value:
            return Path(buf.value)
    except Exception:
        pass
    return None


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / '.histoanalyzer_runtime_probe'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _resolve_home() -> Path:
    system = platform.system().lower()
    candidates = []
    if system == 'windows':
        candidates.extend([
            _env_path('USERPROFILE'),
            _windows_known_folder(40),  # CSIDL_PROFILE
        ])
        drive = os.environ.get('HOMEDRIVE', '')
        part = os.environ.get('HOMEPATH', '')
        if drive and part:
            candidates.append(Path(drive + part))
        for appdata in (_env_path('LOCALAPPDATA'), _windows_known_folder(28), _env_path('APPDATA')):
            if appdata is not None:
                try:
                    candidates.append(appdata.parents[1])
                except IndexError:
                    pass
    else:
        candidates.append(_env_path('HOME'))
    candidates.extend([
        Path(tempfile.gettempdir()) / 'HistoAnalyzer-user',
        Path.cwd() / '.histoanalyzer-user',
    ])
    for candidate in candidates:
        if candidate is not None and _writable(candidate):
            return candidate.resolve()
    raise RuntimeError('Could not determine a writable HistoAnalyzer runtime home.')


def _resolve_local_appdata(home: Path) -> Path:
    if platform.system().lower() == 'windows':
        candidates = [_env_path('LOCALAPPDATA'), _windows_known_folder(28), home / 'AppData' / 'Local']
    elif platform.system().lower() == 'darwin':
        candidates = [home / 'Library' / 'Caches']
    else:
        candidates = [_env_path('XDG_CACHE_HOME'), home / '.cache']
    candidates.append(Path(tempfile.gettempdir()))
    for candidate in candidates:
        if candidate is not None and _writable(candidate):
            return candidate.resolve()
    return Path(tempfile.gettempdir()).resolve()


def _install_home_fallback(home: Path) -> None:
    """Make all common Python home-resolution paths deterministic."""
    global _RUNTIME_HOME
    _RUNTIME_HOME = home

    def safe_home(cls):  # noqa: ANN001 - classmethod protocol
        return cls(str(_RUNTIME_HOME))

    def safe_path_expanduser(self):  # noqa: ANN001
        raw = str(self)
        if raw == '~' or raw.startswith('~/') or raw.startswith('~\\'):
            return type(self)(str(_RUNTIME_HOME) + raw[1:])
        try:
            return _ORIGINAL_PATH_EXPANDUSER(self)
        except RuntimeError:
            return type(self)(str(_RUNTIME_HOME) + raw[1:]) if raw.startswith('~') else self

    def safe_os_expanduser(value):  # noqa: ANN001
        raw = os.fspath(value)
        if raw == '~' or raw.startswith('~/') or raw.startswith('~\\'):
            return str(_RUNTIME_HOME) + raw[1:]
        expanded = _ORIGINAL_OS_EXPANDUSER(raw)
        if expanded == raw and raw.startswith('~'):
            return str(_RUNTIME_HOME) + raw[1:]
        return expanded

    Path.home = classmethod(safe_home)  # type: ignore[method-assign]
    Path.expanduser = safe_path_expanduser  # type: ignore[method-assign]
    os.path.expanduser = safe_os_expanduser  # type: ignore[assignment]


def bootstrap_runtime_environment() -> Dict[str, str]:
    global _BOOTSTRAPPED
    home = _resolve_home()
    local = _resolve_local_appdata(home)
    base = _env_path('HISTOANALYZER_CACHE_DIR') or (local / 'HistoAnalyzer')
    model_cache = _env_path('INSTANSEG_BIOIMAGEIO_PATH') or (base / 'models' / 'instanseg' / 'bioimageio_models')
    torch_cache = _env_path('TORCH_HOME') or (base / 'models' / 'torch')
    for folder in (base, model_cache, torch_cache):
        folder.mkdir(parents=True, exist_ok=True)

    # Assign, rather than setdefault, because a malformed inherited value can
    # be worse than a missing value in frozen GUI/worker processes.
    os.environ['HOME'] = str(home)
    os.environ['USERPROFILE'] = str(home)
    os.environ['LOCALAPPDATA'] = str(local)
    if platform.system().lower() == 'windows' and home.drive:
        os.environ['HOMEDRIVE'] = home.drive
        os.environ['HOMEPATH'] = str(home)[len(home.drive):] or '\\'
    os.environ['HISTOANALYZER_CACHE_DIR'] = str(base)
    os.environ['INSTANSEG_BIOIMAGEIO_PATH'] = str(model_cache)
    os.environ['TORCH_HOME'] = str(torch_cache)
    os.environ.setdefault('MPLCONFIGDIR', str(base / 'matplotlib'))
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

    _install_home_fallback(home)
    _BOOTSTRAPPED = True
    return {
        'home': str(home),
        'local_appdata': str(local),
        'cache_base': str(base),
        'instanseg_model_cache': str(model_cache),
        'torch_home': str(torch_cache),
    }
