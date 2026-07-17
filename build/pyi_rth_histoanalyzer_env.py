# PyInstaller runtime hook: execute before application imports.
from histoanalyzer.runtime_env import bootstrap_runtime_environment
bootstrap_runtime_environment()
