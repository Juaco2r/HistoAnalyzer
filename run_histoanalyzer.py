#!/usr/bin/env python3
from histoanalyzer.runtime_env import bootstrap_runtime_environment
bootstrap_runtime_environment()

from histoanalyzer.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
