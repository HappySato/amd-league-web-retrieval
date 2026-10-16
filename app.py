#!/usr/bin/env python3
"""Container entry point.

The evaluation harness starts this file directly::

    python3 /app/app.py

One JSON file is written per input question. See ``iwir.adapter`` for the
schema and ``README.md`` for environment variables.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iwir.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
