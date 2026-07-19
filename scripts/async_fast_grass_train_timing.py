"""
Async Fast-GRASS — TRAINER timing entry point (spec item 1).

This is the ``async_fast_grass_*`` named entry point for trainer-only timing. The
implementation already lives in ``scripts/fast_grass_train_timing.py`` (referenced
by name in async_fast_grass_implementation_details.md, "Timing Calibration"), so
this file delegates to it rather than duplicating ~450 lines. All CLI flags,
JSON output (``analysis/async_fast_grass_timing/train_timing_*.json``), and
``--synthetic`` behavior are identical.

Measures ``seconds_per_train_step`` / ``steps_per_hour`` for one trainer-only
fresh-loss optimizer step on pre-mined data (no mining, no cache, no maintenance).

Usage:
  python scripts/async_fast_grass_train_timing.py --synthetic
  python scripts/async_fast_grass_train_timing.py --steps 500
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_grass_train_timing import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
