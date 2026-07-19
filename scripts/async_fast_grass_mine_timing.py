"""
Async Fast-GRASS — MINER timing entry point (spec item 2).

This is the ``async_fast_grass_*`` named entry point for miner-round timing. The
implementation already lives in ``scripts/fast_grass_mine_timing.py`` (referenced
by name in async_fast_grass_implementation_details.md, "Timing Calibration"), so
this file delegates to it rather than duplicating ~600 lines. All CLI flags,
JSON output (``analysis/async_fast_grass_timing/mine_timing_*.json``), and
``--synthetic`` behavior are identical.

Measures ``t_mine_round`` for one full Fast-GRASS mined round: cache-init from the
stale pickle, ``_mine_batch_mcdp`` over the mixture, periodic in-round
``cache.maintain`` at ``cache_update_interval * batch_size`` mined queries with
``step=source_checkpoint_step``, plus one end-of-round maintenance cycle. Reports
``queries_per_second``, MCDP encoder calls, unique top-L docs, and
refresh/replace counts. NO training, NO per-query stale FAISS top-P, NO full-corpus
ANN rebuild.

Usage:
  python scripts/async_fast_grass_mine_timing.py --synthetic
  python scripts/async_fast_grass_mine_timing.py --B_doc 32000 --L 64 --T 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_grass_mine_timing import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
