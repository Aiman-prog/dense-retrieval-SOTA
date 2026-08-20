#!/usr/bin/env python
"""
Preprocessor regression check for the branch consolidation.

`run_setup()` in src/data/preprocessor.py is deterministic (MD5 dedupe,
drop_duplicates, insertion order, no timestamps, no RNG), so byte-identical
output is a fair before/after criterion for it. This script pins that:

  1. rebuild a fixed 600-record fixture mixture (deterministic, no network),
  2. WIPE run_setup()'s three outputs -- it short-circuits when they already
     exist, which would make the diff vacuously clean,
  3. run run_setup() against the fixture,
  4. print the sha256 of each output.

Run it before consolidation to record the baseline, then after every step.
Any hash difference means the step changed preprocessor behaviour.

Scope: run_setup() only. prepare_msmarco_train_data() calls random.shuffle()
unseeded and is deliberately NOT covered.

Environment overrides:
  PREPROC_FIXTURE_ROOT    fixture DATA_BASE_DIR   (default: $TMPDIR/dense_retrieval_preproc_fixture)
  PREPROC_FIXTURE_SOURCE  mixture records source  (default: $DATA_BASE_DIR/data/processed/train_reasonir.jsonl)

Usage:  python scripts/consolidation_preproc_check.py
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The three fixture mixture files, in the layout run_setup()'s glob("*.jsonl")
# walks. 200 records each, taken in file order -- no sampling, no RNG.
FIXTURE_SPLITS = [("train_msmarco.jsonl", 0, 200),
                  ("train_vl.jsonl", 200, 400),
                  ("train_hq.jsonl", 400, 600)]

# Real records alone do not exercise run_setup()'s duplicate-text branch: in the
# first 600 there is no text that appears under two different docids (the first
# is at record ~14474, far outside any workable fixture). This fourth file
# re-emits texts already present in the window under fresh docids, so both
# `docid_remap` and the qrels canonicalization that consumes it actually run.
# Derived from the same 600 records -- no extra source data, no RNG.
DUPE_FILE = "train_zz_dupes.jsonl"
DUPE_COUNT = 5

OUTPUTS = ["reasonir_corpus.jsonl", "train_queries.jsonl", "train_qrels.txt"]


def default_source() -> Path:
    base = os.environ.get("DATA_BASE_DIR")
    if not base:
        user = os.environ.get("USER", os.environ.get("USERNAME", "user"))
        base = f"/scratch/{user}/dense-retrieval-SOTA"
    return Path(base) / "data" / "processed" / "train_reasonir.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixture(source: Path, mixture_dir: Path) -> None:
    needed = FIXTURE_SPLITS[-1][2]
    with open(source, "r", encoding="utf-8") as handle:
        records = [next(handle) for _ in range(needed)]
    mixture_dir.mkdir(parents=True, exist_ok=True)
    for name, start, end in FIXTURE_SPLITS:
        with open(mixture_dir / name, "w", encoding="utf-8") as handle:
            handle.writelines(records[start:end])

    parsed = [json.loads(line) for line in records[:DUPE_COUNT + 1]]
    with open(mixture_dir / DUPE_FILE, "w", encoding="utf-8") as handle:
        for i in range(DUPE_COUNT):
            handle.write(json.dumps({
                "query_id": f"fixture_dupe_{i}",
                "query": f"fixture duplicate-text probe {i}",
                "positive_passages": [{
                    "docid": f"fixture_dupe_pos_{i}",
                    "text": parsed[i]["positive_passages"][0]["text"],
                }],
                "negative_passages": [{
                    "docid": f"fixture_dupe_neg_{i}",
                    "text": parsed[i + 1]["positive_passages"][0]["text"],
                }],
            }, ensure_ascii=False) + "\n")


def main() -> int:
    root = Path(os.environ.get(
        "PREPROC_FIXTURE_ROOT",
        Path(tempfile.gettempdir()) / "dense_retrieval_preproc_fixture"))
    source = Path(os.environ.get("PREPROC_FIXTURE_SOURCE", default_source()))
    if not source.is_file():
        print(f"FIXTURE_SOURCE_MISSING {source}", file=sys.stderr)
        return 2

    processed = root / "data" / "processed"
    build_fixture(source, processed / "training_mixture")

    # run_setup() returns early if all three outputs exist and are non-empty.
    for name in OUTPUTS:
        (processed / name).unlink(missing_ok=True)

    os.environ["DATA_BASE_DIR"] = str(root)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from data.preprocessor import run_setup

    run_setup()

    print()
    print(f"PREPROC_FIXTURE_ROOT   {root}")
    print(f"PREPROC_FIXTURE_SOURCE {source}  sha256={sha256(source)[:16]}...")
    for name in OUTPUTS:
        path = processed / name
        if not path.is_file():
            print(f"MISSING_OUTPUT {name}", file=sys.stderr)
            return 1
        print(f"PREPROC_SHA256 {name} {sha256(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
