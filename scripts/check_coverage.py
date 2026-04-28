"""
Corpus coverage check — verifies every positive docid in train_qrels.txt has an entry
in reasonir_corpus.jsonl. Missing entries mean the FAISS index can never retrieve that
positive, making the qrel filter for that query partially blind during ANCE mining.

Usage:
    python scripts/check_coverage.py
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
from utils.helpers import get_path

processed   = get_path("processed")
corpus_file = processed / "reasonir_corpus.jsonl"
qrels_file  = processed / "train_qrels.txt"

if not corpus_file.exists():
    print(f"ERROR: {corpus_file} not found — run preprocessing first.")
    sys.exit(1)
if not qrels_file.exists():
    print(f"ERROR: {qrels_file} not found — run preprocessing first.")
    sys.exit(1)

print(f"Loading corpus docids from {corpus_file}...")
corpus_docids = set()
with open(corpus_file) as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            corpus_docids.add(d['docid'])
print(f"  {len(corpus_docids):,} unique docids in corpus")

print(f"Loading qrel positives from {qrels_file}...")
qrel_docids = set()
with open(qrels_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            qrel_docids.add(parts[2])
print(f"  {len(qrel_docids):,} unique positive docids in qrels")

missing = qrel_docids - corpus_docids
print("\n" + "=" * 55)
print(f"  Positive docids missing from corpus: {len(missing):>8,}")
if missing:
    print(f"  Sample missing: {list(missing)[:5]}")
    print("  ⚠️  FAIL — these positives cannot be retrieved during mining.")
else:
    print("  ✅ PASS — all positive docids present in corpus.")
print("=" * 55)
