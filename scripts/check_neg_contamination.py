"""
Negative-in-qrels contamination check.
Scans raw training_mixture/ JSONL files to find examples where a negative passage docid
matches a known positive for that query (according to train_qrels.txt).

These are true positives mislabelled as negatives — bad training signal that ANCE can
only partially repair after the first ANN refresh.

Usage:
    python scripts/check_neg_contamination.py
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
from utils.helpers import get_path

processed   = get_path("processed")
qrels_file  = processed / "train_qrels.txt"
mixture_dir = processed / "training_mixture"

if not qrels_file.exists():
    print(f"ERROR: {qrels_file} not found — run preprocessing first.")
    sys.exit(1)

print(f"Loading qrels from {qrels_file}...")
qrels = defaultdict(set)
with open(qrels_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            qrels[parts[0]].add(parts[2])
print(f"  {len(qrels):,} queries with known positives")

mix_files = sorted(mixture_dir.glob("*.jsonl"))
if not mix_files:
    print(f"ERROR: No JSONL files in {mixture_dir}")
    sys.exit(1)

total        = 0
contaminated = 0
examples     = []

for f_path in mix_files:
    with open(f_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            total += 1
            qid        = str(d.get('query_id', ''))
            pos_ids    = qrels.get(qid, set())
            neg_docids = [n['docid'] for n in d.get('negative_passages', [])]
            leaked     = [did for did in neg_docids if did in pos_ids]
            if leaked:
                contaminated += 1
                if len(examples) < 3:
                    examples.append({'qid': qid, 'leaked': leaked})

print("\n" + "=" * 55)
print(f"  Total training examples scanned: {total:>8,}")
print(f"  Contaminated (neg is true pos):  {contaminated:>8,}  ({100*contaminated/max(total,1):.2f}%)")
if examples:
    print(f"\n  Sample contaminated examples:")
    for ex in examples:
        print(f"    qid={ex['qid']}  leaked docids={ex['leaked']}")
    print("  ⚠️  FAIL — true positives are being used as hard negatives.")
else:
    print("  ✅ PASS — no contamination found.")
print("=" * 55)
