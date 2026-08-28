"""
False negatives in the training mixture, from two independent sources.

1. EXPLICIT contamination — a negative passage docid that is a known positive for the
   SAME query (train_qrels.txt). True positives mislabelled as negatives; bad training
   signal that ANCE can only partially repair after the first ANN refresh.

2. INCIDENTAL false negatives — in-batch and cross-batch score every query against every
   OTHER query's passages in the same optimizer step, so any passage that is a positive
   for two different queries becomes an unlabelled false negative whenever both land in
   the same batch. Nothing in the code can prevent this; the point is to know its size.
   The explicit hard negative per query (train_group_size 2) is NOT changed by this.

Report only. Usage:
    python scripts/dev/check_neg_contamination.py
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent.parent
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

# ── 2. incidental false negatives from OTHER queries' positives ──────────────
# A docid that is a positive for k > 1 queries contributes a false negative to each of
# the other k-1 whenever they share a batch. Deterministic: no sampling, no RNG.
from collections import Counter                                    # noqa: E402
from utils.helpers import load_config                              # noqa: E402

doc_owners = Counter()
for qid, docids in qrels.items():
    for docid in docids:
        doc_owners[docid] += 1

n_queries = max(len(qrels), 1)
shared_docs = {d: k for d, k in doc_owners.items() if k > 1}
# Expected collisions per query per step = sum over the query's positives of (k-1),
# scaled by the chance the colliding query is in the same batch: (B-1)/(N-1).
# For each query, how many OTHER queries claim one of its positives.
collisions = sum(sum(doc_owners[d] - 1 for d in docids)
                 for docids in qrels.values())
queries_affected = sum(1 for docids in qrels.values()
                       if any(doc_owners[d] > 1 for d in docids))

cfg = load_config()['training']
batches = [("in-batch", cfg['inbatch']['batch_size'] * cfg['inbatch']['train_group_size']),
           ("cross-batch", cfg['crossbatch']['target_batch_size']
            * cfg['crossbatch']['train_group_size'])]

print("\n" + "=" * 55)
print(f"  Total training examples scanned: {total:>8,}")
print(f"  Contaminated (neg is true pos):  {contaminated:>8,}  ({100*contaminated/max(total,1):.2f}%)")
if examples:
    print(f"\n  Sample contaminated examples:")
    for ex in examples:
        print(f"    qid={ex['qid']}  leaked docids={ex['leaked']}")
    print("  ⚠️  FAIL — true positives are being used as hard negatives.")
else:
    print("  ✅ PASS — no explicit contamination found.")

print("-" * 55)
print("  INCIDENTAL false negatives (other queries' positives)")
print(f"  Judged queries:                  {n_queries:>8,}")
print(f"  Docids positive for >1 query:    {len(shared_docs):>8,}")
print(f"  Queries touching a shared docid: {queries_affected:>8,}  "
      f"({100*queries_affected/n_queries:.2f}%)")
for label, pool in batches:
    # (pool_passages - 1) scored per query; the share that is someone else's positive.
    expected = collisions / n_queries * (min(pool, n_queries) - 1) / max(n_queries - 1, 1)
    print(f"  {label:<12} pool {pool - 1:>5} negs/query -> "
          f"~{expected:.3f} incidental false negative(s) per query per step")
print("  Report only: the one explicit hard negative per query is unchanged.")
print("=" * 55)
