"""
Corpus deduplication diagnostic.
Run on login node before and after the docid fix to measure improvement.

Usage:
    python scripts/check_corpus_dupes.py
"""

import json
import glob
import hashlib
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_path

mixture_dir = get_path("processed") / "training_mixture"
files = sorted(mixture_dir.glob("*.jsonl"))

if not files:
    print(f"No JSONL files found in {mixture_dir}")
    sys.exit(1)

print(f"Scanning {len(files)} file(s) in {mixture_dir}...\n")

total_passages  = 0
unique_by_docid = set()
unique_by_text  = set()
pos_neg_dupes   = 0
neg_neg_dupes   = 0
total_examples  = 0

for f in files:
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total_examples += 1

            pos  = d.get('positive_passages', [])
            negs = d.get('negative_passages', [])

            # Collect all passages for corpus stats
            for p in pos + negs:
                text   = p.get('text', '').strip()
                docid  = p.get('docid', '')
                h      = hashlib.md5(text.encode()).hexdigest()
                total_passages += 1
                unique_by_docid.add(docid)
                unique_by_text.add(h)

            # pos == neg (exact text match)
            if pos and negs:
                pos_text = pos[0].get('text', '').strip()
                if any(n.get('text', '').strip() == pos_text for n in negs):
                    pos_neg_dupes += 1

            # neg == neg (any two negatives share text)
            neg_texts = [n.get('text', '').strip() for n in negs]
            neg_hashes = [hashlib.md5(t.encode()).hexdigest() for t in neg_texts]
            if len(neg_hashes) != len(set(neg_hashes)):
                neg_neg_dupes += 1

print("=" * 55)
print(f"  Total training examples:       {total_examples:>10,}")
print(f"  Total passages (pos + neg):    {total_passages:>10,}")
print("-" * 55)
print(f"  Unique by docid:               {len(unique_by_docid):>10,}")
print(f"  Unique by text hash:           {len(unique_by_text):>10,}")
print(f"  Text dupes missed by docid:    {len(unique_by_docid) - len(unique_by_text):>10,}")
print("-" * 55)
print(f"  Examples where pos == neg1:    {pos_neg_dupes:>10,}  ({100*pos_neg_dupes/total_examples:.1f}%)")
print(f"  Examples with dup negatives:   {neg_neg_dupes:>10,}  ({100*neg_neg_dupes/total_examples:.1f}%)")
print("=" * 55)
