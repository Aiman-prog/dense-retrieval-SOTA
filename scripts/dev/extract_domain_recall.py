#!/usr/bin/env python3
"""Per-domain R@1000 for every arm, read from the BRIGHT eval summaries (read-only).

    ssh delftblue python3 - < scripts/dev/extract_domain_recall.py > domain_recall.json

JSON goes to stdout, the same numbers as a table to stderr.
"""
import glob
import json
import os
import sys

ROOT = '/scratch/%s/dense-retrieval-SOTA/results/bright_benchmark/' % os.environ['USER']
ROWS = [
    ('in-batch', 'inbatch_mixed_bge_m3__*/*/summary.json'),
    ('cross-batch', 'crossbatch_mixed_bge_m3_step600__*/*/summary.json'),
    ('ance', 'ance_mixed_bge_m3__*/*/summary.json'),
    ('grass l=0', 'async_fast_grass_mixed_bge_m3_s6_lam0__*/*/summary.json'),
    ('grass l=0.5', 'async_fast_grass_mixed_bge_m3_s6_lam05__*/*/summary.json'),
    ('grass l=0.3', 'async_fast_grass_mixed_bge_m3_s6_lam03__*/*/summary.json'),
    ('bge-m3 untrained', 'main__*/*/summary.json'),
    ('bm25', 'bm25_k1-0.9_b-0.4/summary.json'),
]

out = {}
for name, pattern in ROWS:
    runs = [json.load(open(p)) for p in glob.glob(ROOT + pattern)]
    runs = [r for r in runs if len(r['per_domain']) == 12]
    assert len(runs) == 1, (name, len(runs))
    dom = {d['domain']: d['recall_1000'] for d in runs[0]['per_domain']}
    out[name] = dict(dom, macro=runs[0]['macro_recall_1000'],
                     mean_excl_pony=sum(v for k, v in dom.items() if k != 'pony') / 11)

json.dump(out, sys.stdout, indent=2)

names = [name for name, _ in ROWS]
sys.stderr.write('%-22s' % 'domain' + ''.join('%17s' % n for n in names) + '\n')
for domain in out[names[0]]:
    sys.stderr.write('%-22s' % domain + ''.join('%17.4f' % out[n][domain] for n in names) + '\n')
