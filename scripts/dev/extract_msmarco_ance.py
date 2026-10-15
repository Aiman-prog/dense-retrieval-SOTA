#!/usr/bin/env python3
"""MS MARCO Dev ANCE: our ance_paper vs the published paper numbers (read-only).

    ssh delftblue python3 - < scripts/dev/extract_msmarco_ance.py > msmarco_ance.json
"""
import glob
import json
import os
import sys

pattern = '/scratch/%s/dense-retrieval-SOTA/results/bright_benchmark/ance_paper_roberta__*/msmarco_summary.json' % os.environ['USER']
runs = [json.load(open(p)) for p in glob.glob(pattern)]
runs = [r for r in runs if r['num_judged_queries'] == 6980]
assert len(runs) == 1, len(runs)
r = runs[0]

out = {'ours': r['metrics'], 'published': r['paper_reference'], 'num_judged_queries': r['num_judged_queries']}
json.dump(out, sys.stdout, indent=2)

sys.stderr.write('%-28s %8s %8s\n' % ('', 'MRR@10', 'R@1000'))
sys.stderr.write('%-28s %8.4f %8.4f\n' % ('published ANCE (600K)', r['paper_reference']['mrr_at_10'], r['paper_reference']['recall_1000']))
sys.stderr.write('%-28s %8.4f %8.4f\n' % ('ours ance_paper (300K)', r['metrics']['mrr_at_10'], r['metrics']['recall_1000']))
