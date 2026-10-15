#!/usr/bin/env python3
"""Read completed BRIGHT summaries and print six paired Recall@1000 comparisons.

Run read-only on DelftBlue:
    ssh delftblue python3 - < scripts/dev/extract_paired_bright_recall.py
"""

import glob
import json
import os
import random
import sys
from pathlib import Path


ROOT = Path('/scratch') / os.environ['USER'] / 'dense-retrieval-SOTA/results/bright_benchmark'
PATTERNS = {
    'in_batch': 'inbatch_mixed_bge_m3__*/*/summary.json',
    'cross_batch': 'crossbatch_mixed_bge_m3_step600__*/*/summary.json',
    'ance': 'ance_mixed_bge_m3__*/*/summary.json',
    'grass_0': 'async_fast_grass_mixed_bge_m3_s6_lam0__*/*/summary.json',
    'grass_0_3': 'async_fast_grass_mixed_bge_m3_s6_lam03__*/*/summary.json',
    'grass_0_5': 'async_fast_grass_mixed_bge_m3_s6_lam05__*/*/summary.json',
}
PAIRS = [
    ('cross_batch', 'in_batch'),
    ('ance', 'in_batch'),
    ('grass_0', 'in_batch'),
    ('ance', 'grass_0'),
    ('grass_0_3', 'grass_0'),
    ('grass_0_5', 'grass_0'),
]
RESAMPLES = 100_000
SEED = 42


def load_scores(name, pattern):
    complete = []
    for path in glob.glob(str(ROOT / pattern)):
        summary = json.loads(Path(path).read_text())
        rows = summary.get('per_domain', [])
        if len(rows) != 12:
            continue
        scores = {row['domain']: row['recall_1000'] for row in rows}
        if len(scores) != 12:
            raise ValueError('%s has duplicate domains: %s' % (name, path))
        if abs(sum(scores.values()) / 12 - summary['macro_recall_1000']) > 1e-9:
            raise ValueError('%s macro score differs from its domains: %s' % (name, path))
        complete.append(scores)
    if len(complete) != 1:
        raise ValueError('%s: expected one complete summary, found %d' % (name, len(complete)))
    return complete[0]


def percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def main():
    scores = {name: load_scores(name, pattern) for name, pattern in PATTERNS.items()}
    domains = list(scores['in_batch'])
    if any(set(arm) != set(domains) for arm in scores.values()):
        raise ValueError('BRIGHT domain sets differ between methods')

    differences = {(first, second): [scores[first][d] - scores[second][d] for d in domains]
                   for first, second in PAIRS}
    bootstrap = {pair: [] for pair in PAIRS}
    rng = random.Random(SEED)
    for _ in range(RESAMPLES):
        sample = [rng.randrange(12) for _ in domains]
        for pair in PAIRS:
            values = differences[pair]
            bootstrap[pair].append(sum(values[i] for i in sample) / 12)

    results = []
    for first, second in PAIRS:
        values = differences[first, second]
        distribution = sorted(bootstrap[first, second])
        results.append({
            'first': first,
            'second': second,
            'mean_difference': sum(values) / 12,
            'ci95': [percentile(distribution, 0.025), percentile(distribution, 0.975)],
            'wins': sum(value > 0 for value in values),
            'losses': sum(value < 0 for value in values),
            'ties': sum(value == 0 for value in values),
        })
    json.dump({'metric': 'recall_1000', 'bootstrap': {'unit': 'domain',
               'resamples': RESAMPLES, 'seed': SEED, 'interval': 'percentile'},
               'comparisons': results}, sys.stdout, indent=2)
    print()


if __name__ == '__main__':
    main()
