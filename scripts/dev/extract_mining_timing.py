#!/usr/bin/env python3
"""Mean mining and training time: ANCE vs GRASS (mean of lambda 0 and 0.5). Read-only.

    ssh delftblue python3 - < scripts/dev/extract_mining_timing.py > mining_timing.json

JSON goes to stdout, the same table to stderr. Mining and training overlap, so they are not summed.
"""
import json
import os
import sys
from pathlib import Path

DATA = Path('/scratch') / os.environ['USER'] / 'dense-retrieval-SOTA'
ANCE_PHASES = ('encode_corpus_s', 'encode_query_s', 'index_s', 'search_s')
ANCE_WORK = DATA / 'temp_ance_workdir' / 'b9aec029b000-1788363020-j70367-6f85893f'


def mean(xs):
    return sum(xs) / len(xs)


def run(model, work, final, summary, grass):
    models = DATA / 'models' / model
    meta, prefix = (('mining_meta', 'mining_meta') if grass else ('round_meta', 'round_meta'))
    rounds = [json.load(open(p)) for p in sorted(work.glob(prefix + '_[0-9]*.json'))]
    mining = [r['t_mine_round'] if grass else sum(r[k] for k in ANCE_PHASES) for r in rounds]
    first = json.load(open(work / (meta + '_initial.json')))
    steps = list(range(1000, 10001, 1000))  # the partial last interval (to `final`) is left out
    t = [(models / ('checkpoint-%d' % s) / 'optimizer.pt').stat().st_mtime for s in steps]
    return {
        'startup_mining_s': first['t_cache_mc_init'] + first['t_initial_mine'] if grass else None,
        'mining_s_per_round': mean(mining),
        'train_s_per_1000_steps': mean([b - a for a, b in zip(t, t[1:])]),
        'rounds_consumed': json.load(open(models / summary))['rounds_consumed'],
        'training_s': (models / ('checkpoint-%d' % final)).stat().st_mtime - (work / 'ready_initial').stat().st_mtime,
    }


ance = run('ance_mixed_bge_m3', ANCE_WORK, 10312, 'ance_trainer_summary.json', False)
grass = [run('async_fast_grass_mixed_bge_m3_s6_' + s, DATA / 'temp_fast_grass_workdir' / ('async_mining_s6_' + s),
             10314, 'async_trainer_summary.json', True) for s in ('lam0', 'lam05')]
out = {'ance': ance, 'grass': {k: mean([g[k] for g in grass]) for k in ance if k != 'startup_mining_s'}}
out['grass']['startup_mining_s'] = mean([g['startup_mining_s'] for g in grass])
json.dump(out, sys.stdout, indent=2)

sys.stderr.write('%-24s %10s %10s\n' % ('', 'ANCE', 'GRASS'))
for k in ance:
    sys.stderr.write('%-24s %10s %10s\n' % (k, *('-' if v is None else '%.0f' % v for v in (ance[k], out['grass'][k]))))
