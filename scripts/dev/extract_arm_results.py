#!/usr/bin/env python3
"""Extract pre-training, training and R@1000 per arm from DelftBlue's own records.

Read-only, stdlib only. A value that cannot be read is null with a reason, and the exit
code is 1. Run on the cluster, JSON goes to stdout:

    ssh delftblue python3 - < scripts/dev/extract_arm_results.py > arm_results.json
"""
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LOGS = Path.home() / 'dense-retrieval-SOTA' / 'logs'
DATA = Path('/scratch') / os.environ['USER'] / 'dense-retrieval-SOTA'
ANCE_RUN = 'b9aec029b000-1788363020-j70367-6f85893f'

# arm -> evaluated model dir, training-log runs (file, total steps)
ARMS = {
    'in-batch': ('inbatch_mixed_bge_m3', [('inbatch_neg_18996.err', 10314)]),
    'cross-batch': ('crossbatch_mixed_bge_m3_step600',
                    [('crossbatch_bge_18995.err', 646), ('crossbatch_bge_244736.err', 646)]),
    'ance': ('ance_mixed_bge_m3', None),
    'grass lambda 0': ('async_fast_grass_mixed_bge_m3_s6_lam0', None),
    'grass lambda 0.5': ('async_fast_grass_mixed_bge_m3_s6_lam05', None),
    'grass lambda 0.3': ('async_fast_grass_mixed_bge_m3_s6_lam03', None),
}
GRASS_STEPS = 10314
ANCE_STEPS = 10312


def mtime(path):
    return Path(path).stat().st_mtime


def hms(text):
    parts = [int(x) for x in text.split(':')]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def last_bar(log, total):
    """Last tqdm bar 'N/total [elapsed<' in a log. Returns (steps, seconds)."""
    text = (LOGS / log).read_text(errors='ignore').replace('\r', '\n')
    bars = re.findall(r'(\d+)/%d \[([\d:]+)<' % total, text)
    if not bars:
        raise ValueError('no progress bar in ' + log)
    return int(bars[-1][0]), hms(bars[-1][1])


def sacct_start(job):
    out = subprocess.check_output(['sacct', '-j', str(job), '-X', '-n', '-P', '--format=Start'],
                                  universal_newlines=True).strip()
    return datetime.strptime(out, '%Y-%m-%dT%H:%M:%S').timestamp()


def recall_1000(model):
    """Macro R@1000 from the single complete 12-domain run_all_evals summary of `model`."""
    hits = []
    for path in glob.glob('%s/results/bright_benchmark/%s__*/*/summary.json' % (DATA, model)):
        d = json.load(open(path))
        if len(d['per_domain']) == 12 and d['model'].rstrip('/').endswith('/models/' + model):
            hits.append((path, d))
    if len(hits) != 1:
        raise ValueError('%d complete summaries for %s' % (len(hits), model))
    path, d = hits[0]
    mean = sum(x['recall_1000'] for x in d['per_domain']) / 12
    if abs(mean - d['macro_recall_1000']) > 1e-9:
        raise ValueError('macro_recall_1000 != mean of the 12 domains in ' + path)
    return d['macro_recall_1000'], path


def timings(arm, model, runs):
    """-> (pre_training_s, training_s, notes) for one arm."""
    if runs:  # Tevatron baselines: no mining phase, training from the progress bars
        done = [dict(zip(('log', 'steps', 'seconds'), (log,) + last_bar(log, total)))
                for log, total in runs]
        return 0, sum(r['seconds'] for r in done), {'pre': 'no mining phase', 'runs': done}
    if arm == 'ance':
        ready = DATA / 'temp_ance_workdir' / ANCE_RUN / 'ready_initial'
        end = DATA / 'models' / model / ('checkpoint-%d' % ANCE_STEPS)
        pre = mtime(ready) - sacct_start(70367)
        return pre, mtime(end) - mtime(ready), {
            'pre': 'sacct Start of job 70367 -> ready_initial; INCLUDES startup, round-0 time is not logged',
            'training': 'ready_initial mtime -> checkpoint-%d mtime' % ANCE_STEPS}
    suffix = model.rsplit('_s6_', 1)[1]
    mining = DATA / 'temp_fast_grass_workdir' / ('async_mining_s6_' + suffix)
    meta = json.load(open(mining / 'mining_meta_initial.json'))
    end = DATA / 'models' / model / ('checkpoint-%d' % GRASS_STEPS)
    return meta['t_cache_mc_init'] + meta['t_initial_mine'], mtime(end) - mtime(mining / 'ready_initial'), {
        'pre': 't_cache_mc_init + t_initial_mine from mining_meta_initial.json',
        'training': 'ready_initial mtime -> checkpoint-%d mtime' % GRASS_STEPS}


def main():
    rows, errors = [], []
    for arm, (model, runs) in ARMS.items():
        row = {'arm': arm, 'model': model, 'pre_training_s': None, 'training_s': None,
               'total_s': None, 'recall_1000': None, 'source': {}}
        try:
            row['pre_training_s'], row['training_s'], row['source'] = timings(arm, model, runs)
            row['total_s'] = row['pre_training_s'] + row['training_s']
        except Exception as e:
            errors.append('%s timing: %s' % (arm, e))
        try:
            row['recall_1000'], row['source']['recall_1000'] = recall_1000(model)
        except Exception as e:
            errors.append('%s recall: %s' % (arm, e))
        rows.append(row)
        print('%-17s pre=%s train=%s total=%s R@1000=%s' % (
            arm, row['pre_training_s'], row['training_s'], row['total_s'], row['recall_1000']),
            file=sys.stderr)
    json.dump({'arms': rows, 'errors': errors}, sys.stdout, indent=2)
    for e in errors:
        print('ERROR', e, file=sys.stderr)
    return 1 if errors else 0


sys.exit(main())
