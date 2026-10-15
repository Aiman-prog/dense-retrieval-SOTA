#!/usr/bin/env python3
"""Phase B: does representation uncertainty predict embedding drift beyond age and momentum?

Uncertainty-guided negative SELECTION has failed twice -- EMA disagreement at lambda=1 and
MC-dropout score variance at lambda=0.3/0.5 -- and neither pilot ever validated its estimator
against an independent target, which is why neither failure could be diagnosed afterwards.
This screen does that validation, for the estimator that
`docs/grass_uncertainty_refresh_architecture.md` proposes to allocate the REFRESH budget with.

No training. Three saved checkpoints supply a drift label and a momentum control:

    label      y_d = 1 - cos(e_4000(d), e_5000(d))          forward drift
    momentum   b_d = 1 - cos(e_3000(d), e_4000(d))          backward drift
    treatment  u_rep(d)   = mean_t [1 - cos(z_t, z_bar)]    representation dispersion
    comparator u_score(d) = spread of q.z_t over query probes
    control    v_d = mean_q |q.e_3000(d) - q.e_4000(d)|     score volatility

Every document carries the same age, so age is constant by construction and cannot explain
the correlation. The question is whether `u_rep` adds information ON TOP of `b_d` and `v_d`,
which is what the partial (incremental) Spearman measures.

Statistics are numpy-only on purpose: neither scipy nor sklearn is in the pinned container.

No calibrated uncertainty->drift mapping is fitted. Every Phase B decision -- the incremental
correlation and the captured-drift-at-budget proxy -- is rank-based, and a monotone mapping
cannot change a rank. Absolute predicted drift is a Phase C concern.

SPLIT DISCIPLINE. Calibration and validation are always read. The held-out test split is read
only under --open-test, ONCE, after T is chosen on validation. `screen.json` records which
splits were read so a later reader can tell.

Usage:
  python scripts/grass_uncertainty_screen.py                 # calibration + validation
  python scripts/grass_uncertainty_screen.py --open-test     # the single held-out read
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.helpers import (  # noqa: E402
    get_path, load_config, set_seed, atomic_write, _code_revision,
    encode_batch_tensor, encode_mc,
)
from utils.runtime_timing import PhaseTimer  # noqa: E402
from data.preprocessor import require_derived_artifacts  # noqa: E402

# The drift horizon is fixed by what is on disk: ance_mixed_bge_m3 checkpoints exist at
# 1000-step intervals, so the shortest measurable horizon is 1000 steps. Deployment refresh
# windows are ~100 steps, so the estimator is fitted on a coarser horizon than it is used at.
# That gap is exactly what the architecture document's Phase C transfer gate tests.
DEFAULT_CKPT_STEPS = (3000, 4000, 5000)
DRIFT_HORIZON_STEPS = 1000
SPLIT_SIZES = {'calibration': 10_000, 'validation': 5_000, 'test': 5_000}
T_GRID = (3, 8)
BUDGET_GRID = (256, 512, 1024, 2048)


# ── pure functions (the CPU test suite exercises everything below) ────────────

def sample_and_split_ids(all_ids, split_sizes, seed):
    """Deterministic disjoint splits of a uniform sample of `all_ids`.

    Sorting the chosen positions keeps the sample in corpus order, so two runs with the
    same seed encode documents in the same order and the MC dropout stream lines up.
    """
    total = sum(split_sizes.values())
    if total > len(all_ids):
        raise ValueError(f"need {total} documents, corpus has {len(all_ids)}")
    rng = np.random.default_rng(int(seed))
    chosen = np.sort(rng.choice(len(all_ids), size=total, replace=False))
    # Permute before cutting, otherwise the splits would be ordered corpus segments
    # rather than exchangeable samples.
    order = rng.permutation(total)
    out, start = {}, 0
    for name in ('calibration', 'validation', 'test'):
        n = split_sizes[name]
        out[name] = [all_ids[chosen[i]] for i in sorted(order[start:start + n])]
        start += n
    return out


def cosine_drift(a, b):
    """1 - cos(a_i, b_i) per row, as float64 numpy."""
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return (1.0 - cos).double().cpu().numpy()


def representation_dispersion(Z):
    """Mean cosine distance of each MC sample from the sample mean. Z is [T, n, D] -> [n].

    On the same `1 - cos` scale as the drift label, so estimator and target are comparable
    without rescaling.
    """
    Z = torch.as_tensor(Z, dtype=torch.float32)
    if Z.ndim != 3:
        raise ValueError(f"expected [T, n, D], got {tuple(Z.shape)}")
    centre = torch.nn.functional.normalize(Z.mean(dim=0), dim=-1)
    cos = torch.nn.functional.cosine_similarity(Z, centre.unsqueeze(0), dim=-1)
    return (1.0 - cos).mean(dim=0).double().cpu().numpy()


def score_dispersion(Z, Q):
    """Spread of the relevance score across MC samples. Z [T,n,D], Q [nq,D] -> [n].

    This is the estimator family the lambda pilots used, kept as a comparator so the screen
    can say whether representation uncertainty beats score uncertainty on the same samples.
    """
    Z = torch.as_tensor(Z, dtype=torch.float32)
    Q = torch.as_tensor(Q, dtype=torch.float32)
    scores = torch.einsum('tnd,qd->tnq', Z, Q)
    return scores.std(dim=0, unbiased=False).mean(dim=-1).double().cpu().numpy()


def score_volatility(e_a, e_b, Q):
    """Mean |q.e_a - q.e_b| over query probes -> [n]. Queries are held fixed on purpose:
    varying only the document isolates the document-side effect."""
    e_a = torch.as_tensor(e_a, dtype=torch.float32)
    e_b = torch.as_tensor(e_b, dtype=torch.float32)
    Q = torch.as_tensor(Q, dtype=torch.float32)
    return (e_a @ Q.T - e_b @ Q.T).abs().mean(dim=-1).double().cpu().numpy()


def rankdata(x):
    """Average ranks, ties shared. numpy-only stand-in for scipy.stats.rankdata."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # share ranks within each run of equal values
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _residualize(y, controls):
    """Residuals of y after least-squares removal of `controls` (plus an intercept)."""
    if controls is None or len(controls) == 0:
        return y - y.mean()
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=np.float64) for c in controls])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(u, y, controls=()):
    """Spearman correlation of u with y after removing `controls`, all rank-transformed.

    With no controls this is plain Spearman. With controls it is the INCREMENTAL information
    u carries about y beyond them, which is the quantity the screen is gated on.
    """
    ru, ry = rankdata(u), rankdata(y)
    rc = [rankdata(c) for c in controls]
    return _pearson(_residualize(ru, rc), _residualize(ry, rc))


def bootstrap_ci(stat_fn, n, n_boot, seed, alpha=0.05):
    """Percentile CI of `stat_fn(idx)` resampling document INDICES, so every quantity in a
    replicate comes from the same documents and paired differences stay paired."""
    rng = np.random.default_rng(int(seed))
    draws = np.array([stat_fn(rng.integers(0, n, size=n)) for _ in range(int(n_boot))])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {'lower': float(lo), 'upper': float(hi), 'n_boot': int(n_boot)}


def captured_drift(u, y, budget):
    """Total forward drift of the top-`budget` documents ranked by u.

    The screen's proxy for refresh value: a budget spent on the documents an estimator ranks
    highest should cover more real movement than the same budget spent at random.
    """
    budget = min(int(budget), len(u))
    top = np.argsort(-np.asarray(u, dtype=np.float64), kind='mergesort')[:budget]
    return float(np.asarray(y, dtype=np.float64)[top].sum())


# ── corpus access ─────────────────────────────────────────────────────────────

def read_corpus_ids(corpus_file):
    """Pass 1: docids only. Reading ids before texts keeps peak memory off the 16 GiB host
    ceiling on gpu-a100-small, where the full corpus lookup would not be needed anyway."""
    ids = []
    with open(corpus_file) as handle:
        for line in handle:
            if line.strip():
                ids.append(json.loads(line)['docid'])
    return ids


def read_texts_for(corpus_file, wanted):
    """Pass 2: texts for the sampled ids only."""
    want = set(wanted)
    found = {}
    with open(corpus_file) as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec['docid'] in want:
                found[rec['docid']] = rec['text']
    missing = [d for d in wanted if d not in found]
    if missing:
        raise KeyError(f"{len(missing)} sampled docids absent from corpus, e.g. {missing[:5]}")
    return [found[d] for d in wanted]


def read_query_probes(queries_file, n_probes, seed):
    """Query probes for the score-dispersion comparator and the volatility control.

    `train_queries.jsonl` carries `query_id`/`query`; the BRIGHT eval files carry
    `qid`/`text`. Accept either shape and raise on anything else rather than silently
    probing with `None`, which would make every score identical and the control inert.
    """
    qids, texts = [], []
    with open(queries_file) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec.get('query_id', rec.get('qid'))
            text = rec.get('query', rec.get('text'))
            if qid is None or text is None:
                raise KeyError(
                    f"{queries_file}:{line_no} has no recognised id/text pair "
                    f"(saw {sorted(rec)}); expected query_id/query or qid/text")
            qids.append(str(qid))
            texts.append(text)
    rng = np.random.default_rng([int(seed), 1])
    idx = np.sort(rng.choice(len(texts), size=min(n_probes, len(texts)), replace=False))
    return [qids[i] for i in idx], [texts[i] for i in idx]


# ── GPU work ──────────────────────────────────────────────────────────────────

def load_encoder(ckpt_dir, device):
    from transformers import AutoModel
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    return AutoModel.from_pretrained(str(ckpt_dir), torch_dtype=dtype).to(device).eval()


def encode_deterministic(model, tokenizer, texts, device, max_len, batch_size):
    """Deterministic encode, with eval mode ASSERTED rather than assumed.

    `encode_batch_tensor` only suppresses gradients; it does not touch module modes. A
    model left in training mode would apply dropout here and the "deterministic" drift
    label would itself be stochastic, which would quietly destroy the whole screen. Entry
    modes are saved and restored, matching `_pool_and_fresh_rerank`.
    """
    entry = {mod: mod.training for mod in model.modules()}
    try:
        model.eval()
        return encode_batch_tensor(model, tokenizer, texts, device, max_len, batch_size,
                                   requires_grad=False).float().cpu()
    finally:
        for mod, was_training in entry.items():
            mod.train(was_training)


def mc_scalars(model, tokenizer, texts, device, max_len, batch_size, chunk, Q, t_grid):
    """MC pass over the documents in chunks, keeping only per-document scalars.

    `encode_mc` returns [T, chunk, D]; the full [8, 20000, 1024] tensor is never materialised.
    The first t samples of the same draw serve smaller t, which makes the T comparison PAIRED
    rather than two independent draws.
    """
    t_max = max(t_grid)
    out = {f'u_rep_T{t}': [] for t in t_grid}
    out.update({f'u_score_T{t}': [] for t in t_grid})
    cost = {'mc_passes': int(t_max), 'examples_encoded': 0, 'forward_batches': 0}
    for i in range(0, len(texts), chunk):
        Z, stats = encode_mc(model, tokenizer, texts[i:i + chunk], t_max, device,
                             max_len, batch_size)
        Z = Z.float().cpu()
        # mc_passes is T and does not accumulate; chunking changes batch boundaries, so
        # forward_batches legitimately varies with `chunk` while examples_encoded does not.
        for key in ('examples_encoded', 'forward_batches'):
            cost[key] += int(stats[key])
        for t in t_grid:
            out[f'u_rep_T{t}'].append(representation_dispersion(Z[:t]))
            out[f'u_score_T{t}'].append(score_dispersion(Z[:t], Q))
        del Z
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return {k: np.concatenate(v) for k, v in out.items()}, cost


# ── analysis ──────────────────────────────────────────────────────────────────

def analyse_split(cols, idx, t_grid, budget_grid, n_boot, seed):
    """Every reported number for one split. Controls are momentum and score volatility."""
    y = cols['y'][idx]
    b = cols['b'][idx]
    v = cols['v'][idx]
    n = len(idx)
    report = {'n': int(n),
              'mean_forward_drift': float(y.mean()),
              'mean_backward_drift': float(b.mean()),
              'momentum_spearman': partial_spearman(b, y),
              'volatility_spearman': partial_spearman(v, y)}
    for t in t_grid:
        u = cols[f'u_rep_T{t}'][idx]
        us = cols[f'u_score_T{t}'][idx]
        entry = {
            'raw_spearman': partial_spearman(u, y),
            'incremental_spearman': partial_spearman(u, y, (b, v)),
            'score_raw_spearman': partial_spearman(us, y),
            'score_incremental_spearman': partial_spearman(us, y, (b, v)),
            'captured_drift': {},
        }
        entry['incremental_ci95'] = bootstrap_ci(
            lambda ix, u=u, y=y, b=b, v=v: partial_spearman(u[ix], y[ix], (b[ix], v[ix])),
            n, n_boot, seed)
        for budget in budget_grid:
            if budget <= n:
                entry['captured_drift'][str(budget)] = {
                    'uncertainty': captured_drift(u, y, budget),
                    'momentum': captured_drift(b, y, budget),
                    'random_expectation': float(y.mean() * budget),
                    'oracle': captured_drift(y, y, budget),
                }
        report[f'T{t}'] = entry
    if len(t_grid) == 2:
        lo_t, hi_t = min(t_grid), max(t_grid)
        u_lo, u_hi = cols[f'u_rep_T{lo_t}'][idx], cols[f'u_rep_T{hi_t}'][idx]
        report['paired_T_difference'] = {
            'comparison': f'T{hi_t} - T{lo_t}',
            'point': (partial_spearman(u_hi, y, (b, v))
                      - partial_spearman(u_lo, y, (b, v))),
            'ci95': bootstrap_ci(
                lambda ix: (partial_spearman(u_hi[ix], y[ix], (b[ix], v[ix]))
                            - partial_spearman(u_lo[ix], y[ix], (b[ix], v[ix]))),
                n, n_boot, seed + 1),
        }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_dir', default=None,
                    help='ANCE run holding the checkpoints (default: models/ance_mixed_bge_m3)')
    ap.add_argument('--ckpt_steps', type=int, nargs=3, default=list(DEFAULT_CKPT_STEPS),
                    metavar=('EARLY', 'MID', 'LATE'))
    ap.add_argument('--n_probes', type=int, default=128, help='query probes for u_score / v')
    ap.add_argument('--batch_size', type=int, default=32,
                    help='encode batch (a memory knob, not a scientific parameter). '
                         'gpu-a100-small is a ~9.5 GiB MIG SLICE, not a whole A100: XLM-R '
                         'is stuck on eager attention (transformers 4.40.2 reports '
                         '_supports_sdpa False), so the [B, 16, 512, 512] softmax scales '
                         'linearly with B and 128 needs ~2.95 GiB for that tensor alone.')
    ap.add_argument('--mc_chunk', type=int, default=2048,
                    help='documents per MC chunk; bounds the [T, chunk, D] tensor')
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--open-test', dest='open_test', action='store_true',
                    help='read the held-out split. ONCE, after T is chosen on validation.')
    ap.add_argument('--tag', default=None, help='output subdirectory name')
    args = ap.parse_args()

    config = load_config()
    seed = int(config.get('seed', 42))
    set_seed(seed)
    p_max_len = int(config['model']['passage_max_len'])
    q_max_len = int(config['model']['query_max_len'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[screen] GPU: {torch.cuda.get_device_name(0)}, {total:.2f} GiB total, "
              f"batch_size={args.batch_size}", flush=True)
        if total < 16 and args.batch_size > 48:
            print(f"[screen] WARNING: batch_size {args.batch_size} on a {total:.1f} GiB "
                  "device will likely OOM in eager attention at p512.", flush=True)
    else:
        print("[screen] WARNING: no CUDA — this will be very slow.", flush=True)

    model_dir = Path(args.model_dir) if args.model_dir else get_path('models') / 'ance_mixed_bge_m3'
    early, mid, late = args.ckpt_steps
    ckpts = {step: model_dir / f'checkpoint-{step}' for step in (early, mid, late)}
    missing = [str(p) for p in ckpts.values() if not p.is_dir()]
    if missing:
        print("[screen] ERROR: checkpoints absent:\n  " + "\n  ".join(missing), flush=True)
        return 2

    corpus_file, queries_file, _qrels = require_derived_artifacts()

    tag = args.tag or f'screen_{early}_{mid}_{late}'
    out_dir = project_root / 'analysis' / 'grass_estimator_screen' / tag
    timer = PhaseTimer()

    timer('sample')
    all_ids = read_corpus_ids(corpus_file)
    splits = sample_and_split_ids(all_ids, SPLIT_SIZES, seed)
    docids = [d for name in ('calibration', 'validation', 'test') for d in splits[name]]
    split_of = {d: name for name, ids in splits.items() for d in ids}
    texts = read_texts_for(corpus_file, docids)
    qids, q_texts = read_query_probes(queries_file, args.n_probes, seed)
    print(f"[screen] {len(all_ids):,} corpus docs -> {len(docids):,} sampled "
          f"({ {k: len(v) for k, v in splits.items()} }), {len(qids)} query probes", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ckpts[mid]))

    embs, cost = {}, {'mc_passes': 0, 'examples_encoded': 0, 'forward_batches': 0}  # replaced by mc_scalars
    mc = None
    for step in (early, mid, late):
        timer(f'encode_{step}')
        model = load_encoder(ckpts[step], device)
        embs[step] = encode_deterministic(model, tokenizer, texts, device,
                                          p_max_len, args.batch_size)
        if step == mid:
            Q = encode_deterministic(model, tokenizer, q_texts, device,
                                     q_max_len, args.batch_size)
            timer(f'mc_{step}')
            mc, cost = mc_scalars(model, tokenizer, texts, device, p_max_len,
                                  args.batch_size, args.mc_chunk, Q, T_GRID)
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    timer.close()

    cols = {'y': cosine_drift(embs[mid], embs[late]),
            'b': cosine_drift(embs[early], embs[mid]),
            'v': score_volatility(embs[early], embs[mid], Q)}
    cols.update(mc)

    read = ['calibration', 'validation'] + (['test'] if args.open_test else [])
    pos = {d: i for i, d in enumerate(docids)}
    results = {}
    for name in read:
        idx = np.array([pos[d] for d in splits[name]])
        results[name] = analyse_split(cols, idx, T_GRID, BUDGET_GRID, args.n_boot, seed)

    report = {
        'kind': 'grass_uncertainty_estimator_screen',
        'code_revision': _code_revision(),
        'seed': seed,
        'model_dir': str(model_dir),
        'checkpoint_steps': {'early': early, 'mid': mid, 'late': late},
        'drift_horizon_steps': DRIFT_HORIZON_STEPS,
        'horizon_caveat': (
            'Estimator fitted on a 1000-step horizon because ance_mixed_bge_m3 checkpoints '
            'exist only at 1000-step intervals; deployment refresh windows are ~100 steps. '
            'Transfer across that gap is the architecture document Phase C gate, not this one.'),
        'split_sizes': SPLIT_SIZES,
        'splits_read': read,
        'test_split_opened': bool(args.open_test),
        'n_query_probes': len(qids),
        'passage_max_len': p_max_len,
        'query_max_len': q_max_len,
        'T_grid': list(T_GRID),
        'budget_grid': list(BUDGET_GRID),
        'mc_cost': cost,
        'timings_s': timer.timings,
        'results': results,
        'gate': (
            'Promotion requires incremental_spearman with a 95% bootstrap lower bound above '
            'zero on the held-out test split, after T is chosen on validation.'),
    }

    with atomic_write(out_dir / 'screen.json') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    with atomic_write(out_dir / 'per_doc.jsonl') as handle:
        for i, docid in enumerate(docids):
            row = {'docid': docid, 'split': split_of[docid]}
            row.update({k: float(v[i]) for k, v in cols.items()})
            handle.write(json.dumps(row) + '\n')

    print(f"\n[screen] wrote {out_dir}/screen.json", flush=True)
    for name in read:
        r = results[name]
        print(f"  {name:12s} n={r['n']:,}  momentum_rho={r['momentum_spearman']:+.4f}")
        for t in T_GRID:
            e = r[f'T{t}']
            ci = e['incremental_ci95']
            print(f"    T={t}  raw={e['raw_spearman']:+.4f}  "
                  f"incremental={e['incremental_spearman']:+.4f} "
                  f"[{ci['lower']:+.4f}, {ci['upper']:+.4f}]  "
                  f"(score-based incremental={e['score_incremental_spearman']:+.4f})")
    if not args.open_test:
        print("\n  test split NOT opened — rerun with --open-test once T is chosen.", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
