"""
Async Fast-GRASS — cached-MCDP SIGNAL probe (Phase 0, REPORT-ONLY).

Asks one narrow question before the async miner is built:

    is the cached uncertainty term numerically ALIVE, or is it inert?

i.e. does ``g = s_hat + lambda*sigma`` with ``lambda>0`` select different negatives
than ``lambda=0`` (pure mean score), on identical cached MC states? Our EMA runs
already showed ``lambda=1`` behaving like ``lambda=0``, so this is not hypothetical.
If sigma is zero, badly scaled, or its effect is seed-unstable, the cached-MCDP
mechanism is falsified early and cheaply.

**This is NOT a quality gate and never blocks the build.** A frozen base-model probe
cannot establish whether cached-MCDP improves Recall/NDCG after training, and BRIGHT
qrels do not label the relative usefulness of mined negatives. Treat it as a
non-degeneracy diagnostic only. Downstream retrieval quality belongs to the Phase-3
ablation runs on trained checkpoints. **Do not quote any Recall/NDCG claim from this
script.**

It deliberately does NOT invoke the sequential lazy top-``L`` miner
(``run_fast_grass._mine_batch_mcdp``). That path confounds three different effects in
a single disagreement number — top-``L`` truncation, fresh document resampling, and
caching — and at a base checkpoint freshly cached document states are not
meaningfully model-stale, so it would mostly measure finite-``T`` sampling variation.
A fresh-MCDP oracle, if wanted later, belongs in Phase 3 against a TRAINED checkpoint
and an AGED cache, reporting rank correlation / fresh-score regret rather than
selected-document overlap.

Metrics, over ``--seeds`` independent MC draws on the same fixed cache:
  flip_rate        fraction of queries whose top-1 negative changes when lambda>0
  overlap_at_1     1 - flip_rate
  overlap_at_m     mean Jaccard of the two m-negative sets
  sigma stats      mean / std / p50 / p90 of sigma over all (query, doc) pairs
  lambda_sigma_vs_margin
                   mean of (lambda*sigma_top1) / (s_hat margin between the top-2
                   documents by mean score). << 1 means sigma cannot flip anything
                   and the uncertainty term is inert regardless of its raw scale.
  seed stability   std of flip_rate across seeds: a real signal is stable, sampling
                   noise is not.

Modes:
  --synthetic (default): CPU mock model, no downloads.
  --real                : GPU; needs the stale index pickle + processed mixture.

Usage:
  python scripts/async_fast_grass_quality_probe.py --synthetic
  python scripts/async_fast_grass_quality_probe.py --synthetic --lambda_val 1.0 --seeds 8
  python scripts/async_fast_grass_quality_probe.py --real --max_queries 256 \
      --B_doc 32000 --T 3 --lambda_val 0.5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.negative_cache import NegativeCache  # noqa: E402
from async_fast_grass_cached_mcdp import (  # noqa: E402
    init_Z_mc, encode_queries_mc, score_cached_mcdp,
)

OUT_DIR = project_root / 'analysis' / 'async_fast_grass_timing'


# ---- the probe -------------------------------------------------------------

def _select(g, m):
    return torch.topk(g, k=m, dim=1).indices


def probe_once(cache, Z_mc, student, tokenizer, qids, qid_to_text, qrels_dict,
               T, lam, m, cfg, device, seed):
    """One MC draw: score ONCE, then combine into lambda=0 and lambda>0 rankings.

    Both arms share the SAME s_hat and sigma, so any difference is attributable to
    the uncertainty term alone — not to a different dropout draw.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    texts = [qid_to_text[q] for q in qids]
    q_mc, _ = encode_queries_mc(student, tokenizer, texts, T, device, cfg)

    _g, s_hat, sigma = score_cached_mcdp(q_mc, Z_mc, 0.0)
    g0 = cache.mask_positives(s_hat.clone(), qids, qrels_dict, inplace=True)
    gl = cache.mask_positives(s_hat + lam * sigma, qids, qrels_dict, inplace=True)

    sel0, sell = _select(g0, m), _select(gl, m)

    flips = (sel0[:, 0] != sell[:, 0])
    flip_rate = float(flips.float().mean())

    jac = []
    for i in range(len(qids)):
        a, b = set(sel0[i].tolist()), set(sell[i].tolist())
        jac.append(len(a & b) / len(a | b))

    # how big is the uncertainty nudge relative to the gap it must close?
    top2 = torch.topk(g0, k=min(2, g0.shape[1]), dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).clamp(min=1e-9) if top2.shape[1] > 1 else None
    sigma_top1 = torch.gather(sigma, 1, sel0[:, :1]).squeeze(1)
    ratio = float((lam * sigma_top1 / margin).mean()) if margin is not None else None

    finite_sigma = sigma[torch.isfinite(sigma)]
    leaked = sum(1 for i, q in enumerate(qids)
                 for s in sell[i].tolist()
                 if cache.docids[s] in qrels_dict.get(q, set()))
    return {
        'seed': seed,
        'flip_rate': flip_rate,
        'overlap_at_1': 1.0 - flip_rate,
        'overlap_at_m': float(np.mean(jac)) if jac else 0.0,
        'sigma_mean': float(finite_sigma.mean()),
        'sigma_std': float(finite_sigma.std()),
        'sigma_p50': float(finite_sigma.median()),
        'sigma_p90': float(torch.quantile(finite_sigma.float(), 0.9)),
        'sigma_max': float(finite_sigma.max()),
        's_hat_mean': float(s_hat.mean()),
        'lambda_sigma_vs_margin': ratio,
        'positives_leaked': int(leaked),
        'sigma_all_zero': bool(float(finite_sigma.abs().max()) == 0.0),
        'sigma_nonfinite': bool(not torch.isfinite(sigma).all()),
    }


def aggregate(runs, lam, m, n_queries, T, B_doc):
    fr = [r['flip_rate'] for r in runs]
    return {
        'kind': 'cached_mcdp_signal_probe',
        'report_only': True,
        'not_a_quality_gate': (
            "Non-degeneracy diagnostic on a frozen base model. Does NOT support any "
            "Recall/NDCG claim; downstream quality belongs to Phase-3 ablations."),
        'lambda_val': lam, 'm': m, 'T': T, 'B_doc': B_doc,
        'num_queries': n_queries, 'num_seeds': len(runs),
        'flip_rate_mean': float(np.mean(fr)),
        'flip_rate_std': float(np.std(fr)),
        'flip_rate_min': float(np.min(fr)),
        'flip_rate_max': float(np.max(fr)),
        'overlap_at_1_mean': float(np.mean([r['overlap_at_1'] for r in runs])),
        'overlap_at_m_mean': float(np.mean([r['overlap_at_m'] for r in runs])),
        'sigma_mean': float(np.mean([r['sigma_mean'] for r in runs])),
        'sigma_p90_mean': float(np.mean([r['sigma_p90'] for r in runs])),
        'sigma_max': float(np.max([r['sigma_max'] for r in runs])),
        'lambda_sigma_vs_margin_mean': (
            float(np.mean([r['lambda_sigma_vs_margin'] for r in runs]))
            if runs[0]['lambda_sigma_vs_margin'] is not None else None),
        'positives_leaked_total': int(sum(r['positives_leaked'] for r in runs)),
        'sigma_all_zero_any': any(r['sigma_all_zero'] for r in runs),
        'sigma_nonfinite_any': any(r['sigma_nonfinite'] for r in runs),
        'per_seed': runs,
    }


def _verdict(a):
    """Interpretation only — never an exit code."""
    notes = []
    if a['sigma_all_zero_any']:
        notes.append("sigma is identically ZERO: the cached uncertainty term is dead "
                     "(check that dropout is actually active during MC passes).")
    elif a['flip_rate_mean'] == 0.0:
        notes.append("lambda>0 NEVER changed a selection: sigma is inert at this "
                     "lambda — exactly the EMA lambda=1 ~ lambda=0 failure mode.")
    elif a['flip_rate_mean'] < 0.01:
        notes.append(f"lambda>0 changed only {a['flip_rate_mean']:.1%} of selections: "
                     f"sigma is nearly inert; a larger lambda may be needed for the "
                     f"term to matter at all.")
    else:
        notes.append(f"sigma is ACTIVE: it changes {a['flip_rate_mean']:.1%} of top-1 "
                     f"selections. (Active != better — that is a Phase-3 question.)")
    if a['flip_rate_std'] > 0.5 * max(a['flip_rate_mean'], 1e-9):
        notes.append(f"flip rate is seed-UNSTABLE (mean {a['flip_rate_mean']:.3f} +- "
                     f"{a['flip_rate_std']:.3f}): consistent with sampling noise "
                     f"rather than a stable signal. Consider larger T.")
    r = a['lambda_sigma_vs_margin_mean']
    if r is not None and r < 0.1:
        notes.append(f"lambda*sigma is only {r:.3f}x the top-1/top-2 score margin: "
                     f"the uncertainty term is too small to reorder the ranking.")
    return notes


def _print(a):
    print("\n" + "=" * 68)
    print("  CACHED-MCDP SIGNAL PROBE — lambda=0 vs lambda>0  (REPORT ONLY)")
    print("=" * 68)
    print(f"  queries {a['num_queries']} | m {a['m']} | T {a['T']} | "
          f"B_doc {a['B_doc']:,} | lambda {a['lambda_val']} | seeds {a['num_seeds']}")
    print("-" * 68)
    print(f"  flip rate (top-1 changed) : {a['flip_rate_mean']:.4f} "
          f"± {a['flip_rate_std']:.4f}  [{a['flip_rate_min']:.3f}, {a['flip_rate_max']:.3f}]")
    print(f"  overlap@1 / overlap@m     : {a['overlap_at_1_mean']:.4f} / "
          f"{a['overlap_at_m_mean']:.4f}")
    print(f"  sigma  mean / p90 / max   : {a['sigma_mean']:.6f} / "
          f"{a['sigma_p90_mean']:.6f} / {a['sigma_max']:.6f}")
    if a['lambda_sigma_vs_margin_mean'] is not None:
        print(f"  lambda*sigma / margin     : {a['lambda_sigma_vs_margin_mean']:.4f} "
              f"(legend: a value << 1 means sigma cannot reorder)")
    print(f"  positives leaked          : {a['positives_leaked_total']} (must be 0)")
    print("-" * 68)
    for n in _verdict(a):
        print(f"  • {n}")
    print("-" * 68)
    print("  REPORT ONLY — not a build/no-build gate, and no Recall/NDCG claim")
    print("  follows from a frozen base model. See Phase-3 ablations.")
    print("=" * 68)


def _hard_failures(a):
    """Only genuine correctness breakage is an error — never a signal verdict."""
    bad = []
    if a['positives_leaked_total']:
        bad.append(f"{a['positives_leaked_total']} known positives selected as negatives")
    if a['sigma_nonfinite_any']:
        bad.append("sigma contained non-finite values")
    return bad


# ---- synthetic CPU mode ----------------------------------------------------

def run_synthetic(args):
    from fast_grass_test import make_cfg, DropoutMockModel, MockTokenizer

    device = torch.device('cpu')
    dim, n_corpus, T = 16, 60, args.T or 3
    c_ids = [f"d{i}" for i in range(n_corpus)]
    corpus_lookup = {d: f"document {d} body text" for d in c_ids}
    embs = np.random.default_rng(0).standard_normal((n_corpus, dim)).astype('float32')

    n_q = args.max_queries or 24
    qid_to_text = {f"q{i}": f"query number {i}" for i in range(n_q)}
    qids = list(qid_to_text)
    qrels_dict = {q: {c_ids[i % n_corpus]} for i, q in enumerate(qids)}

    cfg = make_cfg(uncertainty='cached_mcdp', B_doc=args.B_doc or 30, T=T,
                   m=args.m or 2, lambda_val=args.lambda_val, mc_dropout_p=0.3,
                   selection_mode='topk', query_max_len=8, passage_max_len=8,
                   mc_batch_size=32)
    cache = NegativeCache.init_uniform(embs, c_ids, cfg, device, dim=dim)
    student = DropoutMockModel(hidden=dim, p=cfg['mc_dropout_p'])
    tok = MockTokenizer()
    Z_mc, _ = init_Z_mc(cache, corpus_lookup, student, tok, T, cfg, device)

    runs = [probe_once(cache, Z_mc, student, tok, qids, qid_to_text, qrels_dict,
                       T, args.lambda_val, cfg['m'], cfg, device, seed=100 + s)
            for s in range(args.seeds)]
    a = aggregate(runs, args.lambda_val, cfg['m'], len(qids), T, cache.B_doc)
    a['mode'] = 'synthetic'
    _print(a)

    bad = _hard_failures(a)
    if bad:
        for b in bad:
            print(f"  FAIL: {b}")
    print(f"  {'PASS' if not bad else 'FAIL'}  signal-probe harness runs end to end")
    print("=" * 68)
    return 0 if not bad else 1


# ---- real GPU mode ---------------------------------------------------------

def run_real(args):
    import gc
    import pickle
    from types import SimpleNamespace
    from transformers import AutoTokenizer, AutoModel
    from utils.helpers import (get_training_context, load_config, get_path,
                               _load_corpus_lookup, _load_qrels, set_seed)
    from data.preprocessor import run_setup
    import run_fast_grass

    config = load_config()
    ctx = get_training_context('fast_grass')
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("[signal-probe] WARNING: no CUDA — use --synthetic for the CPU smoke.",
              flush=True)

    corpus_file, _q, qrels_file = run_setup()
    stale_pkl = get_path("temp_grass") / "stale_index" / "corpus.pkl"
    if not stale_pkl.exists():
        print(f"[signal-probe] ERROR: stale index not found at {stale_pkl}. Build it "
              "once via run_fast_grass.py, then re-run.", flush=True)
        return 2
    with open(stale_pkl, 'rb') as f:
        cd = pickle.load(f)
    stale_embs, c_ids = cd[0], [str(x) for x in cd[1]]
    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict = _load_qrels(qrels_file)

    train_items = run_fast_grass._load_train_items()
    qid_to_text = {it['query_id']: it['query'] for it in train_items}
    qids = list(qid_to_text)[:args.max_queries]

    batch_size = config['training']['fast_grass'].get('batch_size', 64)
    steps_per_epoch = max(len(qid_to_text) // batch_size, 1)
    ns = SimpleNamespace(B_doc=args.B_doc, uncertainty='cached_mcdp', T=args.T,
                         L=None, m=args.m, lambda_val=args.lambda_val,
                         ema_alpha=None, mc_dropout_p=None, selection_mode=None,
                         num_epochs=None, no_registry=False)
    cfg = run_fast_grass._build_fast_grass_cfg(config, ns, steps_per_epoch)
    cfg.pop('L', None)
    T = int(cfg.get('T', 3))

    base = ctx['base_model']
    tok = AutoTokenizer.from_pretrained(base)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    student = AutoModel.from_pretrained(base, torch_dtype=dtype).to(device).eval()
    for p in student.parameters():
        p.requires_grad_(False)
    for mod in student.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = cfg.get('mc_dropout_p', 0.3)

    cache = NegativeCache.init_uniform(stale_embs, c_ids, cfg, device)
    del stale_embs
    gc.collect()
    Z_mc, _ = init_Z_mc(cache, corpus_lookup, student, tok, T, cfg, device)

    runs = [probe_once(cache, Z_mc, student, tok, qids, qid_to_text, qrels_dict,
                       T, cfg['lambda_val'], cfg['m'], cfg, device, seed=100 + s)
            for s in range(args.seeds)]
    a = aggregate(runs, cfg['lambda_val'], cfg['m'], len(qids), T, cache.B_doc)
    a['mode'] = 'real'
    _print(a)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    p = OUT_DIR / f"signal_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(a, indent=2))
    print(f"[signal-probe] wrote {p}", flush=True)

    bad = _hard_failures(a)
    for b in bad:
        print(f"  FAIL: {b}")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--synthetic', action='store_true', default=True,
                    help='CPU mock smoke (default)')
    ap.add_argument('--real', dest='synthetic', action='store_false',
                    help='real GPU probe (needs stale index + processed mixture)')
    ap.add_argument('--lambda_val', type=float, default=0.5,
                    help='nonzero lambda to compare against lambda=0 (async default 0.5)')
    ap.add_argument('--seeds', type=int, default=5,
                    help='independent MC draws; separates a stable signal from noise')
    ap.add_argument('--max_queries', type=int, default=256, help='query sample size')
    ap.add_argument('--B_doc', type=int, default=None)
    ap.add_argument('--T', type=int, default=None)
    ap.add_argument('--m', type=int, default=None)
    args = ap.parse_args()
    return run_synthetic(args) if args.synthetic else run_real(args)


if __name__ == "__main__":
    sys.exit(main())
