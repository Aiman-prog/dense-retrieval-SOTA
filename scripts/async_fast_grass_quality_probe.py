"""
Async Fast-GRASS — mined-negative quality probe (spec item 6).

Asks, before committing to the async build: does mining from a FROZEN checkpoint
(what the async miner does) select negatives of similar hardness and diversity to
mining from the CURRENT model (what sequential Fast-GRASS does)? Large divergence
would mean async staleness hurts negative quality regardless of wall-clock wins.

Both paths use the SAME Fast-GRASS miner (``_mine_batch_mcdp`` from
run_fast_grass.py) against caches initialized from an identical H. The only
difference is the weights: "sequential" uses the current model; "async" uses a
frozen/stale checkpoint. NO full-corpus ANN rebuild, NO per-query stale FAISS.

Metrics (sequential vs async, over the same fixed query sample):
  s_hat_mean, sigma_mean, g_mean (selected), selected-doc diversity,
  overlap@1 (top-1 negative agreement), overlap@m (Jaccard of negative sets),
  positives-masked check.

Modes:
  --synthetic (default): CPU mock models, deterministic. With zero staleness the
    two paths must agree exactly (overlap@1 == overlap@m == 1.0); with staleness
    noise they diverge — a correctness smoke for the probe itself.
  real: GPU. Sequential = --seq_checkpoint (or base model). Async = frozen
    --async_checkpoint, or a copy of sequential perturbed by --staleness_noise.

Usage:
  python scripts/async_fast_grass_quality_probe.py --synthetic
  python scripts/async_fast_grass_quality_probe.py --synthetic --staleness_noise 0.3
  python scripts/async_fast_grass_quality_probe.py --real --max_queries 256 \
      --seq_checkpoint models/ckpt-current --async_checkpoint models/ckpt-stale
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.negative_cache import NegativeCache  # noqa: E402
import run_fast_grass  # noqa: E402


# ---- shared metric computation ---------------------------------------------

def _mine_sample(cache, model, tokenizer, qids, qid_to_text, corpus_lookup,
                 qrels_dict, cfg, device, seed):
    """Mine the sample with _mine_batch_mcdp under a fixed RNG so that, for
    identical weights, dropout draws align and selection is reproducible."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mined, slots, q_det, _teacher, stats = run_fast_grass._mine_batch_mcdp(
        cache, model, tokenizer, qids, qid_to_text, corpus_lookup, qrels_dict,
        cfg, device)
    return mined, stats


def _quality_metrics(mined_seq, mined_async, stats_seq, stats_async, qids,
                     qrels_dict, m):
    def flat(md):
        return [d for q in qids for d in md.get(q, [])]

    seq_flat, async_flat = flat(mined_seq), flat(mined_async)
    div_seq = len(set(seq_flat)) / len(seq_flat) if seq_flat else 0.0
    div_async = len(set(async_flat)) / len(async_flat) if async_flat else 0.0

    # overlap@1: top-1 negative agreement per query
    top1 = sum(1 for q in qids
               if mined_seq.get(q) and mined_async.get(q)
               and mined_seq[q][0] == mined_async[q][0])
    overlap_at_1 = top1 / len(qids) if qids else 0.0

    # overlap@m: mean Jaccard of the two negative sets per query
    jac = []
    for q in qids:
        a, b = set(mined_seq.get(q, [])), set(mined_async.get(q, []))
        if not a and not b:
            continue
        jac.append(len(a & b) / len(a | b))
    overlap_at_m = float(np.mean(jac)) if jac else 0.0

    # positives masked (must be zero in both)
    def leaks(md):
        return sum(1 for q in qids for d in md.get(q, [])
                   if d in qrels_dict.get(q, set()))

    return {
        'seq': {
            's_hat_mean': stats_seq.get('sel_s_hat_mean'),
            'sigma_mean': stats_seq.get('sel_sigma_mean'),
            'g_mean': (stats_seq.get('sel_s_hat_mean', 0.0)
                       + stats_seq.get('sel_lambda_sigma_mean', 0.0)),
            'diversity': div_seq,
        },
        'async': {
            's_hat_mean': stats_async.get('sel_s_hat_mean'),
            'sigma_mean': stats_async.get('sel_sigma_mean'),
            'g_mean': (stats_async.get('sel_s_hat_mean', 0.0)
                       + stats_async.get('sel_lambda_sigma_mean', 0.0)),
            'diversity': div_async,
        },
        'overlap_at_1': overlap_at_1,
        'overlap_at_m': overlap_at_m,
        'positives_leaked_seq': leaks(mined_seq),
        'positives_leaked_async': leaks(mined_async),
        'm': m,
        'num_queries': len(qids),
    }


def _print(metrics, title):
    s, a = metrics['seq'], metrics['async']
    print("\n" + "=" * 64)
    print(f"  ASYNC FAST-GRASS — NEGATIVE QUALITY PROBE ({title})")
    print("=" * 64)
    print(f"  queries: {metrics['num_queries']} | m: {metrics['m']}")
    print(f"  {'metric':<16}{'sequential':>14}{'async(frozen)':>16}")
    for k in ('s_hat_mean', 'sigma_mean', 'g_mean', 'diversity'):
        sv = s[k] if s[k] is not None else float('nan')
        av = a[k] if a[k] is not None else float('nan')
        print(f"  {k:<16}{sv:>14.4f}{av:>16.4f}")
    print("-" * 64)
    print(f"  overlap@1 (top-1 agree) : {metrics['overlap_at_1']:.3f}")
    print(f"  overlap@m (mean Jaccard): {metrics['overlap_at_m']:.3f}")
    print(f"  positives leaked        : seq={metrics['positives_leaked_seq']} "
          f"async={metrics['positives_leaked_async']}")
    print("=" * 64)


# ---- synthetic CPU smoke ---------------------------------------------------

def run_synthetic(args):
    from fast_grass_train_smoke import DropoutMockModel, MockTokenizer

    device = torch.device('cpu')
    dim = 8
    n_corpus = 40
    c_ids = [f"d{i}" for i in range(n_corpus)]
    corpus_lookup = {d: f"document {d} body text" for d in c_ids}
    embs = np.random.default_rng(0).standard_normal((n_corpus, dim)).astype('float32')

    n_q = 10
    qid_to_text = {f"q{i}": f"query number {i}" for i in range(n_q)}
    qids = list(qid_to_text)
    qrels_dict = {q: {c_ids[i % n_corpus]} for i, q in enumerate(qids)}

    from fast_grass_test import make_cfg  # complete registry/utility cfg surface
    cfg = make_cfg(uncertainty='mcdp', B_doc=20, m=2, selection_mode='topk',
                   lambda_val=1.0, L=6, T=3, mc_dropout_p=0.3,
                   query_max_len=128, passage_max_len=128, mc_batch_size=16)

    # identical H for both paths (same embs + seed)
    cache_seq = NegativeCache.init_uniform(embs, c_ids, cfg, device, dim=dim)
    cache_async = NegativeCache.init_uniform(embs, c_ids, cfg, device, dim=dim)

    seq_model = DropoutMockModel(hidden=dim, p=cfg['mc_dropout_p']).eval()
    async_model = copy.deepcopy(seq_model).eval()  # frozen "stale" checkpoint
    if args.staleness_noise > 0:
        with torch.no_grad():
            for p in async_model.parameters():
                p.add_(torch.randn_like(p) * args.staleness_noise)
    tok = MockTokenizer()

    mined_seq, stats_seq = _mine_sample(cache_seq, seq_model, tok, qids,
                                        qid_to_text, corpus_lookup, qrels_dict,
                                        cfg, device, seed=123)
    mined_async, stats_async = _mine_sample(cache_async, async_model, tok, qids,
                                            qid_to_text, corpus_lookup, qrels_dict,
                                            cfg, device, seed=123)
    metrics = _quality_metrics(mined_seq, mined_async, stats_seq, stats_async,
                               qids, qrels_dict, cfg['m'])
    _print(metrics, f"synthetic, noise={args.staleness_noise}")

    # correctness asserts
    ok = True
    if metrics['positives_leaked_seq'] or metrics['positives_leaked_async']:
        print("  FAIL: positives leaked into negatives"); ok = False
    # every selected negative must come from H
    H = set(cache_seq.docids)
    if not all(d in H for q in qids for d in mined_seq.get(q, [])):
        print("  FAIL: sequential negatives not drawn from H"); ok = False
    if args.staleness_noise == 0:
        if metrics['overlap_at_1'] != 1.0 or metrics['overlap_at_m'] != 1.0:
            print(f"  FAIL: identical weights must agree exactly "
                  f"(got @1={metrics['overlap_at_1']}, @m={metrics['overlap_at_m']})")
            ok = False
    else:
        # with staleness the paths should not be perfectly identical
        if metrics['overlap_at_1'] == 1.0 and metrics['overlap_at_m'] == 1.0:
            print("  NOTE: staleness noise did not change selection (small model / "
                  "coarse H); increase --staleness_noise for a visible split.")
    print("=" * 64)
    print(f"  {'PASS' if ok else 'FAIL'}  quality-probe harness runs end to end")
    print("=" * 64)
    return 0 if ok else 1


# ---- real GPU probe --------------------------------------------------------

def run_real(args):
    import gc
    import pickle
    from transformers import AutoTokenizer, AutoModel
    from types import SimpleNamespace
    from utils.helpers import (get_training_context, load_config, get_path,
                               _load_corpus_lookup, _load_qrels, set_seed)
    from data.preprocessor import run_setup

    config = load_config()
    ctx = get_training_context('fast_grass')
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("[quality-probe] WARNING: no CUDA — real-mode numbers are not "
              "representative; use --synthetic for the correctness smoke.", flush=True)

    corpus_file, _q, qrels_file = run_setup()
    stale_pkl = get_path("temp_grass") / "stale_index" / "corpus.pkl"
    if not stale_pkl.exists():
        print(f"[quality-probe] ERROR: stale index not found at {stale_pkl}. Build "
              "it once via run_fast_grass.py, then re-run.", flush=True)
        return 2
    with open(stale_pkl, 'rb') as f:
        cd = pickle.load(f)
    stale_embs, c_ids = cd[0], [str(x) for x in cd[1]]
    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict = _load_qrels(qrels_file)

    train_items = run_fast_grass._load_train_items()
    qid_to_text = {it['query_id']: it['query'] for it in train_items}
    qids = list(qid_to_text)[:args.max_queries]

    steps_per_epoch = max(len(qid_to_text) // config['training']['fast_grass'].get('batch_size', 64), 1)
    ns = SimpleNamespace(B_doc=args.B_doc, uncertainty='mcdp', T=args.T, L=args.L,
                         m=args.m, lambda_val=None, ema_alpha=None,
                         mc_dropout_p=None, selection_mode=None, num_epochs=None,
                         no_registry=False)
    cfg = run_fast_grass._build_fast_grass_cfg(config, ns, steps_per_epoch)

    base = ctx['base_model']
    tok = AutoTokenizer.from_pretrained(base)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

    def _load(path):
        src = path or base
        mdl = AutoModel.from_pretrained(src, torch_dtype=dtype).to(device).eval()
        for p in mdl.parameters():
            p.requires_grad_(False)
        for mod in mdl.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = cfg.get('mc_dropout_p', 0.3)
        return mdl

    seq_model = _load(args.seq_checkpoint)
    if args.async_checkpoint:
        async_model = _load(args.async_checkpoint)
    else:
        async_model = copy.deepcopy(seq_model)
        if args.staleness_noise > 0:
            with torch.no_grad():
                for p in async_model.parameters():
                    p.add_(torch.randn_like(p) * args.staleness_noise)

    cache_seq = NegativeCache.init_uniform(stale_embs, c_ids, cfg, device)
    cache_async = NegativeCache.init_uniform(stale_embs, c_ids, cfg, device)
    del stale_embs
    gc.collect()

    mined_seq, stats_seq = _mine_sample(cache_seq, seq_model, tok, qids,
                                        qid_to_text, corpus_lookup, qrels_dict,
                                        cfg, device, seed=123)
    mined_async, stats_async = _mine_sample(cache_async, async_model, tok, qids,
                                            qid_to_text, corpus_lookup, qrels_dict,
                                            cfg, device, seed=123)
    metrics = _quality_metrics(mined_seq, mined_async, stats_seq, stats_async,
                               qids, qrels_dict, cfg['m'])
    _print(metrics, f"real, {len(qids)} queries")

    out = project_root / 'analysis' / 'async_fast_grass_timing'
    out.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    p = out / f"quality_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(metrics, indent=2))
    print(f"[quality-probe] wrote {p}", flush=True)
    leaked = metrics['positives_leaked_seq'] + metrics['positives_leaked_async']
    return 0 if leaked == 0 else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--synthetic', action='store_true', default=True,
                    help='CPU mock smoke (default)')
    ap.add_argument('--real', dest='synthetic', action='store_false',
                    help='real GPU probe (needs stale index + checkpoints)')
    ap.add_argument('--staleness_noise', type=float, default=0.0,
                    help='Gaussian weight noise added to the async (frozen) model '
                         'to simulate staleness when no async_checkpoint is given')
    ap.add_argument('--seq_checkpoint', default=None, help='real: current model dir')
    ap.add_argument('--async_checkpoint', default=None, help='real: frozen/stale model dir')
    ap.add_argument('--max_queries', type=int, default=256, help='real: sample size')
    ap.add_argument('--B_doc', type=int, default=None)
    ap.add_argument('--L', type=int, default=None)
    ap.add_argument('--T', type=int, default=None)
    ap.add_argument('--m', type=int, default=None)
    args = ap.parse_args()
    return run_synthetic(args) if args.synthetic else run_real(args)


if __name__ == "__main__":
    sys.exit(main())
