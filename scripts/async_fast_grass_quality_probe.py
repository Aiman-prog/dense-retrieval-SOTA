"""
Async Fast-GRASS — cached-MCDP LAMBDA DOSAGE probe (REPORT-ONLY).

Asks one narrow, cheap question before any 2-GPU pilot is submitted:

    at what lambda does ``g = s_hat + lambda*sigma`` start changing which negative
    gets mined, and by how much?

Every lambda on the grid is evaluated from the **same** ``s_hat`` and ``sigma`` draw, so
differences between them are attributable to the uncertainty term alone rather than to a
different dropout sample. Our EMA runs already showed ``lambda=1`` behaving like
``lambda=0``, so an inert term is not hypothetical.

**REGIME CAVEAT — read before quoting any number.** This runs on a base checkpoint with a
freshly built ``Z_mc``, i.e. ZERO cache staleness. The ``sigma`` measured here is pure
dropout noise; during training the cached document states are stale relative to the model
and sigma's scale shifts. The flip-rate bands below are approximate **dosage
calibration**, not transferable constants, and say nothing about retrieval quality.
Whether uncertainty helps is decided by the pilot arms and their BRIGHT evaluation, never
here. **Do not quote any Recall/NDCG claim from this script.**

It deliberately does NOT invoke the sequential lazy top-``L`` miner
(``run_fast_grass._mine_batch_mcdp``). That path confounds top-``L`` truncation, fresh
document resampling and caching in one disagreement number.

Selection rule (``select_lambdas``), over nonzero grid values only:
  reject   flip-rate SD across seeds > 0.05, or any known positive selected
  low      flip rate in [0.10, 0.20)
  medium   flip rate in [0.20, 0.35]
  ties     prefer the smaller lambda
  fallback nearest to the band centre, flagged ``band_satisfied: false``
  distinct the two arms are never the same lambda; if only one candidate survives,
           ``n_arms: 1`` and only one nonzero pilot should be submitted

Modes:
  --synthetic (default): CPU mock model, no downloads.
  --real                : GPU; needs the stale index pickle + processed mixture.

Usage:
  python scripts/async_fast_grass_quality_probe.py --synthetic
  python scripts/async_fast_grass_quality_probe.py --real \\
      --recipe async_fast_grass_pilot --manifest <pilot10.jsonl> \\
      --lambda_grid 0,0.1,0.2,0.3,0.5,0.7,1.0 --seeds 3 \\
      --max_queries 2048 --B_doc 32000 --T 3 --query_batch_size 128
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
    init_Z_mc, encode_queries_mc, score_cached_mcdp, build_async_cfg,
    steps_per_epoch,
)

OUT_DIR = project_root / 'analysis' / 'async_fast_grass_timing'

DEFAULT_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
LOW_BAND = (0.10, 0.20)      # half-open [lo, hi)
MEDIUM_BAND = (0.20, 0.35)   # closed   [lo, hi]
MAX_FLIP_SD = 0.05

REGIME_CAVEAT = (
    "Measured on a base checkpoint with a freshly built Z_mc (zero cache staleness). "
    "sigma here is pure dropout noise; mid-training it is also model-staleness. These "
    "flip-rate bands are dosage calibration only and imply nothing about retrieval "
    "quality — that is decided by the pilot arms and their BRIGHT evaluation.")


# ---- one seed, whole grid ---------------------------------------------------

def probe_grid(cache, Z_mc, student, tokenizer, qids, qid_to_text, qrels_dict,
               T, lambdas, m, cfg, device, seed, query_batch_size=128):
    """One MC draw over all queries: score ONCE, evaluate every lambda from it.

    ``s_hat`` is masked once; because a masked slot is ``-inf`` and ``sigma`` is finite,
    ``masked_s_hat + lambda*sigma`` stays ``-inf`` there, so every lambda inherits the
    same positive masking without re-masking per lambda.

    Queries are processed in chunks of ``query_batch_size`` so the ``[T, B_q, B_doc]``
    score buffers stay bounded at ``B_doc=32k``.

    **``query_batch_size`` is part of the experiment, not a free memory knob.** Each
    chunk draws its own dropout masks, so changing it changes the MC sample and shifts
    the measured flip rates slightly. Document chunking (``score_chunk_size``) is
    different — that one is exactly invariant, because it only splits a matmul. Hold
    ``query_batch_size`` fixed when comparing probe runs.

    Returns ``{lambda: stats}`` for this seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    acc = {float(l): {'flips': 0, 'n': 0, 'sum_s_hat': 0.0, 'sum_sigma': 0.0,
                      'sum_ratio': 0.0, 'n_ratio': 0, 'leaked': 0}
           for l in lambdas}
    sigma_chunks = []

    for start in range(0, len(qids), int(query_batch_size)):
        batch_qids = qids[start:start + int(query_batch_size)]
        texts = [qid_to_text[q] for q in batch_qids]
        q_mc, _ = encode_queries_mc(student, tokenizer, texts, T, device, cfg)

        _g, s_hat, sigma = score_cached_mcdp(q_mc, Z_mc, 0.0,
                                             chunk_size=cfg.get('score_chunk_size'))
        s0 = cache.mask_positives(s_hat.clone(), batch_qids, qrels_dict, inplace=True)
        sel0 = torch.topk(s0, k=m, dim=1).indices

        # how big is the nudge relative to the gap it has to close?
        top2 = torch.topk(s0, k=min(2, s0.shape[1]), dim=1).values
        margin = ((top2[:, 0] - top2[:, 1]).clamp(min=1e-9)
                  if top2.shape[1] > 1 else None)

        finite = sigma[torch.isfinite(sigma)]
        if finite.numel():
            sigma_chunks.append(finite.float().cpu())

        for lam in lambdas:
            lam = float(lam)
            g = s0 + lam * sigma                     # masked slots stay -inf
            sel = torch.topk(g, k=m, dim=1).indices
            a = acc[lam]
            a['flips'] += int((sel[:, 0] != sel0[:, 0]).sum())
            a['n'] += len(batch_qids)
            a['sum_s_hat'] += float(torch.gather(s_hat, 1, sel).float().sum())
            a['sum_sigma'] += float(torch.gather(sigma, 1, sel).float().sum())
            if margin is not None:
                s_top1 = torch.gather(sigma, 1, sel[:, :1]).squeeze(1)
                a['sum_ratio'] += float((lam * s_top1 / margin).sum())
                a['n_ratio'] += len(batch_qids)
            for i, qid in enumerate(batch_qids):
                pos = qrels_dict.get(qid) or ()
                a['leaked'] += sum(1 for s in sel[i].tolist()
                                   if cache.docids[s] in pos)

    all_sigma = (torch.cat(sigma_chunks) if sigma_chunks
                 else torch.zeros(1, dtype=torch.float32))
    out = {}
    for lam, a in acc.items():
        n = max(a['n'], 1)
        # m selections per query, so per-selection means divide by n*m
        nm = max(a['n'] * m, 1)
        out[lam] = {
            'lambda_val': lam,
            'seed': seed,
            'num_queries': a['n'],
            'flip_rate': a['flips'] / n,
            'sel_s_hat_mean': a['sum_s_hat'] / nm,
            'sel_sigma_mean': a['sum_sigma'] / nm,
            'lambda_sigma_vs_margin': (a['sum_ratio'] / a['n_ratio']
                                       if a['n_ratio'] else None),
            'positives_leaked': a['leaked'],
        }
    sigma_stats = {
        'sigma_mean': float(all_sigma.mean()),
        'sigma_p50': float(all_sigma.median()),
        'sigma_p90': float(torch.quantile(all_sigma, 0.9)),
        'sigma_max': float(all_sigma.max()),
        'sigma_all_zero': bool(float(all_sigma.abs().max()) == 0.0),
        'sigma_nonfinite': bool(not torch.isfinite(all_sigma).all()),
    }
    return out, sigma_stats


def aggregate_grid(per_seed, lambdas, m, n_queries, T, B_doc):
    """Fold the per-seed dicts into one row per lambda (mean + SD over seeds)."""
    rows = []
    for lam in lambdas:
        lam = float(lam)
        runs = [s[lam] for s in per_seed]
        fr = [r['flip_rate'] for r in runs]
        ratios = [r['lambda_sigma_vs_margin'] for r in runs
                  if r['lambda_sigma_vs_margin'] is not None]
        rows.append({
            'lambda_val': lam,
            'flip_rate_mean': float(np.mean(fr)),
            'flip_rate_std': float(np.std(fr)),
            'flip_rate_min': float(np.min(fr)),
            'flip_rate_max': float(np.max(fr)),
            'sel_s_hat_mean': float(np.mean([r['sel_s_hat_mean'] for r in runs])),
            'sel_sigma_mean': float(np.mean([r['sel_sigma_mean'] for r in runs])),
            'sel_lambda_sigma_mean': lam * float(
                np.mean([r['sel_sigma_mean'] for r in runs])),
            'lambda_sigma_vs_margin_mean': float(np.mean(ratios)) if ratios else None,
            'positives_leaked_total': int(sum(r['positives_leaked'] for r in runs)),
            'num_seeds': len(runs),
        })
    return rows


# ---- lambda selection -------------------------------------------------------

def _in_band(flip, band, closed_high):
    lo, hi = band
    return (lo <= flip <= hi) if closed_high else (lo <= flip < hi)


# Flip rates are proportions over ~2k queries, so differences below this are noise.
# Distances are QUANTIZED to it before ranking: without that, "prefer the smaller
# lambda on a tie" could never fire, because two genuinely equidistant candidates
# differ by ~1e-17 in floating point and the comparison silently becomes arbitrary.
TIE_TOLERANCE = 1e-4


def _rank_key(row, centre):
    """Closest to the band centre first; on a (near-)tie prefer the SMALLER lambda."""
    dist = abs(row['flip_rate_mean'] - centre)
    return (round(dist / TIE_TOLERANCE), row['lambda_val'])


def select_lambdas(rows, low_band=LOW_BAND, medium_band=MEDIUM_BAND,
                   max_flip_sd=MAX_FLIP_SD):
    """Choose the low-dose and medium-dose nonzero lambdas. Pure function.

    ``lambda=0`` is the CONTROL arm and is never selectable. A candidate is rejected
    outright if its flip rate is unstable across seeds (SD > ``max_flip_sd``, i.e. the
    "signal" is sampling noise) or if it ever selected a known positive as a negative.

    Bands are ``[0.10, 0.20)`` and ``[0.20, 0.35]`` so they partition rather than
    overlap at 0.20. Within a band, the candidate closest to the band centre wins, and
    ties go to the smaller lambda.

    If a band is empty the nearest surviving candidate is taken instead and
    ``band_satisfied`` is set false — pretending the band was met would hide that the
    grid never reached the intended dosage.

    The two arms are always distinct: a fallback that would duplicate the low arm
    advances to the next-closest surviving candidate. With fewer than two survivors,
    ``n_arms`` reports how many nonzero arms are actually justified; submitting two
    identical arms would burn a 4-hour job to reproduce a run we already have.
    """
    rejected, survivors = [], []
    for r in rows:
        if r['lambda_val'] <= 0:
            continue                       # control, not a candidate
        why = []
        if r['flip_rate_std'] > max_flip_sd:
            why.append(f"flip-rate SD {r['flip_rate_std']:.3f} > {max_flip_sd}")
        if r['positives_leaked_total'] > 0:
            why.append(f"{r['positives_leaked_total']} known positives selected")
        (rejected if why else survivors).append(
            {**r, 'rejected_because': why} if why else r)

    result = {
        'low_band': list(low_band), 'medium_band': list(medium_band),
        'max_flip_sd': max_flip_sd,
        'rejected': rejected,
        'num_survivors': len(survivors),
        'selected_low': None, 'selected_medium': None,
        'band_satisfied': {'low': None, 'medium': None},
        'n_arms': 0, 'notes': [],
    }
    if not survivors:
        result['notes'].append(
            "no nonzero lambda survived the stability/leakage checks; do not submit a "
            "nonzero pilot arm on this evidence")
        return result

    def pick(band, closed_high, exclude):
        pool = [r for r in survivors if r['lambda_val'] not in exclude]
        if not pool:
            return None, None
        centre = (band[0] + band[1]) / 2.0
        in_band = [r for r in pool if _in_band(r['flip_rate_mean'], band, closed_high)]
        if in_band:
            return min(in_band, key=lambda r: _rank_key(r, centre)), True
        return min(pool, key=lambda r: _rank_key(r, centre)), False

    low, low_ok = pick(low_band, closed_high=False, exclude=set())
    result['selected_low'] = low['lambda_val'] if low else None
    result['band_satisfied']['low'] = low_ok

    med, med_ok = pick(medium_band, closed_high=True,
                       exclude={low['lambda_val']} if low else set())
    result['selected_medium'] = med['lambda_val'] if med else None
    result['band_satisfied']['medium'] = med_ok

    result['n_arms'] = sum(1 for v in (result['selected_low'],
                                       result['selected_medium']) if v is not None)
    if result['n_arms'] == 1:
        result['notes'].append(
            "only one nonzero lambda survived; submit ONE nonzero pilot arm — two "
            "identical arms would just reproduce the same run")
    if low_ok is False:
        result['notes'].append(
            f"no surviving lambda lands in the low band {list(low_band)}; fell back to "
            f"lambda={result['selected_low']} at flip rate "
            f"{low['flip_rate_mean']:.3f} (band NOT satisfied)")
    if med is not None and med_ok is False:
        result['notes'].append(
            f"no surviving lambda lands in the medium band {list(medium_band)}; fell "
            f"back to lambda={result['selected_medium']} at flip rate "
            f"{med['flip_rate_mean']:.3f} (band NOT satisfied)")
    return result


# ---- reporting --------------------------------------------------------------

def _print(report):
    a = report
    print("\n" + "=" * 78)
    print("  CACHED-MCDP LAMBDA DOSAGE PROBE  (REPORT ONLY)")
    print("=" * 78)
    print(f"  queries {a['num_queries']} | m {a['m']} | T {a['T']} | "
          f"B_doc {a['B_doc']:,} | seeds {a['num_seeds']} | recipe {a.get('recipe')}")
    print("-" * 78)
    print(f"  {'lambda':>7} {'flip':>8} {'±sd':>7} {'sel s_hat':>11} "
          f"{'sel sigma':>11} {'l*s/margin':>11} {'leak':>5}")
    for r in a['grid']:
        ratio = ('  n/a' if r['lambda_sigma_vs_margin_mean'] is None
                 else f"{r['lambda_sigma_vs_margin_mean']:11.4f}")
        print(f"  {r['lambda_val']:7.2f} {r['flip_rate_mean']:8.4f} "
              f"{r['flip_rate_std']:7.4f} {r['sel_s_hat_mean']:11.4f} "
              f"{r['sel_sigma_mean']:11.6f} {ratio} "
              f"{r['positives_leaked_total']:5d}")
    print("-" * 78)
    s = a['selection']
    print(f"  sigma mean/p90/max : {a['sigma']['sigma_mean']:.6f} / "
          f"{a['sigma']['sigma_p90']:.6f} / {a['sigma']['sigma_max']:.6f}")
    print(f"  selected LOW       : {s['selected_low']}  "
          f"(band {s['low_band']} satisfied={s['band_satisfied']['low']})")
    print(f"  selected MEDIUM    : {s['selected_medium']}  "
          f"(band {s['medium_band']} satisfied={s['band_satisfied']['medium']})")
    print(f"  nonzero arms to run: {s['n_arms']}")
    for r in s['rejected']:
        print(f"  ✗ lambda={r['lambda_val']}: {'; '.join(r['rejected_because'])}")
    for n in s['notes']:
        print(f"  ⚠️  {n}")
    print("-" * 78)
    for line in _verdict(a):
        print(f"  • {line}")
    print("-" * 78)
    print("  REGIME CAVEAT: " + REGIME_CAVEAT)
    print("=" * 78)


def _verdict(a):
    notes = []
    if a['sigma']['sigma_all_zero']:
        notes.append("sigma is identically ZERO: the cached uncertainty term is dead "
                     "(check that dropout is actually active during MC passes).")
        return notes
    nonzero = [r for r in a['grid'] if r['lambda_val'] > 0]
    best = max(nonzero, key=lambda r: r['flip_rate_mean']) if nonzero else None
    if best is None or best['flip_rate_mean'] == 0.0:
        notes.append("NO lambda on the grid changed a single selection: sigma is inert "
                     "— exactly the EMA lambda=1 ~ lambda=0 failure mode.")
    elif best['flip_rate_mean'] < 0.01:
        notes.append(f"the largest lambda ({best['lambda_val']}) changes only "
                     f"{best['flip_rate_mean']:.1%} of selections: sigma is nearly "
                     f"inert across the whole grid.")
    else:
        notes.append(f"sigma is ACTIVE: lambda={best['lambda_val']} changes "
                     f"{best['flip_rate_mean']:.1%} of top-1 selections. "
                     f"(Active != better — that is the pilot's question.)")
    return notes


def _hard_failures(a):
    """Only genuine correctness breakage is an error — never a dosage verdict."""
    bad = []
    leaked = sum(r['positives_leaked_total'] for r in a['grid'])
    if leaked:
        bad.append(f"{leaked} known positives selected as negatives")
    if a['sigma']['sigma_nonfinite']:
        bad.append("sigma contained non-finite values")
    return bad


def run_probe(cache, Z_mc, student, tokenizer, qids, qid_to_text, qrels_dict,
              T, lambdas, m, cfg, device, seeds, query_batch_size, recipe=None):
    """Shared body of the synthetic and real modes."""
    per_seed, sigma_stats = [], None
    for s in range(seeds):
        grid, sig = probe_grid(cache, Z_mc, student, tokenizer, qids, qid_to_text,
                               qrels_dict, T, lambdas, m, cfg, device,
                               seed=100 + s, query_batch_size=query_batch_size)
        per_seed.append(grid)
        sigma_stats = sig if sigma_stats is None else sigma_stats

    rows = aggregate_grid(per_seed, lambdas, m, len(qids), T, cache.B_doc)
    return {
        'kind': 'cached_mcdp_lambda_dosage_probe',
        'report_only': True,
        'regime_caveat': REGIME_CAVEAT,
        'recipe': recipe,
        'lambda_grid': [float(l) for l in lambdas],
        'm': m, 'T': T, 'B_doc': cache.B_doc,
        'num_queries': len(qids), 'num_seeds': seeds,
        'query_batch_size': query_batch_size,
        'grid': rows,
        'sigma': sigma_stats,
        'selection': select_lambdas(rows),
        'per_seed': [{str(k): v for k, v in g.items()} for g in per_seed],
    }


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
                   m=args.m or 2, lambda_val=0.0, mc_dropout_p=0.3,
                   selection_mode='topk', query_max_len=8, passage_max_len=8,
                   mc_batch_size=32)
    cache = NegativeCache.init_uniform(embs, c_ids, cfg, device, dim=dim)
    student = DropoutMockModel(hidden=dim, p=cfg['mc_dropout_p'])
    tok = MockTokenizer()
    Z_mc, _ = init_Z_mc(cache, corpus_lookup, student, tok, T, cfg, device)

    report = run_probe(cache, Z_mc, student, tok, qids, qid_to_text, qrels_dict,
                       T, args.lambda_grid, cfg['m'], cfg, device, args.seeds,
                       args.query_batch_size, recipe='synthetic')
    report['mode'] = 'synthetic'
    _print(report)

    bad = _hard_failures(report)
    for b in bad:
        print(f"  FAIL: {b}")
    print(f"  {'PASS' if not bad else 'FAIL'}  dosage-probe harness runs end to end")
    print("=" * 78)
    return 0 if not bad else 1


# ---- real GPU mode ---------------------------------------------------------

def run_real(args):
    import gc
    import pickle
    from transformers import AutoTokenizer, AutoModel
    from utils.helpers import (get_training_context, load_config, get_path,
                               _load_corpus_lookup, _load_qrels, set_seed)
    from data.preprocessor import run_setup
    import run_fast_grass
    from async_fast_grass_cached_mcdp import canonicalize_positives
    from async_fast_grass_pilot import maybe_apply_manifest

    config = load_config()
    # The ASYNC recipe, not the sequential one: this probe calibrates lambda for
    # cached-MCDP mining, so it must use that block's model, dropout rate, B_doc, T and
    # sequence lengths. Reading `fast_grass` here would silently calibrate against a
    # different configuration than the pilot runs.
    ctx = get_training_context(args.recipe)
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("[dosage-probe] WARNING: no CUDA — use --synthetic for the CPU smoke.",
              flush=True)

    corpus_file, _q, qrels_file = run_setup()
    stale_pkl = get_path("temp_grass") / "stale_index" / "corpus.pkl"
    if not stale_pkl.exists():
        print(f"[dosage-probe] ERROR: stale index not found at {stale_pkl}. Build it "
              "once via run_fast_grass.py, then re-run.", flush=True)
        return 2
    with open(stale_pkl, 'rb') as f:
        cd = pickle.load(f)
    stale_embs, c_ids = cd[0], [str(x) for x in cd[1]]
    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict = _load_qrels(qrels_file)

    train_items = run_fast_grass._load_train_items()
    train_items, _canon = canonicalize_positives(train_items, qrels_dict,
                                                 corpus_lookup,
                                                 log=lambda m: print(f"  {m}"))
    # Manifest BEFORE the max_queries slice: the mixture is HQ-first on disk, so
    # slicing raw file order would probe an HQ-only sample and call it stratified.
    train_items, _mmeta = maybe_apply_manifest(
        train_items, args.manifest, log=lambda m: print(f"[dosage-probe] {m}"))
    qid_to_text = {it['query_id']: it['query'] for it in train_items}
    qids = list(qid_to_text)[:args.max_queries]

    batch_size = ctx['args'].get('batch_size', 64)
    spe = steps_per_epoch(len(train_items), batch_size)
    cfg = build_async_cfg(config, ctx, spe)
    for key, val in (('B_doc', args.B_doc), ('T', args.T), ('m', args.m)):
        if val is not None:
            cfg[key] = val
    T = int(cfg['T'])

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

    report = run_probe(cache, Z_mc, student, tok, qids, qid_to_text, qrels_dict,
                       T, args.lambda_grid, cfg['m'], cfg, device, args.seeds,
                       args.query_batch_size, recipe=args.recipe)
    report['mode'] = 'real'
    report['base_model'] = str(base)
    report['mc_dropout_p'] = cfg.get('mc_dropout_p')
    report['manifest'] = str(args.manifest) if args.manifest else None
    _print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    p = OUT_DIR / f"lambda_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(report, indent=2))
    print(f"[dosage-probe] wrote {p}", flush=True)

    bad = _hard_failures(report)
    for b in bad:
        print(f"  FAIL: {b}")
    return 0 if not bad else 1


def _parse_grid(spec):
    vals = sorted({float(x) for x in str(spec).split(',') if x.strip() != ''})
    if not vals:
        raise argparse.ArgumentTypeError("lambda grid is empty")
    if 0.0 not in vals:
        # every flip rate is measured AGAINST lambda=0, so the control has to be there
        vals = [0.0] + vals
    return vals


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--synthetic', action='store_true', default=True,
                    help='CPU mock smoke (default)')
    ap.add_argument('--real', dest='synthetic', action='store_false',
                    help='real GPU probe (needs stale index + processed mixture)')
    ap.add_argument('--recipe', default='async_fast_grass_pilot',
                    help='training.<recipe> block supplying model, dropout, B_doc, '
                         'T and sequence lengths')
    ap.add_argument('--manifest', default=None,
                    help='pilot manifest JSONL; applied BEFORE --max_queries so the '
                         'sample is stratified rather than mixture-file order')
    ap.add_argument('--lambda_grid', type=_parse_grid, default=list(DEFAULT_GRID),
                    help='comma-separated; 0 is always included as the control')
    ap.add_argument('--seeds', type=int, default=3,
                    help='independent MC draws; separates a stable signal from noise')
    ap.add_argument('--max_queries', type=int, default=2048, help='query sample size')
    ap.add_argument('--query_batch_size', type=int, default=128,
                    help='queries scored at once; bounds the [T, B_q, B_doc] buffer')
    ap.add_argument('--B_doc', type=int, default=None)
    ap.add_argument('--T', type=int, default=None)
    ap.add_argument('--m', type=int, default=None)
    args = ap.parse_args()
    return run_synthetic(args) if args.synthetic else run_real(args)


if __name__ == "__main__":
    sys.exit(main())
