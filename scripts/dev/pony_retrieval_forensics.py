"""Why pony collapses from the untrained baseline to the fine-tuned arms.

Post-hoc forensics on results that already exist: baseline 0.7618, ANCE 0.3809,
in-batch 0.2898 Recall@1000, against 0.6326 / 0.6138 / 0.5421 macro. Pony is the only
domain where a trained arm loses badly, and excluding it the ordering inverts.

Reads the embedding pickles run_all_evals already cached, so this is CPU-only and
re-encodes nothing. Nothing here produces a reported result.

Hypotheses, and the column that separates them:
  H1 junk flooding   18.3% of pony's 7,894 docs are line-number gutters ("12\\n13\\n14").
                     Top-1000 is 12.7% of the corpus, so junk that ranks high evicts
                     real documents. -> junk share of top-1000, and recall with junk
                     dropped from the index.
  H2 collapse        fine-tuning made every doc similar to every query. -> anisotropy,
                     and the relevant-vs-junk similarity gap.
  H3 length bias     pony docs are short (median 169 chars; relevant ones 588).
                     -> rank/length correlation.
  H4 lost norm       IndexFlatIP ranks by inner product and assumes unit vectors.
                     A non-unit fine-tuned encoder would rank by magnitude. -> norms.
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import faiss

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / 'src'))
from utils.helpers import (get_data_base_dir, load_config, load_excluded_ids,
                           search_depth, apply_exclusions, _load_qrels)
from evaluation.trec_eval_wrapper import TrecEvalWrapper

# Pure digits and whitespace. Printed with its count so the definition stays auditable
# rather than buried -- this predicate decides the headline number.
JUNK_RE = re.compile(r'[0-9\s]+')

# The reported runs, by cached-embedding tag. The attempt id pins the exact evaluation
# that produced the published number, not merely the newest one for that model.
ARMS = [
    ('baseline', 'main__f182c9e3', 'fe0fa996541a43bdb9c3fcc009ef9a6f'),
    ('ANCE', 'ance_mixed_bge_m3__9c19ca04', '22a516c3022d4d54878273b27f9a30f2'),
    ('in-batch', 'inbatch_mixed_bge_m3__49e7d4da', 'd9abf03dcf6947a48a13512059df27ed'),
]


def is_junk(text):
    s = (text or '').strip()
    return bool(s) and JUNK_RE.fullmatch(s) is not None


def load_embeddings(base, tag, attempt, domain):
    """(embeddings, ids) as evaluate.py wrote them: pickle of (array, id list)."""
    d = Path(base) / 'data' / 'evaluation' / tag / attempt / domain
    with open(d / 'corpus_emb' / 'corpus.pkl', 'rb') as f:
        c = pickle.load(f)
    with open(d / 'query_emb' / 'query.pkl', 'rb') as f:
        q = pickle.load(f)
    return (c[0].astype(np.float32), [str(x) for x in c[1]],
            q[0].astype(np.float32), [str(x) for x in q[1]])


def retrieve(corpus_embs, corpus_ids, query_embs, query_ids, depth):
    """The same IndexFlatIP path evaluate.py uses, so numbers are comparable."""
    index = faiss.IndexFlatIP(corpus_embs.shape[1])
    index.add(corpus_embs)
    scores, idx = index.search(query_embs, depth)
    run = {}
    for i, qid in enumerate(query_ids):
        run[qid] = {corpus_ids[idx[i][j]]: float(scores[i][j])
                    for j in range(depth) if idx[i][j] >= 0}
    return run, scores, idx


def recall_at_k(run, qrels, excluded, k):
    filtered = apply_exclusions(run, excluded, k)
    return TrecEvalWrapper(qrels).evaluate(filtered, {'recall_1000'})['recall_1000']


def analyse(domain, base, processed, config, reported):
    top_k = config['evaluation']['top_k']
    corpus_file = processed / f'{domain}_corpus.jsonl'
    texts, order = {}, []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                did = str(row['docid'])
                texts[did] = row.get('text') or ''
                order.append(did)

    junk_ids = {d for d in order if is_junk(texts[d])}
    qrels = _load_qrels(processed / f'{domain}_qrels.txt')
    excluded = load_excluded_ids(domain, processed)
    rel_ids = set().union(*qrels.values()) if qrels else set()

    print(f'\n{"=" * 78}\n  {domain}\n{"=" * 78}')
    print(f'  corpus            : {len(order):,} docs')
    print(f'  numeric-only      : {len(junk_ids):,} ({100 * len(junk_ids) / len(order):.1f}%)')
    print(f'  distinct relevant : {len(rel_ids)} over {len(qrels)} queries')
    print(f'  junk that is also judged relevant: {len(junk_ids & rel_ids)}')

    for name, tag, attempt in ARMS:
        try:
            c_emb, c_ids, q_emb, q_ids = load_embeddings(base, tag, attempt, domain)
        except FileNotFoundError as e:
            print(f'\n  [{name}] no cached embeddings ({e.filename}) -- skipped')
            continue

        depth = min(search_depth(top_k, excluded), len(c_ids))
        run, scores, idx = retrieve(c_emb, c_ids, q_emb, q_ids, depth)

        # Gate: unless this reproduces the published number, nothing below is evidence.
        got = recall_at_k(run, qrels, excluded, top_k)
        want = reported.get((name, domain))
        flag = ''
        if want is not None:
            flag = ' ✅' if abs(got - want) < 1e-4 else f' ❌ expected {want:.4f}'
        print(f'\n  [{name}] Recall@{top_k} reproduced: {got:.4f}{flag}')

        # H4: IndexFlatIP is cosine only on unit vectors.
        cn = np.linalg.norm(c_emb, axis=1)
        qn = np.linalg.norm(q_emb, axis=1)
        print(f'    norms       corpus {cn.mean():.4f}±{cn.std():.4f}   '
              f'query {qn.mean():.4f}±{qn.std():.4f}')

        # H1: how much of the retrieved budget goes to junk.
        shares = [sum(1 for d in list(hits)[:top_k] if d in junk_ids) / min(top_k, len(hits))
                  for hits in run.values()]
        print(f'    junk share of top-{top_k}: mean {100 * np.mean(shares):.1f}%  '
              f'min {100 * min(shares):.1f}%  max {100 * max(shares):.1f}%')

        # H1 decisive: same retrieval over an index with the junk removed.
        keep = [i for i, d in enumerate(c_ids) if d not in junk_ids]
        c2 = np.ascontiguousarray(c_emb[keep])
        ids2 = [c_ids[i] for i in keep]
        run2, _, _ = retrieve(c2, ids2, q_emb, q_ids, min(depth, len(ids2)))
        clean = recall_at_k(run2, qrels, excluded, top_k)
        print(f'    Recall@{top_k} with junk removed: {clean:.4f}  '
              f'({clean - got:+.4f})')

        # H1/H3: where the relevant documents actually land.
        pos = {d: r for r, d in enumerate(c_ids)}
        ranks, missed = [], 0
        for qid, gold in qrels.items():
            if qid not in q_ids:
                continue
            row = q_ids.index(qid)
            order_row = {c_ids[idx[row][j]]: j for j in range(depth) if idx[row][j] >= 0}
            for g in gold:
                r = order_row.get(g)
                if r is None:
                    missed += 1
                else:
                    ranks.append(r)
        if ranks:
            rr = np.array(ranks)
            print(f'    relevant-doc rank: median {int(np.median(rr))}  '
                  f'p90 {int(np.percentile(rr, 90))}  beyond depth: {missed}')

        # H2: is everything close to everything.
        rng = np.random.default_rng(0)
        samp = c_emb[rng.choice(len(c_emb), size=min(2000, len(c_emb)), replace=False)]
        samp = samp / np.clip(np.linalg.norm(samp, axis=1, keepdims=True), 1e-12, None)
        sim = samp @ samp.T
        n = len(samp)
        aniso = (sim.sum() - np.trace(sim)) / (n * (n - 1))
        qn_ = q_emb / np.clip(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12, None)
        cn_ = c_emb / np.clip(np.linalg.norm(c_emb, axis=1, keepdims=True), 1e-12, None)
        junk_mask = np.array([d in junk_ids for d in c_ids])
        rel_mask = np.array([d in rel_ids for d in c_ids])
        qc = qn_ @ cn_.T
        print(f'    mean cos  all {qc.mean():+.4f}   '
              f'junk {qc[:, junk_mask].mean() if junk_mask.any() else float("nan"):+.4f}   '
              f'relevant {qc[:, rel_mask].mean() if rel_mask.any() else float("nan"):+.4f}')
        print(f'    corpus anisotropy (mean pairwise cos): {aniso:+.4f}')

        # H3: do short documents win.
        lens = np.array([len(texts[d]) for d in c_ids], dtype=np.float64)
        top_lens, top_ranks = [], []
        for i in range(len(q_ids)):
            for j in range(min(top_k, depth)):
                if idx[i][j] >= 0:
                    top_lens.append(lens[idx[i][j]])
                    top_ranks.append(j)
        if top_lens:
            a = np.argsort(np.argsort(top_lens)).astype(float)
            b = np.argsort(np.argsort(top_ranks)).astype(float)
            rho = np.corrcoef(a, b)[0, 1]
            print(f'    Spearman(rank, doc length) over top-{top_k}: {rho:+.4f}  '
                  f'(positive => shorter docs rank higher)')


def margins(c_emb, c_ids, q_emb, rel_ids, block=4096):
    """Mean query-doc cosine, overall and to judged-relevant docs, plus anisotropy.

    Blocked over the corpus: leetcode is 413,932 x 1024, so the full query x corpus
    matrix is avoided rather than merely tolerated.
    """
    qn = q_emb / np.clip(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12, None)
    rel_mask = np.fromiter((d in rel_ids for d in c_ids), bool, len(c_ids))
    tot = tot_n = rel = rel_n = 0.0
    for s in range(0, len(c_ids), block):
        e = min(s + block, len(c_ids))
        cn = c_emb[s:e] / np.clip(np.linalg.norm(c_emb[s:e], axis=1, keepdims=True),
                                  1e-12, None)
        sim = qn @ cn.T
        tot += sim.sum(); tot_n += sim.size
        m = rel_mask[s:e]
        if m.any():
            rel += sim[:, m].sum(); rel_n += sim[:, m].size
    rng = np.random.default_rng(0)
    samp = c_emb[rng.choice(len(c_emb), size=min(2000, len(c_emb)), replace=False)]
    samp = samp / np.clip(np.linalg.norm(samp, axis=1, keepdims=True), 1e-12, None)
    sim = samp @ samp.T
    n = len(samp)
    aniso = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    mean_all = tot / tot_n
    mean_rel = rel / rel_n if rel_n else float('nan')
    return mean_all, mean_rel, mean_rel - mean_all, aniso


def published_recall(base, config):
    """Recall@1000 as actually reported, so the sweep needs no retrieval at all."""
    import glob
    results = Path(base) / config['paths']['results_dir']
    out = {}
    for name, tag, attempt in ARMS:
        hits = glob.glob(str(results / tag / attempt / 'summary.json')) or \
               glob.glob(str(results / tag / '*' / 'summary.json'))
        if not hits:
            continue
        for row in json.loads(Path(sorted(hits)[-1]).read_text())['per_domain']:
            out[(name, row['domain'])] = row['recall_1000']
    return out


def sweep(base, processed, config):
    """Does margin collapse predict recall loss across all 12 domains, or only pony?"""
    domains = config['evaluation']['eval_domains']
    recall = published_recall(base, config)
    rows = {}
    print(f'\n{"=" * 92}\n  MARGIN SWEEP — all {len(domains)} domains\n{"=" * 92}')
    print(f'  {"domain":<22}{"arm":<10}{"cos(all)":>10}{"cos(rel)":>10}'
          f'{"margin":>9}{"aniso":>9}{"R@1000":>9}')
    for domain in domains:
        qrels = _load_qrels(processed / f'{domain}_qrels.txt')
        rel_ids = set().union(*qrels.values()) if qrels else set()
        for name, tag, attempt in ARMS:
            try:
                c_emb, c_ids, q_emb, _ = load_embeddings(base, tag, attempt, domain)
            except FileNotFoundError as e:
                print(f'  {domain:<22}{name:<10}  MISSING {e.filename}')
                continue
            ma, mr, mg, an = margins(c_emb, c_ids, q_emb, rel_ids)
            r = recall.get((name, domain), float('nan'))
            rows[(name, domain)] = (mg, r)
            print(f'  {domain:<22}{name:<10}{ma:>+10.4f}{mr:>+10.4f}'
                  f'{mg:>+9.4f}{an:>+9.4f}{r:>9.4f}')
            del c_emb, q_emb

    # The question this job exists to answer.
    print(f'\n{"=" * 92}\n  Does margin collapse predict recall loss?\n{"=" * 92}')
    for arm in ('ANCE', 'in-batch'):
        pairs = [(rows[(arm, d)][0] - rows[('baseline', d)][0],
                  rows[(arm, d)][1] - rows[('baseline', d)][1])
                 for d in domains
                 if (arm, d) in rows and ('baseline', d) in rows]
        if len(pairs) < 3:
            print(f'  {arm}: too few domains ({len(pairs)}) to correlate')
            continue
        dm = np.array([p[0] for p in pairs]); dr = np.array([p[1] for p in pairs])
        pear = np.corrcoef(dm, dr)[0, 1]
        spear = np.corrcoef(np.argsort(np.argsort(dm)).astype(float),
                            np.argsort(np.argsort(dr)).astype(float))[0, 1]
        print(f'\n  {arm} vs baseline over {len(pairs)} domains:')
        print(f'    Pearson(Δmargin, ΔRecall@1000)  = {pear:+.4f}')
        print(f'    Spearman(Δmargin, ΔRecall@1000) = {spear:+.4f}')
        for d, (a, b) in zip([d for d in domains if (arm, d) in rows], pairs):
            print(f'      {d:<22} Δmargin {a:+.4f}   ΔR@1000 {b:+.4f}')


def check_corpus_fidelity(processed, domain='pony'):
    """Is our eval corpus byte-faithful to BRIGHT's own documents split?

    Constancy already rules eval preprocessing out -- all four arms read this one file,
    and BM25/baseline score fine on it. This replaces that inference with evidence.
    """
    print(f'\n{"=" * 92}\n  CORPUS FIDELITY — {domain} vs raw BRIGHT\n{"=" * 92}')
    try:
        from data.bright_loader import BRIGHTLoader
        loader = BRIGHTLoader()
        loader.load_dataset()
        raw = loader.get_corpus(domain)
    except Exception as e:
        print(f'  SKIPPED (could not load BRIGHT offline): {type(e).__name__}: {e}')
        return
    raw_map = {str(r): t for r, t in zip(raw['doc_id'], raw['text'])}
    ours = {}
    with open(processed / f'{domain}_corpus.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                ours[str(row['docid'])] = row.get('text') or ''
    only_raw = set(raw_map) - set(ours)
    only_ours = set(ours) - set(raw_map)
    differing = [d for d in set(ours) & set(raw_map) if ours[d] != raw_map[d]]
    print(f'  raw BRIGHT docs : {len(raw_map):,}')
    print(f'  our corpus docs : {len(ours):,}')
    print(f'  ids only in raw : {len(only_raw)}')
    print(f'  ids only in ours: {len(only_ours)}')
    print(f'  differing text  : {len(differing)}')
    print('  VERDICT:', 'FAITHFUL ✅' if not (only_raw or only_ours or differing)
          else 'MISMATCH ❌')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domains', default='pony,leetcode',
                    help='comma separated; leetcode is the control (code, but no junk)')
    ap.add_argument('--sweep', action='store_true',
                    help='all-12-domain margin sweep + corpus fidelity, no retrieval')
    args = ap.parse_args()

    if args.sweep:
        base = Path(get_data_base_dir())
        config = load_config()
        processed = base / config['paths']['processed_dir']
        sweep(base, processed, config)
        check_corpus_fidelity(processed)
        return

    base = Path(get_data_base_dir())
    config = load_config()
    processed = base / config['paths']['processed_dir']

    # Published numbers, so the probe proves itself before it explains anything.
    reported = {
        ('baseline', 'pony'): 0.7617839635,
        ('ANCE', 'pony'): 0.3808575923,
        ('in-batch', 'pony'): 0.2897798654,
        ('baseline', 'leetcode'): 0.6469,
        ('ANCE', 'leetcode'): 0.6898,
        ('in-batch', 'leetcode'): 0.6802,
    }
    print(f'junk predicate: fullmatch {JUNK_RE.pattern!r} on the stripped text')
    for domain in [d.strip() for d in args.domains.split(',') if d.strip()]:
        analyse(domain, base, processed, config, reported)


if __name__ == '__main__':
    main()
