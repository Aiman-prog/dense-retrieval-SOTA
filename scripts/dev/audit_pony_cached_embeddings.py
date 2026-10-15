"""Analyze trusted, saved Pony eval embeddings locally, without re-encoding.

The length-only interval holds short-document/query scores fixed and lets every
long competitor move arbitrarily. It is a counterfactual bound, not a new eval.
"""
import argparse
import collections
import json
import pickle
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tokenizer-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    import yaml
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / 'config/config.yaml').read_text())
    cap = config['model']['passage_max_len']
    k = config['evaluation']['top_k']
    corpus = {d['docid']: d['text'] for d in map(json.loads,
              (args.root / 'pony/pony_corpus.jsonl').open())}
    lengths = {key: len(tokenizer.encode(text).ids) for key, text in corpus.items()}
    qrels = collections.defaultdict(set)
    for line in (args.root / 'pony/pony_qrels.txt').open():
        qid, _, did, rel = line.split()
        if int(rel) > 0:
            qrels[qid].add(did)
    excluded = json.loads((args.root / 'pony/pony_excluded.json').read_text())
    assert all(lengths[d] <= cap for ids in qrels.values() for d in ids)
    result = {'long_documents': sum(v > cap for v in lengths.values()), 'models': {}}
    for folder in sorted((args.root / 'emb').iterdir()):
        # Input files are trusted artifacts from the user's own evaluation jobs.
        with (folder / 'pony/corpus_emb/corpus.pkl').open('rb') as handle:
            c, cids = pickle.load(handle)
        with (folder / 'pony/query_emb/query.pkl').open('rb') as handle:
            q, qids = pickle.load(handle)
        c, q = np.asarray(c, dtype=np.float32), np.asarray(q, dtype=np.float32)
        cids, qids = list(map(str, cids)), list(map(str, qids))
        assert len(set(cids)) == len(cids) and set(cids) == set(corpus)
        assert set(qids) == set(qrels)
        scores = q @ c.T
        recalls, lower, upper, best_ranks, top_ten = [], [], [], [], collections.Counter()
        per_query = {}
        for i, qid in enumerate(qids):
            ex = set(excluded[qid])
            order = sorted((j for j in range(len(cids)) if cids[j] not in ex),
                           key=lambda j: (float(scores[i, j]), cids[j]), reverse=True)
            ranking = [cids[j] for j in order]
            gold = qrels[qid]
            recall = len(set(ranking[:k]) & gold) / len(gold)
            short_ranking = [d for d in ranking if lengths[d] <= cap]
            n_long = sum(lengths[d] > cap for d in ranking)
            lo = len(set(short_ranking[:max(0, k - n_long)]) & gold) / len(gold)
            hi = len(set(short_ranking[:k]) & gold) / len(gold)
            best = min(j + 1 for j, d in enumerate(ranking) if d in gold)
            recalls.append(recall)
            lower.append(lo)
            upper.append(hi)
            best_ranks.append(best)
            top_ten.update(ranking[:10])
            per_query[qid] = {'recall_1000': recall, 'best_relevant_rank': best,
                              'top10': ranking[:10], 'length_only_lower': lo,
                              'length_only_upper': hi}
        result['models'][folder.name] = {
            'recall_1000': float(np.mean(recalls)),
            'length_only_interval': [float(np.mean(lower)), float(np.mean(upper))],
            'median_best_relevant_rank': float(np.median(best_ranks)),
            'mean_query_norm': float(np.linalg.norm(q, axis=1).mean()),
            'mean_corpus_norm': float(np.linalg.norm(c, axis=1).mean()),
            'corpus_centroid_norm': float(np.linalg.norm(c.mean(axis=0))),
            'top10_hubs': [{'docid': d, 'queries': n, 'text': corpus[d][:300]}
                           for d, n in top_ten.most_common(10)],
            'per_query': per_query,
        }
        print(folder.name, {k: v for k, v in result['models'][folder.name].items()
                            if k not in ('per_query', 'top10_hubs')}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
