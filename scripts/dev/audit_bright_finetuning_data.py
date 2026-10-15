"""Read-only local audit of original/split mixtures and Pony; no model inference.

Pass local copies, never run a data scan on a cluster login node. Tokenizer JSON
is loaded directly with tokenizers, without network or transformers side effects.
"""
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path


def records(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def digest(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()


def describe(values, cap):
    values = sorted(values)
    return {
        'n': len(values), 'mean': sum(values) / len(values),
        'p50': values[len(values) // 2], 'p90': values[int(len(values) * .9)],
        'p99': values[int(len(values) * .99)], 'max': values[-1],
        'above_cap': sum(v > cap for v in values),
        'tokens_retained_at_cap': sum(min(v, cap) for v in values) / sum(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tokenizer-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    config_path = Path(__file__).resolve().parents[2] / 'config/config.yaml'
    import yaml
    config = yaml.safe_load(config_path.read_text())
    cap = config['model']['passage_max_len']
    pony = {d['docid']: d['text'] for d in records(args.root / 'pony/pony_corpus.jsonl')}
    relevant = {line.split()[2] for line in (args.root / 'pony/pony_qrels.txt').open()}
    pony_train = collections.Counter()
    report = {'tokenizer_sha256': hashlib.sha256(args.tokenizer_json.read_bytes()).hexdigest(),
              'passage_cap': cap, 'files': {}, 'original': {}, 'split': {}}
    for kind in ('original', 'split'):
        for path in sorted((args.root / kind).glob('train_*.jsonl')):
            groups = collections.Counter()
            positives = collections.Counter()
            lengths = {key: [] for key in ('query', 'positive', 'negative')}
            raw_sha = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    raw_sha.update(block)
            report['files'][f'{kind}/{path.name}'] = raw_sha.hexdigest()
            iterator = records(path)
            while batch := list(itertools.islice(iterator, 128)):
                for d in batch:
                    groups[d['query_id'].split('__c')[0]] += 1
                    p = d['positive_passages'][0]
                    positives[digest(p['text'])] += 1
                    if kind == 'original' and p['docid'] in pony:
                        pony_train[p['docid']] += 1
                for field, texts in (
                    ('query', [d['query'] for d in batch]),
                    ('positive', [d['positive_passages'][0]['text'] for d in batch]),
                    ('negative', [d['negative_passages'][0]['text'] for d in batch]),
                ):
                    lengths[field].extend(len(e.ids) for e in tokenizer.encode_batch(texts))
            n = sum(groups.values())
            report[kind][path.name] = {
                'rows': n, 'parents': len(groups),
                'split_parents': sum(v > 1 for v in groups.values()),
                'rows_from_split_parents': sum(v for v in groups.values() if v > 1),
                'largest_groups': groups.most_common(10),
                'same_parent_pairs': sum(v * (v - 1) // 2 for v in groups.values()),
                'identical_positive_pairs': sum(v * (v - 1) // 2 for v in positives.values()),
                'lengths': {k: describe(v, cap) for k, v in lengths.items()},
            }
            print(kind, path.name, json.dumps(report[kind][path.name]), flush=True)
    # Verify the actual export, including which fields were replicated.
    checks = collections.Counter()
    examples = []
    for original_path in sorted((args.root / 'original').glob('train_*.jsonl')):
        split_path = args.root / 'split' / original_path.name
        group_iterator = itertools.groupby(records(split_path),
                                           lambda d: d['query_id'].split('__c')[0])
        for original, grouped in itertools.zip_longest(records(original_path), group_iterator):
            assert original is not None and grouped is not None
            qid, group = grouped
            group = list(group)
            assert qid == original['query_id'], (qid, original['query_id'])
            checks['parent_rows_checked'] += 1
            checks['changed_query_groups'] += any(d['query'] != original['query'] for d in group)
            checks['changed_negative_groups'] += any(
                d['negative_passages'] != original['negative_passages'] for d in group)
            joined = ''.join(d['positive_passages'][0]['text'] for d in group)
            if joined != original['positive_passages'][0]['text']:
                checks['nonidentical_reconstruction'] += 1
                checks['nonidentical_after_whitespace_removal'] += (
                    ''.join(joined.split()) != ''.join(original['positive_passages'][0]['text'].split()))
            if len(group) >= 8 and len(examples) < 3:
                examples.append({'query_id': qid, 'query': original['query'],
                                 'chunks': [d['positive_passages'][0]['text'] for d in group]})
    pony_lengths = {k: len(tokenizer.encode(v).ids) for k, v in pony.items()}
    report['transform_checks'] = dict(checks)
    report['long_positive_examples'] = examples
    report['pony'] = {
        'corpus': describe(list(pony_lengths.values()), cap),
        'relevant': describe([pony_lengths[k] for k in relevant], cap),
        'training_rows': sum(pony_train.values()), 'training_unique_docs': len(pony_train),
        'training_relevant_docs': len(set(pony_train) & relevant),
        'training_relevant_rows': sum(pony_train[k] for k in relevant),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
