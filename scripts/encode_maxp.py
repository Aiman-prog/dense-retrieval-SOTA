"""MaxP encode child: a JSONL corpus or query file -> one pickle, for ANCE mining.

Spawned by `helpers.encode_to_pickle(..., maxp=True)` with the same environment and
timeout as the Tevatron encode child it replaces for the BGE arm. Documents become up
to `max_passages` passage vectors (`utils.maxp`); queries one vector each.

    corpus pickle: (embs [n_passages, D], parent docids, passage ordinals, contract)
    query  pickle: (embs [n_queries, D],  query ids,     None,              contract)

`helpers.build_faiss_index` reads elements 0 and 1, so the index rows carry PARENT ids.
"""
import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import load_config
from utils.maxp import doc_text, split_passages, encode_ids, representation_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_pkl', required=True)
    parser.add_argument('--is_query', action='store_true')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--data_sha256', default=None)
    args = parser.parse_args()

    config = load_config()
    contract = representation_contract(config, args.model_path, data_sha256=args.data_sha256)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    encoder = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
                                        attn_implementation='eager').to(device).eval()
    with open(args.input_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.is_query:
        ids = [str(r['query_id']) for r in rows]
        tok = tokenizer([r.get('query_text', r.get('query', '')) for r in rows],
                        truncation=True, max_length=contract['query_max_len'])['input_ids']
        embs = encode_ids(encoder, tokenizer, [np.asarray(t, dtype=np.int32) for t in tok],
                          args.batch_size, device).numpy()
        payload = (embs, ids, None, contract)
    else:
        passages, parents, ordinals = [], [], []
        for start in range(0, len(rows), 1024):
            batch = rows[start:start + 1024]
            for row, ps in zip(batch, split_passages(
                    tokenizer, [doc_text(r) for r in batch],
                    contract['passage_max_len'], contract['max_passages'])):
                passages.extend(ps)
                parents.extend([str(row['docid'])] * len(ps))
                ordinals.extend(range(len(ps)))
        embs = encode_ids(encoder, tokenizer, passages, args.batch_size, device).numpy()
        payload = (embs, parents, ordinals, contract)
    tmp = Path(args.output_pkl).with_suffix('.pkl.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp, args.output_pkl)          # a reader never sees a partial pickle
    print(f"[encode_maxp] {len(payload[1]):,} vectors -> {args.output_pkl}", flush=True)


if __name__ == "__main__":
    main()
