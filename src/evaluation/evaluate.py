"""
BRIGHT evaluation of one domain, MaxP (utils/maxp.py): each document is its first
model.max_passages passages and ranks by its best one. FirstP, from each document's
passage 0, is recorded alongside as the control.
LOGIC: Uses standard FAISS CPU indexing.
DIAGNOSTICS: Kept prints to monitor environment status in .out file.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import json
import datetime
import faiss
import torch
from transformers import AutoTokenizer
from tevatron.retriever.modeling import DenseModel

# Hardware Fix
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / 'src'))
from utils.eval_attempt import configure_cache, identity_digest, model_identity
if __name__ == '__main__':
    configure_cache()

from utils.helpers import (load_config, get_data_base_dir, load_excluded_ids,
                           search_depth, apply_exclusions, model_run_tag,
                           _load_qrels, require_eval_files, check_eval_artifacts,
                           atomic_write, eval_artifact_hashes, _code_revision)
from evaluation.trec_eval_wrapper import TrecEvalWrapper
from utils.maxp import doc_text, split_passages, encode_ids, collapse_to_docs

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--domain", type=str, default="biology")
    parser.add_argument('--attempt_manifest', type=Path)
    args = parser.parse_args()

    # Read k and batch_size from config.yaml
    args.k = config['evaluation'].get('top_k', 1000)
    args.batch_size = config['evaluation'].get('batch_size', 128)
    args.dataloader_num_workers = config['evaluation'].get('dataloader_num_workers', 4)
    args.bf16 = config['evaluation'].get('bf16', True)
    # --- HARDWARE DIAGNOSTIC ---
    # Kept this so you can verify the environment in the .out log
    print("\n" + "="*40, flush=True)
    print("🔍 HARDWARE DIAGNOSTIC", flush=True)
    print("="*40, flush=True)
    
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version:  {torch.__version__}", flush=True)
    print(f"CUDA Available:   {cuda_available}", flush=True)
    if cuda_available:
        print(f"GPU Device:       {torch.cuda.get_device_name(0)}", flush=True)
    
    print(f"FAISS Version:    {faiss.__version__}", flush=True)
    print("="*40 + "\n", flush=True)

    # --- PATH SETUP ---
    base_dir = Path(get_data_base_dir())
    processed_dir = base_dir / config['paths']['processed_dir']
    # Key the embedding cache by model AND domain so concurrent evals of different
    # models can't clobber each other's corpus.pkl/query.pkl (domain-only paths race).
    # The tag hashes the ABSOLUTE model path: two runs can both end in
    # `checkpoint-500`, and a basename-only tag let them share this directory.
    model_tag = model_run_tag(args.model_path)
    attempt_id = configure_cache()
    eval_dir = base_dir / 'data' / 'evaluation' / model_tag / attempt_id / args.domain
    eval_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = processed_dir / f"{args.domain}_corpus.jsonl"
    queries_file = processed_dir / f"{args.domain}_queries.jsonl"
    qrels_file = processed_dir / f"{args.domain}_qrels.txt"

    # Preflight before any encoding: a missing corpus otherwise surfaces as an opaque
    # failure an hour of GPU time later.
    require_eval_files(args.domain, [
        corpus_file, queries_file, qrels_file,
        processed_dir / f"{args.domain}_excluded.json"])
    if args.attempt_manifest:
        attempt = json.loads(args.attempt_manifest.read_text())
        if attempt['attempt_id'] != attempt_id:
            raise ValueError('Evaluation attempt does not match inherited environment')
        if (attempt['identity']['model_config'] != config['model'] or
                attempt['identity']['evaluation_config'] != config['evaluation']):
            raise ValueError('Evaluation configuration changed after attempt started')
        results_base = args.attempt_manifest.parent
    else:
        attempt = {'attempt_id': attempt_id, 'identity': {
            'model': model_identity(args.model_path), 'model_config': config['model'],
            'evaluation_config': config['evaluation'],
            'artifacts': eval_artifact_hashes(processed_dir, [args.domain]),
            'evaluator_revision': _code_revision()}}
        results_base = base_dir / config['paths']['results_dir'] / model_tag / attempt_id
    
    # --- Detect LoRA checkpoint ---
    is_lora = (Path(args.model_path) / "adapter_config.json").exists()
    if is_lora:
        from utils.helpers import get_training_context
        ctx = get_training_context("crossbatch")
        encode_model_path = ctx['base_model']
        lora_adapter_path = args.model_path
        print(f"🔗 LoRA adapter detected. Base model: {encode_model_path}", flush=True)
    else:
        encode_model_path = args.model_path
        lora_adapter_path = None

    # --- STEP 1: SPLIT + ENCODE (MaxP, utils/maxp.py) ---
    # Every document is its first max_passages passages, tokenized once and sliced on
    # tokens, each with its own special tokens -- the passages train_maxp.py trains on.
    # Encoded in-process: Tevatron's encode driver takes text, and re-tokenizing a
    # sliced string shifts its edges.
    p_len, max_p = config['model']['passage_max_len'], config['model']['max_passages']
    q_len = config['model']['query_max_len']
    print(f"🚀 Step 1: MaxP encoding, {max_p} x {p_len}-token passages per document...",
          flush=True)
    device = torch.device('cuda' if cuda_available else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(encode_model_path)
    model = DenseModel.load(
        encode_model_path, pooling=config['model'].get('pooling', 'cls'),
        normalize=config['model'].get('normalize', False),
        lora_name_or_path=lora_adapter_path, attn_implementation='eager',
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device).eval()

    with open(corpus_file) as f:
        docs = [json.loads(line) for line in f]
    passages, parents, firsts, n_full = [], [], [], 0
    for start in range(0, len(docs), 1024):
        batch = docs[start:start + 1024]
        for doc, ps in zip(batch, split_passages(tokenizer, [doc_text(d) for d in batch],
                                                 p_len, max_p)):
            firsts.append(len(passages))
            passages.extend(ps)
            parents.extend([str(doc['docid'])] * len(ps))
            n_full += len(ps) < max_p or len(ps[-1]) < p_len
    corpus_ids = [str(d['docid']) for d in docs]
    print(f"   {len(docs):,} documents -> {len(passages):,} passages; "
          f"{100 * n_full / len(docs):.2f}% of documents fully inside {max_p} passages",
          flush=True)
    corpus_embs = encode_ids(model, tokenizer, passages, args.batch_size, device)

    with open(queries_file) as f:
        queries = [json.loads(line) for line in f]
    query_ids = [str(q['query_id']) for q in queries]
    query_tok = tokenizer([q.get('query_text', q.get('query', '')) for q in queries],
                          truncation=True, max_length=q_len)['input_ids']
    query_embs = encode_ids(model, tokenizer, [np.asarray(t, dtype=np.int32) for t in query_tok],
                            args.batch_size, device)
    del model
    torch.cuda.empty_cache()

    # --- STEP 2: FAISS SEARCH (Standard CPU) ---
    # BRIGHT removes each query's excluded documents before scoring, so retrieve
    # deeper than k and filter afterwards -- otherwise excluded hits consume result
    # slots and nothing refills them (aops excludes up to 11,224 docs for one query).
    excluded = load_excluded_ids(args.domain, processed_dir)
    qrels = _load_qrels(qrels_file)
    # Query ids == exclusion keys, qrels covered by the query set, and the encoder
    # neither dropped nor invented a query. Any of those would move the metric
    # without moving anything visible.
    check_eval_artifacts(args.domain, qrels, excluded, queries_file=queries_file,
                         encoded_query_ids=query_ids)
    depth = min(search_depth(args.k, excluded), len(corpus_ids))
    print(f"🔍 Step 2: FAISS IndexFlatIP (CPU), {max_p * depth:,} passages deep "
          f"for {depth:,} documents, top-{args.k}...", flush=True)

    def evaluate_run(embs, rows_parent, search_k):
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        scores_mat, indices_mat = index.search(query_embs, min(search_k, len(embs)))
        run = {qid: collapse_to_docs(rows, scores, rows_parent,
                                     search_depth(args.k, excluded, qid))
               for qid, rows, scores in zip(query_ids, indices_mat, scores_mat)}
        return TrecEvalWrapper(qrels).evaluate(apply_exclusions(run, excluded, args.k),
                                               {'recip_rank', 'ndcg_cut_10', 'recall_1000'})

    # --- STEP 3: EVALUATION ---
    # MaxP is the reported run. FirstP reuses each document's passage 0 -- the exact
    # FirstP-512 input -- as the control, with no second encoding pass.
    print("📊 Step 3: Running Evaluation Wrapper...", flush=True)
    metrics = evaluate_run(corpus_embs, parents, max_p * depth)
    firstp_metrics = evaluate_run(corpus_embs[firsts], corpus_ids, depth)

    print(f"\nFINAL RESULTS: {args.domain}\n" + "*"*40, flush=True)
    for name, m in (('MaxP', metrics), ('FirstP', firstp_metrics)):
        print(f"{name:<7} MRR {m.get('recip_rank', 0):.4f} | NDCG@10 "
              f"{m.get('ndcg_cut_10', 0):.4f} | Recall@1000 {m.get('recall_1000', 0):.4f}",
              flush=True)
    print("*"*40, flush=True)

    # Save results to JSON for downstream aggregation
    results_base.mkdir(parents=True, exist_ok=True)
    result_file = results_base / f"{args.domain}_results.json"
    # model_path is resolved and run_tag recorded so a collector can prove this file
    # belongs to the run it asked for, not to another model of the same basename.
    result_data = {
        "domain": args.domain,
        "model_path": str(Path(args.model_path).resolve()),
        "run_tag": model_tag,
        "metrics": metrics,
        "firstp_metrics": firstp_metrics,
    }
    result_data.update(attempt_id=attempt_id,
                       identity_sha256=identity_digest(attempt['identity']),
                       completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with atomic_write(result_file) as f:
        json.dump(result_data, f, indent=2)
    print(f"📄 Results saved to: {result_file}", flush=True)

if __name__ == "__main__":
    main()
