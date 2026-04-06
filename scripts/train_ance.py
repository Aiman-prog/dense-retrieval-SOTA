import os
import sys
import math
import json
import argparse
import subprocess
import pickle
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from tevatron.retriever.modeling import DenseModel

# Hardware & Project Setup
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_path, get_training_context, load_config, \
                          encode_to_pickle, build_faiss_index, count_jsonl_examples
from data.preprocessor import BRIGHTPreprocessor
from evaluation.trec_eval_wrapper import TrecEvalWrapper

# 🩹 Tevatron Bug Patch
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)


def run_setup(recipe_args):
    """Build corpus/queries/qrels. Skips if the 3 core files already exist."""
    p            = get_path("processed")
    corpus_path  = p / recipe_args['corpus_file']
    queries_path = p / recipe_args['train_queries_file']
    qrels_path   = p / recipe_args['train_qrels_file']

    if all(x.exists() and x.stat().st_size > 0 for x in [corpus_path, queries_path, qrels_path]):
        print("⏩ Skipping setup: files already exist.", flush=True)
        return corpus_path, queries_path, qrels_path

    preprocessor = BRIGHTPreprocessor()

    if recipe_args['setup_mode'] == 'tevatron_msmarco':
        cache = str(get_path("bright"))
        if not corpus_path.exists() or corpus_path.stat().st_size == 0:
            preprocessor.prepare_msmarco_full_corpus(cache_dir=cache)
        mixture_path = p / recipe_args['mixture_dir'] / 'train_msmarco.jsonl'
        if not mixture_path.exists() or mixture_path.stat().st_size == 0:
            preprocessor.prepare_msmarco_tevatron_train(cache_dir=cache)
        if recipe_args.get('eval_queries_file'):
            eval_q = p / recipe_args['eval_queries_file']
            if not eval_q.exists() or eval_q.stat().st_size == 0:
                preprocessor.prepare_msmarco_dev(cache_dir=cache)

    else:  # reasonir_mixture: build from local training_mixture/ JSONL files
        mixture_dir = p / recipe_args['mixture_dir']
        mix_files = [f for f in mixture_dir.glob("*.jsonl") if not f.name.startswith('.')]

        mix_dfs = []
        for f in mix_files:
            df = pd.read_json(f, lines=True)
            if 'query_text' in df.columns:
                df = df.rename(columns={'query_text': 'query'})
            mix_dfs.append(df)
        mix_df = pd.concat(mix_dfs, ignore_index=True)

        # Corpus
        all_passages = []
        for col in ['positive_passages', 'negative_passages']:
            for record_list in mix_df[col]:
                all_passages.extend(record_list)
        corpus_df = (pd.DataFrame(all_passages)
                       .rename(columns={'docid': 'doc_id'})[['doc_id', 'text']]
                       .drop_duplicates(subset=['doc_id']))
        preprocessor.prepare_tevatron_corpus(corpus_df, filename=recipe_args['corpus_file'])
        print(f"Corpus: {len(corpus_df)} passages", flush=True)

        # Queries
        queries_df = mix_df[['query_id', 'query']].drop_duplicates(subset=['query_id'])
        preprocessor.prepare_tevatron_queries(queries_df, filename=recipe_args['train_queries_file'])

        # Qrels
        pos_pairs = []
        for _, row in mix_df.iterrows():
            for pos in row['positive_passages']:
                pos_pairs.append({'query_id': str(row['query_id']), 'doc_id': str(pos['docid']), 'relevance': 1})
        preprocessor.prepare_trec_qrels(pd.DataFrame(pos_pairs).drop_duplicates(),
                                        filename=recipe_args['train_qrels_file'])

    return corpus_path, queries_path, qrels_path


def _load_qrels(qrels_file):
    data = []
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                data.append({'qid': parts[0], 'did': parts[2]})
    return pd.DataFrame(data).groupby('qid')['did'].apply(set).to_dict() if data else {}


def _encode_and_mine_initial(ctx, config, corpus_file, query_file, corpus_lookup,
                              qrels_dict, initial_data_dir, base_model, mixture_dir):
    """Initial ANN mine using the base model so the Trainer has data from step 0."""
    print(f"[ANCE] Initial encode+mine using base model: {base_model}", flush=True)

    work_dir = initial_data_dir / "_work"
    work_dir.mkdir(exist_ok=True)
    encode_to_pickle(base_model, corpus_file, work_dir / "corpus.pkl", False, ctx, config)
    encode_to_pickle(base_model, query_file,  work_dir / "query.pkl",  True,  ctx, config)

    idx, _, c_ids = build_faiss_index(work_dir / "corpus.pkl")
    with open(work_dir / "query.pkl", 'rb') as f:
        q_data = pickle.load(f)

    mining_depth = ctx['args']['mining_depth']
    n_negs = ctx['args']['train_group_size'] - 1
    _, indices = idx.search(q_data[0].astype(np.float32), mining_depth)

    mined_negs = {}
    for i, qid in enumerate([str(x) for x in q_data[1]]):
        pot = [c_ids[j] for j in indices[i] if j >= 0]
        true_negs = [d for d in pot if d not in qrels_dict.get(qid, set())]
        candidates = true_negs if true_negs else pot
        if len(candidates) >= n_negs:
            mined_negs[qid] = candidates[:n_negs]
        else:
            mined_negs[qid] = (candidates * (n_negs // max(len(candidates), 1) + 1))[:n_negs]

    for f_path in mixture_dir.glob("*.jsonl"):
        if f_path.name.startswith('.'):
            continue
        with open(f_path) as f_in, open(initial_data_dir / f_path.name, 'w') as f_out:
            for line in f_in:
                d = json.loads(line)
                if str(d['query_id']) in mined_negs:
                    d['negative_passages'] = [
                        {"docid": nid, "text": corpus_lookup.get(nid, "")}
                        for nid in mined_negs[str(d['query_id'])]
                    ]
                f_out.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f"[ANCE] Initial data written to {initial_data_dir}", flush=True)


def _evaluate(ctx, config, model_path):
    """Evaluate final model. Single-file eval if eval_corpus_file set, else multi-domain."""
    args     = ctx['args']
    temp_dir = get_path(args['temp_workdir'])

    if args.get('eval_corpus_file'):
        # Single eval set (e.g. MS MARCO dev)
        p          = get_path("processed")
        d_corpus   = p / args['eval_corpus_file']
        d_queries  = p / args['eval_queries_file']
        d_qrels    = p / args['eval_qrels_file']

        if not all(x.exists() for x in [d_corpus, d_queries, d_qrels]):
            print("[Eval] Skipping: eval files not found", flush=True)
            return

        eval_dir = temp_dir / "final_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        encode_to_pickle(str(model_path), d_corpus,  eval_dir / "c.pkl", False, ctx, config)
        encode_to_pickle(str(model_path), d_queries, eval_dir / "q.pkl", True,  ctx, config)

        with open(eval_dir / "c.pkl", 'rb') as f: dc = pickle.load(f)
        with open(eval_dir / "q.pkl", 'rb') as f: dq = pickle.load(f)

        idx_e = faiss.IndexFlatIP(dc[0].shape[1])
        idx_e.add(dc[0].astype(np.float32))
        s_e, i_e = idx_e.search(dq[0].astype(np.float32), args.get('eval_top_k', 1000))

        results = {
            str(dq[1][j]): {
                str(dc[1][i_e[j][k]]): float(s_e[j][k])
                for k in range(len(i_e[j])) if i_e[j][k] >= 0
            }
            for j in range(len(dq[1]))
        }

        eval_qrels_data = []
        with open(d_qrels) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    eval_qrels_data.append({'query_id': parts[0], 'doc_id': parts[2], 'relevance': parts[3]})
        metric = args['eval_metric']
        evaluator = TrecEvalWrapper(pd.DataFrame(eval_qrels_data))
        metrics = evaluator.evaluate(results, {metric})
        print(f"\n📈 Eval — {metric}={metrics.get(metric, 0):.4f}", flush=True)

    else:
        # Multi-domain eval (BRIGHT)
        eval_summary = []
        for domain in config['evaluation'].get('eval_domains', []):
            d_corpus  = get_path("processed") / f"{domain}_corpus.jsonl"
            d_queries = get_path("processed") / f"{domain}_queries.jsonl"
            d_qrels   = get_path("processed") / f"{domain}_qrels.txt"

            if not all(p.exists() for p in [d_corpus, d_queries, d_qrels]):
                print(f"[Eval] Skipping {domain}: files not found", flush=True)
                continue

            eval_dir = temp_dir / "final_eval" / domain
            eval_dir.mkdir(parents=True, exist_ok=True)

            encode_to_pickle(str(model_path), d_corpus,  eval_dir / "c.pkl", False, ctx, config)
            encode_to_pickle(str(model_path), d_queries, eval_dir / "q.pkl", True,  ctx, config)

            with open(eval_dir / "c.pkl", 'rb') as f: dc = pickle.load(f)
            with open(eval_dir / "q.pkl", 'rb') as f: dq = pickle.load(f)

            idx_e = faiss.IndexFlatIP(dc[0].shape[1])
            idx_e.add(dc[0].astype(np.float32))
            eval_top_k = args.get('eval_top_k', 10)
            s_e, i_e = idx_e.search(dq[0].astype(np.float32), eval_top_k)

            results = {
                str(dq[1][j]): {
                    str(dc[1][i_e[j][k]]): float(s_e[j][k])
                    for k in range(len(i_e[j])) if i_e[j][k] >= 0
                }
                for j in range(len(dq[1]))
            }

            eval_qrels_data = []
            with open(d_qrels) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        eval_qrels_data.append({'query_id': parts[0], 'doc_id': parts[2], 'relevance': parts[3]})
            evaluator = TrecEvalWrapper(pd.DataFrame(eval_qrels_data))
            metrics = evaluator.evaluate(results, {'recip_rank', 'ndcg_cut_10'})
            eval_summary.append({'domain': domain, 'ndcg10': metrics.get('ndcg_cut_10', 0)})
            print(f"[Eval] {domain}: NDCG@10={metrics.get('ndcg_cut_10', 0):.4f}", flush=True)

        if eval_summary:
            mean_ndcg = pd.DataFrame(eval_summary)['ndcg10'].mean()
            print(f"\n📈 Final Mean NDCG@10: {mean_ndcg:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', default='ance')
    recipe = parser.parse_args().recipe

    ctx    = get_training_context(recipe)
    config = load_config()
    corpus_file, query_file, qrels_file = run_setup(ctx['args'])

    # Detect GPU count BEFORE restricting visibility.
    # With --gpus-per-task=2, SLURM sets CUDA_VISIBLE_DEVICES=0,1.
    # Tevatron encode raises NotImplementedError on multi-GPU, so we pin the
    # orchestrator to GPU 0 for all encode_to_pickle calls (initial mine, eval).
    # Inferencer/Trainer subprocesses override this with their own assignments.
    import torch as _torch
    n_gpus = _torch.cuda.device_count()
    infer_gpu = '1' if n_gpus >= 2 else '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"[ANCE] {n_gpus} GPU(s) detected — Trainer→GPU 0, Inferencer→GPU {infer_gpu}", flush=True)

    temp_workdir     = get_path(ctx['args']['temp_workdir'])
    mixture_dir      = get_path("processed") / ctx['args']['mixture_dir']
    ann_dir          = temp_workdir / "ann_data"
    initial_data_dir = temp_workdir / "initial_data"
    ann_dir.mkdir(exist_ok=True, parents=True)
    initial_data_dir.mkdir(exist_ok=True)

    # Total training steps
    n_examples = count_jsonl_examples(str(mixture_dir / "*.jsonl"))
    if n_examples == 0:
        raise RuntimeError(f"No training examples found in {mixture_dir}. Run preprocessing first.")
    steps_per_epoch = math.ceil(n_examples / ctx['args']['batch_size'])
    total_epochs    = ctx['args']['total_epochs']
    max_steps       = steps_per_epoch * total_epochs
    print(f"[ANCE] {n_examples} examples | {steps_per_epoch} steps/epoch | "
          f"{total_epochs} epochs | {max_steps} total steps", flush=True)

    ance_base_model = ctx['args'].get('base_model', ctx['base_model'])
    print(f"[ANCE] Starting from model: {ance_base_model}", flush=True)

    output_model_dir = get_path("models") / ctx['args']['model_name']

    # ── INITIAL ENCODE + MINE (once, before training starts) ─────────────────
    existing_jsonl = list(initial_data_dir.glob("*.jsonl"))
    if existing_jsonl:
        print(f"[ANCE] Skipping initial mine: {len(existing_jsonl)} JSONL files already in "
              f"{initial_data_dir}", flush=True)
    else:
        corpus_lookup = {}
        with open(corpus_file) as f:
            for line in f:
                d = json.loads(line)
                corpus_lookup[d['docid']] = d['text']
        qrels_dict = _load_qrels(qrels_file)
        _encode_and_mine_initial(ctx, config, corpus_file, query_file,
                                  corpus_lookup, qrels_dict, initial_data_dir,
                                  ance_base_model, mixture_dir)

    # ── LAUNCH INFERENCER (background, never blocks) ──────────────────────────
    infer_env  = {**os.environ, 'CUDA_VISIBLE_DEVICES': infer_gpu}
    infer_proc = subprocess.Popen([
        sys.executable, str(Path(__file__).parent / "run_ance_data_gen.py"),
        '--output_model_dir', str(output_model_dir),
        '--ann_dir',          str(ann_dir),
        '--corpus_file',      str(corpus_file),
        '--query_file',       str(query_file),
        '--qrels_file',       str(qrels_file),
        '--recipe',           recipe,
    ], env=infer_env)
    print(f"[ANCE] Inferencer started on GPU {infer_gpu} (pid {infer_proc.pid})", flush=True)

    # ── LAUNCH TRAINER on GPU 0 (foreground — blocks until training completes) ─
    train_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
    try:
        subprocess.run([
            sys.executable, str(Path(__file__).parent / "run_ance_train.py"),
            '--model_name_or_path', ance_base_model,
            '--initial_data_dir',   str(initial_data_dir),
            '--ann_dir',            str(ann_dir),
            '--output_dir',         str(output_model_dir),
            '--max_steps',          str(max_steps),
        ], env=train_env, check=True)
    finally:
        infer_proc.terminate()
        infer_proc.wait()
        print("[ANCE] Inferencer terminated.", flush=True)

    # ── EVALUATE (final model only) ───────────────────────────────────────────
    _evaluate(ctx, config, output_model_dir)


if __name__ == "__main__":
    main()
