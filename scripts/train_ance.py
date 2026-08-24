import os
import sys
import math
import json
import random
import argparse
import subprocess
import pickle
import shutil
import numpy as np
import faiss
from pathlib import Path
from tevatron.retriever.modeling import DenseModel

# Hardware & Project Setup
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_path, get_training_context, load_config, \
                          encode_to_pickle, build_faiss_index, count_jsonl_examples, \
                          _load_qrels, evaluate_bright, log_startup_config
from data.preprocessor import (BRIGHTPreprocessor, MIXTURE_FILES,
                               MSMARCO_ONLY_FILES, require_derived_artifacts,
                               require_mixture_files)

# 🩹 Tevatron Bug Patch
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)


def run_setup(recipe_args):
    """Resolve corpus/queries/qrels for the recipe.

    The reasonir_mixture path only *verifies* -- those files are built by
    `python src/data/preprocessor.py`, so training never regenerates its own inputs.
    """
    p = get_path("processed")

    if recipe_args['setup_mode'] == 'tevatron_msmarco':
        corpus_path  = p / recipe_args['corpus_file']
        queries_path = p / recipe_args['train_queries_file']
        qrels_path   = p / recipe_args['train_qrels_file']
        mixture_path = p / recipe_args['mixture_dir'] / MSMARCO_ONLY_FILES[0]
        train_set = (mixture_path, queries_path, qrels_path)
        if all(x.exists() and x.stat().st_size > 0 for x in train_set) and \
                corpus_path.exists() and corpus_path.stat().st_size > 0:
            print("⏩ Skipping setup: files already exist.", flush=True)
            require_mixture_files(mixture_path.parent, MSMARCO_ONLY_FILES)
            return corpus_path, queries_path, qrels_path

        preprocessor = BRIGHTPreprocessor(output_dir=p)
        cache = str(get_path("bright"))
        if not corpus_path.exists() or corpus_path.stat().st_size == 0:
            preprocessor.prepare_msmarco_full_corpus(cache_dir=cache)
        if not all(x.exists() and x.stat().st_size > 0 for x in train_set):
            preprocessor.prepare_msmarco_tevatron_train(
                cache_dir=cache,
                mixture_filename=f"{recipe_args['mixture_dir']}/{MSMARCO_ONLY_FILES[0]}",
                queries_filename=recipe_args['train_queries_file'],
                qrels_filename=recipe_args['train_qrels_file'])
        if recipe_args.get('eval_queries_file'):
            eval_q = p / recipe_args['eval_queries_file']
            if not eval_q.exists() or eval_q.stat().st_size == 0:
                preprocessor.prepare_msmarco_dev(cache_dir=cache)
        require_mixture_files(mixture_path.parent, MSMARCO_ONLY_FILES)
        return require_derived_artifacts(
            output_dir=p, corpus_file=recipe_args['corpus_file'],
            queries_file=recipe_args['train_queries_file'],
            qrels_file=recipe_args['train_qrels_file'])

    require_mixture_files(p / recipe_args['mixture_dir'], MIXTURE_FILES)
    return require_derived_artifacts(
        output_dir=p,
        corpus_file=recipe_args['corpus_file'],
        queries_file=recipe_args['train_queries_file'],
        qrels_file=recipe_args['train_qrels_file'],
    )


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
        pool = candidates[:mining_depth]
        if len(pool) >= n_negs:
            mined_negs[qid] = random.sample(pool, n_negs)
        else:
            mined_negs[qid] = (pool * (n_negs // max(len(pool), 1) + 1))[:n_negs]

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', default='ance')
    recipe = parser.parse_args().recipe

    ctx    = get_training_context(recipe)
    config = load_config()
    log_startup_config(recipe, ctx)
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

    # Remove checkpoints from any previous run. get_last_checkpoint() returns the
    # highest step-numbered checkpoint in the directory, so a stale checkpoint-17202
    # from a prior run will permanently shadow all new checkpoint-500/1000/... saves,
    # keeping the inferencer stuck forever on the old weights.
    stale = sorted(output_model_dir.glob("checkpoint-*"))
    if stale:
        for ckpt in stale:
            shutil.rmtree(ckpt, ignore_errors=True)
        print(f"[ANCE] Removed {len(stale)} stale checkpoint(s) from {output_model_dir.name}", flush=True)

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
            '--recipe',             recipe,
        ], env=train_env, check=True)
    finally:
        infer_proc.terminate()
        infer_proc.wait()
        print("[ANCE] Inferencer terminated.", flush=True)

    # ── EVALUATE (final model only) ───────────────────────────────────────────
    evaluate_bright(ctx, config, output_model_dir)


if __name__ == "__main__":
    main()
