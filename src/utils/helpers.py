"""Helper utility functions for Path and Context Management."""

import sys
import subprocess
import pickle
import json
import math
import yaml
import os
import contextlib
import tempfile
import numpy as np
import faiss
import torch
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml"):
    """
    Finds the project root and loads the config file.
    """
    # 1. Get the directory where THIS file (helpers.py) lives
    # 2. Go up two levels to reach the project root (src/utils -> project_root)
    project_root = Path(__file__).resolve().parent.parent.parent
    
    full_path = project_root / config_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"❌ Config not found at {full_path}. Check your folder structure!")
        
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)

def get_data_base_dir() -> Path:
    """Get base directory for all data, returning a Path object."""
    if 'DATA_BASE_DIR' in os.environ:
        return Path(os.environ['DATA_BASE_DIR'])
    
    user = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
    return Path(f'/scratch/{user}/dense-retrieval-SOTA')

def get_path(key: str, model_name: str = None) -> Path:
    """
    Centralized path resolver.
    Example: get_path('processed') -> /scratch/user/.../data/processed
    """
    config = load_config()
    base = get_data_base_dir()
    p_cfg = config['paths']
    
    path_map = {
        "base": base,
        "data": base / p_cfg['data_dir'],
        "processed": base / p_cfg['processed_dir'],
        "bright": base / p_cfg['bright_cache'],
        "models": base / p_cfg['models_dir'],
        "results": base / p_cfg['results_dir'],
        "temp_ance": base / "temp_ance_workdir",
        # ance_msmarco sets temp_workdir: "temp_ance_msmarco"; without this key
        # get_path returned None and train_ance.py:161 raised TypeError on
        # `None / "ann_data"` seconds into the job.
        "temp_ance_msmarco": base / "temp_ance_msmarco_workdir",
        "temp_grass": base / "temp_grass_workdir",
        # async Fast-GRASS handoff root: temp_fast_grass_workdir/async_mining/
        "temp_fast_grass": base / "temp_fast_grass_workdir",
    }
    
    if model_name:
        return path_map["models"] / model_name
    return path_map.get(key)

def get_training_context(training_type: str = "inbatch") -> Dict[str, Any]:
    config = load_config()
    recipe = config['training'][training_type]
    model_name = recipe.get('base_model') or config['model']['base_model']
    
    # Force absolute path resolution
    cache_base = get_path("bright").resolve() / "hub"
    repo_id = model_name.replace("/", "--")
    snapshot_dir = cache_base / f"models--{repo_id}" / "snapshots"
    
    final_base_model = model_name # Default fallback

    if snapshot_dir.exists():
        # Filter out hidden files and get actual directories
        snapshots = [d for d in snapshot_dir.iterdir() if d.is_dir()]
        if snapshots:
            # Sort to get the most recent or consistent one
            chosen_snapshot = sorted(snapshots)[-1]
            # Check if config.json is there (exists() or is_symlink() for HF cache)
            cfg = chosen_snapshot / "config.json"
            if cfg.exists() or cfg.is_symlink():
                final_base_model = str(chosen_snapshot)

    return {
        "args": recipe,
        "base_model": final_base_model,
        "max_q": config['model']['query_max_len'],
        "max_p": config['model']['passage_max_len'],
        "pooling": config['model'].get('pooling', 'cls'),
        "normalize": config['model'].get('normalize', False),
        "temperature": config['model'].get('temperature', 0.02),
        "processed_dir": get_path("processed"),
        "output_dir": get_path("models", recipe['model_name']),
        "cache_dir": str(get_path("bright").resolve())
    }


def log_startup_config(recipe_name: str, ctx: Dict[str, Any], recipe: Dict[str, Any] = None):
    """Print the resolved training configuration before training begins.

    ``base_model`` is the value AFTER get_training_context()'s HF-snapshot
    resolution, which falls back to the raw configured string when no snapshot
    directory holds a config.json. That fallback is silent, so a run can train
    cleanly against the wrong weights; this block is what catches it. The four
    recipes that train from /scratch/.../models/inbatch_mixed_bge_m3 are the
    ones at risk, hence the explicit on-disk existence check.

    ``recipe`` defaults to ctx['args']; callers that apply CLI overrides pass
    their effective config so the block reports what will actually run.
    """
    args = recipe if recipe is not None else (ctx.get('args') or {})
    resolved = ctx.get('base_model')

    try:
        configured = args.get('base_model') or load_config()['model']['base_model']
    except Exception:
        configured = None

    if configured is not None and resolved != configured:
        source = f"HF snapshot resolved from {configured!r}"
    else:
        source = "as configured (no HF snapshot dir with config.json)"
    if isinstance(resolved, str) and resolved.startswith("/") and not Path(resolved).exists():
        source += "  [PATH DOES NOT EXIST]"

    # Recipes do not share one spelling: crossbatch has no batch_size (it sizes by
    # per_device_batch_size / target_batch_size) and the ANCE recipes use
    # total_epochs rather than num_epochs. Report the key that is actually present
    # rather than inventing one.
    def first_present(*keys):
        for key in keys:
            if args.get(key) is not None:
                return key, args[key]
        return keys[0], None

    batch_key, batch_value = first_present("batch_size", "per_device_batch_size",
                                           "target_batch_size")
    epoch_key, epoch_value = first_present("num_epochs", "total_epochs")

    rows = [
        ("recipe", recipe_name),
        ("base_model", resolved),
        ("base_model source", source),
        ("temperature", ctx.get('temperature')),
        ("query_max_len", ctx.get('max_q')),
        ("passage_max_len", ctx.get('max_p')),
        (batch_key, batch_value),
        ("learning_rate", args.get('learning_rate')),
        (epoch_key, epoch_value),
    ]
    print("=" * 66, flush=True)
    print("RESOLVED TRAINING CONFIG", flush=True)
    for label, value in rows:
        print(f"  {label:<21}: {'<absent>' if value is None else value}", flush=True)
    print("=" * 66, flush=True)



def encode_to_pickle(model_path, input_file, output_pkl, is_query, ctx, config):
    """Run Tevatron encode subprocess and save embeddings to a pickle file."""
    cmd = [
        sys.executable, '-m', 'tevatron.retriever.driver.encode',
        '--output_dir', str(output_pkl.parent),
        '--model_name_or_path', model_path,
        '--bf16', 'True', '--fp16', 'False',
        '--per_device_eval_batch_size', str(ctx['args']['per_device_eval_batch_size']),
        '--dataset_name', 'json', '--dataset_path', str(input_file),
        '--encode_output_path', str(output_pkl),
        '--attn_implementation', 'eager',
        '--dataloader_num_workers', str(ctx['args']['dataloader_num_workers']),
        '--pooling', ctx['pooling'],
        '--normalize', str(ctx['normalize']),
    ]
    if is_query:
        q_len = str(config['model'].get('query_max_len', 128))
        try:
            subprocess.run(cmd + ['--encode_is_query', '--query_max_len', q_len], check=True)
        except subprocess.CalledProcessError:
            subprocess.run(cmd + ['--encode_is_qry', '--q_max_len', q_len], check=True)
    else:
        subprocess.run(cmd + ['--passage_max_len', str(config['model'].get('passage_max_len', 512))], check=True)


def build_faiss_index(corpus_pkl_path):
    """Load corpus pickle and build a FAISS IndexFlatIP. Returns (index, embeddings, ids)."""
    with open(corpus_pkl_path, 'rb') as f:
        c_data = pickle.load(f)
    embeddings = c_data[0].astype(np.float32)
    ids = [str(x) for x in c_data[1]]
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, ids


def patch_tevatron_loss(temperature):
    """Monkey-patch Tevatron's GradCache trainer to use temperature-scaled contrastive loss."""
    from models.temperature_scaled_loss import (
        TemperatureScaledContrastiveLoss,
        DistributedTemperatureScaledContrastiveLoss,
    )
    import tevatron.retriever.gc_trainer as gc_module

    class SimpleContrastiveLossPatched(TemperatureScaledContrastiveLoss):
        def __init__(self):
            super().__init__(temperature=temperature)

    class DistributedContrastiveLossPatched(DistributedTemperatureScaledContrastiveLoss):
        def __init__(self, n_target=0, scale_loss=True):
            super().__init__(temperature=temperature, n_target=n_target, scale_loss=scale_loss)

    gc_module.SimpleContrastiveLoss = SimpleContrastiveLossPatched
    gc_module.DistributedContrastiveLoss = DistributedContrastiveLossPatched


def set_seed(seed: int):
    import random as _random
    import numpy as _np
    import torch as _torch
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)


def encode_batch_tensor(model, tokenizer, texts, device, max_len, batch_size,
                        *, requires_grad, autocast_enabled=True):
    """
    Single in-process encoder for GRASS. Encodes texts in mini-batches into
    L2-normalised CLS embeddings, returned as a torch.Tensor on `device`.

    CLS pooling (last_hidden_state[:, 0, :]) + L2 normalize, with bf16 autocast
    on CUDA (no-op on CPU so tests run locally).

      requires_grad=False — forward under torch.no_grad(); deterministic mining
                            encodes (Algorithm 2 lines 1–3, 6–13).
      requires_grad=True  — gradients flow; the Algorithm 1 training step.

    See encode_batch() for the CPU float32 numpy variant used for FAISS/scoring.
    """
    if not texts:
        # Empty pool — return a well-formed empty tensor instead of crashing in
        # torch.cat. dim from model.config when available (real AutoModel), else
        # 0 (mocks). Callers already guard, this just hardens the helper.
        dim = getattr(getattr(model, 'config', None), 'hidden_size', 0)
        return torch.zeros((0, dim), device=device)
    use_autocast = autocast_enabled and device.type == 'cuda'
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors='pt').to(device)
        grad_ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()
        with grad_ctx, torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                      enabled=use_autocast):
            out = model(**inputs)
        embs = out.last_hidden_state[:, 0, :]
        embs = torch.nn.functional.normalize(embs, dim=-1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


@contextlib.contextmanager
def dropout_only(model):
    """Put ``model`` in eval mode but re-enable ONLY ``nn.Dropout`` modules.

    Used by cached-MCDP: "frozen for the round" means no parameter updates and no
    gradients, NOT dropout-off (async_fast_grass_implementation_details.md, "Miner
    Loop"). A plain ``model.train()`` would also switch on any other stateful
    training-mode module (BatchNorm running stats, etc.), which the miner must not do.

    Every module's entry mode is captured and restored on exit, including on
    exception.
    """
    entry_modes = {mod: mod.training for mod in model.modules()}
    try:
        model.eval()
        for mod in model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.train(True)
        yield model
    finally:
        for mod, was_training in entry_modes.items():
            mod.train(was_training)


def encode_mc(model, tokenizer, texts, T, device, max_len, batch_size, dtype=None):
    """Encode ``texts`` through ``T`` genuine dropout passes -> ``[T, n, D]``.

    Each pass is a separate stochastic forward over the full text list, so the ``T``
    states of a document are independent dropout samples rather than one
    deterministic embedding repeated ``T`` times. No gradients.

    Returns ``(Z, stats)``. ``stats`` uses the cached-MCDP accounting vocabulary:
    ``mc_passes`` is the logical pass count ``T``; ``examples_encoded`` is ``n*T``;
    ``forward_batches`` is the number of real encoder forward calls. ``n*T`` is
    examples, NOT encoder calls — keep the three distinct when reporting cost.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    passes = []
    with dropout_only(model):
        for _ in range(int(T)):
            z = encode_batch_tensor(model, tokenizer, texts, device, max_len,
                                    batch_size, requires_grad=False)
            passes.append(z.detach())
    Z = torch.stack(passes, dim=0)
    if dtype is not None:
        Z = Z.to(dtype=dtype)
    n = len(texts)
    stats = {
        'mc_passes': int(T),
        'examples_encoded': int(n * T),
        'forward_batches': int(math.ceil(n / max(batch_size, 1)) * T) if n else 0,
    }
    return Z, stats


def encode_batch(model, tokenizer, texts, device, max_len, batch_size):
    """
    Deterministic no-grad encode returning CPU float32 numpy (FAISS search,
    fresh-rerank scoring, MC/EMA σ). Thin wrapper over encode_batch_tensor —
    see it for the encoding contract.
    """
    return encode_batch_tensor(
        model, tokenizer, texts, device, max_len, batch_size,
        requires_grad=False,
    ).cpu().float().numpy()


def count_jsonl_examples(pattern: str) -> int:
    """Count total lines across all JSONL files matching a glob pattern."""
    import glob as glob_module
    total = 0
    for path in glob_module.glob(pattern):
        with open(path) as f:
            total += sum(1 for line in f if line.strip())
    return total


@contextlib.contextmanager
def atomic_write(path):
    """Write to a unique temp file beside the destination, then `os.replace`.

    Unique per invocation: a shared `<name>.tmp` meant two writers to the same
    destination shared one scratch file and truncated each other mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            yield handle
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


# ── BRIGHT excluded_ids: one filter shared by every evaluation path ──────────

def load_excluded_ids(domain: str, processed_dir=None) -> Dict[str, frozenset]:
    """Per-query excluded doc ids, as written next to the other eval files.

    A missing file raises: treating it as "no exclusions" would silently reproduce
    the pre-filter numbers.
    """
    path = Path(processed_dir or get_path("processed")) / f"{domain}_excluded.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Regenerate the BRIGHT evaluation files with "
            f"`python src/data/preprocessor.py`.")
    with open(path) as f:
        return {str(q): frozenset(map(str, ids)) for q, ids in json.load(f).items()}


def search_depth(base_k: int, excluded: Dict[str, frozenset], qid=None) -> int:
    """Retrieve this deep so `base_k` results survive filtering.

    BRIGHT excludes up to 11,224 documents for a single aops query, so retrieving
    exactly base_k and then filtering would return a short list.
    """
    if qid is not None:
        return base_k + len(excluded.get(str(qid), ()))
    return base_k + max((len(v) for v in excluded.values()), default=0)


def apply_exclusions(run_results: dict, excluded: Dict[str, frozenset],
                     top_k: int) -> dict:
    """Drop each query's excluded documents, then keep its top_k by score.

    Filtering precedes truncation, so eligible lower-ranked documents refill the
    slots the exclusions freed.
    """
    out = {}
    for qid, hits in run_results.items():
        drop = excluded.get(str(qid), ())
        kept = sorted(((d, s) for d, s in hits.items() if d not in drop),
                      key=lambda ds: -ds[1])
        out[str(qid)] = dict(kept[:top_k])
    return out


# ── Shared IPC / IO utilities (used by ANCE) ────────────────────────────────

def is_valid_checkpoint(ckpt_path: str) -> bool:
    """Checkpoint is fully written once optimizer.pt exists (trainer writes it last)."""
    return (Path(ckpt_path) / "optimizer.pt").exists()


def get_latest_marker_no(directory: Path, prefix: str = "ready_") -> int:
    """Return the highest N from files named {prefix}{N} in directory, or 0 if none."""
    nos = [int(f.name[len(prefix):]) for f in directory.glob(f"{prefix}*")
           if f.name[len(prefix):].isdigit()]
    return max(nos) if nos else 0


def _load_qrels(qrels_file) -> dict:
    """Load TREC qrels file. Returns {qid: set(docids)}."""
    import pandas as pd
    data = []
    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                data.append({'qid': parts[0], 'did': parts[2]})
    return pd.DataFrame(data).groupby('qid')['did'].apply(set).to_dict() if data else {}


def _load_corpus_lookup(corpus_file) -> dict:
    """Load corpus JSONL. Returns {docid: text}."""
    lookup = {}
    with open(corpus_file) as f:
        for line in f:
            d = json.loads(line)
            lookup[d['docid']] = d['text']
    return lookup


def _pool_and_fresh_rerank(model, tokenizer, batch_qids, batch_q_embs_det,
                           faiss_indices, qrels_dict, c_ids, corpus_lookup,
                           p_max_len, mc_batch_size, device,
                           L, max_pool_per_query):
    """Build candidate pool from FAISS, encode with the CURRENT model in
    eval+no_grad, pick top-L per query by current_q . current_d.

    Pool construction per query:
      1. Take FAISS hits in rank order, capped at max_pool_per_query.
      2. Filter qrels positives.

    The pool docs are encoded ONCE per batch (deduped across queries). Model
    mode is saved at entry, set to eval, restored at exit; encoding runs under
    torch.no_grad(). The returned embeddings are NOT reused for MCDP/EMA
    uncertainty scoring — callers re-encode top-L cleanly.

    Returns:
      batch_shortlist: {qid: [docid] top-L}
      pool_stats:      {qid: {retrieved, pool_count, positives_filtered}}
    """
    batch_pool      = {}
    pool_stats      = {}
    all_pool_docids = set()

    for i, qid in enumerate(batch_qids):
        faiss_ids      = [c_ids[j] for j in faiss_indices[i] if j >= 0]
        positives      = qrels_dict.get(qid, set())

        ordered        = []
        seen           = set()
        n_pos_filtered = 0

        for docid in faiss_ids:
            if len(ordered) >= max_pool_per_query:
                break
            if docid in positives:
                n_pos_filtered += 1
                continue
            if docid in seen:
                continue
            seen.add(docid)
            ordered.append(docid)

        batch_pool[qid] = ordered
        all_pool_docids.update(ordered)
        pool_stats[qid] = {
            'retrieved':          len(faiss_ids),
            'pool_count':         len(ordered),
            'positives_filtered': n_pos_filtered,
        }

    # Encode pool once across the batch — eval + no_grad, restore train state
    pool_docids = list(all_pool_docids)
    if not pool_docids:
        empty_shortlist = {qid: [] for qid in batch_qids}
        return empty_shortlist, pool_stats

    pool_texts = [corpus_lookup.get(d, "") for d in pool_docids]
    pool_idx   = {d: i for i, d in enumerate(pool_docids)}

    prev_training = model.training
    model.eval()
    try:
        # encode_batch already wraps in no_grad
        pool_embs = encode_batch(model, tokenizer, pool_texts,
                                 device, p_max_len, mc_batch_size)
    finally:
        if prev_training:
            model.train()

    # Per query: dot product current_q . current_d on its pool slice; top-L
    batch_shortlist = {}
    for i, qid in enumerate(batch_qids):
        pool_for_qid = batch_pool[qid]
        if not pool_for_qid:
            batch_shortlist[qid] = []
            continue
        idxs   = [pool_idx[d] for d in pool_for_qid]
        scores = pool_embs[idxs] @ batch_q_embs_det[i]
        top_l  = np.argsort(scores)[::-1][:L]
        batch_shortlist[qid] = [pool_for_qid[k] for k in top_l]

    return batch_shortlist, pool_stats


def evaluate_bright(ctx, config, model_path, temp_workdir_key=None):
    """Multi-domain BRIGHT evaluation (or single-set if eval_corpus_file set in ctx.args)."""
    import pickle
    import pandas as pd
    from evaluation.trec_eval_wrapper import TrecEvalWrapper

    args = ctx['args']
    if temp_workdir_key is None:
        temp_workdir_key = args.get('temp_workdir', 'temp_grass')
    temp_dir = get_path(temp_workdir_key)

    if args.get('eval_corpus_file'):
        p         = get_path("processed")
        d_corpus  = p / args['eval_corpus_file']
        d_queries = p / args['eval_queries_file']
        d_qrels   = p / args['eval_qrels_file']
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
            str(dq[1][j]): {str(dc[1][i_e[j][k]]): float(s_e[j][k])
                             for k in range(len(i_e[j])) if i_e[j][k] >= 0}
            for j in range(len(dq[1]))
        }
        eval_qrels_data = []
        with open(d_qrels) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    eval_qrels_data.append({'query_id': parts[0], 'doc_id': parts[2],
                                            'relevance': parts[3]})
        metric = args.get('eval_metric', 'ndcg_cut_10')
        evaluator = TrecEvalWrapper(pd.DataFrame(eval_qrels_data))
        metrics = evaluator.evaluate(results, {metric})
        print(f"\n📈 Eval — {metric}={metrics.get(metric, 0):.4f}", flush=True)
    else:
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
            # Filter BRIGHT exclusions before the top-k cut, as evaluate.py does.
            excluded = load_excluded_ids(domain)
            depth = min(search_depth(eval_top_k, excluded), len(dc[1]))
            s_e, i_e = idx_e.search(dq[0].astype(np.float32), depth)
            results = {
                str(dq[1][j]): {str(dc[1][i_e[j][k]]): float(s_e[j][k])
                                 for k in range(len(i_e[j])) if i_e[j][k] >= 0}
                for j in range(len(dq[1]))
            }
            results = apply_exclusions(results, excluded, eval_top_k)
            eval_qrels_data = []
            with open(d_qrels) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        eval_qrels_data.append({'query_id': parts[0], 'doc_id': parts[2],
                                                'relevance': parts[3]})
            evaluator = TrecEvalWrapper(pd.DataFrame(eval_qrels_data))
            metrics = evaluator.evaluate(results, {'recip_rank', 'ndcg_cut_10'})
            eval_summary.append({'domain': domain, 'ndcg10': metrics.get('ndcg_cut_10', 0)})
            print(f"[Eval] {domain}: NDCG@10={metrics.get('ndcg_cut_10', 0):.4f}", flush=True)
        if eval_summary:
            mean_ndcg = pd.DataFrame(eval_summary)['ndcg10'].mean()
            print(f"\n📈 Final Mean NDCG@10: {mean_ndcg:.4f}", flush=True)
