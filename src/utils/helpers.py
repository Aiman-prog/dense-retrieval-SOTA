"""Helper utility functions for Path and Context Management."""

import time
import sys
import subprocess
import pickle
import json
import math
import yaml
import os
import contextlib
import hashlib
import shutil
import tempfile
import datetime
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
        "temp_ance_paper": base / "temp_ance_paper_workdir",
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

    model_cfg = effective_model_config(config, recipe)
    return {
        "args": recipe,
        "base_model": final_base_model,
        # Per-recipe when declared, global otherwise. Read through ctx, never from
        # config['model'] directly, or a recipe override would apply in training and
        # silently not in encoding.
        "max_q": model_cfg['query_max_len'],
        "max_p": model_cfg['passage_max_len'],
        "model_cfg": model_cfg,
        # From the EFFECTIVE config, not the global block: a recipe that overrides the
        # objective's geometry would otherwise apply it in training and silently not in
        # encoding. No existing recipe declares these, so this is a no-op for them.
        "pooling": model_cfg.get('pooling', 'cls'),
        "normalize": model_cfg.get('normalize', False),
        "temperature": model_cfg.get('temperature', 0.02),
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
    """Run Tevatron encode subprocess and save embeddings to a pickle file.

    The paper-fidelity ANCE arm branches here and nowhere else. Its encoder carries a
    projection head that Tevatron's encode driver cannot load -- the driver rebuilds a
    stock DenseModel, so the head would be silently dropped and every embedding would
    be a bare CLS. Because BOTH sides emit the same ``(embeddings, ids)`` pickle, this
    single branch serves every consumer: the ANCE miner, the MS MARCO evaluator and the
    BRIGHT evaluators all call this function and none of them needs a paper branch.
    """
    if (ctx.get('args') or {}).get('paper_fidelity'):
        sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'scripts'))
        from ance_paper import encode_jsonl_to_pickle
        encode_jsonl_to_pickle(
            model_path, input_file, output_pkl, is_query=is_query,
            max_len=ctx['max_q'] if is_query else ctx['max_p'],
            batch_size=ctx['args']['per_device_eval_batch_size'])
        return

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
    # Lengths come from ctx, which has already applied any recipe override. Reading
    # config['model'] here again would encode at the global cap while training used
    # the recipe's, and nothing downstream would notice.
    if is_query:
        q_len = str(ctx.get('max_q') or config['model'].get('query_max_len', 128))
        try:
            subprocess.run(cmd + ['--encode_is_query', '--query_max_len', q_len], check=True)
        except subprocess.CalledProcessError:
            subprocess.run(cmd + ['--encode_is_qry', '--q_max_len', q_len], check=True)
    else:
        p_len = str(ctx.get('max_p') or config['model'].get('passage_max_len', 512))
        subprocess.run(cmd + ['--passage_max_len', p_len], check=True)


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


# ── The optimizer both compared arms must share ──────────────────────────────
#
# BRIGHT ANCE and GRASS are compared to isolate NEGATIVE SELECTION, so every other
# moving part has to be pinned. The optimizer was not: run_grass.py chose between
# `bnb.optim.AdamW8bit` and `torch.optim.AdamW` on whether bitsandbytes happened to
# import, so installing it as a transitive dependency would have switched one arm to
# a quantized optimizer with no visible signal. Betas and eps were also left implicit
# on both sides, which matches only for as long as torch's defaults do not move.

ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8


def build_adamw(params, *, lr, weight_decay, label):
    """The one optimizer for the compared arms. Returns ``(optimizer, spec)``.

    Every hyperparameter is explicit, including the two that are usually left to
    torch's defaults, so a future change to those defaults cannot silently move one
    arm. ``spec`` is what the caller prints and records; comparing two specs is how
    the arms are shown to agree.

    Deliberately NOT configurable. A different optimizer class is a different
    experiment, and a knob here is exactly how the arms would drift apart again.
    """
    lr, weight_decay = float(lr), float(weight_decay)
    optimizer = torch.optim.AdamW(params, lr=lr, betas=ADAMW_BETAS, eps=ADAMW_EPS,
                                  weight_decay=weight_decay)
    spec = {"label": label, "optimizer": "torch.optim.AdamW", "lr": lr,
            "betas": list(ADAMW_BETAS), "eps": ADAMW_EPS,
            "weight_decay": weight_decay}
    print(f"[optim] {label}: {spec['optimizer']} lr={lr} betas={ADAMW_BETAS} "
          f"eps={ADAMW_EPS} weight_decay={weight_decay}", flush=True)
    return optimizer, spec


def optimizer_specs_agree(a, b):
    """Fields on which two arms must match. `label` names the arm, so it is excluded."""
    keys = [k for k in ("optimizer", "lr", "betas", "eps", "weight_decay")]
    return {k: a.get(k) for k in keys} == {k: b.get(k) for k in keys}


def encode_batch_tensor(model, tokenizer, texts, device, max_len, batch_size,
                        *, requires_grad, autocast_enabled=True, normalize=True):
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
        # A model that already pools returns the embedding itself (the paper ANCE
        # encoder returns norm(embeddingHead(CLS))); an HF backbone returns a
        # ModelOutput to take CLS from.
        embs = (out.last_hidden_state[:, 0, :]
                if hasattr(out, 'last_hidden_state') else out)
        if normalize:
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


def _sha256(path) -> str:
    """Streamed sha256 of a file's bytes.

    Streamed rather than read-whole: the corpus and mixture files this hashes are
    hundreds of megabytes, and a manifest must never be the reason a job OOMs.
    """
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


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


# ── Evaluation boundary: run identity, strict qrels, artifact consistency ────

def model_run_tag(model_path) -> str:
    """Directory tag that is unique per model PATH, not per basename.

    Two runs can both end in `checkpoint-500`; keying embeddings, results and
    summaries by the basename alone let them share -- and clobber -- directories.
    """
    resolved = Path(model_path).resolve()
    digest = hashlib.sha1(str(resolved).encode('utf-8')).hexdigest()[:8]
    return f"{resolved.name}__{digest}"


def require_eval_files(label, files) -> None:
    """Every path must exist and be non-empty, or raise naming every offender.

    Cheap, and it runs before any encoding: discovering a missing corpus through a
    Tevatron subprocess failure costs a GPU hour and names nothing useful.
    """
    bad = []
    for f in files:
        f = Path(f)
        if not f.is_file():
            bad.append(f"{f} (missing)")
        elif f.stat().st_size == 0:
            bad.append(f"{f} (empty)")
    if bad:
        raise FileNotFoundError(
            f"[{label}] required evaluation file(s) unusable: " + "; ".join(bad))


def _read_query_ids(queries_file) -> list:
    """Query ids in file order, from a Tevatron-format queries JSONL."""
    ids = []
    with open(queries_file) as f:
        for line in f:
            if line.strip():
                ids.append(str(json.loads(line)['query_id']))
    return ids


def _id_sample(ids, n=5) -> str:
    ordered = sorted(ids)
    return (", ".join(ordered[:n]) + (" ..." if len(ordered) > n else "")) or "none"


def check_eval_artifacts(domain, qrels, excluded, *, query_ids=None,
                         queries_file=None, encoded_query_ids=None) -> None:
    """Cross-artifact consistency at the evaluation boundary.

    * query ids == exclusion-map keys -- a query missing from the map is scored with
      no exclusions at all, silently reproducing pre-filter numbers. ``excluded=None``
      declares that this benchmark HAS no exclusion map (MS MARCO) and skips only
      this check; it is not a way to opt a BRIGHT domain out of exclusion filtering,
      and `load_excluded_ids` raises on a missing file so BRIGHT cannot reach it;
    * qrels query ids are covered by the query set -- a judged query that is never
      retrieved would otherwise contribute an invisible zero;
    * encoded ids == source ids, AS SETS -- the encoder may reorder, but a query it
      dropped or invented breaks the correspondence between run and qrels.

    Source ids come from `query_ids` when the caller already has them, otherwise
    from `queries_file`. `encoded_query_ids=None` skips the last check, so BM25 --
    which never encodes -- can call this without inventing ids.
    """
    if query_ids is None:
        if queries_file is None:
            raise ValueError("check_eval_artifacts needs query_ids or queries_file")
        query_ids = _read_query_ids(queries_file)
    source = {str(q) for q in query_ids}

    if excluded is not None:
        keys = {str(q) for q in excluded}
        if keys != source:
            raise ValueError(
                f"[{domain}] query set and excluded_ids keys disagree: {len(source)} "
                f"queries, {len(keys)} exclusion entries; queries with no entry: "
                f"{_id_sample(source - keys)}; entries with no query: "
                f"{_id_sample(keys - source)}")

    judged = {str(q) for q in qrels}
    if not judged <= source:
        raise ValueError(
            f"[{domain}] {len(judged - source)} judged query id(s) absent from the "
            f"query set: {_id_sample(judged - source)}")

    if encoded_query_ids is not None:
        enc = {str(q) for q in encoded_query_ids}
        if enc != source:
            raise ValueError(
                f"[{domain}] encoded query ids do not match {len(source)} source "
                f"queries; not encoded: {_id_sample(source - enc)}; encoded but not "
                f"in the query file: {_id_sample(enc - source)}")


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
    """Load TREC qrels file. Returns {qid: set(docids)}.

    The one strict reader for mining AND evaluation: a line that is not four columns
    raises, and an empty file raises rather than judging nothing and scoring every
    query zero. `TrecEvalWrapper` takes this mapping directly; BRIGHT and MS MARCO
    qrels are binary, so the relevance column carries no extra information.
    """
    qrels = {}
    with open(qrels_file) as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(
                    f"{qrels_file}:{line_no}: expected four columns, found "
                    f"{len(parts)}")
            qrels.setdefault(parts[0], set()).add(parts[2])
    if not qrels:
        raise ValueError(f"{qrels_file}: no qrels rows; nothing to evaluate")
    return qrels


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
    """Multi-domain BRIGHT evaluation (or single-set if eval_corpus_file set in ctx.args).

    Every requested domain is preflighted -- files present, and query/qrel/exclusion
    ids mutually consistent -- BEFORE the first encode. An incomplete domain set
    fails there, costing no GPU time; it never prints a mean over the domains that
    happened to have files.
    """
    import pickle
    from evaluation.trec_eval_wrapper import TrecEvalWrapper

    args = ctx['args']
    if temp_workdir_key is None:
        temp_workdir_key = args.get('temp_workdir', 'temp_grass')
    temp_dir = get_path(temp_workdir_key)
    # Run-tagged so two recipes sharing one temp workdir cannot swap c.pkl/q.pkl.
    run_tag = model_run_tag(model_path)

    if args.get('eval_corpus_file'):
        p         = get_path("processed")
        d_corpus  = p / args['eval_corpus_file']
        d_queries = p / args['eval_queries_file']
        d_qrels   = p / args['eval_qrels_file']
        # Not BRIGHT: three artifacts, no exclusion file, no exclusions applied.
        require_eval_files(args['eval_corpus_file'], [d_corpus, d_queries, d_qrels])
        eval_dir = temp_dir / "final_eval" / run_tag
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
        metric = args.get('eval_metric', 'ndcg_cut_10')
        evaluator = TrecEvalWrapper(_load_qrels(d_qrels))
        metrics = evaluator.evaluate(results, {metric})
        print(f"\n📈 Eval — {metric}={metrics.get(metric, 0):.4f}", flush=True)
        return

    domains = config['evaluation'].get('eval_domains', [])
    if not domains:
        raise ValueError(
            "evaluation.eval_domains is empty; there is nothing to evaluate")

    # ── Preflight EVERY domain before encoding the first one ─────────────────
    prepared = {}
    for domain in domains:
        d_corpus  = get_path("processed") / f"{domain}_corpus.jsonl"
        d_queries = get_path("processed") / f"{domain}_queries.jsonl"
        d_qrels   = get_path("processed") / f"{domain}_qrels.txt"
        require_eval_files(domain, [
            d_corpus, d_queries, d_qrels,
            get_path("processed") / f"{domain}_excluded.json"])
        qrels = _load_qrels(d_qrels)
        excluded = load_excluded_ids(domain)
        check_eval_artifacts(domain, qrels, excluded, queries_file=d_queries)
        prepared[domain] = (d_corpus, d_queries, qrels, excluded)

    eval_summary = []
    for domain in domains:
        d_corpus, d_queries, qrels, excluded = prepared[domain]
        eval_dir = temp_dir / "final_eval" / run_tag / domain
        eval_dir.mkdir(parents=True, exist_ok=True)
        encode_to_pickle(str(model_path), d_corpus,  eval_dir / "c.pkl", False, ctx, config)
        encode_to_pickle(str(model_path), d_queries, eval_dir / "q.pkl", True,  ctx, config)
        with open(eval_dir / "c.pkl", 'rb') as f: dc = pickle.load(f)
        with open(eval_dir / "q.pkl", 'rb') as f: dq = pickle.load(f)
        q_ids = [str(x) for x in dq[1]]
        check_eval_artifacts(domain, qrels, excluded, queries_file=d_queries,
                             encoded_query_ids=q_ids)
        idx_e = faiss.IndexFlatIP(dc[0].shape[1])
        idx_e.add(dc[0].astype(np.float32))
        eval_top_k = args.get('eval_top_k', 10)
        # Filter BRIGHT exclusions before the top-k cut, as evaluate.py does.
        depth = min(search_depth(eval_top_k, excluded), len(dc[1]))
        s_e, i_e = idx_e.search(dq[0].astype(np.float32), depth)
        results = {
            q_ids[j]: {str(dc[1][i_e[j][k]]): float(s_e[j][k])
                        for k in range(len(i_e[j])) if i_e[j][k] >= 0}
            for j in range(len(q_ids))
        }
        results = apply_exclusions(results, excluded, eval_top_k)
        evaluator = TrecEvalWrapper(qrels)
        metrics = evaluator.evaluate(results, {'recip_rank', 'ndcg_cut_10'})
        eval_summary.append(metrics.get('ndcg_cut_10', 0))
        print(f"[Eval] {domain}: NDCG@10={metrics.get('ndcg_cut_10', 0):.4f}", flush=True)

    mean_ndcg = sum(eval_summary) / len(eval_summary)
    print(f"\n📈 Final Mean NDCG@10: {mean_ndcg:.4f} over {len(eval_summary)} domains",
          flush=True)


# ── Run identity: manifest, fresh-start gate, success validation ─────────────
#
# Tevatron's driver resumes unconditionally -- `train.py` does
# `trainer.train(resume_from_checkpoint=(get_last_checkpoint(output_dir) is not None))`
# and `--overwrite_output_dir` does not suppress it. A re-run into a finished
# output directory therefore resumes, executes ZERO optimizer steps, rewrites the
# stale weights and exits 0. Clearing `checkpoint-*` is what makes
# `get_last_checkpoint` return None, so fresh-by-default needs no Tevatron patch.

RUN_MANIFEST_NAME = "run_manifest.json"
TRAINING_LOG_NAME = "training_log.jsonl"

# Every package whose version can change a result. Absent ones record None rather
# than raising: a manifest must never be the reason a job dies.
# Distribution names, not import names: GradCache installs as "GradCache" (a
# "grad-cache" lookup silently records null), and faiss ships as faiss-cpu locally
# but faiss-gpu on the cluster, so both are asked for.
_MANIFEST_PACKAGES = ("torch", "transformers", "accelerate", "datasets", "peft",
                      "safetensors", "tevatron", "GradCache", "pyserini",
                      "faiss-cpu", "faiss-gpu")

# What resume compatibility is judged on. Deliberately excludes code revision and
# dependency versions: a docstring edit must not invalidate a resumable run, and
# those are recorded separately so a mismatch is still visible.
_FINGERPRINT_KEYS = ("effective_config", "base_model", "data_sha256", "seed",
                     "world_size")


class RunDirectoryError(RuntimeError):
    """The output directory cannot be used as requested."""


def _package_versions(names=_MANIFEST_PACKAGES) -> Dict[str, Any]:
    from importlib import metadata
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except Exception:                                          # noqa: BLE001
            out[name] = None
    return out


def _code_revision() -> Dict[str, Any]:
    """Git HEAD and whether the tree is dirty, or nulls outside a checkout."""
    root = Path(__file__).resolve().parent.parent.parent
    def git(*args):
        try:
            out = subprocess.run(["git", "-C", str(root), *args],
                                 capture_output=True, text=True, check=False)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:                                          # noqa: BLE001
            return None
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"git_sha": head,
            "git_dirty": None if status is None else bool(status.strip())}


def build_run_manifest(recipe_name, ctx, recipe, *, data_files, world_size,
                       negative_pool_size, optimizer_steps, extra=None) -> Dict[str, Any]:
    """Everything needed to say what produced a checkpoint. Pure but for hashing.

    ``data_files`` is the tuple ``require_mixture_files`` already returns, so the
    manifest hashes exactly the files training will read -- not a glob that might
    resolve differently later.
    """
    config = load_config()
    files = []
    for path in data_files:
        path = Path(path)
        files.append({"name": path.name,
                      "bytes": path.stat().st_size,
                      "lines": count_jsonl_examples(str(path)),
                      "sha256": _sha256(path)})

    configured = recipe.get('base_model') or config['model']['base_model']
    resolved = ctx.get('base_model')
    now = datetime.datetime.now(datetime.timezone.utc)

    manifest = {
        "recipe": recipe_name,
        "started_at": now.isoformat(),
        # Kept alongside the ISO form purely so the "was a checkpoint written by
        # THIS run" check is an mtime comparison and not a date parse.
        "started_at_epoch": now.timestamp(),
        "seed": config.get('seed', 42),
        "effective_config": {"recipe": dict(recipe),
                             "model": effective_model_config(config, recipe)},
        "base_model": resolved,
        "base_model_configured": configured,
        "base_model_exists": bool(resolved and Path(resolved).exists()),
        "data_files": files,
        "data_sha256": [f["sha256"] for f in files],
        "code_revision": _code_revision(),
        "dependencies": _package_versions(),
        "world_size": int(world_size),
        "negative_pool_size": int(negative_pool_size),
        "optimizer_steps_planned": int(optimizer_steps),
    }
    if extra:
        manifest.update(extra)

    payload = {k: manifest[k] for k in _FINGERPRINT_KEYS}
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()
    return manifest


def _fingerprint_diff(prior, manifest) -> str:
    """Which fingerprint inputs disagree -- the error is useless without this."""
    differing = [k for k in _FINGERPRINT_KEYS if prior.get(k) != manifest.get(k)]
    return ", ".join(differing) if differing else "(unreadable prior manifest)"


def prepare_output_dir(output_dir, manifest, *, resume=False, overwrite=False) -> Path:
    """Gate the output directory, then publish the manifest. The one entry point.

    Fresh is the default. Resume requires ``--resume`` AND a prior manifest whose
    fingerprint matches; an incompatible directory is refused rather than silently
    reused, because the failure it causes -- training zero steps and re-saving old
    weights -- otherwise reports success.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / RUN_MANIFEST_NAME

    prior, unreadable = None, False
    if manifest_path.is_file():
        # An IO error and malformed JSON mean different things and must not share a
        # branch. Malformed JSON is a corrupt manifest: unreadable-and-blocking is
        # right, since we cannot tell what produced the directory. An OSError is
        # /scratch being flaky, and treating THAT as "prior completed, incompatible"
        # advised --overwrite, which then deletes every checkpoint -- the opposite of
        # what a crashed run needs. Retry it, and fail loudly at startup if it stands.
        raw = {}

        def _read():
            raw['text'] = manifest_path.read_text()

        if not retry_io(_read, f"read {manifest_path}"):
            raise RunDirectoryError(
                f"could not read {manifest_path} after repeated attempts. This is an "
                f"IO failure, not a configuration mismatch -- do NOT pass --overwrite, "
                f"which would discard the checkpoints in {output_dir}.")
        try:
            prior = json.loads(raw['text'])
        except ValueError:
            # Unreadable is treated as present-and-incompatible, never as absent:
            # "absent" would silently permit a fresh run over unknown state.
            prior, unreadable = {}, True
    same = prior is not None and prior.get('fingerprint') == manifest['fingerprint']
    checkpoints = sorted(output_dir.glob("checkpoint-*"))

    # Only a run that FINISHED is worth protecting. `finished_at` is stamped by
    # assert_training_succeeded and by nothing else, so its absence means the run died
    # before producing anything. The manifest is written BEFORE training, so without
    # this a job that dies at startup locks its own output directory against the very
    # fix it needs -- which is exactly what happened once the HF cache was seeded and
    # base_model went from an unresolvable repo id to a real path.
    # An unreadable manifest stays blocking: we cannot tell what it was.
    prior_completed = unreadable or bool(prior and prior.get('finished_at'))

    if resume:
        if prior is None:
            raise RunDirectoryError(
                f"--resume was given but {manifest_path} does not exist; there is "
                f"nothing to resume from. Drop --resume to start fresh.")
        if not same:
            raise RunDirectoryError(
                f"--resume was given but {output_dir} was produced by a different "
                f"configuration; differing: {_fingerprint_diff(prior, manifest)}. "
                f"Start fresh with --overwrite, or point at the matching directory.")
        if not checkpoints:
            raise RunDirectoryError(
                f"--resume was given and the manifest matches, but {output_dir} "
                f"holds no checkpoint-* to resume from.")
        # A resume needs a baseline, or "did this invocation train?" is unanswerable:
        # --resume keeps training_log.jsonl, so max(global_step) is the PREVIOUS
        # run's final step and a run that trains nothing still reads it back as
        # success. trainer_state.json is HF's own record of where the checkpoint
        # stopped; without it there is no baseline and the resume is refused.
        start_step, start_src = _trainer_state_step(output_dir)
        if start_step < 1:
            raise RunDirectoryError(
                f"--resume was given but no checkpoint in {output_dir} carries a "
                f"readable trainer_state.json, so the invocation start step cannot "
                f"be established and progress could not be verified afterwards. "
                f"Start fresh instead of resuming from an unidentifiable state.")
        _record_invocation(manifest, manifest_path, start_step)
        print(f"[run] resuming from {len(checkpoints)} checkpoint(s) in "
              f"{output_dir.name} at step {start_step} ({start_src.name}); "
              f"fingerprint {manifest['fingerprint'][:12]}",
              flush=True)
        return output_dir

    if prior_completed and not same and not overwrite:
        raise RunDirectoryError(
            f"{output_dir} already holds a run with a different configuration "
            f"(differing: {_fingerprint_diff(prior, manifest)}). Pass --overwrite to "
            f"discard it, or --resume if you meant to continue a matching run.")

    if prior is None and checkpoints and not overwrite:
        # Checkpoints with no manifest are unidentifiable: they predate this gate, so
        # nothing says which configuration produced them. Deleting them silently is how
        # a previous run's artifacts get discarded by someone who only meant to start a
        # new one -- refuse and make the discard explicit instead.
        raise RunDirectoryError(
            f"{output_dir} holds {len(checkpoints)} checkpoint(s) but no "
            f"{RUN_MANIFEST_NAME}, so they cannot be matched against this run's "
            f"configuration. Pass --overwrite to discard them, or move them aside "
            f"first. (--resume needs a manifest and cannot be used here.)")

    if prior is not None and not prior_completed and not same:
        print(f"[run] discarding an unfinished run's manifest "
              f"(differing: {_fingerprint_diff(prior, manifest)})", flush=True)

    # ignore_errors=True used to hide this: on EREMEOTEIO (P11) the directory stayed,
    # get_last_checkpoint() returned its step number, Tevatron resumed from it and the
    # run still printed "fresh run". Removal is retried, and then VERIFIED by
    # re-globbing -- the claim being made is that nothing is left to resume from.
    for ckpt in checkpoints:
        retry_io(lambda c=ckpt: shutil.rmtree(c), f"remove {ckpt.name}")
    survivors = sorted(p.name for p in output_dir.glob("checkpoint-*"))
    if survivors:
        raise RunDirectoryError(
            f"could not remove stale checkpoint(s) from {output_dir}: "
            f"{', '.join(survivors)}. Tevatron would resume from the highest of "
            f"these and shadow every new save, so this is not a fresh run. Remove "
            f"them by hand and re-submit.")
    if checkpoints:
        print(f"[run] removed {len(checkpoints)} stale checkpoint(s) from "
              f"{output_dir.name}", flush=True)

    # The log is append-only, so a fresh run into a used directory would inherit the
    # previous attempt's records. assert_training_succeeded reads max(global_step)
    # from it, so a run that died at step 50 would report the old run's 3000 and pass.
    # Kept on --resume, where the earlier records belong to the same run.
    stale_log = output_dir / TRAINING_LOG_NAME
    if stale_log.exists():
        retry_io(stale_log.unlink, f"remove stale {stale_log.name}")
    if stale_log.exists():
        raise RunDirectoryError(
            f"could not remove stale {stale_log.name} from {output_dir}. Its old "
            f"optimizer steps and ranking probes would be indistinguishable from "
            f"this fresh invocation and could validate a zero-step run. Remove it "
            f"by hand and re-submit.")

    _record_invocation(manifest, manifest_path, 0)
    print(f"[run] fresh run in {output_dir.name}; fingerprint "
          f"{manifest['fingerprint'][:12]}", flush=True)
    return output_dir


def _safetensors_tensor_count(path) -> int:
    """Tensors declared in a safetensors header, without loading any weights.

    Reads the 8-byte length prefix and the JSON header only, so a truncated or
    half-written file fails here instead of at the next job's model load.
    """
    with open(path, 'rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path} is shorter than its 8-byte length prefix")
        length = int.from_bytes(raw, 'little')
        header = f.read(length)
        if len(header) != length:
            raise ValueError(
                f"{path} header is truncated: declared {length} bytes, read {len(header)}")
    return len([k for k in json.loads(header) if k != "__metadata__"])


def _newest_model_artifact(output_dir):
    """The model file this run would have written, root first then checkpoints."""
    candidates = [output_dir] + sorted(output_dir.glob("checkpoint-*"))
    best = None
    for directory in candidates:
        for name in ("model.safetensors", "pytorch_model.bin"):
            f = directory / name
            if f.is_file() and (best is None or f.stat().st_mtime > best.stat().st_mtime):
                best = f
    return best


def _finite(value) -> bool:
    """True only for a real, finite number. None / NaN / non-numeric are not signals."""
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# The encoding contract: the only training settings evaluation actually depends on.
# Pooling and normalization decide what an embedding MEANS, and the two lengths decide
# what of the text survives truncation. Everything else in a training recipe (batch
# size, LR, epochs) cannot change how a finished checkpoint encodes text.
ENCODING_CONTRACT = ("pooling", "normalize", "query_max_len", "passage_max_len")


# Model settings a recipe may override. Sequence lengths and the objective's
# geometry: everything downstream reads through effective_model_config, so a key
# added here becomes part of the manifest, the fingerprint and the eval cache
# identity at once. No existing recipe declares any of these, which is what makes
# adding one a byte-level no-op for every arm but the declarer.
_MODEL_OVERRIDE_KEYS = ('query_max_len', 'passage_max_len', 'pooling', 'normalize',
                        'temperature')


def effective_model_config(config, recipe=None):
    """The model settings a run ACTUALLY uses: globals, with recipe overrides applied.

    Sequence lengths are global by default because every BGE arm shares one encoding
    contract. The paper-fidelity ANCE arm does not: RoBERTa on MS MARCO wants q64/p512
    where BRIGHT wants q1024/p512, and inheriting the global caps would make the
    reproduction cost BGE-M3 money.

    The same applies to the objective's geometry. `normalize` and `temperature` are
    global because every BGE arm trains scaled cosine; the paper arm trains raw
    unscaled dot, and inheriting the globals would make its manifest and its embedding
    cache identity describe an objective it does not use.

    A recipe that declares none of these keys gets `dict(config['model'])` back
    unchanged, byte for byte, which is what keeps every existing manifest fingerprint
    stable.
    """
    model = dict(config['model'])
    for key in _MODEL_OVERRIDE_KEYS:
        if recipe and recipe.get(key) is not None:
            model[key] = recipe[key]
    return model


def load_training_manifest(model_path):
    """The run manifest for a checkpoint, or None for a checkpoint that predates it.

    Resolves from the model directory, or its parent for a `checkpoint-*` subdir.
    A legacy checkpoint has no manifest and stays evaluable; the caller records
    training_manifest: null rather than refusing.
    """
    model_path = Path(model_path)
    for candidate in (model_path, model_path.parent):
        manifest = candidate / RUN_MANIFEST_NAME
        if manifest.is_file():
            raw = {}

            def _read():
                raw['text'] = manifest.read_text()

            if not retry_io(_read, f"read {manifest}"):
                raise RunDirectoryError(
                    f"could not read {manifest} after repeated attempts. The "
                    f"checkpoint's encoding contract cannot be verified, so this "
                    f"is not treated as a legacy checkpoint.")
            try:
                return json.loads(raw['text'])
            except ValueError as exc:
                raise RunDirectoryError(
                    f"{manifest} could not be read as valid JSON; the checkpoint's "
                    f"encoding contract cannot be verified, so this is not treated "
                    f"as a legacy checkpoint.") from exc
        if not model_path.name.startswith("checkpoint-"):
            break
    return None


def encoding_contract_drift(manifest, model_cfg):
    """Where the live encoding settings differ from the checkpoint's.

    Deliberately NOT compared: the training data hashes. A checkpoint stays valid
    when the mixture that produced it is no longer on disk, so requiring equality
    there would fail good checkpoints. Training provenance is recorded instead.
    """
    if not manifest:
        return {}
    trained = (manifest.get('effective_config') or {}).get('model') or {}
    drift = {}
    for key in ENCODING_CONTRACT:
        if key not in trained:
            continue
        before, after = trained.get(key), model_cfg.get(key)
        if before != after:
            drift[key] = {"checkpoint": before, "evaluation": after}
    return drift


def training_provenance(manifest):
    """What to record about the checkpoint's origin, whether or not it drifted."""
    if not manifest:
        return None
    return {
        "fingerprint": manifest.get('fingerprint'),
        "recipe": manifest.get('recipe'),
        "base_model": manifest.get('base_model'),
        "final_global_step": manifest.get('final_global_step'),
        "training_data_sha256": manifest.get('data_sha256'),
        "encoding_contract": {k: (manifest.get('effective_config') or {})
                              .get('model', {}).get(k) for k in ENCODING_CONTRACT},
    }


def eval_artifact_hashes(processed_dir, domains):
    """Digest every file a domain's score depends on, so two runs can be compared.

    Domain-set equality alone does not make a dense and a sparse run comparable: the
    corpus, queries, judgments or exclusion lists can each have been regenerated
    between them.
    """
    processed_dir = Path(processed_dir)
    out = {}
    for domain in sorted(domains):
        entry = {}
        for label, name in (("corpus", f"{domain}_corpus.jsonl"),
                            ("queries", f"{domain}_queries.jsonl"),
                            ("qrels", f"{domain}_qrels.txt"),
                            ("excluded", f"{domain}_excluded.json")):
            path = processed_dir / name
            entry[label] = _sha256(path) if path.is_file() else None
        out[domain] = entry
    return out


def _record_invocation(manifest, manifest_path, start_step):
    """Stamp where THIS invocation began, then publish the manifest.

    `invocation_start_step` is the baseline progress is measured against; a resume
    inherits the previous run's log, so without it "did this invocation train?"
    cannot be answered. Neither key is in _FINGERPRINT_KEYS, so recording them
    cannot invalidate a resumable run.
    """
    manifest['invocation_start_step'] = int(start_step)
    manifest['invocation_started_at'] = datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    with atomic_write(manifest_path) as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def _trainer_state_step(output_dir):
    """Highest global_step HF itself recorded, from any checkpoint's trainer_state.json.

    Independent of our diagnostics callback, whose writes are best-effort on BeeGFS.
    Returns (step, source) with step 0 when nothing readable is found.
    """
    best, source = 0, None
    for ckpt in sorted(Path(output_dir).glob("checkpoint-*")):
        state_file = ckpt / "trainer_state.json"
        if not state_file.is_file():
            continue
        try:
            step = int(json.loads(state_file.read_text()).get('global_step', 0))
        except (ValueError, OSError, TypeError):
            continue
        if step > best:
            best, source = step, state_file
    return best, source


def assert_training_succeeded(output_dir, manifest, *,
                              required_final_step=None) -> Dict[str, Any]:
    """Refuse to call a run successful without evidence that it trained.

    A clean exit proves nothing here: Tevatron's unconditional resume can finish a
    run having taken zero optimizer steps and re-saved the weights it started from.

    Success requires NEW progress past this invocation's start step AND reaching
    the planned step count. `required_final_step` overrides the latter, for a run
    that is deliberately short; neither baseline passes it.
    """
    output_dir = Path(output_dir)
    log_path = output_dir / TRAINING_LOG_NAME
    records = []
    if log_path.is_file():
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                # A torn final line: appends are not atomic and a SIGKILL can cut one
                # mid-write. That is not evidence of a failed run, so skip it rather
                # than raise a JSONDecodeError the caller cannot interpret.
                print(f"[diag] skipping an unparseable line in {log_path}",
                      file=sys.stderr, flush=True)

    steps = [r for r in records if r.get('phase') is None]
    probes = [r for r in records if r.get('phase') is not None]
    final_step = max((int(r.get('global_step', 0)) for r in records), default=0)

    # Where this invocation began. Absent on manifests written before this key
    # existed (job 18816), which is treated as a fresh start at 0.
    start_step = int(manifest.get('invocation_start_step', 0) or 0)
    planned = int(manifest.get('optimizer_steps_planned', 0) or 0)
    target = required_final_step if required_final_step is not None else planned

    # Our own log is written by the diagnostics callback, whose writes are
    # best-effort on a flaky filesystem. trainer_state.json is written by HF itself
    # into every checkpoint and carries the same global_step, so it is the
    # independent witness -- a lost log must not condemn a run that did train.
    state_step, state_src = _trainer_state_step(output_dir)
    if state_step > final_step:
        print(f"[diag] {log_path} holds {final_step} step(s); {state_src} reports "
              f"{state_step}. Using it as the evidence of training.",
              file=sys.stderr, flush=True)
        final_step = state_step

    # Progress, not position. On --resume the log is kept, so max(global_step) is
    # the PREVIOUS invocation's final step and would pass on its own.
    if final_step <= start_step:
        raise RunDirectoryError(
            f"training reported success with no new optimizer steps: this "
            f"invocation started at step {start_step} and ended at {final_step}. "
            f"{log_path} holds {len(steps)} step record(s). A stale checkpoint in "
            f"{output_dir} that Tevatron resumed from is the usual cause.")

    # Completion. A wall-clock kill or a mid-run crash leaves a real but partial
    # run, which must not be reported as the configured experiment.
    if target and final_step < target:
        raise RunDirectoryError(
            f"training stopped at step {final_step} of the {target} planned "
            f"({100.0 * final_step / target:.1f}%); the run is incomplete, not "
            f"successful. Resume it, or pass an explicit required_final_step to "
            f"accept a short run deliberately.")

    for r in steps:
        loss = r.get('loss')
        if loss is not None and not math.isfinite(float(loss)):
            raise RunDirectoryError(
                f"non-finite loss={loss} at step {r.get('global_step')} in "
                f"{log_path}; the optimization diverged.")
        # grad_norm warns rather than raises. It is the PRE-clipping norm, so a
        # transient inf in bf16 is absorbed by max_grad_norm and the step that
        # followed it was still valid. Failing the run here would discard a whole
        # training run over a value clipping already handled.
        gn = r.get('grad_norm')
        if gn is not None and not math.isfinite(float(gn)):
            print(f"[diag] non-finite grad_norm={gn} at step {r.get('global_step')} "
                  f"(pre-clipping; max_grad_norm applies after this)",
                  file=sys.stderr, flush=True)

    artifact = _newest_model_artifact(output_dir)
    if artifact is None:
        raise RunDirectoryError(f"no model.safetensors or pytorch_model.bin under {output_dir}")
    started = float(manifest['started_at_epoch'])
    if artifact.stat().st_mtime < started:
        raise RunDirectoryError(
            f"{artifact} predates the start of this run "
            f"({manifest['started_at']}); no new checkpoint was written.")

    directory = artifact.parent
    if directory.name.startswith("checkpoint-") and not is_valid_checkpoint(directory):
        raise RunDirectoryError(
            f"{directory} has no optimizer.pt; the trainer writes it last, so the "
            f"checkpoint is incomplete.")
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(str(directory))
    except Exception as e:                                         # noqa: BLE001
        raise RunDirectoryError(f"{directory} is not loadable: {type(e).__name__}: {e}")
    if artifact.name == "model.safetensors" and _safetensors_tensor_count(artifact) == 0:
        raise RunDirectoryError(f"{artifact} declares no tensors")

    # A probe that raised is recorded as {"phase":…, "error":…} so its absence is
    # never mistaken for a pass -- but it is not a signal. Counting those, in-batch
    # 14990 shipped {"phase":"begin","error":"TypeError…"} and still validated.
    # Two probes at the SAME step are one point measured twice, not two points.
    ok_steps = {
        int(r.get('global_step', 0)) for r in probes
        if _finite(r.get('rank_acc')) and _finite(r.get('margin_mean'))}
    if len(ok_steps) < 2:
        raise RunDirectoryError(
            f"the run left {len(ok_steps)} successful ranking probe point(s) at "
            f"distinct steps (minimum 2), from {len(probes)} probe record(s) in "
            f"{log_path}. Loss alone is not evidence of learning: trivial negatives "
            f"give low loss and near-zero gradients.")

    manifest_path = output_dir / RUN_MANIFEST_NAME
    stored = json.loads(manifest_path.read_text()) if manifest_path.is_file() else dict(manifest)
    stored.update({
        "final_global_step": final_step,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": str(directory),
        "probe": probes,
        "probe_ok_steps": sorted(ok_steps),
    })
    with atomic_write(manifest_path) as f:
        json.dump(stored, f, indent=2, default=str)

    print(f"[run] validated: {final_step} optimizer steps, checkpoint {directory.name}, "
          f"{len(probes)} probe point(s)", flush=True)
    return stored


# ── Learning observability: shared by in-batch, cross-batch and ANCE ─────────
#
# Loss, learning rate and the PRE-CLIPPING gradient norm are already computed by
# HF Trainer -- `grad_norm` in its logs is the return of accelerator.clip_grad_norm_,
# i.e. the norm before clipping. Nothing new is computed here; the records are
# merely persisted next to the checkpoint so the trajectory survives checkpoint
# rotation. The one genuinely missing signal is the fixed-probe ranking metric.


def retry_io(op, what, attempts=3, delay=0.5):
    """Run ``op()``, retrying transient filesystem errors. True if it landed.

    /scratch is BeeGFS and returns EREMOTEIO intermittently. The helper itself reports
    failure without raising: best-effort diagnostics may ignore False, while critical
    callers must verify their postcondition and raise. A failed diagnostic write must
    never end training -- job 14990 died at step 3000 of 10314 because an OSError from
    a one-line append propagated out of Trainer.log.
    """
    for attempt in range(attempts):
        try:
            op()
            return True
        except OSError as exc:
            if attempt + 1 == attempts:
                print(f"[diag] giving up on {what} after {attempts} attempts: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                return False
            time.sleep(delay * (2 ** attempt))
    return False


def append_jsonl(path, record) -> bool:
    """Append one JSON record. Append, not atomic_write: this is a growing log.

    Returns whether the record landed; never raises on an IO failure. See retry_io.
    """
    path = Path(path)
    line = json.dumps(record, ensure_ascii=False, default=str) + '\n'

    def _write():
        # mkdir here rather than per-record at module scope: it is one metadata
        # round-trip to a shared filesystem, and it must be inside the retry.
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line)

    return retry_io(_write, f"append to {path}")


def probe_triples_from_mixture(data_files, n=64):
    """A fixed (query, positive, negative) probe set. Deterministic, no RNG.

    The last ``n`` usable records of the mixture, preferring train_hq.jsonl. These
    records ARE seen during training, so the probe measures ranking fit rather than
    generalization -- which is what a fixed-probe signal is for. Held-out retrieval
    quality is what the BRIGHT evaluation is for, and it stays a separate job.
    """
    paths = [Path(p) for p in data_files]
    chosen = next((p for p in paths if p.name == "train_hq.jsonl"), paths[0] if paths else None)
    if chosen is None:
        raise ValueError("probe_triples_from_mixture needs at least one data file")

    from collections import deque
    keep = deque(maxlen=n)
    with open(chosen, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pos, neg = row.get('positive_passages') or [], row.get('negative_passages') or []
            if not pos or not neg:
                continue
            keep.append((str(row.get('query') or ''), str(pos[0].get('text') or ''),
                         str(neg[0].get('text') or '')))
    if not keep:
        raise ValueError(f"no record in {chosen} carries both a positive and a negative")
    return list(keep)


def _probe_encoder(model):
    """The HF encoder inside whatever wrapper the caller has.

    Tevatron's EncoderModel.forward takes (query=, passage=), not input_ids, so
    encode_batch_tensor cannot call it directly. Detected by `encode_query`, which
    EncoderModel defines and a bare HF model does not -- checking for `.encoder`
    alone would wrongly unwrap XLMRobertaModel, which has an `.encoder` of its own.
    """
    model = getattr(model, 'module', model)                  # DistributedDataParallel
    if hasattr(model, 'encode_query') and hasattr(model, 'encoder'):
        return model.encoder
    return model


def ranking_probe(model, tokenizer, triples, device, max_q, max_p, batch_size=8,
                  normalize=True) -> Dict[str, Any]:
    """Positive-negative margin and ranking accuracy on a fixed probe set.

    Explicitly eval() + no_grad, entry mode restored in `finally`: a probe must not
    leave dropout off for the next training step. Encoding goes through
    encode_batch_tensor, so pooling and normalization match training exactly
    (CLS + L2, per config.model.pooling / normalize). `normalize=False` measures the
    margin in raw dot space, which is what the paper-fidelity ANCE arm trains in --
    an L2-normalized probe there would report a different geometry than the loss.
    """
    if not triples:
        raise ValueError("ranking_probe needs at least one triple")
    if tokenizer is None:
        # Tevatron builds Trainer without tokenizer=, so the callback kwarg is None on
        # every Tevatron pipeline. The caller must supply one; a bare TypeError from
        # deep inside encode_batch_tensor said nothing about why.
        raise ValueError(
            "ranking_probe got tokenizer=None -- Tevatron's Trainer carries no "
            "tokenizer, so the caller must pass one explicitly")
    encoder = _probe_encoder(model)
    was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            enc = lambda texts, ml: encode_batch_tensor(
                encoder, tokenizer, texts, device, ml, batch_size,
                requires_grad=False, normalize=normalize)
            q = enc([t[0] for t in triples], max_q)
            p = enc([t[1] for t in triples], max_p)
            n = enc([t[2] for t in triples], max_p)
            margin = ((q * p).sum(-1) - (q * n).sum(-1)).float()
    finally:
        if was_training:
            encoder.train()
    return {
        "n": len(triples),
        "margin_mean": float(margin.mean()),
        "margin_p10": float(margin.quantile(0.10)),
        "rank_acc": float((margin > 0).float().mean()),
    }


def attach_training_diagnostics(output_dir, probe_fn=None, *, probe_fractions=(0.5,)):
    """Persist loss/LR/grad-norm and run ``probe_fn`` at >= 2 points.

    Registered by appending to transformers.trainer.DEFAULT_CALLBACKS, which
    Trainer.__init__ reads. This is the same monkey-patch idiom as
    patch_tevatron_loss, and it is the only way to reach a Trainer that Tevatron
    constructs from an argv list. transformers is imported here rather than at
    module scope so BM25 -- which imports this module -- does not acquire the
    dependency.
    """
    import transformers.trainer as _trainer_module
    from transformers import TrainerCallback

    if not hasattr(_trainer_module, 'DEFAULT_CALLBACKS'):
        raise RuntimeError(
            "transformers.trainer.DEFAULT_CALLBACKS is absent; training diagnostics "
            "cannot be attached and the run would report success without evidence.")

    log_path = Path(output_dir) / TRAINING_LOG_NAME

    class TrainingDiagnosticsCallback(TrainerCallback):
        """Loss/LR/pre-clipping grad norm every logging step, plus probe points."""

        def __init__(self):
            self._probed = set()

        # Rank 0 only: every rank would otherwise interleave lines into one file.
        @staticmethod
        def _main(state):
            return getattr(state, 'is_world_process_zero', True)

        def _run_probe(self, phase, state, kwargs):
            if probe_fn is None or not self._main(state):
                return
            model = kwargs.get('model')
            tokenizer = kwargs.get('processing_class') or kwargs.get('tokenizer')
            if model is None:
                return
            try:
                result = probe_fn(model, tokenizer)
            except Exception as e:                                 # noqa: BLE001
                # A probe failure must not kill a training run; it is diagnostic.
                # It is recorded so its absence is never mistaken for a passing probe.
                result = {"error": f"{type(e).__name__}: {e}"}
            append_jsonl(log_path, {"global_step": int(state.global_step),
                                    "phase": phase, **result})

        def on_train_begin(self, args, state, control, **kwargs):
            self._run_probe("begin", state, kwargs)

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if 'loss' not in logs or not self._main(state):
                return
            loss = logs['loss']
            if not math.isfinite(float(loss)):
                raise RunDirectoryError(
                    f"non-finite loss={loss} at step {state.global_step}; stopping "
                    f"rather than saving a diverged checkpoint.")
            append_jsonl(log_path, {
                "global_step": int(state.global_step),
                "epoch": logs.get('epoch'),
                "loss": loss,
                "learning_rate": logs.get('learning_rate'),
                # HF logs the norm returned by clip_grad_norm_, i.e. PRE-clipping.
                "grad_norm": logs.get('grad_norm'),
            })

        def on_step_end(self, args, state, control, **kwargs):
            total = getattr(state, 'max_steps', 0) or 0
            for fraction in probe_fractions:
                if fraction in self._probed or total <= 0:
                    continue
                if state.global_step >= max(1, int(fraction * total)):
                    self._probed.add(fraction)
                    self._run_probe(f"step_{int(fraction * 100)}pct", state, kwargs)

        def on_train_end(self, args, state, control, **kwargs):
            self._run_probe("end", state, kwargs)

    # Idempotent: a second call must not double every record.
    _trainer_module.DEFAULT_CALLBACKS[:] = [
        cb for cb in _trainer_module.DEFAULT_CALLBACKS
        if getattr(cb, '__name__', '') != 'TrainingDiagnosticsCallback']
    _trainer_module.DEFAULT_CALLBACKS.append(TrainingDiagnosticsCallback)
    return log_path


def require_recipe_keys(recipe_name, recipe, consumed, optional=()) -> None:
    """config.yaml is the source of truth: every declared key must be consumed.

    Same contract as require_mixture_files' strict mode, applied to config keys
    instead of files. An unused key is how `target_batch_size`, `gc_p_chunk_size`,
    `grad_cache` and `bf16` came to be declared for cross-batch and silently
    ignored while the code used literals; a missing key is how a recipe rename
    reaches training as a KeyError mid-job. ``optional`` keys may be absent --
    cross-batch's LoRA block is opt-in and the recipe declares it only when used.
    """
    declared, consumed = set(recipe), set(consumed)
    missing = sorted(consumed - declared)
    unused = sorted(declared - consumed - set(optional))
    if missing or unused:
        parts = []
        if missing:
            parts.append(f"consumed but not declared: {', '.join(missing)}")
        if unused:
            parts.append(f"declared but never consumed: {', '.join(unused)}")
        raise ValueError(
            f"config.yaml training.{recipe_name} is inconsistent with the code -- "
            + "; ".join(parts))
