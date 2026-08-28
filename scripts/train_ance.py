"""ANCE orchestrator: initial mine, then a supervised Trainer/Inferencer pair.

Architecture (paper Figure 2, Appendix A.3): the Trainer learns continuously on
GPU 0 while the Inferencer re-encodes the whole corpus on GPU 1 and publishes a
refreshed ANN round. Training never pauses for a refresh.

Three properties this file is responsible for, none of which used to hold:

* **Provenance.** Every round a run trains on was mined by THIS run, against THIS
  run's corpus/queries/qrels. The work root is unique per invocation and its id is
  written into the run manifest; rounds carry that id and the trainer refuses any
  round that does not. A leftover `ready_7` from an earlier run is how another run's
  negatives used to reach the trainer at the first logging step.
* **Refresh.** A dead Inferencer leaves the Trainer cycling base-model negatives to
  `max_steps` and exiting 0 -- static hard-negative training wearing ANCE's name.
  Both workers are supervised, and the run is not successful without a consumed,
  checkpoint-derived round.
* **Success.** Validated by `assert_training_succeeded`, not inferred from exit 0.

The initial round is mined here, by the parent, and marked `ready_initial`. It is a
base-model round (`checkpoint_step 0`), so it is deliberately NOT counted as a
refresh -- the same distinction `async_fast_grass_handoff.latest_committed_round`
draws by ignoring `ready_initial`.
"""

import os
import sys
import json
import random
import argparse
import subprocess
import pickle
import uuid
from pathlib import Path
from tevatron.retriever.modeling import DenseModel

# Hardware & Project Setup
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import get_path, get_training_context, load_config, \
                          encode_to_pickle, build_faiss_index, \
                          _load_qrels, _load_corpus_lookup, log_startup_config, \
                          build_run_manifest, prepare_output_dir, set_seed, \
                          assert_training_succeeded, require_recipe_keys, _sha256, \
                          RUN_MANIFEST_NAME, atomic_write
from data.preprocessor import (BRIGHTPreprocessor, MIXTURE_FILES, set_msmarco_revisions,
                               MSMARCO_ONLY_FILES, require_derived_artifacts,
                               require_mixture_files)
from ance_mining import (INITIAL_ROUND, assert_ance_refresh, build_round_records,
                         mine_from_index, publish_round)

# 🩹 Tevatron Bug Patch
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)

TRAINER_SUMMARY_NAME = "ance_trainer_summary.json"

# Every training.<recipe> key this pipeline reads, across all three processes.
# require_recipe_keys fails on a declared-but-unread key, which is how
# `gradient_checkpointing` and `save_total_limit` came to be declared while the
# trainer implemented neither.
CONSUMED_KEYS = (
    'base_model', 'model_name', 'train_group_size', 'mining_depth', 'total_epochs',
    'learning_rate', 'batch_size', 'per_device_eval_batch_size', 'bf16',
    'dataloader_num_workers', 'warmup_ratio', 'weight_decay', 'max_grad_norm',
    'save_steps', 'logging_steps', 'eval_top_k', 'data_gen_poll_interval',
    'corpus_file', 'train_queries_file', 'train_qrels_file', 'mixture_dir',
    'temp_workdir', 'setup_mode', 'eval_corpus_file', 'eval_queries_file',
    'eval_qrels_file', 'eval_metric',
)

# Keys only the paper-fidelity recipe declares. require_recipe_keys fails on a
# declared-but-unread key, so they are listed rather than silently tolerated.
PAPER_KEYS = ('paper_fidelity', 'lamb_eps', 'warmup_steps', 'query_max_len',
              'passage_max_len', 'normalize', 'temperature', 'pooling',
              'forbidden_init_sha256')


def assert_permitted_init(model_path, forbidden_sha256):
    """Identify the init and refuse any configured finished-checkpoint hash.

    Training from it would 'reproduce' 0.330 by construction while reproducing
    nothing. Matched on the weight file's content hash, because a filename check is
    defeated by a copy or a rename. The deny-list may be empty until the released
    artifact is downloaded; the returned hash always records the actual init in the
    run manifest for comparison with the locally evaluated warm-up.
    """
    weights = next((Path(model_path) / n for n in ('model.safetensors',
                                                   'pytorch_model.bin')
                    if (Path(model_path) / n).is_file()), None)
    if weights is None:
        raise RuntimeError(f"{model_path} holds no weight file; it cannot be "
                           f"identified, so it cannot initialize a reproduction.")
    digest = _sha256(weights)
    if digest in {h.lower() for h in (forbidden_sha256 or ()) if h}:
        raise RuntimeError(
            f"{model_path} is the released, finished ANCE checkpoint (sha256 "
            f"{digest[:16]}...). It is an evaluation reference only. Initialize from "
            f"the BM25 warm-up checkpoint instead.")
    return digest


def run_setup(recipe_args):
    """Resolve corpus/queries/qrels for the recipe.

    The reasonir_mixture path only *verifies* -- those files are built by
    `python src/data/preprocessor.py`, so training never regenerates its own inputs.
    """
    p = get_path("processed")

    if recipe_args['setup_mode'] == 'tevatron_msmarco':
        # Pin the HF dataset revision before anything is built. "Tevatron/msmarco-passage
        # at revision X" is the reproduction's data provenance; unpinned, a rebuild
        # months later is a different corpus with the same filename.
        revisions = load_config()['data'].get('msmarco_reproduction') or {}
        set_msmarco_revisions(
            passage=revisions.get('passage_revision'),
            corpus=revisions.get('corpus_revision'))
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


def preflight_inputs(mixture_files, query_file, qrels_file, corpus_lookup, qrels_dict):
    """Every mixture query must be encodable, judged and resolvable in the corpus.

    Discovering any of this mid-run costs the whole allocation: an unmined query
    stops the round from publishing, and an unresolvable positive would have been
    padded into the loss by the old loader.
    """
    encodable = set()
    with open(query_file, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                encodable.add(str(json.loads(line)['query_id']))

    missing_q, missing_qrel, missing_doc = [], [], []
    n_records = 0
    for path in mixture_files:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                n_records += 1
                record = json.loads(line)
                qid = str(record['query_id'])
                if qid not in encodable:
                    missing_q.append(qid)
                if qid not in qrels_dict:
                    missing_qrel.append(qid)
                for p in record.get('positive_passages') or []:
                    if str(p['docid']) not in corpus_lookup:
                        missing_doc.append(str(p['docid']))

    problems = []
    if missing_q:
        problems.append(f"{len(missing_q)} mixture query id(s) absent from "
                        f"{Path(query_file).name}, e.g. {missing_q[:5]}")
    if missing_qrel:
        problems.append(f"{len(missing_qrel)} mixture query id(s) absent from "
                        f"{Path(qrels_file).name}, e.g. {missing_qrel[:5]}")
    if missing_doc:
        problems.append(f"{len(missing_doc)} positive docid(s) absent from the "
                        f"corpus, e.g. {missing_doc[:5]}")
    if problems:
        raise RuntimeError(
            "ANCE input preflight failed: " + "; ".join(problems) +
            ". Rebuild the derived artifacts against this mixture before training.")
    return n_records


def calculate_training_budget(n_examples, recipe):
    """Return the optimizer budget for the dataset the trainer actually iterates.

    BGE ANCE keeps all passages for a query in one grouped dataset item. Paper mode
    expands each of its mined negatives into a separate pairwise triplet, so an epoch
    contains ``n_examples * (train_group_size - 1)`` items. Keeping this calculation in
    the orchestrator lets the manifest and worker share one exact max_steps value.
    """
    n_examples = int(n_examples)
    batch_size = int(recipe['batch_size'])
    total_epochs = int(recipe['total_epochs'])
    triplets_per_query = (int(recipe['train_group_size']) - 1
                          if recipe.get('paper_fidelity') else 1)
    training_instances = n_examples * triplets_per_query
    steps_per_epoch = max(training_instances // batch_size, 1)
    max_steps = steps_per_epoch * total_epochs
    return {
        'query_records': n_examples,
        'triplets_per_query': triplets_per_query,
        'training_instances': training_instances,
        'steps_per_epoch': steps_per_epoch,
        'total_epochs': total_epochs,
        'max_steps': max_steps,
        'triplets_processed': max_steps * batch_size,
    }


def negative_pool_manifest(recipe, batch_size):
    """Describe mined candidate diversity separately from loss-pool width."""
    mined = int(recipe['train_group_size']) - 1
    if recipe.get('paper_fidelity'):
        return {'negative_pool_size': 1, 'mined_negatives_per_query': mined,
                'triplets_per_query': mined}
    return {'negative_pool_size': int(batch_size) * int(recipe['train_group_size']) - 1,
            'mined_negatives_per_query': mined, 'triplets_per_query': 1}


def mine_initial_round(ctx, config, *, corpus_file, query_file, mixture_files,
                       corpus_lookup, qrels_dict, work_root, base_model, run_id,
                       rng):
    """Round `initial`: mined by the base model so the Trainer has data at step 0.

    Kept in the parent, not moved into the Inferencer. Doing it here needs no
    blocking startup wait, cannot deadlock on an Inferencer that dies before
    publishing, and uses GPU 0 while the Trainer has not started.
    """
    print(f"[ANCE] Initial encode+mine using base model: {base_model}", flush=True)
    staging = work_root / "initial_encode"
    staging.mkdir(parents=True, exist_ok=True)
    encode_to_pickle(base_model, corpus_file, staging / "corpus.pkl", False, ctx, config)
    encode_to_pickle(base_model, query_file,  staging / "query.pkl",  True,  ctx, config)

    index, _, corpus_ids = build_faiss_index(staging / "corpus.pkl")
    with open(staging / "query.pkl", 'rb') as f:
        q_data = pickle.load(f)

    mined, failures = mine_from_index(
        index, corpus_ids, q_data, mixture_files, qrels_dict,
        n_negs=ctx['args']['train_group_size'] - 1,
        mining_depth=ctx['args']['mining_depth'], rng=rng)

    publish_round(
        work_root, INITIAL_ROUND,
        records_by_file=build_round_records(mixture_files, mined, corpus_lookup,
                                            n_negs=ctx['args']['train_group_size'] - 1),
        meta={'run_id': run_id, 'ann_no': INITIAL_ROUND,
              'checkpoint': str(base_model), 'checkpoint_step': 0,
              'n_queries_mined': len(mined), 'n_sampling_failures': len(failures),
              'sampling_failures': failures[:20],
              'corpus_sha256': _sha256(corpus_file)})
    import shutil
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[ANCE] Initial round committed in {work_root}", flush=True)


def build_run_id(manifest):
    """A work-root name no concurrent invocation can collide with.

    The fingerprint identifies the CONFIGURATION, so it is identical across reruns by
    construction; a wall-clock second is not enough to separate them. A SLURM array
    launches its tasks in the same second with the same recipe, which is exactly the
    case that produced one shared work root where each task refused the other's
    rounds. The uuid is what actually makes this unique; the job id is carried for
    traceability from a work root back to a log file.
    """
    parts = [manifest['fingerprint'][:12], str(int(manifest['started_at_epoch']))]
    job = os.environ.get('SLURM_JOB_ID')
    if job:
        parts.append(f"j{job}")
    parts.append(uuid.uuid4().hex[:8])
    return "-".join(parts)


def require_ance_gpus(n_gpus):
    """ANCE is a 1:1 Trainer:Inferencer pipeline; refuse a degraded allocation."""
    n_gpus = int(n_gpus)
    if n_gpus < 2:
        raise RuntimeError(
            f"ANCE requires 2 visible GPUs (Trainer GPU 0 / Inferencer GPU 1), "
            f"but torch sees {n_gpus}. Refusing to run both workers on one device.")
    return n_gpus


def supervise(trainer, inferencer, poll_seconds=5.0, grace=120, log=print):
    """Run until the trainer exits, failing the run if the inferencer stops first.

    **Any** inferencer exit before we ask it to stop is a failure, a clean rc 0
    included. `run_ance_data_gen.main()` loops until it is terminated and has no
    `--max_rounds` equivalent, so there is no code path on which it finishes early
    and legitimately -- an early exit means refreshes stopped, which degenerates the
    run into static hard-negative training on whatever round was current while it
    still looks like a successful ANCE run. (`train_async_fast_grass.supervise`
    tolerates a clean miner exit because `--max_rounds` makes one legitimate there.
    Copying that rule to ANCE imported an exemption for a flag ANCE does not have.)

    Checked inside the loop and again after the trainer finishes, so an inferencer
    that dies in the same window as a trainer exit is still caught.

    Returns ``(inferencer_failure_returncode_or_None, trainer_returncode)``. The
    failure code may be 0, so callers must test `is not None`, never truthiness.
    """
    failed = None
    stop_requested = False

    def _early_exit():
        """The inferencer's return code if it stopped on its own, else None."""
        return None if stop_requested or inferencer.poll() is None \
            else inferencer.returncode

    try:
        while True:
            failed = _early_exit()
            if failed is not None:
                log(f"[ANCE] ERROR: inferencer exited with code {failed} while the "
                    f"trainer was still running — no further ANN refresh is "
                    f"possible, so the trainer is being terminated rather than left "
                    f"to finish on stale negatives")
                _stop(trainer, grace)
                break
            try:
                trainer.wait(timeout=poll_seconds)
                break
            except subprocess.TimeoutExpired:
                pass
    finally:
        # Re-check AFTER the loop: the inferencer may have died during the same
        # window in which the trainer exited, which the in-loop check would miss.
        if failed is None:
            failed = _early_exit()
            if failed is not None:
                log(f"[ANCE] ERROR: inferencer exited with code {failed}, detected "
                    f"after the trainer finished — the run consumed stale mined data")
        if inferencer.poll() is None:
            stop_requested = True          # our termination is not an early exit
            _stop(inferencer, grace)
    return failed, trainer.returncode


def _stop(proc, grace):
    """terminate, then kill if it will not go."""
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', default='ance')
    recipe_name = parser.parse_args().recipe

    config = load_config()
    seed = config.get('seed', 42)
    set_seed(seed)

    ctx = get_training_context(recipe_name)
    recipe = ctx['args']
    consumed = CONSUMED_KEYS
    if recipe.get('paper_fidelity'):
        # Paper mode uses an absolute warmup_steps value, not warmup_ratio.
        consumed = tuple(k for k in consumed if k != 'warmup_ratio') + PAPER_KEYS
    require_recipe_keys(recipe_name, recipe, consumed)
    log_startup_config(recipe_name, ctx)
    corpus_file, query_file, qrels_file = run_setup(recipe)

    # Detect GPU count BEFORE restricting visibility.
    # With --gpus-per-task=2, SLURM sets CUDA_VISIBLE_DEVICES=0,1.
    # Tevatron encode raises NotImplementedError on multi-GPU, so we pin the
    # orchestrator to GPU 0 for all encode_to_pickle calls (initial mine).
    # Inferencer/Trainer subprocesses override this with their own assignments.
    import torch as _torch
    n_gpus = require_ance_gpus(_torch.cuda.device_count())
    infer_gpu = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"[ANCE] {n_gpus} GPU(s) detected — Trainer→GPU 0, Inferencer→GPU {infer_gpu}",
          flush=True)

    mixture_dir   = get_path("processed") / recipe['mixture_dir']
    expected      = (MSMARCO_ONLY_FILES if recipe['setup_mode'] == 'tevatron_msmarco'
                     else MIXTURE_FILES)
    mixture_files = list(require_mixture_files(mixture_dir, expected))

    corpus_lookup = _load_corpus_lookup(corpus_file)
    qrels_dict    = _load_qrels(qrels_file)
    n_examples    = preflight_inputs(mixture_files, query_file, qrels_file,
                                     corpus_lookup, qrels_dict)
    if n_examples == 0:
        raise RuntimeError(f"No training examples found in {mixture_dir}.")

    # Floor matches drop_last=True. Paper mode expands every mined negative into its
    # own triplet; ordinary BGE mode keeps one grouped record per query.
    batch_size = recipe['batch_size']
    budget = calculate_training_budget(n_examples, recipe)
    steps_per_epoch = budget['steps_per_epoch']
    total_epochs = budget['total_epochs']
    max_steps = budget['max_steps']
    print(f"[ANCE] {n_examples} query records × {budget['triplets_per_query']} "
          f"triplet(s)/query = {budget['training_instances']} training instances | "
          f"{steps_per_epoch} steps/epoch (floor) | {total_epochs} epochs | "
          f"{max_steps} total steps", flush=True)

    ance_base_model  = recipe.get('base_model', ctx['base_model'])
    output_model_dir = get_path("models") / recipe['model_name']
    print(f"[ANCE] Starting from model: {ance_base_model}", flush=True)

    # Run identity. The derived corpus/queries/qrels are hashed alongside the
    # mixture: they are what every ANN round is mined against, and P-PRE-02 is the
    # record of them silently predating the code that reads them.
    derived = {'corpus': corpus_file, 'queries': query_file, 'qrels': qrels_file}
    paper_provenance = None
    if recipe.get('paper_fidelity'):
        paper_provenance = {
            'upstream_code_executed': False,
            'init_sha256': assert_permitted_init(
                ance_base_model, recipe.get('forbidden_init_sha256')),
            'similarity': 'dot', 'normalize': False, 'temperature': None,
            'note': ("Microsoft supplied initialization weights and behavioural "
                     "specifications. The run, miner, trainer and evaluator are "
                     "this repository's."),
        }
    pool = negative_pool_manifest(recipe, batch_size)
    manifest = build_run_manifest(
        recipe_name, ctx, recipe,
        data_files=mixture_files,
        world_size=1,
        negative_pool_size=pool['negative_pool_size'],
        optimizer_steps=max_steps,
        extra={'derived_artifacts': {k: {'path': str(v), 'sha256': _sha256(v)}
                                     for k, v in derived.items()},
               'training_budget': budget,
               'steps_per_epoch': steps_per_epoch,
               'steps_per_epoch_rule': ('floor(query_records * triplets_per_query / '
                                        'batch_size)'),
               'mined_negatives_per_query': pool['mined_negatives_per_query'],
               'triplets_per_query': pool['triplets_per_query'],
               'paper_provenance': paper_provenance})
    # overwrite=True keeps ANCE's existing behaviour: it always starts fresh and
    # never refuses a dir. get_last_checkpoint() returns the highest step-numbered
    # checkpoint, so a stale checkpoint-17202 from a prior run would shadow every
    # new save and keep the inferencer stuck forever on the old weights.
    prepare_output_dir(output_model_dir, manifest, overwrite=True)

    run_id = build_run_id(manifest)
    work_root = get_path(recipe['temp_workdir']) / run_id
    try:
        # exist_ok=False: a collision must be a startup error, never a silently
        # shared work root in which each run is the other's "foreign run".
        work_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"work root {work_root} already exists. Two invocations cannot share "
            f"one: each would publish rounds the other refuses, and the initial "
            f"round would be overwritten mid-training.") from exc
    manifest['run_id'] = run_id
    manifest['work_root'] = str(work_root)
    with atomic_write(output_model_dir / RUN_MANIFEST_NAME) as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[ANCE] run_id={run_id} | work_root={work_root}", flush=True)

    # ── INITIAL ROUND (always re-mined; there is no resume in this pipeline) ──
    mine_initial_round(ctx, config, corpus_file=corpus_file, query_file=query_file,
                       mixture_files=mixture_files, corpus_lookup=corpus_lookup,
                       qrels_dict=qrels_dict, work_root=work_root,
                       base_model=ance_base_model, run_id=run_id,
                       rng=random.Random(seed))
    del corpus_lookup

    common = ['--work_root', str(work_root), '--run_id', run_id,
              '--recipe', recipe_name]
    infer_proc = subprocess.Popen([
        sys.executable, str(Path(__file__).parent / "run_ance_data_gen.py"),
        '--output_model_dir', str(output_model_dir),
        '--corpus_file',      str(corpus_file),
        '--query_file',       str(query_file),
        '--qrels_file',       str(qrels_file),
        *common,
    ], env={**os.environ, 'CUDA_VISIBLE_DEVICES': infer_gpu})
    print(f"[ANCE] Inferencer started on GPU {infer_gpu} (pid {infer_proc.pid})",
          flush=True)

    train_proc = subprocess.Popen([
        sys.executable, str(Path(__file__).parent / "run_ance_train.py"),
        '--model_name_or_path', ance_base_model,
        '--output_dir',         str(output_model_dir),
        '--max_steps',          str(max_steps),
        '--seed',               str(seed),
        *common,
    ], env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})
    print(f"[ANCE] Trainer started on GPU 0 (pid {train_proc.pid})", flush=True)

    infer_failed, train_rc = supervise(train_proc, infer_proc)
    if infer_failed is not None:
        raise RuntimeError(
            f"the ANCE inferencer exited early with code {infer_failed}. It is meant "
            f"to run until terminated, so ANY early exit — a clean one included — "
            f"means refreshes stopped and the trainer continued on whatever round "
            f"was current. That is static hard-negative training, not ANCE. See the "
            f"traceback in the .err log.")
    if train_rc != 0:
        raise RuntimeError(f"the ANCE trainer exited with code {train_rc}")

    # ── VALIDATE: refresh happened, then that training happened ──────────────
    summary_path = output_model_dir / TRAINER_SUMMARY_NAME
    if not summary_path.is_file():
        raise RuntimeError(
            f"{summary_path} was not written, so no round consumption evidence "
            f"exists and the run cannot be shown to have refreshed.")
    summary = json.loads(summary_path.read_text())
    fresh = assert_ance_refresh(summary)
    print(f"[ANCE] {len(fresh)} checkpoint-derived round(s) consumed: "
          f"{[r['ann_no'] for r in fresh]}", flush=True)

    stored = assert_training_succeeded(output_model_dir, manifest)
    stored.update({'ann_rounds': summary.get('rounds', []),
                   'fresh_rounds_consumed': len(fresh),
                   'final_refresh': fresh[-1] if fresh else None,
                   # what the trainer REALLY built, read back rather than recomputed,
                   # so the manifest cannot claim an optimizer the run did not use
                   'optimizer': summary.get('optimizer'),
                   'run_id': run_id, 'work_root': str(work_root)})
    with atomic_write(output_model_dir / RUN_MANIFEST_NAME) as f:
        json.dump(stored, f, indent=2, default=str)

    print(f"\n✅ ANCE training validated. Model: {output_model_dir}", flush=True)
    _print_eval_instructions(recipe_name, recipe, output_model_dir)


def _print_eval_instructions(recipe_name, recipe, output_model_dir):
    """No in-job BRIGHT evaluation. It printed a bare mean NDCG@10 with no summary,
    no artifact hashes and no link to the checkpoint's manifest -- the exact shape of
    number that became the quarantined 0.1683 (P-ANCE-01). The reportable path is
    run_all_evals.py, which covers all twelve domains, applies the BRIGHT exclusion
    filter and writes eval_artifact_sha256 alongside training provenance.
    """
    print("=" * 72, flush=True)
    if recipe.get('eval_corpus_file'):
        print("  MS MARCO evaluation (MRR@10 + Recall@1000):\n", flush=True)
        print(f"    python scripts/eval_msmarco.py --recipe {recipe_name} "
              f"--model_path {output_model_dir}",
              flush=True)
    else:
        print("  BRIGHT evaluation — all 12 domains, exclusion-aware, hashed:\n",
              flush=True)
        print(f"    EVAL_REQUIRE_EXISTING=1 EVAL_DOMAINS=all "
              f"EVAL_MODEL_PATH={output_model_dir} \\\n"
              f"      sbatch scripts/launchers/run_evaluate_singularity.sh", flush=True)
        print("\n  EVAL_REQUIRE_EXISTING=1 is not optional: without it a missing "
              "domain is\n  rebuilt from HuggingFace mid-evaluation, regenerating "
              "processed data\n  underneath the comparison.", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
