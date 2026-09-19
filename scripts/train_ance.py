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
import shutil
import uuid
import datetime
import hashlib
from pathlib import Path
from tevatron.retriever.modeling import DenseModel

# Hardware & Project Setup
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.runtime_timing import PhaseTimer
from utils.helpers import get_path, get_training_context, load_config, \
                          _load_qrels, _load_corpus_lookup, log_startup_config, \
                          build_run_manifest, prepare_output_dir, set_seed, \
                          assert_training_succeeded, require_recipe_keys, _sha256, \
                          RUN_MANIFEST_NAME, atomic_write, _code_revision
from data.preprocessor import (BRIGHTPreprocessor, MIXTURE_FILES, set_msmarco_revisions,
                               MSMARCO_ONLY_FILES, require_derived_artifacts,
                               require_mixture_files, _trec_safe_docid)
from ance_mining import (INITIAL_ROUND, adopt_prepared_round, assert_ance_refresh,
                         build_round_records, encode_and_mine, publish_round)

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
    'eval_qrels_file', 'eval_metric', 'ann_chunk_factor', 'max_encode_seconds',
    'gradient_checkpointing', 'min_fresh_rounds',
)

# Keys only the paper-fidelity recipe declares. require_recipe_keys fails on a
# declared-but-unread key, so they are listed rather than silently tolerated.
# `total_epochs` is deliberately absent: the reproduction is step-budgeted against
# the released checkpoint, not epoch-budgeted against our expanded dataset.
PAPER_KEYS = ('paper_fidelity', 'lamb_eps', 'warmup_steps', 'query_max_len',
              'passage_max_len', 'normalize', 'temperature', 'pooling',
              'expected_init_sha256', 'scheduler_max_steps', 'train_stop_steps')


def assert_permitted_init(model_path, expected_sha256):
    """Require the documented BM25 warm-up, by exact content hash.

    NOT Microsoft's released 60K warm-up: that artifact no longer exists. Both blob
    URLs return HTTP 409, microsoft/ANCE issues #23/#24/#26 have been open since 2022,
    and the only surviving mirror is bit-identical to the released 600K FINAL. So the
    pinned hash is OUR warm-up, trained by `train_ance_warmup.py` from roberta-base on
    the same kind of BM25 negatives -- a deviation recorded in paper_provenance.

    That makes this check MORE necessary, not less. A deny-list can only refuse hashes
    someone thought to list, so an empty one passes any structurally compatible
    checkpoint -- the finished 600K weights included, which would 'reproduce' 0.330 by
    construction while reproducing nothing, and which are now the easiest ANCE
    checkpoint on the internet to download by mistake. An allow-list of one inverts
    that: every checkpoint that is not the documented starting point is refused,
    whatever it is. Matched on content, because a filename check is defeated by a copy
    or a rename.
    """
    weights = next((Path(model_path) / n for n in ('model.safetensors',
                                                   'pytorch_model.bin')
                    if (Path(model_path) / n).is_file()), None)
    if weights is None:
        raise RuntimeError(f"{model_path} holds no weight file; it cannot be "
                           f"identified, so it cannot initialize a reproduction.")
    digest = _sha256(weights)
    expected = (expected_sha256 or "").strip().lower()
    if not expected:
        raise RuntimeError(
            f"training.ance_paper.expected_init_sha256 is empty, so the reproduction "
            f"cannot prove which warm-up it started from. The init at {model_path} "
            f"hashes to {digest}. Verify this is the warm-up produced by "
            f"train_ance_warmup.py -- NOT the released 600K final, which is what a "
            f"downloaded 'ANCE checkpoint' almost always is -- then record it in "
            f"config/config.yaml.")
    if digest != expected:
        raise RuntimeError(
            f"{weights.name} hashes to {digest}, not the allow-listed warm-up "
            f"{expected}. A reproduction does not start from an unidentified "
            f"checkpoint.")
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
        # The dev split is part of what setup owes this recipe. Without it the reuse
        # path could skip a build whose dev artifacts were never produced, and the run
        # would train for 24h before the evaluator discovered it had nothing to score
        # against.
        dev_set = tuple(p / recipe_args[k] for k in
                        ('eval_queries_file', 'eval_qrels_file') if recipe_args.get(k))
        if all(x.exists() and x.stat().st_size > 0
               for x in train_set + dev_set + (corpus_path,)):
            print("⏩ Skipping setup: files already exist.", flush=True)
            require_mixture_files(mixture_path.parent, MSMARCO_ONLY_FILES)
            return require_derived_artifacts(
                output_dir=p, corpus_file=recipe_args['corpus_file'],
                queries_file=recipe_args['train_queries_file'],
                qrels_file=recipe_args['train_qrels_file'])

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
        if dev_set and not all(x.exists() and x.stat().st_size > 0 for x in dev_set):
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


def _digest(text):
    """The text digest `preprocessor._derive` canonicalizes duplicate passages by.

    Must stay byte-identical to that function's
    `hashlib.md5(text.strip().encode()).hexdigest()`, or a legitimately collapsed
    positive reads as a missing one.
    """
    return hashlib.md5(("" if text is None else str(text))
                       .strip().encode()).hexdigest()


def _canonical_docids(corpus_lookup, wanted):
    """Corpus docid for each wanted text digest.

    Built from the already-loaded lookup rather than a second read of the corpus file,
    and only for the digests a positive actually asked for: on MS MARCO a full reverse
    map would be 8.8M entries built to answer a few hundred questions.
    """
    found = {}
    if not wanted:
        return found
    for docid, text in corpus_lookup.items():
        digest = _digest(text)
        if digest in wanted:
            found[digest] = docid
    return found


def _resolve_positives(misses, corpus_lookup, qrels_dict):
    """Classify positives whose raw docid the corpus does not hold.

    `preprocessor._derive` publishes a corpus and qrels that BOTH carry canonical
    docids while the training mixture keeps the raw id it was built with. Two
    remappings produce a raw id the corpus cannot resolve, and neither is a defect:

    * whitespace -- `_trec_safe_docid` percent-encodes an id that would otherwise
      break a TREC column;
    * duplicate text -- identical passages collapse onto one canonical owner, and
      only that owner is emitted.

    The same defect, and the same remap-through-the-qrels answer, is already handled
    for the async arm by `async_fast_grass_cached_mcdp.canonicalize_positives`.

    Requiring the canonical owner to be one of the query's qrels is what keeps this
    STRONGER than a bare corpus-membership test rather than weaker: a corpus or qrels
    regenerated against a DIFFERENT mixture stops resolving and is named here, which
    is the drift the membership test was reaching for and could not distinguish.

    Returns the misses that stay unexplained.
    """
    if not misses:
        return []
    escaped, remaining = [], []
    for qid, raw, digest in misses:
        if _trec_safe_docid(raw) in corpus_lookup:
            escaped.append(raw)
        else:
            remaining.append((qid, raw, digest))

    canonical = _canonical_docids(corpus_lookup,
                                  {digest for _, _, digest in remaining})
    unexplained, collapsed = [], 0
    for qid, raw, digest in remaining:
        owner = canonical.get(digest)
        if owner is None:
            unexplained.append(f"{raw!r} (query {qid}): its text is absent from the "
                               f"corpus under any docid")
        elif owner not in (qrels_dict.get(qid) or ()):
            unexplained.append(f"{raw!r} (query {qid}): its text is in the corpus as "
                               f"{owner!r}, which that query's qrels do not carry")
        else:
            collapsed += 1
    if escaped or collapsed:
        print(f"[ANCE] preflight canonicalized {len(escaped) + collapsed:,} positive(s) "
              f"the corpus holds under another docid ({len(escaped):,} "
              f"whitespace-escaped, {collapsed:,} duplicate-text)", flush=True)
    return unexplained


def preflight_inputs(mixture_files, query_file, qrels_file, corpus_lookup, qrels_dict):
    """Every mixture query must be encodable, judged, textually identical and
    resolvable in the corpus.

    Discovering any of this mid-run costs the whole allocation: an unmined query stops
    the round from publishing, and an unresolvable positive would have been padded into
    the loss by the old loader. Text consistency is checked because it fails SILENTLY
    -- the miner encodes the query file and labels the result with the mixture's qid,
    so a disagreement mines negatives for a different question than the one trained on
    and produces a complete, valid-looking round.

    A positive whose docid the corpus does not hold is NOT automatically a defect:
    `preprocessor._derive` remaps the corpus and the qrels but deliberately leaves the
    mixture carrying raw docids. `_resolve_positives` decides which of those misses
    are legitimate remappings and which are real staleness.
    """
    encodable = {}
    with open(query_file, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                encodable[str(row['query_id'])] = row['query']

    missing_q, missing_qrel, wrong_text, unresolved = [], [], [], []
    misses = []                       # (qid, raw docid, text digest)
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
                elif encodable[qid] != record['query']:
                    # The miner scores the ENCODED query and labels the result with
                    # the mixture's qid. If the two files disagree on the text, every
                    # negative is mined for a different question than the one trained
                    # on, and nothing downstream can tell.
                    wrong_text.append(qid)
                if qid not in qrels_dict:
                    missing_qrel.append(qid)
                positives = record.get('positive_passages') or []
                for p in positives:
                    if str(p['docid']) not in corpus_lookup:
                        misses.append((qid, str(p['docid']), _digest(p.get('text'))))
                # record_positives unions the record's own positives with the qrels'.
                # If that union has nothing the corpus can resolve, the round cannot
                # be mined -- select_ance_negatives has no positive to exclude.
                labelled = {str(p['docid']) for p in positives} | \
                           set(qrels_dict.get(qid) or ())
                if not any(d in corpus_lookup for d in labelled):
                    unresolved.append(qid)

    problems = []
    if missing_q:
        problems.append(f"{len(missing_q)} mixture query id(s) absent from "
                        f"{Path(query_file).name}, e.g. {missing_q[:5]}")
    if wrong_text:
        problems.append(f"{len(wrong_text)} mixture query id(s) whose text differs "
                        f"from {Path(query_file).name}, e.g. {wrong_text[:5]}")
    if missing_qrel:
        problems.append(f"{len(missing_qrel)} mixture query id(s) absent from "
                        f"{Path(qrels_file).name}, e.g. {missing_qrel[:5]}")
    unresolvable = _resolve_positives(misses, corpus_lookup, qrels_dict)
    if unresolvable:
        problems.append(f"{len(unresolvable)} positive(s) the corpus cannot resolve "
                        f"even after canonicalization: {'; '.join(unresolvable[:3])}")
    if unresolved:
        problems.append(f"{len(unresolved)} query(ies) whose every labelled positive "
                        f"is absent from the corpus, e.g. {unresolved[:5]}")
    if problems:
        raise RuntimeError(
            "ANCE input preflight failed: " + "; ".join(problems) +
            ". Rebuild the derived artifacts against this mixture before training.")
    return n_records


def create_work_root(work_root):
    """Claim a per-invocation work root, refusing one that already exists.

    exist_ok=False is the whole point: a collision must be a startup error, never a
    silently shared work root in which each run is the other's "foreign run" and the
    initial round is overwritten mid-training.
    """
    try:
        work_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"work root {work_root} already exists. Two invocations cannot share "
            f"one: each would publish rounds the other refuses, and the initial "
            f"round would be overwritten mid-training.") from exc
    return work_root


def calculate_training_budget(n_examples, recipe):
    """Return the optimizer budget for the dataset the trainer actually iterates.

    BGE ANCE keeps all passages for a query in one grouped dataset item. Paper mode
    expands each of its mined negatives into a separate pairwise triplet, so an epoch
    contains ``n_examples * (train_group_size - 1)`` items. Keeping this calculation in
    the orchestrator lets the manifest and worker share one exact max_steps value.

    Paper mode is STEP-budgeted, not epoch-budgeted: the reference stops at a named
    checkpoint (600K) while its schedule decays toward a separate horizon
    (max_steps default 1,000,000), and an epoch count over OUR expanded dataset is a
    different quantity that happens to be measured in the same unit. `train_stop_steps`
    and `scheduler_max_steps` say both numbers out loud.

    Sharding does not change the budget. `ann_chunk_factor` decides how much of the
    query set one ROUND covers; the optimizer still runs for the same number of steps
    over the same mixture, the loader simply restarts more often inside a round.
    """
    n_examples = int(n_examples)
    batch_size = int(recipe['batch_size'])
    chunk_factor = max(int(recipe['ann_chunk_factor']), 1)
    triplets_per_query = (int(recipe['train_group_size']) - 1
                          if recipe.get('paper_fidelity') else 1)
    training_instances = n_examples * triplets_per_query
    steps_per_epoch = max(training_instances // batch_size, 1)
    budget = {
        'query_records': n_examples,
        'queries_per_round': n_examples // chunk_factor,
        'ann_chunk_factor': chunk_factor,
        'triplets_per_query': triplets_per_query,
        'training_instances': training_instances,
        'steps_per_epoch': steps_per_epoch,
    }
    if recipe.get('train_stop_steps'):
        max_steps = int(recipe['train_stop_steps'])
        budget.update({'total_epochs': None,
                       'train_stop_steps': max_steps,
                       'scheduler_max_steps': int(recipe['scheduler_max_steps'])})
    else:
        total_epochs = int(recipe['total_epochs'])
        max_steps = steps_per_epoch * total_epochs
        budget.update({'total_epochs': total_epochs,
                       'scheduler_max_steps': max_steps})
    budget['max_steps'] = max_steps
    budget['triplets_processed'] = max_steps * batch_size
    return budget


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
    # The refresh rounds have carried phase timings since the inferencer passed its own
    # timer; the initial round passed none, which is why job 204931's 6h38m could not be
    # split into encode, serialization, index build and search -- the numbers the
    # decision about GPU search depends on.
    phase = PhaseTimer(events=Path(work_root) / "initial_mining_timings.jsonl")
    staging = work_root / "initial_encode"
    # ann_no 0. Upstream generates the initial round with --end_output_num 0 from
    # ann_no = -1, so it is output_num 0 and takes shard 0; the rotation then
    # continues 1, 2, ... through the refresh rounds.
    _, _, mined, failures, shard_qids, shard = encode_and_mine(
        base_model, staging, corpus_file=corpus_file, query_file=query_file,
        mixture_files=mixture_files, qrels_dict=qrels_dict, ctx=ctx, config=config,
        rng=rng, ann_no=0, phase=phase)

    publish_round(
        work_root, INITIAL_ROUND,
        records_by_file=build_round_records(
            mixture_files, mined, corpus_lookup,
            n_negs=ctx['args']['train_group_size'] - 1, shard_qids=shard_qids),
        meta={'run_id': run_id, 'ann_no': INITIAL_ROUND,
              'checkpoint': str(base_model), 'checkpoint_step': 0,
              'n_queries_mined': len(mined), 'n_sampling_failures': len(failures),
              'sampling_failures': failures[:20],
              'corpus_sha256': _sha256(corpus_file),
              'mining_seconds': dict(phase.timings), **shard})
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[ANCE] Initial round committed in {work_root}", flush=True)


PREPARED_MANIFEST_NAME = "prepared_initial.json"

# Tokenizer files differ by model family (RoBERTa has vocab/merges, XLM-R has the
# sentencepiece model), so the set is probed rather than required: a prepared artifact
# must match whatever the initialization actually ships, not a fixed list.
_TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                    'merges.txt', 'sentencepiece.bpe.model', 'special_tokens_map.json')


def _tokenizer_hashes(model_dir):
    model_dir = Path(model_dir)
    return {name: _sha256(model_dir / name) for name in _TOKENIZER_FILES
            if (model_dir / name).is_file()}


def initial_round_identity(recipe_name, ctx, recipe, *, base_model, corpus_file,
                           query_file, qrels_file, mixture_files, seed):
    """Everything a prepared round 0 must agree with before a run may adopt it.

    Round 0 is a function of exactly these inputs, so two runs agreeing on all of them
    would mine byte-identical rounds -- that is what makes reuse legitimate. The model
    PATH is recorded for diagnosis only: a path can be rebuilt in place, which is the
    substitution ``assert_permitted_init`` exists to catch.
    """
    base = Path(base_model)
    weights = next((base / n for n in ('model.safetensors', 'pytorch_model.bin')
                    if (base / n).is_file()), None)
    return {
        'recipe': recipe_name,
        'base_model_path': str(base_model),
        'base_model_weights_sha256': _sha256(weights) if weights else None,
        'tokenizer_sha256': _tokenizer_hashes(base),
        'corpus_sha256': _sha256(corpus_file),
        'queries_sha256': _sha256(query_file),
        'qrels_sha256': _sha256(qrels_file),
        'mixture_sha256': {Path(f).name: _sha256(f) for f in mixture_files},
        'seed': int(seed),
        'mining': {
            'mining_depth': int(recipe['mining_depth']),
            'n_negs': int(recipe['train_group_size']) - 1,
            'ann_chunk_factor': int(recipe['ann_chunk_factor']),
            'query_max_len': int(ctx['max_q']),
            'passage_max_len': int(ctx['max_p']),
            'per_device_eval_batch_size': int(recipe['per_device_eval_batch_size']),
            'paper_fidelity': bool(recipe.get('paper_fidelity')),
        },
    }


def prepare_initial_artifact(artifact_dir, *, ctx, config, identity, corpus_file,
                             query_file, mixture_files, corpus_lookup, qrels_dict,
                             base_model, seed):
    """Mine round 0 into a reusable artifact, on ONE GPU, outside a training job.

    Refuses to overwrite a committed artifact: a training job may already be queued
    against it.
    """
    artifact_dir = Path(artifact_dir)
    if (artifact_dir / f"ready_{INITIAL_ROUND}").exists():
        raise RuntimeError(
            f"{artifact_dir} already holds a committed initial round. Refusing to "
            f"overwrite it -- a training job may be queued against it. Point "
            f"--prepare-initial at a new directory if you need to re-mine.")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    prep_id = (f"prepared-{uuid.uuid4().hex[:12]}-"
               f"j{os.environ.get('SLURM_JOB_ID', 'local')}")
    mine_initial_round(ctx, config, corpus_file=corpus_file, query_file=query_file,
                       mixture_files=mixture_files, corpus_lookup=corpus_lookup,
                       qrels_dict=qrels_dict, work_root=artifact_dir,
                       base_model=base_model, run_id=prep_id,
                       rng=random.Random(seed))

    data_dir = artifact_dir / "training_data_initial"
    payload = {p.name: _sha256(p) for p in sorted(data_dir.glob("*.jsonl"))}
    round_meta = json.loads((artifact_dir / f"round_meta_{INITIAL_ROUND}.json")
                            .read_text())
    manifest = {
        'prepared_run_id': prep_id,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'prepared_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'code_revision': _code_revision(),
        'identity': identity,
        'payload_sha256': payload,
        'round_meta': round_meta,
    }
    with atomic_write(artifact_dir / PREPARED_MANIFEST_NAME) as handle:
        json.dump(manifest, handle, indent=2, default=str)
    return manifest


def assert_prepared_initial_matches(artifact_dir, identity):
    """Refuse a prepared round that was not mined from exactly these inputs.

    Reports EVERY mismatch, not the first -- fixing them one job at a time costs a
    queue wait each.
    """
    artifact_dir = Path(artifact_dir)
    path = artifact_dir / PREPARED_MANIFEST_NAME
    if not path.is_file():
        raise RuntimeError(
            f"{path} is missing, so this directory cannot be shown to hold a round "
            f"prepared for this recipe. Mine one with --prepare-initial.")
    try:
        prepared = json.loads(path.read_text())
    except ValueError as exc:
        raise RuntimeError(f"{path} is not valid JSON") from exc

    recorded = prepared.get('identity') or {}
    problems = [f"{key}: prepared={recorded.get(key)!r} current={value!r}"
                for key, value in identity.items() if recorded.get(key) != value]
    if problems:
        raise RuntimeError(
            "the prepared initial round does not match this run:\n  - "
            + "\n  - ".join(problems)
            + f"\nAdopting it would train on negatives mined for a different setup. "
              f"Re-run --prepare-initial for these inputs.")

    data_dir = artifact_dir / "training_data_initial"
    claimed = prepared.get('payload_sha256') or {}
    if not claimed:
        raise RuntimeError(f"{path} records no payload hashes, so the round's "
                           f"contents cannot be verified.")
    present = sorted(p.name for p in data_dir.glob("*.jsonl"))
    if present != sorted(claimed):
        raise RuntimeError(f"prepared payload {present} does not match the manifest "
                           f"{sorted(claimed)}")
    for name, digest in claimed.items():
        if _sha256(data_dir / name) != digest:
            raise RuntimeError(
                f"{data_dir / name} does not match the hash recorded when it was "
                f"prepared. The artifact has changed on disk; re-prepare it.")
    return prepared


def adopt_initial_round(artifact_dir, work_root, *, run_id, identity):
    """Validate a prepared artifact, then publish it as THIS run's round 0."""
    prepared = assert_prepared_initial_matches(artifact_dir, identity)
    meta = dict(prepared['round_meta'])
    meta['run_id'] = run_id
    meta['prepared_from'] = {
        'artifact_dir': str(artifact_dir),
        'prepared_run_id': prepared.get('prepared_run_id'),
        'prepared_at': prepared.get('prepared_at'),
        'slurm_job_id': prepared.get('slurm_job_id'),
        'code_revision': prepared.get('code_revision'),
    }
    adopt_prepared_round(work_root, INITIAL_ROUND,
                         source_dir=Path(artifact_dir) / "training_data_initial",
                         meta=meta)
    print(f"[ANCE] Adopted prepared initial round from {artifact_dir} "
          f"(prepared_run_id={prepared.get('prepared_run_id')}); it remains "
          f"checkpoint_step=0 and is NOT counted as a refresh.", flush=True)
    return prepared


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
    parser.add_argument('--preflight', action='store_true',
                        help='validate the mixture/corpus/qrels against the REAL '
                             'processed data and exit, without touching a GPU and '
                             'without writing anything. Run this before submitting a '
                             'long job. It loads the corpus WITH text (several GB), '
                             'so give it a compute node, not a login node.')
    parser.add_argument('--prepare-initial', metavar='DIR',
                        help='mine round 0 into DIR and exit, on ONE GPU, so a '
                             'training job does not spend hours of a two-GPU '
                             'allocation on it. Upstream splits the same stage out '
                             '(commands/run_train.sh --end_output_num 0).')
    parser.add_argument('--initial-round', metavar='DIR',
                        help='adopt the round 0 prepared in DIR instead of mining '
                             'one. Refused unless it was mined from the same '
                             'initialization, corpus, queries, qrels, mixture, '
                             'mining settings and seed.')
    parser.add_argument('--overwrite', action='store_true',
                        help='clear an existing output directory. Without it a '
                             'directory holding checkpoints from an unfinished run '
                             'is refused rather than silently deleted.')
    args = parser.parse_args()
    if args.prepare_initial and args.initial_round:
        parser.error("--prepare-initial mines round 0 and --initial-round consumes "
                     "one; pass one or the other.")
    if args.prepare_initial and args.preflight:
        parser.error("--preflight writes nothing; --prepare-initial exists to write "
                     "the artifact. Run the preflight first, then prepare.")
    recipe_name = args.recipe

    config = load_config()
    seed = config.get('seed', 42)
    set_seed(seed)

    ctx = get_training_context(recipe_name)
    recipe = ctx['args']
    consumed = CONSUMED_KEYS
    if recipe.get('paper_fidelity'):
        # Paper mode uses an absolute warmup_steps value, not warmup_ratio, and a
        # fixed stop step rather than an epoch count.
        consumed = tuple(k for k in consumed
                         if k not in ('warmup_ratio', 'total_epochs')) + PAPER_KEYS
    require_recipe_keys(recipe_name, recipe, consumed)
    log_startup_config(recipe_name, ctx)
    ance_base_model = recipe.get('base_model', ctx['base_model'])
    paper_init_sha256 = None
    if recipe.get('paper_fidelity'):
        # Fail before preparing/loading millions of records or requesting any GPU work.
        paper_init_sha256 = assert_permitted_init(
            ance_base_model, recipe.get('expected_init_sha256'))
    corpus_file, query_file, qrels_file = run_setup(recipe)

    # Detect GPU count BEFORE restricting visibility.
    # With --gpus-per-task=2, SLURM sets CUDA_VISIBLE_DEVICES=0,1.
    # Tevatron encode raises NotImplementedError on multi-GPU, so we pin the
    # orchestrator to GPU 0 for all encode_to_pickle calls (initial mine).
    # Inferencer/Trainer subprocesses override this with their own assignments.
    # Kept HERE, ahead of the corpus load, so a badly allocated job fails in seconds.
    # --preflight is the one caller that legitimately has no GPU.
    infer_gpu = '1'
    if args.prepare_initial:
        # Preparation is a single full-corpus encode with no trainer beside it, so the
        # 1:1 Trainer:Inferencer rule does not apply and demanding two GPUs here would
        # queue for an allocation half of which sits idle.
        import torch as _torch
        if _torch.cuda.device_count() < 1:
            raise RuntimeError("--prepare-initial needs one visible GPU; torch sees 0.")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print("[ANCE] preparation mode — mining round 0 on GPU 0, no trainer",
              flush=True)
    elif not args.preflight:
        import torch as _torch
        n_gpus = require_ance_gpus(_torch.cuda.device_count())
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print(f"[ANCE] {n_gpus} GPU(s) detected — Trainer→GPU 0, "
              f"Inferencer→GPU {infer_gpu}", flush=True)

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
    max_steps = budget['max_steps']
    scheduler_max_steps = budget['scheduler_max_steps']
    print(f"[ANCE] {n_examples} query records × {budget['triplets_per_query']} "
          f"triplet(s)/query = {budget['training_instances']} training instances | "
          f"{steps_per_epoch} steps/epoch (floor) | {max_steps} total steps | "
          f"decay horizon {scheduler_max_steps} | "
          f"{budget['queries_per_round']} queries per mined round "
          f"(ann_chunk_factor {budget['ann_chunk_factor']})", flush=True)

    if args.preflight:
        # Everything above this line only READ. Returning here is what makes the
        # validator safe to run on a shared node while a real run holds the same
        # output dir: no manifest, no work root, no cleared checkpoints.
        print("[ANCE] ✅ preflight passed — inputs are consistent and the budget "
              "resolves. Nothing was written.", flush=True)
        return 0

    # Only the prepare/adopt paths need it, and it re-hashes multi-GB files that
    # build_run_manifest already hashes. An ordinary run pays nothing.
    identity = None
    if args.prepare_initial or args.initial_round:
        identity = initial_round_identity(
            recipe_name, ctx, recipe, base_model=ance_base_model,
            corpus_file=corpus_file, query_file=query_file, qrels_file=qrels_file,
            mixture_files=mixture_files, seed=seed)

    if args.prepare_initial:
        manifest = prepare_initial_artifact(
            args.prepare_initial, ctx=ctx, config=config, identity=identity,
            corpus_file=corpus_file, query_file=query_file,
            mixture_files=mixture_files, corpus_lookup=corpus_lookup,
            qrels_dict=qrels_dict, base_model=ance_base_model, seed=seed)
        print(f"[ANCE] ✅ prepared initial round in {args.prepare_initial}\n"
              f"[ANCE] {manifest['round_meta'].get('n_records')} record(s) from "
              f"{manifest['round_meta'].get('n_queries_mined')} queries, shard "
              f"{manifest['round_meta'].get('shard_index')} of "
              f"{manifest['round_meta'].get('n_shards')}\n"
              f"[ANCE] Train with: --initial-round {args.prepare_initial}",
              flush=True)
        return 0

    output_model_dir = get_path("models") / recipe['model_name']
    print(f"[ANCE] Starting from model: {ance_base_model}", flush=True)

    # Run identity. The derived corpus/queries/qrels are hashed alongside the
    # mixture: they are what every ANN round is mined against, and P-PRE-02 is the
    # record of them silently predating the code that reads them.
    derived = {'corpus': corpus_file, 'queries': query_file, 'qrels': qrels_file}
    paper_provenance = None
    if recipe.get('paper_fidelity'):
        # Every way this run differs from the supplied Passage command, named here so
        # the number it produces is never presented as an unqualified reproduction.
        deviations = [
            "bf16 autocast over FP32 weights, where the supplied ANCE command runs "
            "FP32 throughout (run_train.sh passes no --fp16; only run_train_warmup.sh "
            "does). The weights are deliberately NOT bf16: LAMB at this learning rate "
            "moves a weight by less than bf16 can represent",
            "initialized from OUR BM25 warm-up (scripts/train_ance_warmup.py, "
            "roberta-base + Tevatron's BM25 negatives, 60K steps at seq 128), not "
            "Microsoft's released 60K warm-up -- that artifact was withdrawn (blob "
            "409; microsoft/ANCE #23/#24/#26 open since 2022) and the only surviving "
            "mirror is bit-identical to the released 600K FINAL",
            "1 Trainer + 1 Inferencer A100, not upstream's 4:4 allocation",
            f"direct batch {batch_size}, not per_gpu 8 x accum 2 x 4 ranks",
            f"stops at {max_steps} steps; the released comparison point is 600000",
        ]
        paper_provenance = {
            'upstream_code_executed': False,
            'init_sha256': paper_init_sha256,
            'similarity': 'dot', 'normalize': False, 'temperature': None,
            'scheduler_max_steps': scheduler_max_steps,
            'train_stop_steps': max_steps,
            'deviations': deviations,
            'note': ("Microsoft supplied initialization weights and behavioural "
                     "specifications. The run, miner, trainer and evaluator are "
                     "this repository's."),
        }
        print("[ANCE] paper-fidelity deviations:", flush=True)
        for item in deviations:
            print(f"         - {item}", flush=True)
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
               'ann_chunk_factor': budget['ann_chunk_factor'],
               'steps_per_epoch': steps_per_epoch,
               'steps_per_epoch_rule': ('floor(query_records * triplets_per_query / '
                                        'batch_size)'),
               'mined_negatives_per_query': pool['mined_negatives_per_query'],
               'triplets_per_query': pool['triplets_per_query'],
               'paper_provenance': paper_provenance})
    # ANCE starts fresh -- there is no resume, and get_last_checkpoint() returns the
    # highest step-numbered checkpoint, so a stale checkpoint-17202 from a prior run
    # would shadow every new save and keep the inferencer stuck forever on the old
    # weights. Clearing is therefore still the only safe way to start. What changed is
    # that it is no longer SILENT: job 204931 died at step 10,100 leaving
    # checkpoint-10000, and the next submission would have deleted it without a word.
    # Refuse instead, and make --overwrite the place that decision is taken.
    stale = sorted(p.name for p in output_model_dir.glob("checkpoint-*")) \
        if output_model_dir.is_dir() else []
    if stale and not args.overwrite:
        raise RuntimeError(
            f"{output_model_dir} already holds {len(stale)} checkpoint(s) "
            f"({', '.join(stale[:3])}{'...' if len(stale) > 3 else ''}). ANCE cannot "
            f"resume, and starting fresh DELETES them. Re-run with --overwrite if "
            f"they are spent, or point training.{recipe_name}.model_name elsewhere to "
            f"keep them.")
    prepare_output_dir(output_model_dir, manifest, overwrite=True)

    run_id = build_run_id(manifest)
    work_root = create_work_root(get_path(recipe['temp_workdir']) / run_id)
    manifest['run_id'] = run_id
    manifest['work_root'] = str(work_root)
    with atomic_write(output_model_dir / RUN_MANIFEST_NAME) as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[ANCE] run_id={run_id} | work_root={work_root}", flush=True)

    # ── INITIAL ROUND: adopted from a preparation job, or mined here ─────────
    # Either way it is round `initial` at checkpoint_step 0 and is never counted as a
    # refresh. There is no automatic discovery: an artifact is used only when the
    # caller names it, so a leftover ready_initial can never be picked up by accident.
    if args.initial_round:
        manifest['initial_round'] = adopt_initial_round(
            args.initial_round, work_root, run_id=run_id, identity=identity)['identity']
        manifest['initial_round_source'] = str(args.initial_round)
        with atomic_write(output_model_dir / RUN_MANIFEST_NAME) as f:
            json.dump(manifest, f, indent=2, default=str)
    else:
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
        '--scheduler_max_steps', str(scheduler_max_steps),
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
    # min_consume_steps is the logging interval, not 1: a round consumed for a single
    # optimizer step before the next one replaced it did not train anything, and
    # counting it lets a nearly static run pass as ANCE.
    fresh = assert_ance_refresh(summary,
                                min_fresh_rounds=int(recipe['min_fresh_rounds']),
                                min_consume_steps=int(recipe['logging_steps']))
    print(f"[ANCE] {len(fresh)} round(s) from distinct checkpoints consumed: "
          f"{[(r['ann_no'], r['checkpoint_step'], r['consumed_steps']) for r in fresh]}",
          flush=True)
    print(f"[ANCE] {summary.get('checkpoint_opportunities')} checkpoint "
          f"opportunit(ies) | {summary.get('rounds_completed')} round(s) completed | "
          f"{summary.get('rounds_consumed')} consumed | {summary.get('rounds_skipped')} "
          f"skipped | {len(summary.get('rounds_refused') or [])} refused | terminal unconsumed: "
          f"{summary.get('terminal_unconsumed')}", flush=True)

    stored = assert_training_succeeded(output_model_dir, manifest)
    stored.update({'ann_rounds': summary.get('rounds', []),
                   'fresh_rounds_consumed': len(fresh),
                   'final_refresh': fresh[-1] if fresh else None,
                   'checkpoint_opportunities': summary.get('checkpoint_opportunities'),
                   'rounds_completed': summary.get('rounds_completed'),
                   'rounds_consumed': summary.get('rounds_consumed'),
                   'rounds_skipped': summary.get('rounds_skipped'),
                   'rounds_refused': summary.get('rounds_refused'),
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
