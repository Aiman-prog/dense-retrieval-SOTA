# ANCE Training Stage

## Purpose

The ANCE stage fine-tunes the same base encoder on the same objective as the
baseline arms, but replaces their static mined negative with one drawn from an
approximate-nearest-neighbour index that is rebuilt from the model's own recent
checkpoints while training continues. It owns optimization and mining. The
mixture is built by the preprocessor stage and retrieval quality is measured by
the evaluation stage.

This is **ANCE-style asynchronous ANN mining under the shared BGE-M3 contrastive
objective**, not Microsoft's pairwise RoBERTa/LAMB reproduction. The mining
schedule, the top-200 uniform sampling and the 1:1 trainer/inferencer split
follow the paper; the loss, the pooling, the temperature and the optimizer are
the ones every other arm in this repository uses. That is deliberate: it makes
the BRIGHT comparison a comparison of *mining*, and it means the numbers here
are not comparable to published ANCE results. See [Comparability](#comparability).

Two recipes share this stage. Everything below describes the shared machinery
once; the two are separated only where they genuinely differ.

| | BRIGHT arm | MS MARCO arm |
|---|---|---|
| recipe | `training.ance` | `training.ance_msmarco` |
| purpose | control arm for the GRASS comparison | port sanity check on a public benchmark |
| corpus | ReasonIR, ~1.6M passages | MS MARCO, 8.8M passages |
| evaluated by | BRIGHT, 12 domains | MS MARCO Dev |

## Training Flow

```text
orchestrator ──> verify inputs, hash them into the run manifest
             ──> mine round `initial` with the base model
             ──> launch two supervised workers
                    │
     GPU 0 ─────────┴──── trainer: consume the current round, never pause
                    │        └─> checkpoint every `save_steps`
     GPU 1 ─────────┴──── inferencer: poll for a checkpoint
                             └─> re-encode the whole corpus
                             └─> FAISS index, mine top-200
                             └─> commit round N, write `ready_N` last
                    │
             ──> trainer swaps its data loader on the next poll
             ──> assert a refresh was consumed, then assert training succeeded
```

Training never stops for a refresh. The trainer checks for a newly committed
round every `logging_steps` and swaps its data loader in place; between swaps it
keeps optimizing on the round it already has.

## Data and Batch Construction

The BRIGHT arm reads the three mixture files described in
[`preprocessor.md`](preprocessor.md), totalling **329,993 examples**. The MS
MARCO arm reads a single-component mixture of roughly **400,000** records
streamed from the public training split. Both carry exactly one positive per
record; their negatives are replaced wholesale at every refresh.

`train_group_size: 2` gives each query a group of two passages, its positive
followed by **one ANN-mined negative**. With `batch_size: 64` a step holds 64
queries and 128 passages, exactly as in the in-batch arm.

Steps per epoch are `floor(examples / batch_size)`, matching both the data
loader's `drop_last` and the convention used by naive GRASS. For the BRIGHT arm
that is 5,156 steps per epoch and **10,312 optimizer steps** over two epochs.

## Negative Pool

The similarity matrix spans all 64 queries against all 128 passages and the
target for query *i* is passage *2i*, so each query is contrasted against
**127 negatives**.

| source | count per query |
|---|---|
| its own ANN-mined negative | 1 |
| other queries' mined negatives | 63 |
| other queries' positives | 63 |
| **total negatives** | **127** |

Only the first is mined. The other 126 are incidental cross-example passages,
and the shared objective performs **no cross-query false-negative masking**, so
a passage relevant to another query in the step still sits in the denominator.
What ANCE changes relative to the baseline arms is the *provenance of that one
mined negative*, not the size or composition of the pool.

The mined negative is drawn uniformly from the query's top-200 ANN candidates
after its positives are removed. Uniform sampling rather than top-1 selection is
the paper's own choice; the reference implementation slices the top only in its
metric-measurement mode. Positives are taken from both the judgments file and
the record's own labels, and a negative is **never** fabricated: no positive, no
duplicate, no corpus-random filler. A query that cannot supply a genuine ANN
negative discards the entire round before it is published.

## Asynchronous Index Refresh

Refresh is what makes this ANCE, and the stage will not report success without
evidence that one happened.

Every invocation gets its own work root, named by the configuration fingerprint,
a timestamp, the scheduler job id when present, and a random suffix. Two
invocations therefore cannot share mined data, which matters because the
directory is created with no overwrite permitted: a collision is a startup
error, never two runs quietly consuming each other's rounds.

A round is committed by staging its files, writing its metadata, and only then
writing the `ready_N` marker. The metadata records which checkpoint mined the
round, at what step, how many queries it covered, and a **SHA-256 for every file
in the round**. Before consuming a round the trainer verifies the marker, the
run identity, the file list, the record count and those hashes, so a round that
was truncated, edited or produced by a different run is refused rather than
trained on.

The base-model round is marked `initial` and carries `checkpoint_step: 0`. It is
deliberately **not** counted as a refresh, because negatives mined by the
starting weights are not ANN negatives in the sense the method requires.

`save_steps` is not a convenience: saving a checkpoint is what triggers the next
refresh, so it *is* the paper's refresh interval *m*. It is a declared
hyperparameter and must be held constant across a sweep. At the BRIGHT setting
of 1,000 it also bounds the run to about ten checkpoints, which is why no
checkpoint pruner exists.

## Run Integrity

A clean exit proves nothing, so the stage asserts the following and fails
nonzero otherwise.

- **Two visible GPUs before any mining.** The trainer and the inferencer are
  pinned to separate devices; the run refuses to start on one, rather than
  discovering the contention after the initial full-corpus encode.
- **Both workers supervised.** *Any* inferencer exit before the orchestrator
  asks it to stop is a failure, a clean zero included. The inferencer is meant
  to run until terminated, so an early exit means refreshes stopped and the
  trainer would have continued to completion on stale negatives while still
  looking successful.
- **At least one refresh consumed.** A round mined from a checkpoint of this run
  must have trained for at least one optimizer step.
- **Finite optimization.** A non-finite loss is rejected before the backward
  pass, and a non-finite pre-clipping gradient norm before the optimizer step.
  Clipping cannot rescue the latter: the coefficient is itself non-finite, so
  the step would write NaN into every parameter.
- **Real progress.** New optimizer steps past this invocation's start, the
  planned step count reached, a freshly written loadable checkpoint, and two
  successful ranking probes at distinct steps.

Consumption evidence is written durably, with retries, because it is the only
record of which rounds were used and the run is failed without it.

## Objective and Hyperparameters

The loss is softmax cross-entropy over cosine similarities divided by a fixed
temperature, identical to the baseline arms. Settings come from
[`config/config.yaml`](../config/config.yaml) under `model` and the two ANCE
recipes.

| setting | BRIGHT | MS MARCO |
|---|---|---|
| base model | in-batch BGE-M3 checkpoint | in-batch BGE-M3 checkpoint |
| pooling / normalization | CLS, L2-normalized | ← |
| temperature | 0.02 | ← |
| query / passage max length | 1024 / 512 tokens | ← |
| batch size | 64 | ← |
| train group size | 2 | ← |
| mining depth | top-200 | ← |
| optimizer | AdamW, betas (0.9, 0.999), eps 1e-8 | ← |
| weight decay | 0.01 | ← |
| gradient clipping | 1.0 | ← |
| precision | bf16 | ← |
| seed | 42 | ← |
| learning rate | 1e-5 | 5e-6 |
| warmup ratio | 0.1 | 0.06 |
| epochs | 2 | 3 |
| refresh interval (`save_steps`) | 1,000 | 1,250 |
| logging interval | 100 | 500 |
| retrieval depth | 10 | 1,000 |

The optimizer is built by one shared factory with every hyperparameter explicit,
including the two usually left to framework defaults, so a future change to
those defaults cannot move one arm relative to another.

Because this stage runs its own training loop rather than the framework's
trainer, the temperature reaches the loss through the model's own scaling, which
already applies it. The gradient-cache patch that the cross-batch arm relies on
must **not** be applied here: it rewrites a loss path this stage never enters, so
it is inert at best, and it exists precisely to add the scaling that is already
present.

## Infrastructure

Two A100 GPUs on one node, 16 CPU cores and roughly 125 GB of host memory,
inside the project Singularity container. The trainer holds GPU 0 for the whole
run; the inferencer holds GPU 1 and is busy only while re-encoding.

Cost is dominated by the full-corpus encode, which runs once before the first
step and once per refresh. On the BRIGHT corpus that is affordable at the
configured cadence. **On MS MARCO it is not**: one pass over 8.8M passages takes
hours, so the configured 1,250-step interval is far tighter than the inferencer
can service and a realistic run reaches the initial round plus one or two
refreshes. That is enough to satisfy the refresh requirement and not enough to
approach convergence, which is the honest framing for any MS MARCO number this
stage produces.

## Comparability

**Against naive GRASS (the intended comparison).** Both arms share a base model,
a mixture, a batch size, a group size, a 127-negative pool, a learning rate, a
schedule shape and — enforced by test — an identical explicit optimizer. Both
run two epochs at `floor(examples / batch_size)` steps per epoch, so the
optimizer-step budget matches. What differs is how the single mined negative is
chosen. Optimizer parity is currently guaranteed **only** for this pair;
sequential and async Fast-GRASS retain their own optimizer paths.

Two limits belong next to any number from this pair. The reference ANCE recipe
is step-budgeted and much longer, so a two-epoch run is a **lower bound on
ANCE** rather than ANCE at convergence; every arm is equally budget-limited, so
this is a limitation to state rather than a confound. And naive GRASS still
writes no run manifest, training log or success assertion, so the comparison is
only as well-evidenced as its weaker arm (recorded as `P-ANCE-02` in
[`CONSOLIDATION_STATUS.md`](../CONSOLIDATION_STATUS.md)).

**Against published ANCE.** Not a reproduction. The published passage result is
RoBERTa initialized from a BM25 warm-up, trained with LAMB on a pairwise
objective for 600K steps. This stage shares none of those. The MS MARCO arm is a
port sanity check and must not be presented beside the paper's figures.

⚠️ **The earlier ANCE result, job 9566838 / 0.1683, is quarantined and
non-reportable** (`P-ANCE-01`). It predates exclusion filtering, its data
provenance cannot be reconstructed, and its single claimed refresh rests on a log
line rather than an artifact.

## Evaluation

The stage runs **no in-job BRIGHT evaluation**; it prints the command instead. A
reportable BRIGHT number comes only from the shared evaluation runner over all
twelve domains, which applies the per-query exclusion filter and records artifact
hashes alongside the checkpoint's training provenance. See
[`evaluation.md`](evaluation.md).

The MS MARCO arm has its own evaluator, reporting **MRR@10** (the run truncated
to depth 10 before scoring, since the metric has no native cutoff) and
**Recall@1000** at full depth. Before searching it requires the encoded query ids
to equal the source ids and the judged ids to be a subset of them, so an encoder
that dropped or invented a query fails loudly rather than producing a quietly
wrong average. Corpus embeddings are cached under a key derived from the model
weights, config, tokenizer, corpus and encoding settings, because this stage
overwrites its output directory on every run and a path-based key would go
stale. The paper's reference figures are printed **only** when exactly 6,980 dev
queries are judged, and every summary records a `paper_comparable` flag so a
number measured on a different split cannot be mistaken for one that is.

## Running the Stage

Submitted through the cluster launchers, which set the data root, force offline
Hugging Face mode, request two GPUs and select the container. There is no resume:
every run re-mines its initial round into a fresh work root, because reusing
mined data across invocations is precisely the failure the run-identity checks
exist to prevent.

Each run writes the trained weights, periodic checkpoints, a manifest recording
the configuration, input hashes, the optimizer actually built and the rounds
actually consumed, a diagnostics log holding loss, learning rate and pre-clipping
gradient norm, and a consumption summary. Progress is visible in the log as the
initial round being committed, rounds being swapped in, and a final validation
line reporting the optimizer steps and checkpoint-derived rounds consumed. The
per-experiment success and failure signals are listed in
[`GPU_CHECKLIST.md`](GPU_CHECKLIST.md).

---

# Appendix: the paper-fidelity arm (`ance_paper`)

A **separate experiment** from everything above, and not a GRASS control arm. It runs
Microsoft's own recipe through this repository's ANCE orchestration, so that "is this
really ANCE?" has an answer that is not just documentation.

```bash
python scripts/train_ance.py --recipe ance_paper          # same entry point
python scripts/eval_msmarco.py --recipe ance_paper --model_path <checkpoint>
```

**The mining algorithm is already the paper's, on both arms.** Verified against
`microsoft/ANCE`, file and line: async refresh with the trainer never stopping
(`drivers/run_ann.py:185-235`), full-corpus re-encode, top-200 FAISS, uniform sampling
from the shuffled top-*k* minus positives (`run_ann_data_gen.py:366-389` — the same
procedure as `select_ance_negatives`), `save_steps` as the refresh interval *m*,
base-model round 0. What this appendix adds is the paper's **model, loss and optimizer**.

| | BRIGHT arm (`ance`) | paper arm (`ance_paper`) |
|---|---|---|
| encoder | BGE-M3 | RoBERTa + `Linear(h,768)` + `LayerNorm` on CLS |
| similarity | CLS, L2-normalized, ÷ 0.02 | raw dot product, no temperature |
| loss | in-batch CE over 127 negatives | `-log_softmax([q·pos, q·neg])[:,0]` |
| negatives | 1 mined (`train_group_size: 2`) | 20 mined (`train_group_size: 21`), each its own triplet |
| optimizer | AdamW, shared with naive GRASS | LAMB, ported quirks included |
| lengths | q1024 / p512 | q64 / p512, dynamic padding |

**Why the BRIGHT arm keeps the shared objective.** ANCE is a negative-*mining* strategy.
Holding the objective constant across ANCE and GRASS is what makes a BRIGHT difference
between them attributable to mining. Giving the BRIGHT arm the paper's pairwise loss
would confound the two, and since in-batch CE over 127 negatives is strictly more
informative than pairwise over 1, it would most likely lower the numbers as well.

**Reuse.** One new module, `scripts/ance_paper.py`. Everything else is the existing
pipeline: `train_ance.py`, `run_ance_train.py`, `run_ance_data_gen.py` (unmodified —
the encoder dispatch lives in `helpers.encode_to_pickle`, which every consumer already
calls), `ance_mining.py`, the round commit with per-file SHA-256, the freshness gate,
process supervision and `assert_training_succeeded`. The paper encoder emits the same
`(embeddings, ids)` pickle the miner already reads, which is what makes that possible.

**No conversion step.** Their checkpoints store `roberta.*`, `embeddingHead.*`,
`norm.*`, so `from_pretrained` maps them straight onto `AnceEncoder`. `load_ance_encoder`
turns HF's missing/unexpected-key *warnings* into errors, tolerating only
`classifier.*` — an unused head transformers 2.3.0 persisted.

**Stated deviations.** bf16 rather than the paper's fp16: bf16 needs no `GradScaler`,
so there are no skipped updates and `global_step` is the successful-step count.
Initialization comes from the released BM25 warm-up checkpoint rather than a locally
trained one.

## Hardware-constrained full MS MARCO run

This repository runs its own ANCE trainer and miner from Microsoft's released 60K
BM25-negative warm-up checkpoint. It does **not** execute Microsoft's training code.
Before training, score their released 600K checkpoint through our evaluator and expect
**0.330 MRR@10 / 0.959 Recall@1000 within ±0.005**. A miss at that stage invalidates the
evaluation harness, not the trainer.

We cannot honestly claim a 600K reproduction on the available DelftBlue allocation.
Microsoft used several inference GPUs; this project has one trainer A100, one miner A100
and a 24-hour limit. The miner must encode 8.8M passages per refresh and may therefore
skip intermediate checkpoints while it is busy. Instead, the configured middle ground
is **two expanded-dataset epoch equivalents over the 20-negative triplet pool**. This
retains the model, raw-dot pairwise loss, LAMB optimizer, top-200 mining, 20 candidate
negatives and asynchronous replacement while reducing optimization exposure.

The table uses the Tevatron export's expected 400,782 usable training records. Runtime
counts in `run_manifest.json` are authoritative: preprocessing may produce a different
count only if the pinned dataset itself differs.

| Aspect | Microsoft ANCE | This repository's ANCE validation |
|---|---:|---:|
| initialization | released 60K BM25-negative warm-up | same released warm-up |
| ANN depth | 200 | 200 |
| mined negatives per query/round | 20 | 20 |
| loss instance | one positive + one negative | same |
| model / similarity / optimizer | RoBERTa projection, raw dot, pairwise NLL, LAMB | same |
| effective training batch | approximately 64 | 64 |
| precision | upstream mixed-precision path | bf16 |
| optimizer steps | 600,000 | ~250,488 |
| triplets processed | 38,400,000 | ~16,031,232 |
| step-equivalent passes over the expanded pool | ~4.79 | 2 |
| checkpoint/refresh opportunity | every 10K steps | every 10K steps |
| checkpoint opportunities | 60 | ~26 including the terminal checkpoint |
| inference hardware | several inference GPUs | one miner GPU |
| allocation | upstream hardware | two A100s, 24 hours |

Twenty mined negatives do **not** make a 21-way loss. Each becomes its own pairwise
triplet, so the durable manifest records `negative_pool_size: 1`,
`mined_negatives_per_query: 20` and `triplets_per_query: 20`. For approximately 400,782
query rows, one epoch is `floor(400782 * 20 / 64) = 125244` optimizer steps; two epochs
are 250,488 steps. This is about 41.7% of the paper's update/triplet exposure.

### Commands and acceptance

```bash
# Train our code from the configured Microsoft warm-up checkpoint.
ANCE_RECIPE=ance_paper sbatch scripts/launchers/run_ance_msmarco_singularity.sh

# Validate the evaluator on Microsoft's released 600K checkpoint.
EVAL_RECIPE=ance_paper EVAL_MODEL_PATH=<released-600K> \
  sbatch scripts/launchers/eval_msmarco_singularity.sh

# Establish the baseline, then score our final model with the identical evaluator.
EVAL_RECIPE=ance_paper EVAL_MODEL_PATH=<released-60K-warm-up> \
  sbatch scripts/launchers/eval_msmarco_singularity.sh
EVAL_RECIPE=ance_paper EVAL_MODEL_PATH=$DATA_BASE_DIR/models/ance_paper_roberta \
  sbatch scripts/launchers/eval_msmarco_singularity.sh
```

The shortened run succeeds as an ANCE validation when its MRR@10 exceeds the locally
measured warm-up MRR@10, Recall@1000 is at least 0.949, and the training manifest proves
that at least one checkpoint-derived ANN round was consumed. It need not beat the
paper's 0.330 MRR@10 and must not be described as reproducing 600K-step convergence.
Report the actual number of produced and consumed refreshes; one miner GPU is expected
to produce fewer than Microsoft's inference fleet.

**Data provenance** uses separate immutable commits because the train/dev and corpus
are separate Hugging Face repositories:
`data.msmarco_reproduction.passage_revision` and `corpus_revision`. The run additionally
hashes the derived corpus, queries, qrels and warm-up weights.
