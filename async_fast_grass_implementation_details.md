# Async Fast-GRASS Implementation Details

This file complements `async_fast_grass_architecture.md`. The architecture is
fixed; the choices below are current engineering defaults and implementation
contracts. Async Fast-GRASS applies the ANCE process split to Fast-GRASS mining:
the trainer and miner are separate processes, and the miner uses cached-MCDP
instead of the existing lazy fresh-MCDP `_mine_batch_mcdp` path.

Async Fast-GRASS is not ANCE mining. It must not rebuild a full-corpus ANN index
during training and must not return to per-query stale FAISS top-P mining.

Cached-MCDP replaces lazy fresh-MCDP as the production MCDP implementation.
Preserve the existing EMA implementation. The old `_mine_batch_mcdp` logic may
remain only as an explicitly invoked small-sample quality oracle; it must not
have production defaults, a normal training mode, or a launcher path. Ordinary
cached mining performs no document encoder calls. Document MC states are
created during cache initialization and updated only by cache maintenance.

## Current Async Defaults

Replace the existing production MCDP defaults under `training.fast_grass` with
the cached-MCDP defaults below. The repository-level production uncertainty
modes become `cached_mcdp` and `ema`; remove lazy `mcdp` from normal CLI/config
mode selection. Preserve the existing sequential EMA configuration and
behavior, while the first async launcher accepts `cached_mcdp` only. Async EMA
remains deferred. Remove `L` from the cached-MCDP configuration and launcher
because cached scoring covers all of `H`.

| Parameter | Default | Notes |
|---|---:|---|
| `async_mine_every_steps` | calibrated fixed value | Trainer checkpoint-save cadence in optimizer steps. This is not the ready-data polling cadence and is not runtime-computed. Initial practical range: `1000` to `2000`, then retune from logs. Avoid tiny values such as `100` because checkpoints are large. |
| `ready_poll_steps` | `logging_steps` | Trainer cadence for checking whether a newer `ready_N` round exists. This is a lightweight directory check, decoupled from checkpoint saving. |
| `trainer_gpu` | `0` | `CUDA_VISIBLE_DEVICES=0` for the optimizer process. |
| `miner_gpu` | `1` | `CUDA_VISIBLE_DEVICES=1` for the miner process; may fall back to `0` on a single-GPU run. |
| `async_poll_interval` | `60` seconds | Miner sleep interval while polling for valid checkpoints, mirroring ANCE's inferencer polling. |
| `cache_update_interval` | `100` trainer steps | Same meaning as sequential Fast-GRASS; trainer-step-equivalent interval used in the maintenance budget. |
| `maintenance_interval_mined_queries` | `cache_update_interval * trainer_batch_size` | Miner execution trigger for maintenance. With `cache_update_interval=100` and `batch_size=64`, maintain every `6400` mined query examples, not every `100`. |
| `uncertainty` | `cached_mcdp` | Async production mode. Existing sequential `ema` remains supported; lazy fresh-MCDP is oracle-only. |
| `B_doc` | `32_000` | First-run cache size. Test `100_000` only after timing and validating the 32k configuration. |
| `T` | `3` | Cached document states and fresh query passes. |
| `lambda_val` | `0.5` | First nonzero uncertainty setting. Always compare with `lambda=0`; do not assume `lambda=1` is optimal. |
| `mc_dropout_p` | `0.3` | Existing Fast-GRASS key; applied to no-grad stochastic encoder passes. |
| `m` | `1` | Inherited negatives per query. |
| `selection_mode` | `topk` | Inherited selection mode. |
| `rho_start` | `0.5` | Fractional cache maintenance budget at training start. |
| `rho_end` | `0.25` | Quality-oriented end budget for the first cached-MCDP run. |
| `max_age_epochs` | `2` | Forces more current cached states than the previous four-epoch setting. |

`L` is not part of cached-MCDP. Every query is scored against all of `H`.
If `selection_mode=softmax` is used as an ablation, call the existing selector
with `L=None` so sampling is over the full finite cache.

**`R_doc` is DEFERRED in the first async implementation.** Cached-MCDP replacement
draws its entire candidate pool uniformly from the corpus, excluding documents
already in `H`, then recertifies against the query-MC reservoir. Registry
admission, nomination, retention, and persistence are not built yet; the `R`
counters are reported as zero and `cached_mcdp_v1` serializes no registry entries.
The `R_doc` knobs (`R_fraction`, `R_size_factor`, `utility_remember_threshold`) are
therefore absent from the async config block rather than present with dead values.
The query-MC reservoir is unaffected and still required, and the existing
EMA/sequential registry stays active on its own path. Every `R` mention below is
scoped by this note.

For BGE-M3's 1024-dimensional embeddings stored in BF16:

| `B_doc` | `Z_mc` with `T=3` | `Z_mean` | Persistent embedding total |
|---:|---:|---:|---:|
| `32_000` | 187.5 MiB | 62.5 MiB | 250.0 MiB |
| `100_000` | 585.9 MiB | 195.3 MiB | 781.2 MiB |

These totals exclude metadata, model weights, query embeddings, and temporary
score buffers. Accumulate score moments in FP32 and chunk over cache slots if
necessary instead of retaining all `T` complete score matrices.

Add a `get_path` key for the async workdir:

```text
temp_fast_grass -> temp_fast_grass_workdir
```

The async handoff root is:

```text
temp_fast_grass_workdir/async_mining/
```

Cache state files are expected to have the same persistent embedding footprint
because they store CPU BF16 tensors for restart.

## Process Roles

`train_async_fast_grass.py` is the orchestrator:

```text
run_setup
load corpus IDs/text lookup; a stale artifact may supply ordering, not MC states
sample B_doc initial cache document IDs
encode every initial document T times with dropout using the base model
compute Z_mean and initialize cache metadata
generate initial cached-MCDP data from that cache
write initial_data/, mining_meta_initial.json, cache_state_initial.pt, ready_initial
start miner subprocess on miner_gpu; it may poll before a checkpoint exists
run trainer subprocess independently on trainer_gpu
in finally: terminate and wait for the miner subprocess
```

`run_async_fast_grass_train.py` is the trainer:

```text
owns gradients, optimizer, scheduler, checkpoints
consumes mined JSONL data
fresh-encodes query, positive, and mined negatives for the loss
swaps dataloaders when a newer ready round appears
never reads or mutates H, R, utility metadata, Z_mc, or Z_mean
```

`run_async_fast_grass_miner.py` is the miner:

```text
owns H, R, Z_mc, Z_mean, utility metadata, and cache age
loads cache_state_initial.pt, or the newest cache_state_N.pt gated by ready_N
keeps H/R/Z_mc/Z_mean in memory across rounds
loads newest valid trainer checkpoints
mines complete JSONL rounds with checkpoint-frozen weights
maintains a rolling recent query-MC reservoir for replacement recertification
runs periodic in-round refresh/replacement
writes cache_state_N.pt and mining_meta_N.json after each round
```

The miner should not reinitialize the cache after startup except when
deliberately starting a fresh async run.

## Handoff Protocol

Use the ANCE ready-marker pattern with Fast-GRASS-specific artifacts:

```text
temp_fast_grass_workdir/
  async_mining/
    initial_data/
      *.jsonl
    mining_meta_initial.json
    cache_state_initial.pt
    ready_initial

    work_N/
      training_data/
        *.jsonl
      cache_state.pt
      mining_meta.json

    training_data_N/
      *.jsonl

    cache_state_N.pt
    mining_meta_N.json
    ready_N
```

Publish a round in this order:

```text
write and close all files under work_N/
os.replace(work_N/training_data, training_data_N)
os.replace(work_N/cache_state.pt, cache_state_N.pt)
os.replace(work_N/mining_meta.json, mining_meta_N.json)
write ready_N.tmp, then os.replace(ready_N.tmp, ready_N)
```

`ready_N` is the only trainer-visible completion signal. The trainer never reads
from `work_N/`. The trainer consumes the newest ready round and skips older ready
rounds. The miner finishes its current round before checking for a newer
checkpoint; between rounds it picks the newest valid checkpoint and skips older
unused checkpoints.

`ready_initial` is handled separately as the trainer's step-0 input. Numeric
rounds start at `ready_1`; `get_latest_marker_no(..., prefix="ready_")` only
discovers numeric markers and intentionally ignores `ready_initial`.

Final-path artifacts without `ready_N` are uncommitted leftovers. On restart,
both processes discover the newest numeric round through ready markers first,
then load only that round's data/metadata/cache state. They may delete or
overwrite orphaned artifacts with larger round numbers.

Checkpoint validity follows ANCE:

```text
checkpoint-N/
  model weights
  tokenizer files
  scheduler.pt
  optimizer.pt   # written last; validity flag
```

The miner treats a checkpoint as valid only after `optimizer.pt` exists.

## Step Vocabulary

Use optimizer-step semantics, not epoch semantics:

```text
checkpoint_step
    trainer optimizer step whose checkpoint weights the miner uses

source_checkpoint_step
    checkpoint_step recorded in miner outputs and trainer logs

consume_step
    trainer optimizer step where a mined round becomes active

async_gap_steps
    consume_step - source_checkpoint_step

data_age_steps
    trainer steps spent reusing the currently active round after consume_step

mining_round
    one complete mined dataset for the full training mixture
```

`async_gap_steps` is fixed when a round is consumed. `data_age_steps` grows while
the trainer continues reusing that same round.

## Trainer Loop

The trainer mirrors the ANCE in-place dataloader swap pattern while keeping the
Fast-GRASS fresh-loss training step:

```text
load initial_data
global_step = 0
active_round_no = 0
active_round_label = initial

while global_step < max_steps:
    if global_step % ready_poll_steps == 0:
        latest = get_latest_marker_no(async_mining_dir, prefix="ready_")
        if latest > active_round_no:
            swap dataloader to training_data_latest
            active_round_no = latest
            active_round_label = latest
            log consume_step, source_checkpoint_step, async_gap_steps

    batch = next active dataloader batch
    fresh-encode queries
    fresh-encode positives
    fresh-encode mined negatives
    compute contrastive loss
    optimizer.step()
    scheduler.step()
    global_step += 1
    log loss, step time, active round, data_age_steps

    if global_step % async_mine_every_steps == 0:
        save checkpoint
        write optimizer.pt last
```

The trainer never uses `Z_mc`, `Z_mean`, or any miner-produced embeddings in the
loss. Gradients come only from fresh query/positive/negative encodings.

## Miner Loop

The miner owns cache state and mines full rounds:

```text
validate cache schema, T, B_doc, and embedding dimension
committed_round = max(get_latest_marker_no(async_mining_dir, prefix="ready_"), 0)
load cache_state_committed_round, or cache_state_initial when committed_round = 0
round_no = committed_round + 1

while true:
    checkpoint = newest valid checkpoint not already mined
    if none:
        sleep async_poll_interval
        continue

    source_checkpoint_step = parse checkpoint step
    load checkpoint weights
    freeze parameters for no-grad mining
    reset per-round counters
    initialize bounded recent query-MC reservoir

    for each virtual mining batch in the full training mixture:
        q_mc = encode each query T times with dropout and no gradients
        score all H with score_cached_mcdp(q_mc, Z_mc)
        mask qrels and select m negatives
        write selected negatives to work_N/training_data/*.jsonl
        push q_mc and query IDs into the reservoir
        record selected slots/docids for utility accounting

        if mined_queries_since_maintenance >= maintenance_interval_mined_queries:
            cache.maintain_cached_mcdp(..., source_checkpoint_step, ...)
            reset mined_queries_since_maintenance

    fold any partial utility interval
    optionally run one final bounded maintenance interval if useful pending state exists
    write work_N/cache_state.pt and work_N/mining_meta.json
    publish round and write ready_N last
```

The checkpoint weights are frozen/no-grad for the mining round, but dropout must
remain active for stochastic encoding. Use a scoped dropout-only context:

```text
put the model in eval mode
temporarily put dropout modules in train mode
run under inference_mode/no_grad
restore every module's entry mode afterward
```

"Frozen for the round" means no parameter updates and no gradients, not
dropout-off. Dropout-only mode avoids accidentally enabling future stateful
training-mode modules.

Do not call the current `_mine_batch_mcdp` implementation here. It performs
deterministic broad scoring followed by fresh query/document dropout encoding
for a top-`L` shortlist, which is exactly the bottleneck cached-MCDP removes.

## Cached-MCDP Scoring Contract

Extend `NegativeCache` without breaking the existing EMA fields and methods.
Cached-MCDP mode replaces the old production MCDP cache behavior and adds:

```text
Z_mc:   [T, B_doc, D], BF16
Z_mean: [B_doc, D], BF16
```

In cached-MCDP mode, keep `Z_student` as an alias of `Z_mean` for compatibility
with existing diagnostics and shared cache utilities; do not allocate a third
embedding bank. Refresh and replacement must mutate `Z_mean` in place so the
alias remains valid. `Z_teacher` remains `None`. Serialization stores
`Z_mc`/`Z_mean` once and recreates the alias on load.

The implementation should expose a dedicated cached scoring function rather
than overloading `cheap_scores`:

```text
score_cached_mcdp(q_mc, lambda_val, chunk_size=None)
```

where `q_mc` has shape `[T, B_query, D]`. For every pass:

```text
scores_t = (q_mc[t].to(Z_mc.dtype) @ Z_mc[t].T).float()
s_hat   = mean_t(scores_t)
sigma   = sqrt(mean_t((scores_t - s_hat)^2))  # population std, correction=0
g       = s_hat + lambda_val * sigma
```

The output is `[B_query, B_doc]`. Apply known-positive masking before `topk`
selection. Use the existing qrel mask semantics and deterministic tie behavior.
The pass index simply pairs independent query and cached document dropout
samples; do not create extra document passes or a fresh shortlist.

The scorer should compute or accumulate moments in FP32. A chunked
implementation must produce the same selected doc IDs and scores as the
unchunked formula within the configured numerical tolerance.

Ordinary mining accounting must satisfy:

```text
mcdp_query_encoder_calls = T per virtual query batch
mcdp_doc_encoder_calls_mining = 0
```

## Cache State And Initialization

The initial cache cannot repeat a deterministic embedding `T` times. After
sampling the initial `B_doc` IDs, encode those documents through `T` genuine
dropout passes with the base model:

```text
Z_mc[t] = normalized document embeddings from dropout pass t
Z_mean  = mean_t(Z_mc[t])  # do not renormalize unless explicitly tested
last_refreshed_step[:] = 0
```

Write `ready_initial` only after the complete MC cache, initial mined JSONLs, and
metadata are durable.

Persist an explicit versioned state dictionary, not the live Python cache
object:

```text
schema_version = "cached_mcdp_v1"
docids
Z_mc and Z_mean as CPU BF16 tensors
utility_ema, peak_utility_ema, selected_indicator
lifetime_selected_count, intervals_since_selected
last_refreshed_step and selection history
replacement registry entries       # DEFERRED — absent from cached_mcdp_v1
NumPy bit-generator state
Torch generator state
T, B_doc, embedding dimension, and dtype
```

On load, reject schema/configuration mismatches before moving tensors to the
miner device. A save/reload round trip must reproduce the next cache-random
decision, not only the current embeddings.

## Cache Maintenance Semantics

The maintenance budget is unchanged from sequential Fast-GRASS:

```text
rho = linear_decay(rho_start, rho_end, source_checkpoint_step / total_steps)
maintenance_budget_interval =
    round(rho * B_doc * cache_update_interval / steps_per_epoch)
```

The miner executes one bounded maintenance interval every:

```text
maintenance_interval_mined_queries =
    cache_update_interval * trainer_batch_size
```

Call the cached-MCDP maintenance path with the checkpoint step as model time:

```text
cache.maintain_cached_mcdp(
    student,
    tokenizer,
    corpus_lookup,
    c_ids,
    query_mc_reservoir,
    source_checkpoint_step,
    cfg,
    device,
    qrels_dict=qrels_dict,
)
```

`query_mc_reservoir` contains recent no-grad query states with shape
`[T, R_query, D]` and their query IDs. If it is absent, maintenance may refresh
existing slots but must skip replacement recertification.

One maintenance interval performs:

```text
fold selected_indicator into interval-based utility
plan refresh and replacement slots under maintenance_budget_interval

for refresh documents:
    encode all selected documents for T dropout passes
    update Z_mc, Z_mean, and last_refreshed_step together
    preserve the document's accumulated utility and lifetime history

for replacement candidates (Phase 1: uniform corpus sampling only, excluding H;
                            R nomination is DEFERRED):
    encode every candidate for T dropout passes exactly once
    score candidates against query_mc_reservoir with cached-MCDP g
    mask qrels during recertification
    insert selected candidates using slices of the already-computed Z_mc
    update docid, Z_mean, utility metadata, and timestamp together
    admit the evicted document to R under the existing registry policy   # DEFERRED
    reset the new slot's utility/history and set its timestamp to source_checkpoint_step
```

An atomic slot update means readers never observe a new `docid` paired with old
MC embeddings or a partially replaced `T` bank. Since the miner is a single
owner, construct all new slot values first and assign the complete indexed
tensors/metadata as one maintenance commit.

Documents outside the selected refresh/replacement slots remain bitwise
unchanged. The complete cache is not re-encoded at every interval.

Inside one mining round:

```text
model-time age stays fixed:
    age = source_checkpoint_step - last_refreshed_step

rho/progress stays fixed:
    rho = linear_decay(..., source_checkpoint_step / total_steps)

refresh/replace timestamps stay model-time:
    last_refreshed_step := source_checkpoint_step

utility state advances per maintenance interval:
    selected_indicator -> utility_ema
    peak_utility_ema, lifetime_selected_count, intervals_since_selected update
    selected_indicator resets
```

Do not pass miner batch index, mined-query count, maintenance-interval count, or
round-local progress as the `step` argument. That would corrupt cache age,
rho/progress, and `last_refreshed_step`, which represent trainer-model
staleness.

Not chosen for the first async implementation:

```text
mine full training_data_N with fixed H
run one large cache maintenance pass at round end
```

Periodic in-round maintenance is closer to sequential Fast-GRASS and lets `H`
refresh or replace stale/low-utility entries while a long mining round is being
generated.

## Metadata And Logs

`mining_meta_N.json` should include:

```text
round_no
source_checkpoint
source_checkpoint_step
B_doc
T
m
lambda_val
mc_dropout_p
selection_mode
async_mine_every_steps
cache_update_interval
maintenance_interval_mined_queries
maintenance_budget_interval
num_maintenance_intervals
maintenance_model_step
num_queries
num_refresh_total
num_replace_total
num_over_age_total
over_age_backlog_final
num_R_entries                  # DEFERRED — reported as 0
num_R_candidates_total         # DEFERRED — reported as 0
num_uniform_candidates_total
num_recertified_candidates_total
cache_turnover_rate_mean
cache_state_path
t_mine_round
queries_per_second
mcdp_query_encoder_calls
mcdp_doc_encoder_calls_mining
mcdp_doc_encoder_calls_maintenance
mcdp_docs_encoded_maintenance
cache_mc_bytes
cache_score_pairs
cache_maintenance_time
cache_age_mean_steps
cache_age_p95_steps
cache_age_max_steps
```

`mcdp_doc_encoder_calls_mining` must be zero. Initialization document calls are
recorded in `mining_meta_initial.json`; later document calls are attributed only
to maintenance.

Trainer per-step or per-swap logs should include:

```text
global_step
active_round_no
source_checkpoint_step
consume_step
async_gap_steps
data_age_steps
loss
step_wall_time
checkpoint_write_time
rounds_consumed
rounds_skipped
miner_idle_time
```

Runtime tuning:

```text
rising async_gap_steps or data_age_steps with miner_idle_time near zero:
    miner is bottlenecked; raise async_mine_every_steps or reduce B_doc/T

large rounds_skipped or miner_idle_time with near-zero async_gap_steps:
    trainer checkpoints faster than useful; raise async_mine_every_steps to cut checkpoint I/O
```

## Timing Calibration

Timing calibration is optional and pre-run only. It can suggest an initial fixed
`async_mine_every_steps`, but it is not required and is not a runtime controller.
Runtime correctness comes from `ready_N` markers, newest-ready consumption, and
skip-stale checkpoint selection.

Measure:

```text
t_train_step = trainer-only wall time per optimizer step on pre-mined data
t_mine_round = miner wall time to produce one full round, including periodic in-round maintenance
```

Existing timing helpers:

```text
scripts/fast_grass_train_timing.py
scripts/fast_grass_mine_timing.py
scripts/run_fast_grass_timing_singularity.sh
```

Optional initial estimate:

```text
async_mine_every_steps ~= ceil((t_mine_round / t_train_step) * safety_margin)
safety_margin = 1.1 to 1.25
```

Retune after the first real run from `async_gap_steps`, `data_age_steps`,
`rounds_skipped`, `miner_idle_time`, checkpoint write cost, and cache/mining wall
time.

## Failure Behavior

- Miner slower than checkpoint cadence: trainer keeps using the current round;
  miner finishes its current round, then skips stale checkpoints; `data_age_steps`
  grows and new consumed rounds may have larger `async_gap_steps`.
- No `ready_N`: trainer stays on `initial_data/` or the current active round.
- Partial `work_N/` or final-path artifacts without `ready_N`: ignored because
  only the ready marker commits a round; orphaned artifacts may be cleaned or
  overwritten on startup.
- Stale checkpoint: miner skips checkpoints older than the newest valid one after
  the current round finishes.
- Cache maintenance too expensive: raise `async_mine_every_steps`, lower
  `B_doc`/`T`, raise `cache_update_interval`, or lower `rho_start`/`rho_end`
  only if the cache is over-maintained.
- Cached score matrix too expensive: use cache-slot chunking and start with
  `B_doc=32k, T=3` before testing `B_doc=100k`.
- Cached states too stale: raise the maintenance budget or lower
  `max_age_epochs`; track per-slot age in metadata.
- Uncertainty promotes irrelevant high-variance documents: lower
  `lambda_val`; compare every nonzero setting against `lambda=0`.
- `T=3` uncertainty is too noisy: test `T=5` only after the cheaper setting
  demonstrates a useful signal.
- EMA requested: defer unless trainer checkpoints include EMA teacher weights.

## Tests And Validation

Async handoff/unit tests:

```text
latest ready_N discovery with get_latest_marker_no
no partial reads before ready_N
stale ready round skipping
newest valid checkpoint selection with is_valid_checkpoint
async_gap_steps and data_age_steps arithmetic
maintenance fires every maintenance_interval_mined_queries
cache.maintain_cached_mcdp receives source_checkpoint_step, not miner-local counters
dataloader swap without optimizer/scheduler reset
```

Cached scoring unit tests:

```text
synthetic q_mc and Z_mc match explicit score mean and population std
lambda=0 produces exactly the mean-score ranking
known positives are masked before selection
chunked and unchunked scoring return the same selected doc IDs
ordinary mining invokes T query passes and zero document encoder calls
dropout-only context restores every module's entry mode
```

Cache maintenance and persistence tests:

```text
initial cache contains T genuine stochastic document states, not repeated means
B_doc invariant after every maintenance interval
refreshed slots update all T embeddings, Z_mean, and timestamp together
unselected slots remain bitwise unchanged
replacement candidates are encoded once and reused for insertion
query-MC reservoir preserves T states and query IDs for qrel masking
cache_state_N.pt reloads into the next round on restart
state round trip preserves the next RNG-driven cache decision
  (registry round-tripping lands with R_doc; deferred)
schema, T, B_doc, and dimension mismatches fail before mining
metadata separates mining and maintenance document encoder calls
metadata totals aggregate across all maintenance intervals
```

Miner smoke tests:

```text
initial cached-MCDP data generation completes before ready_initial
cached-MCDP negatives come only from H
ordinary mining reports mcdp_doc_encoder_calls_mining = 0
periodic maintenance changes later portions of the same mined round
```

Trainer smoke tests:

```text
starts from initial_data
swaps to training_data_1 without resetting optimizer or scheduler
global optimizer step remains continuous across the swap
checkpoint writes optimizer.pt last
```

Cluster acceptance:

```text
trainer runs on GPU 0 and miner runs on GPU 1
no full-corpus ANN rebuild happens during async Fast-GRASS
no per-query stale FAISS top-P path is used
at least one ready_N round is produced and consumed
logs show source_checkpoint_step, consume_step, async_gap_steps, and data_age_steps
```

## Source Cross-Checks

Before implementation, confirm:

```text
NegativeCache state, maintenance metadata, and registry logic in src/utils/negative_cache.py
encode_batch_tensor normalization and dropout mode behavior in scripts/run_fast_grass.py
old _mine_batch_mcdp is unreachable from production mode selection and is oracle-only
is_valid_checkpoint and get_latest_marker_no in src/utils/helpers.py
ANCE dataloader swap pattern in scripts/run_ance_train.py
ANCE ready marker and checkpoint polling pattern in scripts/run_ance_data_gen.py
```

The async orchestration remains ANCE-style, but the production MCDP algorithm
intentionally changes from lazy fresh-MCDP to cached-MCDP. Keep EMA behavior
intact. Fresh-MCDP is permitted only in the bounded quality probe used as an
oracle, not as a training configuration.

## Decisive Experiments

Keep the first evaluation small and attributable:

```text
1. lambda=0 versus cached-MCDP lambda=0.5 at B_doc=32k, T=3
2. EMA versus cached-MCDP under the same cache size and mining budget
3. fresh-MCDP on a small fixed query sample as a ranking-quality oracle
4. T=3 versus T=5 only if cached uncertainty helps but appears noisy
5. refresh schedule 0.5->0.25/max_age=2 versus a cheaper schedule only if maintenance dominates
```

Report mining throughput, ready rounds per epoch, Recall@1000, and NDCG@10.
Cached-MCDP is worth retaining only if the nonzero uncertainty setting improves
retrieval quality over `lambda=0` without returning the miner to the previous
fresh-MCDP throughput bottleneck.
