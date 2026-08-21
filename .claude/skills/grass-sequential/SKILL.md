---
name: grass-sequential
description: Quick recall of the Fast-GRASS pipeline (Negative-Cache-inspired global-cache mining) on the `fast-grass` branch. Two estimators — EMA + teacher-free MCDP. Use when resuming after a memory clear, or to check the shape of run_fast_grass.py / negative_cache.py / the training.fast_grass config.
user-invocable: true
---

# Fast-GRASS — current state (`fast-grass` branch)

> **Read this first** after a context clear. The active branch is now `fast-grass`.
> This supersedes the old per-query-FAISS "sequential GRASS" world; that pipeline
> still lives on the `sequential-grass` branch (`run_grass.py`, `_mine_queries`,
> `_pool_and_fresh_rerank`, `CandidateMemory`) but is NOT what runs here.

## Where I left off (latest work)

**Added teacher-free MCDP as the second estimator; EMA kept as baseline.** Code is
done + all tests green (compile clean, `fast_grass_test.py` **28/28**, smokes
**6/6** + EMA **9/9** + MCDP **10/10**). NOT yet run on the cluster — that's the next
step. Nothing committed. What changed:
- `--uncertainty {ema,mcdp}`, **config default now `mcdp`**. MCDP is teacher-free:
  no `Z_teacher`, σ from top-L MC-dropout; cache maintenance/recert degrade to
  student-only. EMA path unchanged.
- New knobs: `T=3`, `mc_dropout_p=0.3`, `L=128` (was 1024 — MCDP cost ≈
  `batch_size·L·T`; guardrail warns >25k; first real run may want `L≈50–64`).
- Safety guards in `_mine_batch_mcdp`: raise clearly if any query has `<m` finite
  top-L candidates or the whole batch union is empty; strict `corpus_lookup[d]`.

## TL;DR

GRASS scoring `g = ŝ + λσ`, but mining runs against **one bounded global negative
cache `H`** instead of per-query stale FAISS top-P + fresh rerank. Two estimators,
pick at the CLI/config (default **mcdp**), selection default TopK:
- **mcdp** (default, teacher-free): cheap deterministic student score over all `H` →
  top-`L` → full query/document MC-dropout (`T` passes) → σ = std over T.
- **ema** (baseline): σ = |student − EMA-teacher| over all `H`.

```bash
python scripts/run_fast_grass.py --uncertainty {mcdp|ema} \
  [--T 3] [--mc_dropout_p 0.3] [--L 128] [--ema_alpha A] \
  [--B_doc N] [--lambda_val {0|1}] [--selection_mode {topk|softmax}] \
  [--m M] [--no_registry] [--no_eval] [--num_epochs N] [--model_suffix STR] [--debug]
```

Cluster: `FAST_GRASS_UNCERTAINTY=mcdp FAST_GRASS_L=64 FAST_GRASS_B_DOC=32000 sbatch scripts/run_fast_grass_singularity.sh`.

## What the pipeline does (Algorithm 1 over the global cache)

Per minibatch (`run_fast_grass_pipeline` in `scripts/run_fast_grass.py`):

1. **Mine** (`_mine_batch` → `_mine_batch_ema` | `_mine_batch_mcdp`): select `m`
   negs/query against `H`. No FAISS, no rerank.
2. Push the selection query embeddings into a rolling **recert reservoir** (reused
   for maintenance — no extra encode). MCDP pushes `q_teacher=None`.
3. **Train** (identical to GRASS): fresh-encode selected pos+negs
   (`requires_grad=True`) → `TemperatureScaledContrastiveLoss` → `optimizer.step()`
   → `scheduler.step()`.
4. **EMA-update** the teacher (`_update_ema`, imported from `run_grass.py`) — EMA mode only.
5. Every `cache_update_interval` steps: **amortized cache maintenance**.

### Mining detail
**EMA** (`cache.score`, over all of `H`):
```
sigma = |q_stu·Z_studentᵀ − q_tea·Z_teacherᵀ|   # (batch, B_doc), no_grad
g     = s_student + λ·sigma  → mask positives → select
```
**MCDP** (`_mine_batch_mcdp`, teacher-free, lazy top-L):
```
eval:  s_cheap = q_det · Z_studentᵀ  → mask positives → top-L per query   (cheap_scores)
train: dropout-encode the DEDUPED top-L union (T passes, query+doc)
       sims = q_t·d_t ; ŝ=mean_T ; σ=std_T ; g=ŝ+λσ  → scatter to (B,B_doc)
       → mask positives (belt+braces) → select
```
Both end in `select`: `topk` (top-m by g) | `softmax` (Gumbel-top-k over β·g, top-L
prefilter). `select` **raises** if any query has `< m` finite slots (never emits a
masked positive); MCDP **also** raises earlier if any query has `<m` finite top-L or
the batch union is empty.

### Cache maintenance (`NegativeCache.maintain`)
- Update `utility_ema` from a per-interval `selected_indicator`; track
  `peak_utility_ema`, `lifetime_selected_count`, `intervals_since_selected`.
- Budget = `linear_decay(rho_start, rho_end, progress) · B_doc ·
  cache_update_interval / steps_per_epoch`. `num_refresh + num_replace ≤ budget`.
- Action order: urgent over-age (refresh useful / replace low-util) → persistently
  low-util replace → useful stale refresh by `utility·age_norm` → defer rest.
- **Refresh:** re-encode the same docid's `Z_H` with the current student (+ teacher
  in EMA mode; MCDP refreshes student states only).
- **Replace:** evict → candidates = uniform (dominant, `uniform_candidate_fraction`
  ≥ 0.75) ∪ `R.nominate` → recertify against the reservoir under `no_grad`
  (`reentry = mean top-k g`; EMA g = s_stu+λ|s_stu−s_tea|, MCDP g = s_stu only;
  masks positives) → insert top-`num_replace`.
- **Grace window** `K·cache_update_interval`: freshly-inserted / init docs are
  exempt from "persistently low-utility" churn until they've had a chance to prove
  useful.

### RetiredRegistry `R`
Metadata-only (`peak_utility_ema`, `lifetime_selected_count`, `last_seen_step`),
bounded at `R_size_factor·B_doc`. Admits an evicted doc only if it was previously
useful; **nominates** replacement candidates during maintenance but never scores
queries and never stores embeddings. `--no_registry` fully disables it (uniform-only
candidates, `R_size_factor=0`).

Gradients never flow through `Z_H` — the cache is selection-only; the training loss
always uses fresh encodes.

## Files (`fast-grass` branch)

| Path | Role |
|------|------|
| `scripts/run_fast_grass.py` | **The** trainer. `_mine_batch` (dispatcher) + `_mine_batch_ema` + `_mine_batch_mcdp` + `_selection_diag`, `run_fast_grass_pipeline`, `_build_fast_grass_cfg`, `main`. Imports `_update_ema` from `run_grass.py`. |
| `src/utils/negative_cache.py` | `NegativeCache` (`H`/`Z_H`, `score`/`cheap_scores`/mask/select/maintain/refresh/replace; `Z_teacher` optional) + `RetiredRegistry`. |
| `scripts/run_fast_grass_singularity.sh` | SLURM launcher. `FAST_GRASS_*` env passthroughs (incl. `UNCERTAINTY/T/MC_DROPOUT_P/L`). |
| `scripts/fast_grass_test.py` | 28 unit tests (see below). |
| `scripts/fast_grass_smoke.py` | CPU wiring + one init→score→mask→select→maintain cycle. |
| `scripts/fast_grass_train_smoke.py` | Mock end-to-end drive of `run_fast_grass_pipeline` (no GPU/compile/eval). |
| `scripts/fast_grass_feasibility.py` | Fast-GRASS vs GRASS mining-speed benchmark (`--synthetic` = CPU, Fast-GRASS only). |
| `src/utils/helpers.py` | `encode_batch_tensor` (grad/no-grad encode), `build_faiss_index` (init source only), `_load_qrels`, `_load_corpus_lookup`, `evaluate_bright`, path helpers. |
| `config/config.yaml → training.fast_grass` | All Fast-GRASS knobs. |
| `fast_grass_negative_cache_architecture.md` / `fast_grass_implementation_details.md` | Design spec + v0 defaults. |

## Code map (what each symbol does)

**`src/utils/negative_cache.py`**
- `NegativeCache.__init__` — holds `docids`, `docid_to_slot`, `Z_student`/`Z_teacher`
  (detached, `no_grad`), per-slot metadata tensors (`last_refreshed_step`,
  `utility_ema`, `peak_utility_ema`, `selected_indicator`, `selected_count_recent`,
  `lifetime_selected_count`, `intervals_since_selected`), a `RetiredRegistry`, and a
  seeded torch `Generator` for Gumbel selection.
- `init_uniform` (classmethod) — samples `B_doc` corpus docids (numpy seeded RNG),
  copies + L2-normalizes stale-index embeddings into `Z_student` (+ `Z_teacher` iff
  `cfg['uncertainty']=='ema'`, else `None`); bf16 on CUDA. Clamps `B_doc` to corpus.
- `score(q_student, q_teacher, lambda_val)` — EMA mining matmul under `no_grad`:
  returns `(g, s_student, sigma)`, each `(batch, B_doc)`; counts `cache_score_pairs`.
  **Raises** on a teacher-free cache (use `cheap_scores` for MCDP).
- `cheap_scores(q_student)` — student-only `q_student·Z_studentᵀ` under `no_grad`
  (MCDP's cheap top-L ranking); counts `cache_score_pairs`. No teacher needed.
- `mask_positives(g, qids, qrels)` — sets `g[i, slot]=-inf` for a query's known
  positives (by `docid_to_slot`).
- `select(g, m, mode, beta, L)` — `topk` = top-m by `g`; `softmax` = `_gumbel_topk`
  (Gumbel-top-k over `β·g`, optional top-L prefilter, seeded, `-inf` never wins).
  Raises if any query has `< m` finite slots. Returns `(slots, docids)`.
- `record_selection(slots)` — flips `selected_indicator` for chosen slots (utility signal).
- `maintain(...)` — one bounded cycle: `_update_utility` → `_interval_budget` →
  `_plan_actions` → `_refresh` (+ `_replace` if a reservoir is given). Forces both
  models to `eval()` and restores their modes; returns a cost-counter dict.
- `_update_utility` — folds `selected_indicator` into `utility_ema`, updates peak /
  lifetime / `intervals_since_selected`, resets the indicator.
- `_interval_budget` — `round(linear_decay(rho_start,rho_end,progress)·B_doc·interval/steps_per_epoch)`.
- `_plan_actions` — orders slots (over-age refresh/replace → persistent-low-util
  replace → useful-stale refresh) under the shared budget; the grace window
  `K·interval` shields new/init docs; returns `(refresh_slots, replace_slots, diag)`.
- `_refresh` / `_replace` / `_insert` / `_sample_uniform` — refresh re-encodes `Z_H`
  in place (all `Z_teacher` writes guarded — no-ops when teacher-free); replace
  nominates uniform∪R candidates, recertifies under `no_grad` (student-only when
  teacher-free), admits evicted docs to `R`, inserts winners; `_encode_docs` wraps
  `encode_batch_tensor(requires_grad=False)` and returns `zt=None` teacher-free.
- `RetiredRegistry` — `admit` (only if previously useful; bounded by
  `peak_utility_ema`/`lifetime_selected_count`), `nominate` (seeded subset), `__len__`.

**`scripts/run_fast_grass.py`**
- `_mine_batch` — dispatcher on `fg_cfg['uncertainty']` → `_mine_batch_ema` |
  `_mine_batch_mcdp`. Returns `(mined, slots, q_student, q_teacher, stats)`
  (`q_teacher=None` for MCDP). Takes `corpus_lookup` (used only by MCDP).
- `_mine_batch_ema` — `cache.score` over all `H` → mask → select → `record_selection`.
- `_mine_batch_mcdp` — teacher-free: eval `cheap_scores` → top-L (finite-masked) →
  dropout-encode the **deduped top-L union** (T passes, query+doc) → per-query
  ŝ/σ/g scattered into a full grid → mask → select. Mode restored via `try/finally`.
- `_selection_diag` — shared per-step σ-testability + cost stats (incl.
  `flip_rate_vs_lambda0`, MCDP encode-cost fields); no extra encodes.
- `run_fast_grass_pipeline` — Algorithm 1 loop: mine → push queries to recert
  reservoir → fresh-encode selected pos+negs → `TemperatureScaledContrastiveLoss` →
  optimizer/scheduler step → `_update_ema` (EMA only) → periodic `cache.maintain` →
  write `mining_log.jsonl`/`cost_log.jsonl` → checkpoint/save. Loads teacher only in
  EMA mode; sets MCDP dropout-`p`; prints the `batch_size·L·T>25k` guardrail warning.
  `_student_raw` is the uncompiled handle for save/EMA/cache encodes.
- `_build_fast_grass_cfg` — merges `training.fast_grass` + CLI overrides + derived
  `steps_per_epoch`/`total_steps`/`max_age_steps`; `--no_registry` sets
  `uniform_candidate_fraction=1.0`, `R_size_factor=0`.
- `_load_train_items` — flattens the training mixture to `{query_id, query, pos_docid}`.
- `main` — cluster entry: `run_setup` → build/reuse stale index → `build_faiss_index`
  for `(embs, c_ids)` → `init_uniform` cache → free FAISS/embs → run pipeline.

**Reused helpers (`src/utils/helpers.py`)** — `encode_batch_tensor` (grad/no-grad
CLS-pooled + L2-normalized encode, bf16 autocast on CUDA; `requires_grad` toggles
`no_grad`), `build_faiss_index` (returns ordered `(index, embs, c_ids)`; index unused
here), `_load_qrels`, `_load_corpus_lookup`, `encode_to_pickle`, `evaluate_bright`,
`get_path`/`get_training_context`/`set_seed`. `_update_ema` is imported from
`run_grass.py`.

## CLI / SLURM

CLI flags: `--uncertainty {ema,mcdp}` (default from config = mcdp) · `--T` ·
`--mc_dropout_p` · `--L` · `--ema_alpha` (ema only) · `--B_doc` · `--lambda_val` ·
`--selection_mode {topk,softmax}` · `--m` · `--no_registry` · `--no_eval` ·
`--num_epochs` · `--model_suffix` · `--debug`. (Flags default `None` → fall back to config.)

SLURM env vars (all optional, fall back to config): `FAST_GRASS_UNCERTAINTY`,
`FAST_GRASS_T`, `FAST_GRASS_MC_DROPOUT_P`, `FAST_GRASS_L`, `FAST_GRASS_MODEL_SUFFIX`,
`FAST_GRASS_NUM_EPOCHS`, `FAST_GRASS_B_DOC`, `FAST_GRASS_LAMBDA`,
`FAST_GRASS_EMA_ALPHA`, `FAST_GRASS_SELECTION_MODE`, `FAST_GRASS_M`,
`FAST_GRASS_NO_REGISTRY`, `FAST_GRASS_NO_EVAL`.

Output dir: `models/{model_name}_{uncertainty}` (e.g. `fast_grass_mixed_bge_m3_mcdp`).

## Config (`training.fast_grass`)
```yaml
B_doc: 100000            # global cache size; ablate 32k / 100k / 512k
m: 1                     # negs/query
selection_mode: topk     # topk | softmax
lambda_val: 1.0          # g = s_hat + lambda*sigma; baseline ablation = 0
beta: 5.0                # softmax temperature
uncertainty: mcdp        # mcdp (teacher-free, default) | ema (baseline)
ema_alpha: 0.999         # ema mode only; teacher decay (1.0 = frozen base teacher)
L: 128                   # MCDP top-L shortlist / softmax prefilter (cost ~ bs*L*T; was 1024)
T: 3                     # MCDP dropout passes
mc_dropout_p: 0.3        # MCDP dropout probability
rho_start: 0.50 / rho_end: 0.10 / cache_update_interval: 100 / max_age_epochs: 4
utility_ema_decay: 0.95 / utility_floor: 0.01 / utility_remember_threshold: 0.05
K: 3 / R_fraction: 0.25 / uniform_candidate_fraction: 0.75
replacement_candidate_multiplier: 2 / recent_query_reservoir_size: 128
reentry_top_k: 5 / R_size_factor: 0.5 / cache_init_seed: 42
num_epochs: 2 / batch_size: 64 / learning_rate: 1e-5
```

## Logs
- `mining_log.jsonl` — σ-testability per step (both estimators): `s_hat_mean`,
  `sigma_mean`, `sel_s_hat_mean`, `sel_sigma_mean`, `sel_lambda_sigma_mean`,
  `sel_sigma_over_s_hat`, `flip_rate_vs_lambda0` (does σ change the pick vs λ=0),
  `selected_doc_diversity`. **MCDP adds:** `mcdp_L_used`, `mcdp_T`,
  `mcdp_unique_docs`, `mcdp_query_encoder_calls`, `mcdp_doc_encoder_calls`
  (actual = unique·T), `estimated_max_mcdp_doc_encodes_per_step` (= bs·L·T; compare
  to actual to see top-L union dedup savings).
- `cost_log.jsonl` — per-step cost deltas: `doc_encoder_calls_loss`,
  `doc_encoder_calls_cache_refresh/replace`, `doc_encoder_calls_mcdp`,
  `query_encoder_calls_mcdp`, `estimated_max_mcdp_doc_encodes_per_step` (0 for EMA),
  `cache_score_pairs`, `num_refresh`, `num_replace`, `num_over_age`,
  `over_age_backlog`, `num_R_*`, `num_recertified_candidates`,
  `replacement_yield_at_K`, `cache_turnover_rate`, `step_wall_time`.

## Tests / verification
```bash
python -m py_compile scripts/run_fast_grass.py src/utils/negative_cache.py && \
  python scripts/fast_grass_smoke.py && \
  python scripts/fast_grass_train_smoke.py && \
  python scripts/fast_grass_test.py
```
Expect compile clean and **28/28** `fast_grass_test.py`, `fast_grass_smoke.py`
**6/6**, `fast_grass_train_smoke.py` **EMA 9/9 + MCDP 10/10** (runs both estimators
with `KMP_DUPLICATE_LIB_OK=TRUE`). Cache tests cover init/score/**cheap_scores**/
**teacher-free MCDP cache + maintain**/mask/select/utility/budget/grace/refresh/
maintain/replace/registry/recertify; the MCDP train-smoke asserts teacher-free
end-to-end, top-L-only encoding (`unique_docs < B_doc`, `≤ Q·L`), cost identities,
positive masking, negatives-from-`H`, and grad-free `Z`.

## Gotchas / invariants
- **`Z_H` is selection-only, `no_grad`, bf16 on CUDA.** Never let gradients flow
  through it; the loss encodes fresh (`requires_grad=True`).
- **Mining restores model modes via `try/finally`.** EMA/MCDP-cheap encode in
  `eval()`; MCDP dropout passes flip to `train()` then restore. `maintain` restores
  student (+ teacher) modes.
- **age is a global-step counter**, not epochs. Churn is utility-triggered +
  grace-protected, not time-based eviction.
- **Uniform candidate dominance is the binding constraint** (`uniform_candidate_fraction`,
  ≥0.75); `R_fraction` only fills the remainder — config drift in `R_fraction` can't
  break dominance.
- **EMA update uses the uncompiled `_student_raw` handle** (`torch._foreach_*` breaks
  on compiled wrappers). `torch.compile` falls back silently (`suppress_errors=True`).
- **MCDP is teacher-free by design** (no `Z_teacher`, no EMA teacher). `cache.score`
  raises on an MCDP cache — MCDP mining uses `cheap_scores` + top-L dropout. Config
  default is `mcdp`; run EMA with `--uncertainty ema`.
- **MCDP cost ≈ `batch_size·L·T`.** Keep `L` small (default 128; first real run may
  want `L≈50–64`). The trainer warns at launch if `bs·L·T > 25k`. Docs are the source
  of truth — `fast_grass_implementation_details.md` now says `L=128`.
- The R-recertification ablation matrix + Softmax-sampling specifics live in
  `fast_grass_*.md`; `--no_registry` gives the no-R ablation.
- The stale FAISS index is only an **init source** for `H` (via `build_faiss_index`);
  it's freed after cache init — there is no per-query ANN in Fast-GRASS.
- `--no_eval` defers the BRIGHT eval; run it later via the canonical
  `run_all_evals.py` / `run_evaluate_singularity.sh` (in-pipeline eval uses
  `evaluate_bright`). Deferring avoids the shared eval-scratch race in parallel sweeps.

## Status / results
- **EMA:** cluster runs done at `B_doc=32k` — `fast_grass_..._l0_32k_ema` (λ=0) and
  `_l1_32k_ema` (λ=1); see `logs_cluster/`.
- **MCDP:** code done + tests green, **not yet run on the cluster** (next step —
  start with a small `L`, e.g. `FAST_GRASS_L=64`, and watch the `mcdp_*` cost fields).
- Metrics tracked: MRR@10, NDCG@10, Recall@20/100. Headline uncertainty question:
  does σ change the mined negative — read `flip_rate_vs_lambda0` in `mining_log.jsonl`
  (compare λ=0 vs λ=1, and now EMA vs MCDP).
- Nothing committed yet on this MCDP work.
