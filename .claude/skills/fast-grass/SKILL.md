---
name: fast-grass
description: Resume Fast-GRASS work — the negative-cache architecture, its feasibility probe, and (later) the trainer. Use when continuing Fast-GRASS / negative-cache mining development on this repo.
user-invocable: true
argument-hint: [topic]
---

# Fast-GRASS Session

Repo: `dense-retrieval-SOTA`. **Working branch: `fast-grass`** (off `main`). Read this fully
before acting. Canonical design: **`fast_grass_negative_cache_architecture.md`** +
**`fast_grass_implementation_details.md`** (repo root). Architecture is fixed; build to it.
Approved working plan: `~/.claude/plans/smooth-kindling-spindle.md`.

NOTE: the cwd may be the unrelated `markitdown` repo. The real repo is
`/Users/aiamn/PycharmProjects/dense-retrieval-SOTA` (local) ↔ `~/dense-retrieval-SOTA` on DelftBlue.

## ⏩ RESUME HERE (2026-06-21)
Naive GRASS merged; feasibility gate PASSED; **Phase 1 cache core BUILT + review-HARDENED
(25/25)**; **Phase 2 trainer BUILT** — all 4 deliverables + the 1-line config add are in place
and green locally. → **Next action: GIT (user drives)** — commit + push the `fast-grass` branch
(Phase 1 + Phase 2 are all uncommitted), then Phase 3 = ablation RUNS (no code) on DelftBlue.

**Phase 2 files (NEW, done 2026-06-21):**
- `scripts/run_fast_grass.py` — `_mine_batch`, `run_fast_grass_pipeline`, `_build_fast_grass_cfg`,
  `_load_train_items`, `main`. Pipeline owns model loading (`main` passes `models=None`); stale
  index reuse/build-once/never-rebuild; per-step cost DELTAS to `cost_log.jsonl`; reuses
  `run_grass._update_ema` + GRASS optimizer/compile/eval scaffolding by import. NO edits to
  run_grass.py / helpers.py / negative_cache.py.
- `scripts/fast_grass_train_smoke.py` — CPU e2e drive of the pipeline with mocks
  (`compile_model=False, do_eval=False, output_model_dir=<tmpdir>`). **8/8.**
- `scripts/run_fast_grass_singularity.sh` — trainer SLURM launcher (`FAST_GRASS_*` knobs incl. `--no_registry`).
- `scripts/run_fast_grass_feasibility_singularity.sh` — standalone gate benchmark launcher (`FG_FEAS_*`).
- `config/config.yaml` — one-line `batch_size: 64` under `training.fast_grass`.

**Verified green (2026-06-21):** train_smoke 8/8, fast_grass_test 25/25, fast_grass_smoke 5/5,
grass_test 33/33, grass_smoke 9/9; both `.sh` pass `bash -n`; trainer parses + imports clean.
`--no_registry` = `uniform_candidate_fraction=1.0` AND `R_size_factor=0`. Detail of each
milestone follows.

## Status (2026-06-20)

**Naive GRASS = DONE & merged.** The merge-safe `run_grass.py` refactor is committed on `main`
(commit b633376; `encode_batch_tensor` in helpers, `_mine_queries` split). Tests green:
`scripts/grass_test.py` 33/33, `scripts/grass_smoke.py` 9/9. Don't redo it.

**Decision (locked):** eventually build the FULL Fast-GRASS architecture (registry R +
recertification + adaptive rho + Softmax). BUT **feasibility first** — the user's priority is the
**mining speedup RATIO** (Fast-GRASS vs current GRASS) before investing in the trainer.

**This session's work — `scripts/grass_negcache_feasibility.py` made runnable (committed).**
- It imported 4 fns from `grass_twoset_feasibility.py`, which is **not committed in git**. BUT that
  module **exists on DelftBlue** (`~/dense-retrieval-SOTA/scripts/grass_twoset_feasibility.py`, 26KB)
  AND its compiled bytecode survives locally at
  `scripts/__pycache__/grass_twoset_feasibility.cpython-313.pyc`.
- I disassembled the `.pyc` (`marshal` + `dis`) and **recovered + INLINED** the 4 functions
  (`load_train_queries`, `encode_queries`, `build_candidate_matrix`, `cached_grass_sampler`) + the
  `_topk_neighbors` helper into `grass_negcache_feasibility.py`, removed the broken import, added
  `json`/`pickle` imports. Now self-contained (runs without `grass_twoset_feasibility.py`).
- Validated: compiles, imports, synthetic correctness of recovered fns, and **real cluster quick run
  7/7 PASS** (behavior confirms the reconstruction).
- Git: committed as `0605b1f "neg cache feasibility"` on branch `fast-grass`. **NOT pushed** (no
  upstream; push with `git push -u origin fast-grass`). `run_negcache_feasibility_singularity.sh` is
  **staged** (edited to full A100 — see below), not yet committed.
- OPEN: if/when committing the self-contained version permanently, optionally diff the inlined fns
  against the real `grass_twoset_feasibility.py` on the cluster to confirm byte-faithfulness (the
  recovery already passes synthetic + real runs, so this is belt-and-suspenders).

## Feasibility results so far

**Quick run** (gpu-a100-small, `NC_MAXQ=3000 NC_SKIP_ENCODE=1`): **7/7 PASS**, verdict 🟢.
- T1 cache fits huge headroom: 10% of corpus (65.6k docs) = 0.40 GB bf16 (budget 10 GB).
- T3 scoring ~15k q/s; full-epoch scoring ≈ 0 min.
- T8 mining speedup **~1,827×** — but DON'T quote it. Caveats: baseline used a hardcoded
  **1,500 docs/s fallback** (because `--skip_encode_test`); "new" time is matmul vs a **synthetic
  random Z_H** and **excludes cache maintenance**; H here is **freq-ranked** (`select_global_H`), not
  the uniform-random v0 the doc specifies (so T5's overlap looks optimistic — quality caveat).

**NEXT (user is running):** grounded full run on the **full `gpu-a100`** partition, **no skip**:
`NC_MINIBATCH=1 sbatch scripts/run_negcache_feasibility_singularity.sh`. This makes T2 measure the
**real encoder docs/s** → T8 ratio is anchored to measured hardware (not 1,500 fallback), and T7
runs a real minibatch to finite loss. Read the speedup off the resulting `logs/negcache_feas_*.out`.

The launcher (`scripts/run_negcache_feasibility_singularity.sh`) was changed: partition
`gpu-a100-small`→`gpu-a100`, `cpus-per-task` 2→16, `OMP/MKL_NUM_THREADS` 2→8 (match `run_grass` so
CPU-bound parts don't skew the ratio). Same edit must be applied to the cluster copy (sed) or scp'd.

## Phase 1 — cache core + tests BUILT (2026-06-20, staged on `fast-grass`)

Feasibility gate PASSED (grounded full run: 8/8, T2 0.35× ANCE, T6 9× fewer encodes
vs current GRASS). So we executed **Phase 1** of the refined plan (user pasted it;
it's a faithful paraphrase of the two design docs — verified anchors first, all OK:
`_mine_queries` is a clean module-level import at `run_grass.py:194`; `encode_batch_tensor`
does CLS+L2+no_grad; config uses `learning_rate`/`ema_alpha` names).

Built + locally verified (CPU):
- **`src/utils/negative_cache.py`** — `NegativeCache` (init_uniform from stale pickle,
  score Q×Z_H → g=ŝ+λσ, mask_positives, TopK + Gumbel-Softmax select, record_selection,
  adaptive-rho `maintain()` = refresh/replace/recertify + utility/age + cost counters,
  memory_bytes) + `RetiredRegistry` (admit/evict/nominate) + `linear_decay`. EMA v0.
  Z_H is no-grad, bf16 on cuda / fp32 cpu. `maintain()` takes a reservoir dict
  `{q_student,q_teacher,qids}` (None ⇒ refresh-only) and `qrels_dict`.
- **`scripts/fast_grass_test.py`** — 19/19 PASS (init/score/mask/select/utility/budget/
  refresh/maintain-invariant/registry/recertify).
- **`scripts/fast_grass_smoke.py`** — 5/5 PASS (API + config keys + tiny CPU cycle).
- **`scripts/fast_grass_feasibility.py`** — mining-speed benchmark: baseline
  `_mine_queries` vs Fast-GRASS cache path on the SAME batches; RowCounter forward-hook
  + wall-time; amortized maintenance; `--synthetic` (CPU, Fast-GRASS-only, NO ratio)
  works locally; **real GPU run pending on DelftBlue** (needs stale index + processed
  mixture). Flags: `--batches --B_doc --synthetic --steps-per-epoch`.
- **`config/config.yaml`** — added `training.fast_grass` (40 keys, v0 defaults; sibling
  to `grass`, untouched). Training knobs copied in so the block is self-contained.
- Regression: `grass_test.py` 33/33, `grass_smoke.py` 9/9 still green.

### Phase 1 HARDENED after code review (still staged on `fast-grass`)
A review found 9 issues; all fixed in `negative_cache.py` + a follow-up of 4 small fixes,
each **mutation-tested** (reverted the bug, confirmed the test fails). Now **25/25** unit
+ 5/5 smoke + 33/33 & 9/9 regression. The 9+4: (1) grace gate — floor-eviction only after
`K*cache_update_interval` steps (stops new-doc/init churn); (2) restore `teacher.training`
in maintain; (3) `score()` under `no_grad`; (4) seeded `torch.Generator` for Gumbel (was
global rand); (5) `select` raises if <m finite slots; (6) recert skips -inf-reentry
candidates; (7) reinserted R doc popped from R; (8) `uniform_candidate_fraction` is the
binding constraint (not R_fraction); (9) `doc_encoder_calls_cache_replace=0` (reuses recert
encodes); +`no_grad` recert scoring; +`L>=m` softmax guard; +comment clarity. Grace is tied
to `K*cache_update_interval` (no separate config field — user's choice).

All 5 files **staged on `fast-grass`**, NOT committed/pushed (user drives git;
`git push -u origin fast-grass`). Stale launcher edit also pending commit.

## Key learnings (don't relearn these)

- **GPU/ratio:** Fast-GRASS scoring and current-GRASS encoding are both GPU-GEMM-bound → the ratio
  roughly transfers across GPUs; the real driver is algorithmic (encode P≈200 docs/query vs ONE
  matmul), which is hardware-independent. BUT measure **both sides on the same GPU** (no skip), and
  CPU-bound parts (`build_candidate_matrix` Python loop, T5 masking) DON'T scale with GPU — hence
  matching cpus/threads to the real run.
- **Data schema:** raw mixture `train_reasonir_*.jsonl` use `positives`/`negatives` = passage TEXT.
  `run_setup()` (`src/data/preprocessor.py`) rewrites them into `train_hq/msmarco/vl.jsonl` with
  `positive_passages=[{docid,text}]`. Both `load_train_queries` AND `run_grass.py` expect the
  processed `positive_passages` schema. Locally only raw files exist → "0 unique train queries"
  locally; the cluster has the processed files (schema OK confirmed there).
- **Feasibility probe is read-only** and requires pre-existing inputs (no `run_setup`): stale index
  `temp_grass_workdir/stale_index/corpus.pkl` (2.7 GB, 655,644 docs, dim 1024), processed mixture,
  qrels. Exits with a clear "required inputs missing" message if absent.
- **macOS local runs** crash with an OpenMP duplicate-runtime abort (faiss vs torch) →
  prefix with `KMP_DUPLICATE_LIB_OK=TRUE`.
- **DelftBlue:** user `aimanabdulwaha`, `~/dense-retrieval-SOTA`, `/scratch/$USER/dense-retrieval-SOTA`,
  container `/scratch/$USER/containers/pytorch_2.1.sif`. Partitions: `gpu-a100` (full, long queue) vs
  `gpu-a100-small` (≤10 GB MIG, 2 cores, 4 h, short queue). Launcher knobs are `NC_*` env vars
  (`NC_MAXQ`, `NC_SKIP_ENCODE`, `NC_MINIBATCH`, `NC_BDOC_FRAC`, `NC_BATCH`, ...).

## What the probe's tests mean (`grass_negcache_feasibility.py`)
T1 cache budget (Z_H=T·|H|·dim fits) · T2 MCDP-refresh ≤ 0.5× ANCE full-corpus encode · T3 `Q×Z_H`
throughput · T4 synthetic sampler top-m+masking · T5 positive-mask contamination==0 on real H · T6
fresh-encode count drops the B·L/B·P rerank · T7 one real minibatch finite loss (`--minibatch_test`)
· T8 mining-only speedup vs current-GRASS estimate.

## Phase 2 — PLAN APPROVED, ready to implement (NEXT SESSION START HERE)

Full plan in **`~/.claude/plans/smooth-kindling-spindle.md`** (user approved it after 3
review rounds; it EMBEDS both design docs as Appendix A/B because Ultraplan-on-the-web
clones the markitdown cwd and can't read dense-retrieval-SOTA — see [[ultraplan-clones-cwd-repo]]).
Just implement it. **4 new files in `scripts/` + 1-line config add. Do NOT edit
`run_grass.py`/`helpers.py`/`negative_cache.py`.**

Deliverables:
1. **`scripts/run_fast_grass.py`** — `_mine_batch` (encode q no_grad → `cache.score` → mask →
   `select` → record) + `run_fast_grass_pipeline(cache, c_ids, corpus_lookup, qrels_dict,
   qid_to_text, train_items, cfg, config, ctx, device, models=None, compile_model=True,
   do_eval=True, output_model_dir=None, debug=False)` (Algorithm-1: mine → fresh-encode
   pos+negs → InfoNCE → step → `_update_ema` → periodic `cache.maintain` every
   `cache_update_interval` → checkpoint → eval) + `_build_fast_grass_cfg(config,args,steps_per_epoch)`
   + `main()`. Reuse via import: `from run_grass import _update_ema`,
   `TemperatureScaledContrastiveLoss`, helpers, optimizer/compile pattern (run_grass.py:315-336).
2. **`scripts/run_fast_grass_singularity.sh`** — mirror `run_grass_singularity.sh`, `FAST_GRASS_*` knobs.
3. **`scripts/fast_grass_train_smoke.py`** — CPU e2e via pipeline with injected mock models,
   `compile_model=False, do_eval=False, output_model_dir=<tmpdir>`.
4. **`scripts/run_fast_grass_feasibility_singularity.sh`** — sbatch the Phase-1 benchmark; knobs
   `FG_FEAS_{BATCHES,B_DOC,UNCERTAINTY,MAINTAIN_EVERY,STEPS_PER_EPOCH}`.

Review-baked specifics (already in the plan — honor them): **model loading = the pipeline**
(main passes `models=None`, never double-loads); stale index reuse-if-present/build-once/never-
rebuild; **cost_log.jsonl logs per-step DELTAS** (`cache_score_pairs` is cumulative);
`--no_registry` sets `uniform_candidate_fraction=1.0` AND `R_size_factor=0`; add `batch_size:64`
to `training.fast_grass`; `replacement_yield_at_K = num_replace/max(1,num_recertified_candidates)`
(null if no maintenance); cost confirms mining COST in-loop, the speedup RATIO comes from
deliverable 4. EMA v0; raw `AutoModel` in-process; H init uniform-random; v0 defaults in the
config block + `fast_grass_implementation_details.md`.

Phase 3 (runs, not code): ablation sweeps (λ 0/1, B_doc 32k/100k/512k, m 1/4, TopK/Softmax,
no-R vs R+recert, max-g vs avg-top-k reentry).

## Reusable helpers (`src/utils/helpers.py`)
`get_path, load_config, get_training_context, build_faiss_index, _load_qrels, _load_corpus_lookup,
set_seed, encode_batch, encode_batch_tensor, encode_to_pickle, _pool_and_fresh_rerank, evaluate_bright`.

## Working preferences (user)
Prefers to clarify multiple-choice questions conversationally before answering (answer the follow-up
plainly, then proceed once they say "go ahead"). Before refactors: test-impact audit + minimum churn.
Don't delete research artifacts — flag, don't clean. Stage edits for the user to commit/push (they
drive git). The old `grass` SKILL.md is stale (describes `sequential-grass`).
