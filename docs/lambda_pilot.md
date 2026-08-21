# Lambda pilot (Stage 7)

Cheap λ selection for Async Fast-GRASS before committing to a 16-hour run.

**Not a new miner.** A manifest + two recipe blocks + a probe + a validity gate + a
decision rule, all driving the **existing** async cached-MCDP miner on a representative
10% subset. Nothing in the mining algorithm changed.

Why it exists: every async run before the refresh fix had **dead document refresh**
(`max_age_steps` was forced to `max_age_epochs * steps_per_epoch = 10314 == total_steps`,
and `_plan_actions` tests `age >= max_age_steps`). See CLAUDE.md → "The refresh bug".

> ## ✅ OUTCOME — the pilot has RUN and CONCLUDED (August 2026)
>
> Probe selected λ=0.3 (low band, 16.7% flip) and λ=0.5 (medium band, 26.6% flip). All
> three arms trained, passed the validity gate with refresh active, and were evaluated.
> **Neither dose was promoted.** Macro NDCG@10 0.2726 / 0.2698 / 0.2682 and macro
> Recall@1000 0.7089 / 0.7059 / 0.7036 for λ = 0 / 0.3 / 0.5; 0 of 4 domains improved.
> Deltas are inside the noise floor but monotone decreasing in λ on every metric.
>
> **A full-scale confirmation run of these arms is not justified.** The write-up, including
> what is and is not defensible to claim, is `lambda_pilot_experiment_summary.md`.
>
> Everything below remains the authoritative description of the machinery, which is intact
> and reusable for a different σ or a different combination rule.

---

## Recipes

`config.yaml` anchors the base block as `&async_fg_base`; `async_fast_grass_pilot` and
`async_fast_grass_smoke` inherit it with YAML merge keys and override only what differs.
Selected with the existing `--recipe`; no new plumbing.

| recipe | items | spe | max_steps | B_doc | budget/interval | max_age_steps |
|---|---|---|---|---|---|---|
| `async_fast_grass` | 330,000 | 5,157 | 10,314 (2 ep) | 32,000 | 310 | 1,000 |
| `async_fast_grass_pilot` | 33,000 | 516 | 1,032 (2 ep) | 32,000 | 310 | 100 |
| `async_fast_grass_smoke` | 1,024 | 16 | 64 (**4 ep**) | 512 | 16 | 10 |

Pinned by `async_fast_grass_pilot_test.py::recipe blocks match the plan`.

- The pilot's `cache_update_interval: 10` (vs 100) changes only maintenance
  **granularity** — the budget scales with it, so total maintenance work matches the full
  run (`0.5 × 32000 × 10 / 516 ≈ 310 ≡ 0.5 × 32000 × 100 / 5157`).
- `max_age_steps == async_mine_every_steps` in both pilot and smoke, so the **first**
  numeric round is already eligible to refresh over-age slots.
- Smoke uses **4** epochs, not 2. At 32 steps the miner has only ~22 training steps after
  `checkpoint-10` to load it, mine, publish and be consumed, and a longer SLURM
  allocation does not help because the trainer exits at step 32 regardless.

### Two recipe keys change behaviour

**`requires_manifest: true`** — the orchestrator, the miner **and** the main training
launcher refuse to run without `--manifest`.

> ⚠️ **The probe launcher and `quality_probe.run_real` do NOT enforce it.** An empty
> `ASYNC_FG_MANIFEST` there falls back to the first `--max_queries` of the mixture in file
> order, which is `train_hq.jsonl`-first and therefore HQ-biased. Pass the probe manifest
> explicitly, or close that gap first.

**`pilot_gate_min_steps`** — its presence enables the run-validity gate AND a **nonzero
exit** on failure (128 pilot / 1 smoke). Absent from the full run, whose exit behaviour is
unchanged.

---

## Manifest (`scripts/async_fast_grass_pilot.py`)

Deterministic stratified subset keyed on `query_id` alone — globally unique because
`preprocessor.py` emits `msmarco_*`, `reasonir_vl_*`, `reasonir_hq_*`. Uniqueness is
**asserted** at build time rather than worked around with composite keys.

Presets: `pilot10` = 8,303 / 14,997 / 9,700 = **33,000** (10% of each source);
`smoke1k` = **1,024** split by largest remainder so the total is exact.

Per source: sort by `query_id` (independent of file order), seeded draw without
replacement (`default_rng([seed, source_index])`, so a source's draw does not depend on
which others were requested), then **proportional interleave** — item `i` of a source of
size `n` gets key `(i + 0.5) / n` and everything merges on that key.

**Why interleave matters:** the miner walks manifest order in batches of `batch_size` and
does **not** shuffle (unlike `run_fast_grass`, which calls `random.shuffle`). A
concatenated manifest would make whole mining batches single-source, so every
cache-maintenance interval would see one domain at a time.

`apply_manifest` **raises** on any manifest id absent from the mixture — silently dropping
would shrink the pilot by an unrecorded amount and break comparability between arms, the
same reasoning as `canonicalize_positives`.

sha256 is written to a `.meta.json` sidecar and recorded in `async_run_summary.json`. It
is reproducible cross-platform — identical on macOS and DelftBlue
(`pilot10 17dc7446…`, `smoke1k fc64d786…`) — which is what proves all arms consumed the
same data when they run days apart in separate jobs.

Output: `$DATA_BASE_DIR/data/processed/pilot_manifests/<name>.jsonl`. Additive; nothing
under `training_mixture/` or the corpus is touched.

---

## Run-validity gate

`evaluate_pilot_gate` requires **all** of:

1. a **numeric** mined round was consumed — `initial_data` never counts, since it is mined
   from the base model before any checkpoint exists, so a run that only consumed it
   exercised no async loop at all;
2. that round reports `num_refresh_total > 0`;
3. it was active for ≥ `pilot_gate_min_steps` optimizer steps, so the mined negatives
   actually shaped the weights;
4. `miner_failed is None` — note `supervise()` normally *terminates* the miner after the
   trainer finishes, so a zero exit code is not the expected outcome;
5. the final model was saved.

Artifacts:
- `async_trainer_summary.json` — per-round `consume_step` / `steps_active` /
  `async_gap_steps`, from `summarize_round_consumption`.
- `async_run_summary.json` — recipe, λ, manifest sha, refresh info, gate verdict. Written
  **before** any failing exit, so a failed run stays diagnosable.

Re-check a finished run without rerunning it:

```bash
python scripts/async_fast_grass_pilot.py check-gate \
  --async_dir <handoff root> --model_dir <model dir> --min_steps 128
```

---

## Mining diagnostics

Previously computed per batch and thrown away. `mine_batch_cached_mcdp(..., age_step=)`
now returns `sel_s_hat_mean`, `sel_sigma_mean`, `sel_lambda_sigma_mean`,
`sel_age_mean/max` and `flip_rate_vs_lambda0`; `MiningDiagnostics` folds them
**query-weighted** (not per batch, so a short final batch does not count as much as a full
one) into `mining_meta_N.json`.

`age_step` is **model time** (`source_checkpoint_step`), identical for every batch in a
round.

> **`flip_rate_vs_lambda0` is TopK-only.** Under `selection_mode: softmax` it is `null`
> plus a reason, because Gumbel top-k is a *sample* and a "flip" against the λ=0 argmax
> would report sampling noise rather than the uncertainty term. An unmeasured value is
> `None`, never `0.0` — which would read as "no flips".

---

## Lambda dosage probe

`scripts/dev/async_fast_grass_quality_probe.py`. Every λ on the grid is scored from the
**same** `s_hat`/`sigma` draw, so differences are attributable to the uncertainty term
alone rather than to a different dropout sample. One set of MC query encodes per seed
covers the whole grid; `Z_mc` is built once.

> **Regime caveat.** Runs on a base checkpoint with a freshly built `Z_mc`, i.e. **zero**
> cache staleness, so σ here is pure dropout noise. Mid-training the cached states are
> also model-stale and σ's scale shifts. The bands calibrate **dosage** only and imply
> nothing about retrieval quality — that is the pilot arms' question.

`select_lambdas`, over **nonzero** grid values only (λ=0 is the control, never selectable):

- reject any candidate with flip-rate SD > 0.05 across seeds, or any known positive leaked;
- **low** = flip rate in `[0.10, 0.20)`; **medium** = `[0.20, 0.35]` (they partition rather
  than overlap at 0.20);
- within a band, closest to the band centre wins; ties go to the **smaller** λ. Distances
  are quantized to `1e-4` first — otherwise two genuinely equidistant candidates differ by
  ~1e-17 in floating point and the tie-break could never fire;
- if a band is empty, take the nearest surviving candidate and set
  `band_satisfied: false` with a loud warning;
- the two arms are always **distinct**; a fallback that would duplicate the low arm
  advances to the next-closest surviving candidate. With fewer than two survivors,
  `n_arms` reports how many nonzero arms are justified — never emit duplicate arms.

Report: `analysis/async_fast_grass_timing/lambda_probe_<ts>.json`.

---

## Evaluation and decision

`run_all_evals.py` gained `--domains`, `--results_json` and `--require_existing`, and now
**exits nonzero** when a requested domain fails. It used to catch `CalledProcessError` and
continue, so a partial evaluation reported success and a downstream comparison would rest
on fewer domains. `--require_existing` makes missing BRIGHT domain files an error instead
of regenerating them mid-experiment.

`lambda_pilot_decide.py` refuses to decide on partial results and applies:

- **promote** at macro ΔNDCG@10 ≥ 0.005 **and** ≥ 3/4 domain wins;
- **inconclusive** at 0.002–0.005 → needs a second pilot seed, not a full run;
- **stop** below 0.002, negative, or driven by a single domain;
- both promote and differ by < 0.002 → take the **smaller** λ.

> This is a **permissive screening gate, not a statistical test.** Four BRIGHT development
> domains is ~520 queries; macro standard error is ≈ 0.01–0.015, so 0.005 sits inside the
> noise and "3/4 domain wins" has p ≈ 0.31 under the null. It exists to stop clearly
> useless λ from consuming 16-hour jobs. Only the matched full confirmation can show that
> uncertainty helps.

---

## DelftBlue runbook

`$DATA_BASE_DIR` is exported **only inside the sbatch scripts**. In a login shell it is
unset and `$DATA_BASE_DIR/...` collapses to `/data/...`. Set it yourself or use absolute
paths.

```bash
cd /home/$USER/dense-retrieval-SOTA
export DATA_BASE_DIR=/scratch/$USER/dense-retrieval-SOTA
CONTAINER=/scratch/$USER/containers/pytorch_2.1.sif
BIND="--bind /scratch/$USER:/scratch/$USER --bind /home/$USER:/home/$USER"
MANIFEST=$DATA_BASE_DIR/data/processed/pilot_manifests/pilot10_seed42.jsonl
SMOKE_MANIFEST=$DATA_BASE_DIR/data/processed/pilot_manifests/smoke1k_seed42.jsonl

# prerequisites
ls -la $DATA_BASE_DIR/temp_grass_workdir/stale_index/corpus.pkl
ls -d  $DATA_BASE_DIR/models/inbatch_mixed_bge_m3
```

**0. Manifests** (login node, ~1 min)

```bash
singularity exec $BIND $CONTAINER \
  python scripts/async_fast_grass_pilot.py build-manifest --preset pilot10 --seed 42
singularity exec $BIND $CONTAINER \
  python scripts/async_fast_grass_pilot.py build-manifest --preset smoke1k --seed 42
```

**1. Preflight** (optional — it also runs inside every job as step 1b). Loads the full
corpus with text, several GB, which is heavy for a login node.

```bash
singularity exec $BIND $CONTAINER python scripts/train_async_fast_grass.py --preflight \
  --recipe async_fast_grass_pilot --manifest $MANIFEST
```

**2. GPU smoke** (2×A100, ~15 min; request 30 min)

```bash
ASYNC_FG_RECIPE=async_fast_grass_smoke ASYNC_FG_MANIFEST=$SMOKE_MANIFEST \
ASYNC_FG_SUFFIX=smoke ASYNC_FG_FRESH=1 ASYNC_FG_NO_EVAL=1 \
sbatch --time=00:30:00 --job-name=fg_smoke scripts/launchers/run_async_fast_grass_singularity.sh
```

**3. Lambda probe** (1×A100, 15–40 min)

```bash
ASYNC_FG_MANIFEST=$MANIFEST \
sbatch --time=01:00:00 scripts/run_async_fast_grass_probe_singularity.sh
```

**4. λ=0 pilot** (2×A100, 1.5–3 h) — **must exit 0 before step 5**

```bash
ASYNC_FG_RECIPE=async_fast_grass_pilot ASYNC_FG_MANIFEST=$MANIFEST \
ASYNC_FG_LAMBDA=0 ASYNC_FG_SUFFIX=pilot_lam0 ASYNC_FG_FRESH=1 ASYNC_FG_NO_EVAL=1 \
sbatch --time=04:00:00 --job-name=fg_pilot_lam0 scripts/launchers/run_async_fast_grass_singularity.sh
```

**5. Nonzero arms** (concurrent; omit the second if the probe reported `n_arms: 1`)

```bash
LOW=<selected_low>; MED=<selected_medium>
for A in "lamLOW:$LOW" "lamMED:$MED"; do
  S=${A%%:*}; L=${A##*:}
  ASYNC_FG_RECIPE=async_fast_grass_pilot ASYNC_FG_MANIFEST=$MANIFEST \
  ASYNC_FG_LAMBDA=$L ASYNC_FG_SUFFIX=pilot_$S ASYNC_FG_FRESH=1 ASYNC_FG_NO_EVAL=1 \
  sbatch --time=04:00:00 --job-name=fg_pilot_$S scripts/launchers/run_async_fast_grass_singularity.sh
done
```

**6. Four-domain eval + decision** (1×A100, ~1–2 h per model on the FULL `gpu-a100`)

`run_evaluate_singularity.sh` now **defaults `EVAL_DOMAINS` to the four pilot domains**, so
they no longer have to be spelled out and cannot silently disagree with the set the decision
rule reads. `EVAL_DOMAINS=all` is the escape hatch for all twelve.

⚠️ **Use `gpu-a100`, not `gpu-a100-small`.** Measured on the small partition: 2.11 s/it,
~52 min per corpus encode, which overran a 2 h wall clock at one domain and does not
reliably fit four inside the 4 h partition cap. On the full partition all four domains
complete in one job. A timeout is recoverable but wasteful: per-domain
`{domain}_results.json` files are written as each domain finishes, so resubmit with
`EVAL_DOMAINS=<the gaps>`.

⚠️ **`$DATA_BASE_DIR` is unset in a login shell**, so write the model path absolutely in
anything typed interactively. This trap cost three failed eval jobs (`Model path does not
exist: /models/...`).

```bash
for S in pilot_lam0 pilot_lamLOW pilot_lamMED; do
  EVAL_MODEL_PATH=/scratch/$USER/dense-retrieval-SOTA/models/async_fast_grass_pilot_bge_m3_$S \
  EVAL_REQUIRE_EXISTING=1 \
  sbatch --partition=gpu-a100 --time=04:00:00 --job-name=eval_$S \
         scripts/launchers/run_evaluate_singularity.sh
done

singularity exec $BIND $CONTAINER python scripts/dev/lambda_pilot_decide.py \
  --baseline async_fast_grass_pilot_bge_m3_pilot_lam0 \
  --candidates async_fast_grass_pilot_bge_m3_pilot_lamLOW \
               async_fast_grass_pilot_bge_m3_pilot_lamMED \
  --domains biology,economics,stackoverflow,theoremqa_questions
```

### Reading the result

> **Exit codes are necessary but not sufficient.**
> `sacct -j <id> --format=JobID,State,ExitCode,Elapsed` confirms the job executed validly.
> It does **not** tell you the outcome:
> - probe exit 0 ⇒ it ran. Still read `band_satisfied` and `n_arms` — `n_arms == 1` means
>   submit ONE nonzero arm, not two.
> - decision exit 0 ⇒ it compared. Still read the per-candidate verdict; nothing may be
>   promoted.

Per stage: preflight → source counts 8,303/14,997/9,700, `total_steps 1032`, no refresh
error. Smoke/pilot → job exit 0 plus the gate block, and read **both**
`num_refresh_total` and `num_replace_total` (with `max_age_steps == cadence` every slot is
over-age at round start, which shifts the refresh/replace mix; measure it, do not predict
it).

---

## Gotchas

- **`query_batch_size` in the probe is part of the experiment**, not a free memory knob:
  each chunk draws its own dropout masks, so changing it shifts the measured flip rates.
  Hold it fixed across probe runs. `score_chunk_size` (document side) **is** exactly
  invariant — it only splits a matmul.
- **Short runs are dominated by checkpoint I/O.** The smoke writes 7 checkpoints
  (~2–3 GB each) at `async_mine_every_steps: 10` over 64 steps.
- **An empty env var + `${VAR:+--flag $VAR}` expands to nothing.** This silently dropped
  `--manifest` and ran the smoke against the full 330k mixture: `steps_per_epoch` 5,157
  instead of 16 and a budget rounding to **0**. The smoke failed loudly only because
  `B_doc=512` collapsed the budget; the *pilot* would have trained 10,314 steps instead of
  1,032 and looked healthy. Hence `requires_manifest`, enforced in three places plus a
  pre-submit launcher guard.
- **`ASYNC_FG_RUN_TESTS` cannot be turned off.** The launcher does
  `"${ASYNC_FG_RUN_TESTS:-1}"` then `[ -n ... ]`, so empty becomes `1` and even `0` is a
  non-empty string. The CPU gate always runs (~2 min) and the `SKIPPED` branch is dead
  code.

## Follow-up if a λ is promoted

Rerun λ=0 **and** the promoted λ with `--recipe async_fast_grass` (`max_age_steps: 1000`),
full mixture, no manifest, same seed and remaining hyperparameters. The pre-fix λ=0 run is
not a valid control for the corrected experiment.
