# Rerun Baselines, Rebuild Fast-GRASS Index, Then Test Sigma

## Summary

Use two comparison tracks. Track 1 tests warmup/batch-size behavior from raw `BAAI/bge-m3`: rerun InBatch and CrossBatch separately. Track 2 tests Fast-GRASS as a hard-negative mining continuation from the rerun InBatch checkpoint. CrossBatch does not need batch size 64 because its purpose is to test whether a much larger effective batch helps.

## Key Changes

- Rerun InBatch from raw `BAAI/bge-m3`:
  - Keep `batch_size=64`, `num_epochs=2`, `train_group_size=2`, `learning_rate=1e-5`.
  - This becomes the warmup checkpoint for Fast-GRASS.

- Rerun CrossBatch from raw `BAAI/bge-m3`:
  - Keep its method-specific large effective batch setup.
  - `per_device_batch_size=64`, `gradient_accumulation_steps=16`, 2 GPUs.
  - Do not force CrossBatch to match Fast-GRASS batch size; report it as a large-batch baseline.

- Run Fast-GRASS from the rerun InBatch checkpoint:
  - Set `training.fast_grass.base_model` to `models/inbatch_mixed_bge_m3`.
  - Keep shared training knobs aligned with InBatch where relevant: `batch_size=64`, `num_epochs=2`, `train_group_size=2`, `learning_rate=1e-5`, `query_max_len=1024`, `passage_max_len=512`, `temperature=0.02`.

- Rebuild the Fast-GRASS stale index:
  - After rerunning InBatch and pointing Fast-GRASS to the new checkpoint, remove/archive `temp_grass_workdir/stale_index/corpus.pkl`.
  - Let the first Fast-GRASS run rebuild the stale index from the new InBatch model.
  - Reuse that rebuilt index for all Fast-GRASS sigma tests from the same base.

- Add sigma-testability support:
  - Add `--ema_alpha` and launcher env `FAST_GRASS_EMA_ALPHA`.
  - Add mining diagnostics: selected `s_hat`, selected `sigma`, `lambda*sigma`, `sigma/s_hat`, and selection flip rate versus `lambda=0`.

## Run Order

1. **InBatch warmup**
   - Run `scripts/run_inbatch_singularity.sh`.
   - Evaluate/save metrics for the warmup baseline.

2. **CrossBatch large-batch baseline**
   - Run `scripts/run_crossbatch_singularity.sh`.
   - Treat this as "does much larger effective batch help from raw BGE-M3?", not as the Fast-GRASS base.

3. **Prepare Fast-GRASS**
   - Ensure `training.fast_grass.base_model = /scratch/.../models/inbatch_mixed_bge_m3`.
   - Delete/archive old `temp_grass_workdir/stale_index/corpus.pkl`.

4. **Cheap EMA sigma diagnostics**
   - Run Fast-GRASS with `B_doc=32k`, `m=1`, `lambda=1`, `num_epochs=1`, `--no_eval`.
   - First diagnostic: `ema_alpha=1.0` frozen teacher.
   - If frozen teacher produces meaningful sigma/selection flips, run `ema_alpha=0.9999`.
   - If frozen teacher still has tiny sigma and near-zero flips, treat EMA uncertainty as not useful here.

5. **Full Fast-GRASS run**
   - Only run the promising sigma setting for full `2 epochs`.
   - Run BRIGHT eval sequentially or with isolated eval scratch.

## Test Plan

- Before cluster runs:
  - Verify config base models:
    - `inbatch` falls back to `BAAI/bge-m3`.
    - `crossbatch` starts from `BAAI/bge-m3`.
    - `fast_grass` starts from `models/inbatch_mixed_bge_m3`.
  - Run Fast-GRASS smoke/tests after sigma-diagnostic code changes.

- After runs:
  - Confirm stdout logs show the intended base model.
  - Confirm Fast-GRASS stale index was rebuilt after the InBatch checkpoint changed.
  - Compare:
    - InBatch vs CrossBatch: effect of large batch.
    - InBatch vs Fast-GRASS: benefit of hard-negative continuation.
    - Fast-GRASS EMA variants: whether sigma actually changes selection.

## Assumptions

- "Naive" means raw `BAAI/bge-m3`.
- CrossBatch is a separate large-batch baseline, not the base checkpoint for Fast-GRASS.
- Fast-GRASS remains a hard-negative mining stage after InBatch warmup.
