# Cross-Batch Negative Training Stage

## Purpose

The cross-batch stage fine-tunes the same base encoder on the same training
mixture as the in-batch arm, but enlarges the contrastive denominator by pooling
the queries and passages of every GPU into a single loss computation. It owns
optimization only. The mixture is built by the preprocessor stage and retrieval
quality is measured by the evaluation stage. It is the higher-negative arm of
the baseline comparison; the lower-negative arm is described in
[`inbatch.md`](inbatch.md).

## Training Flow

```text
training mixture ──> 512 queries per device, on 2 devices
                 ──> 1 positive + 1 mined negative per query
                 ──> encode without gradients, cache representations
                 ──> all-gather across devices: 1024 queries, 2048 passages
                 ──> temperature-scaled cross-entropy over the pooled matrix
                 ──> re-encode in chunks, backpropagate through the cache
                 ──> checkpoint
```

The two passes exist because the pooled batch is far too large to hold
activations for. The first pass computes representations under no-grad, the loss
is taken on those, and the second pass recomputes each chunk to apply the cached
gradients.

## Data and Batch Construction

Training reads the same three mixture files as the in-batch arm, described in
[`preprocessor.md`](preprocessor.md), totalling **329,993 examples** with
exactly one positive and one mined hard negative per record.

`train_group_size: 2` again gives each query a group of two passages, the
positive and one negative selected from that record's candidate list. With
`per_device_batch_size: 512` on two devices, one optimization step holds **1,024
queries and 2,048 passages**. Gradient accumulation is fixed at 1, and this is a
substantive choice rather than a default: accumulation does not enlarge a
gradient-cached pool, because each accumulated micro-step computes its own
softmax over its own queries before the gradients are summed. The pool is set by
devices times per-device batch size, and by nothing else.

## Negative Pool

Representations are gathered across both devices before the loss, so the
similarity matrix is 1,024 by 2,048 and the target for query *i* is passage
*2i*. Each query is contrasted against **2,047 negatives**.

| source | count per query |
|---|---|
| its own mined hard negative | 1 |
| other queries' mined negatives | 1,023 |
| other queries' positives | 1,023 |
| **total negatives** | **2,047** |

Unlike the in-batch arm, the final step of an epoch is **not** thinner.
Accelerate pads the distributed sampler so every rank receives an equally sized
batch, so the last step still presents a full 2,048-column denominator; it
simply carries 265 new records plus replayed padding rather than 1,024 new ones.
The pool is constant at 2,047 for every step.

As in the in-batch arm, exactly one negative per query was explicitly mined. The
other 2,046 are incidental. The false-negative risk noted for in-batch training
is larger here in absolute terms, because the pool contains sixteen times as
many other queries' positives.

## How the Pool Fits in Memory

Gradient caching is what makes a 2,048-passage denominator affordable; it does
not create the negatives, which come from the cross-device gather. Peak
activation memory is governed by the chunk size, set to 32 for both queries and
passages, rather than by the 512 per-device batch. The encoder therefore never
holds activations for more than 32 sequences at once, while the loss still sees
all 2,048. The resulting gradients are mathematically identical to those of a
single undivided backward pass, so the chunk size is a memory and speed knob
with no effect on the optimization.

## Objective and Hyperparameters

The objective matches the in-batch arm. Settings come from
[`config/config.yaml`](../config/config.yaml) under `model` and
`training.crossbatch`.

| setting | value |
|---|---|
| base model | BGE-M3 |
| pooling / normalization | CLS, L2-normalized |
| temperature | 0.02 |
| query / passage max length | 1024 / 512 tokens |
| per-device batch size | 512 |
| effective query pool | 1,024 |
| gradient accumulation | 1 |
| gradient-cache chunk size | 32 queries, 32 passages |
| train group size | 2 |
| learning rate | 1e-5, warmup ratio 0.1 |
| weight decay | 0.01 |
| gradient clipping | 1.0 |
| epochs | 2 |
| precision | bf16 |
| seed | 42 |

Because gradient caching replaces the encoder's own loss path, the temperature
is applied to the cached-gradient loss rather than by the trainer argument that
the in-batch arm relies on. The scaling is the same 0.02 in both arms, but it
reaches the objective by a different route, which is worth knowing when reading
either implementation. The run refuses to start if gradient caching is disabled,
because the cross-device pooling lives inside that loss: without it the arm
would train on a single device's batch and the enlarged negative pool this
recipe exists to create would be gone.

## Infrastructure

Two A100 GPUs on one node, launched as two processes, with 16 CPU cores and
roughly 125 GB of host memory, inside the project Singularity container. The run
refuses to start unless exactly two processes are present, because a single
process would drop the cross-device gather and halve the negative pool without
any visible error.

Two epochs over 329,993 examples at a 1,024-query pool give **646 optimizer
steps**. Expected wall clock is **8.5 to 11 hours**. This is an estimate rather
than a measurement: it is extrapolated from the first 30 steps of an earlier
run, which include warm-up, and no cross-batch run has yet completed.

## Comparability

The two arms are **complete recipes**, not two settings of one variable. They
share a base model, a training mixture, an epoch count, a learning rate and a
mined-negative count of one per query. Two variables move together between them:
the negative pool grows from 127 to 2,047, and the optimizer-step count falls
from 10,314 to 646, a ratio of roughly 16 to 1.

Because those two move together, neither arm isolates the other and no causal
claim about negative-pool size can be drawn from the pair. The honest reading is
a comparison of two complete training recipes. A thesis reporting these numbers
should state both differences together.

## Running the Stage

Submitted through the cluster launcher, which sets the data root, forces offline
Hugging Face mode and starts two training processes. A run starts fresh by
default, clearing prior checkpoints so training cannot silently resume and take
zero steps. Setting `CROSSBATCH_RESUME=1` continues a run whose recorded
configuration matches, which also bounds the cost of an interruption to the
steps taken since the last checkpoint. `CROSSBATCH_OVERWRITE=1` discards an
output directory built from a different configuration.

Before training begins the run prints and enforces its batch contract, reporting
the process count, query pool, passage pool and negatives per query, so a
silently halved pool fails immediately instead of producing a misattributed
result. Each run writes the trained weights, checkpoints every 100 steps, a
manifest recording the configuration and input hashes, and a diagnostics log
holding loss, learning rate and pre-clipping gradient norm. The resulting
checkpoint is evaluated by the procedure in [`evaluation.md`](evaluation.md).
