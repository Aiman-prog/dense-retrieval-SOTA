# In-Batch Negative Training Stage

## Purpose

The in-batch stage fine-tunes the base encoder on the shared training mixture
using a contrastive objective whose negatives come almost entirely from the
other queries in the same optimization step. It owns optimization only. The
mixture is built by the preprocessor stage, and retrieval quality is measured
separately by the evaluation stage. It is the lower-negative arm of the baseline
comparison; the higher-negative arm is described in
[`crossbatch.md`](crossbatch.md).

## Training Flow

```text
training mixture ──> sample 64 queries
                 ──> 1 positive + 1 mined negative per query
                 ──> tokenize and encode all queries and passages
                 ──> similarity matrix over every passage in the step
                 ──> temperature-scaled cross-entropy
                 ──> checkpoint
```

Everything runs in a single process on one GPU. No gradient synchronization or
cross-device gathering takes place, so the batch the loss sees is exactly the
batch one device produced.

## Data and Batch Construction

Training reads the three mixture files described in
[`preprocessor.md`](preprocessor.md), totalling **329,993 examples**: 97,000
from ReasonIR HQ, 149,963 from ReasonIR VL, and 83,030 sampled from MS MARCO.
Every record carries exactly one positive passage and exactly one mined hard
negative.

`train_group_size: 2` means each query contributes a group of two passages, its
positive followed by one negative selected from that record's candidate list.
With `batch_size: 64` a step therefore holds 64 queries and **128 passages**.
Selection is seeded by record index and epoch rather than random, so it is
reproducible. In the mixture as currently built every record of all three
components carries exactly one positive and one negative candidate, so the
selection is degenerate and the same negative is used in both epochs.

## Negative Pool

The similarity matrix spans all 64 queries against all 128 passages, and the
cross-entropy target for query *i* is passage *2i*. Every other column is
treated as a negative, so each query is contrasted against **127 negatives**.

| source | count per query |
|---|---|
| its own mined hard negative | 1 |
| other queries' mined negatives | 63 |
| other queries' positives | 63 |
| **total negatives** | **127** |

Those figures describe a full batch. Batches are not dropped when the epoch does
not divide evenly, and 329,993 examples at batch size 64 leave a final batch of
**9 queries**, which sees 18 passages and therefore **17 negatives**. One batch
in 5,157 per epoch is contrasted that thinly.

Only one of those 127 was explicitly mined. The remaining 126 are incidental,
drawn from whichever other queries happened to share the step. Because other
queries' positives sit in the denominator, a passage relevant to more than one
query can appear as a negative for a query it actually answers. That
false-negative risk is inherent to in-batch training and grows with pool size.

## Objective and Hyperparameters

The loss is softmax cross-entropy over cosine similarities, divided by a fixed
temperature before the softmax. Settings come from
[`config/config.yaml`](../config/config.yaml) under `model` and
`training.inbatch`.

| setting | value |
|---|---|
| base model | BGE-M3 |
| pooling / normalization | CLS, L2-normalized |
| temperature | 0.02 |
| query / passage max length | 1024 / 512 tokens |
| batch size | 64 |
| train group size | 2 |
| learning rate | 1e-5, warmup ratio 0.1 |
| weight decay | 0.01 |
| gradient clipping | 1.0 |
| epochs | 2 |
| precision | bf16 |
| seed | 42 |

Because this path does not use gradient caching, the temperature reaches the
loss through the trainer's own `--temperature` argument rather than through a
patch.

## Infrastructure

One A100, 16 CPU cores and roughly 125 GB of host memory, inside the project
Singularity container on the DelftBlue cluster. Two epochs over 329,993 examples
at batch size 64 give **10,314 optimizer steps**. A completed run takes
approximately **12.2 hours**, measured from the progress trace of an earlier run
at a steady 4.0 to 4.1 seconds per step. Checkpoints are written every 20
percent of total steps.

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
Hugging Face mode and selects the container. A run starts fresh by default:
prior checkpoints in the output directory are cleared so training cannot
silently resume and take zero steps. Setting `INBATCH_RESUME=1` continues a run
whose recorded configuration matches, and `INBATCH_OVERWRITE=1` discards an
output directory built from a different configuration.

Each run writes the trained weights, periodic checkpoints, a manifest recording
the configuration and input hashes that produced them, and a diagnostics log
holding loss, learning rate and pre-clipping gradient norm. The resulting
checkpoint is evaluated by the procedure in [`evaluation.md`](evaluation.md).
