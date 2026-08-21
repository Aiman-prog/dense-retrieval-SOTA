# DelftBlue runtime environment (measured)

Recorded 2026-08-20 by inspecting the live cluster. This file replaces the
`env_delftblue_actual.txt` that earlier docs referenced but that was never created.

## The one thing to know

**`pytorch_2.1.sif` is a historical name. Nothing in this repo runs the container's torch.**

The container supplies CUDA and a torch 2.1.0 that nothing imports, and it has **no
`transformers` at all**. The entire ML stack resolves from
`~/.local/lib/python3.10/site-packages`, which shadows the container for every job.

| package | version | source |
|---|---|---|
| torch | **2.10.0+cu128** | `~/.local` |
| transformers | 4.40.2 | `~/.local` |
| accelerate | 0.30.1 | `~/.local` |
| datasets | 2.19.2 | `~/.local` |
| tevatron | patched `8f31cd8` + 3 hand-applied patches, unversioned | `~/.local` |
| grad_cache | unversioned | `~/.local` |

⚠️ **Never set `PYTHONNOUSERSITE`.** It hides `~/.local`, and because the container has no
`transformers`, it breaks every pipeline instantly.

## Why torch is 2.10 when everything is pinned for 2.1

`torch-2.10.0.dist-info` carries **no `REQUESTED` marker**, so it was pulled in as a
transitive dependency rather than asked for. Install timestamp: **2026-02-22 23:28**.

`requirements-hpc.txt` still pins the 2.1-era versions of everything else. **That is
correct and must not be "fixed" upward** — those pins are what actually works, and every
result in the repo was produced with exactly this combination.

## Are results comparable across the upgrade?

**Yes.** Every checkpoint under `models/` is dated 2026-04-05 or later, i.e. after the torch
install. The only earlier artifact is `inbatch_mixed_bge_m3.OLD_baseline` (2026-02-22 17:21),
six hours *before* the install, and it is not used by any reported result. There is no
mixed-stack comparability caveat.

## Stale-but-harmless packages

`torchvision 0.16.0` and `torchaudio 2.1.0` remain at the 2.1-era ABI and would fail if
imported under torch 2.10. **Nothing in this repo imports them** (verified by a repo-wide
grep), so they are inert.

## Confirming the stack still works

Six entry points were import-checked on the cluster, including `train_crossbatch`, which
pulls the deepest chain (`transformers` Trainer → `accelerate` 0.30.1 → `GradCacheTrainer`
→ `grad_cache`). To re-verify in ~1 min:

```bash
singularity exec /scratch/$USER/containers/pytorch_2.1.sif python -c "
import torch, transformers
from tevatron.retriever.modeling import DenseModel
print(torch.__version__, transformers.__version__, 'tevatron OK')"
```

Backup of the patched Tevatron: `/scratch/$USER/tevatron_patched_20260820.tgz` (93 K).

To regenerate the full resolved list on the cluster:

```bash
singularity exec /scratch/$USER/containers/pytorch_2.1.sif python -m pip list --format=freeze
```

Tracked as defect **P7** in `CONSOLIDATION_STATUS.md` — recorded deliberately, with no code
fix intended.
