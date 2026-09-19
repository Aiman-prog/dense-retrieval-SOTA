"""The BM25 warm-up `ance_paper` initializes from.

Microsoft's released 60K warm-up is gone: both blob URLs return HTTP 409, issues
#23/#24/#26 on microsoft/ANCE have been open since 2022, and the only surviving mirror
is bit-identical to the 600K FINAL checkpoint -- which `assert_permitted_init` exists
precisely to refuse as an initialization. So the warm-up is trained here instead.

Deliberately NOT a flag on `run_ance_train.py`. That file is the ANCE worker, and its
round-provenance machinery (work roots, run ids, refresh accounting) is what stops a run
finishing on another experiment's negatives. A static-negative branch through it would
add risk to the most safety-critical file in the pipeline to save a hundred lines here.

Same model, same loss and same optimizer as `ance_paper` -- upstream's warm-up trains
`RobertaDot_NLL_LN`, whose `NLL` (model/models.py:58-81) is `ance_paper.pairwise_nll`.
Only three things differ, and all three come from `commands/run_train_warmup.sh`:

    negatives   the BM25 negatives already in the mixture, static. No ANN, no rounds.
    lengths     128 for query AND passage (--max_seq_length 128 feeds one processor),
                not ance_paper's q64/p512
    schedule    lr 2e-4 over a ~2.19M-step horizon, stopping at 60K

RAGGED, not fixed-width. Upstream reads triples.train.small.tsv one triplet per LINE
(drivers/run_warmup.py:743, data/process_fn.py:48-70), so "negatives per query" is not a
constraint anywhere in ANCE. ~1.8% of Tevatron/msmarco-passage records carry fewer than
30 BM25 negatives, and job 70494 died on exactly that.

Run: python scripts/train_ance_warmup.py [--preflight] [--overwrite]
"""
import sys
import json
import hashlib
import argparse
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'scripts'))

from utils.helpers import (get_training_context, load_config, get_path, set_seed,
                           append_jsonl, ranking_probe,
                           build_run_manifest, prepare_output_dir, require_recipe_keys,
                           assert_training_succeeded, TRAINING_LOG_NAME)
from data.preprocessor import MSMARCO_ONLY_FILES, require_mixture_files
from ance_paper import build_ance_encoder_from_base, load_ance_encoder, pairwise_nll, Lamb
from run_ance_train import ANCEDataset, make_paper_collate, build_schedule, \
                           optimization_step

RECIPE = 'ance_paper_warmup'

# Matches probe_triples_from_mixture's n: the last 64 usable records. Taken from the
# already-loaded dataset instead of re-reading the 5.2 GB mixture a fourth time.
PROBE_N = 64
PREFLIGHT_STEPS = 3

# Every training.ance_paper_warmup key this script reads. require_recipe_keys is
# bidirectionally strict, so a key added to the recipe and not read here fails startup.
CONSUMED_KEYS = (
    'paper_fidelity', 'model_name', 'base_model', 'mixture_dir', 'train_group_size',
    'batch_size', 'learning_rate', 'lamb_eps', 'weight_decay', 'max_grad_norm',
    'warmup_steps', 'train_stop_steps', 'scheduler_max_steps', 'query_max_len',
    'passage_max_len', 'normalize', 'temperature', 'pooling', 'bf16',
    'gradient_checkpointing', 'dataloader_num_workers', 'logging_steps', 'save_steps',
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _refuse_to_clobber(output_dir, overwrite):
    """prepare_output_dir(overwrite=True) is shutil.rmtree. Do not silently delete a
    finished 60K warm-up because someone re-ran the script."""
    if overwrite:
        return
    prior = Path(output_dir) / "run_manifest.json"
    if not prior.is_file():
        return
    try:
        done = json.loads(prior.read_text()).get('finished_at')
    except (ValueError, OSError):
        return
    if done:
        raise SystemExit(
            f"❌ {output_dir} already holds a COMPLETED warm-up (finished_at={done}).\n"
            f"   Starting fresh would DELETE it and there is no resume. Re-run with "
            f"--overwrite if that is genuinely what you want.")


def parse_args():
    p = argparse.ArgumentParser(description="Train the BM25 warm-up.")
    p.add_argument('--preflight', action='store_true',
                   help='validate the REAL mixture, model, collate, optimizer, save and '
                        'reload path on CPU, then exit. Writes nothing to the model '
                        'directory. It loads the whole 5.2 GB mixture, so give it a '
                        'compute node, not a login node.')
    p.add_argument('--overwrite', action='store_true',
                   help='replace an existing COMPLETED warm-up (deletes it).')
    return p.parse_args()


def main():
    cli = parse_args()
    config = load_config()
    set_seed(config.get('seed', 42))
    ctx = get_training_context(RECIPE)
    args = ctx['args']
    require_recipe_keys(RECIPE, args, CONSUMED_KEYS)

    device = torch.device('cpu' if cli.preflight else 'cuda')
    max_steps = PREFLIGHT_STEPS if cli.preflight else int(args['train_stop_steps'])
    # 2 under --preflight so the rescue-save branch is actually exercised; in a
    # real run it first fires at step 30000, hours in.
    save_steps = 2 if cli.preflight else int(args['save_steps'])
    log_every = 1 if cli.preflight else int(args['logging_steps'])
    batch_size = int(args['batch_size'])
    mixture_dir = get_path("processed") / args['mixture_dir']
    mixture_files = list(require_mixture_files(mixture_dir, MSMARCO_ONLY_FILES))

    tag = "PREFLIGHT " if cli.preflight else ""
    print(f"[warmup] {tag}{ctx['base_model']} -> {args['model_name']} | {max_steps} "
          f"steps @ batch {batch_size} | q{ctx['max_q']}/p{ctx['max_p']} | lr "
          f"{args['learning_rate']} | decay horizon {args['scheduler_max_steps']}",
          flush=True)

    real_output_dir = get_path("models") / args['model_name']
    if not cli.preflight:
        _refuse_to_clobber(real_output_dir, cli.overwrite)
    manifest = build_run_manifest(RECIPE, ctx, args, data_files=mixture_files,
                                  world_size=1, negative_pool_size=1,
                                  optimizer_steps=max_steps)

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / args['model_name'] if cli.preflight else real_output_dir
        prepare_output_dir(output_dir, manifest, overwrite=True)
        log_path = output_dir / TRAINING_LOG_NAME

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ctx['base_model'])
        # Refuses anything but the projection head being freshly initialized: a RoBERTa
        # body that failed to load would train, converge, and be a warm-up of nothing.
        model = build_ance_encoder_from_base(ctx['base_model']).to(device)
        if args['gradient_checkpointing']:
            model.roberta.gradient_checkpointing_enable()
        model.train()

        # ragged=True: take the negatives each record actually has, capped at n_negs.
        # See ANCEDataset._build_ragged_index -- upstream has no per-query count.
        ds = ANCEDataset(mixture_dir, int(args['train_group_size']), paper_mode=True,
                         ragged=True)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=int(args['dataloader_num_workers']),
                            collate_fn=make_paper_collate(tokenizer, ctx['max_q'],
                                                          ctx['max_p']))
        full = sum(1 for ex in ds.examples
                   if len(ex['negative_passages']) >= ds.n_negs)
        short = len(ds.examples) - full
        print(f"[warmup] {len(ds.examples):,} queries -> {len(ds):,} triplets "
              f"({short:,} queries carry fewer than {ds.n_negs} negatives, "
              f"{100.0 * short / max(len(ds.examples), 1):.2f}%)", flush=True)

        optimizer = Lamb(model.parameters(), lr=float(args['learning_rate']),
                         eps=float(args['lamb_eps']),
                         weight_decay=float(args['weight_decay']))
        scheduler, _ = build_schedule(optimizer, int(args['warmup_steps']), max_steps,
                                      int(args['scheduler_max_steps']))
        # Same selection as probe_triples_from_mixture, without the fourth full read.
        probe_triples = [(ex['query'], ex['positive_passages'][0]['text'],
                          ex['negative_passages'][0]['text'])
                         for ex in ds.examples[-PROBE_N:]]

        def probe(step, phase, *, required=False):
            """Recorded even when it raises, so a failure is never read as a pass."""
            try:
                result = ranking_probe(model, tokenizer, probe_triples, device,
                                       ctx['max_q'], ctx['max_p'],
                                       normalize=ctx['normalize'])
            except Exception as e:                                 # noqa: BLE001
                result = {"error": f"{type(e).__name__}: {e}"}
            append_jsonl(log_path, {"global_step": step, "phase": phase, **result})
            if required and 'error' in result:
                raise RuntimeError(
                    f"the {phase} ranking probe failed: {result['error']}. "
                    f"assert_training_succeeded needs two finite probe points, so this "
                    f"would fail the run AFTER all {max_steps} steps. Failing now.")
            return result

        # Fail fast: the begin probe costs seconds, and a broken probe is otherwise
        # only discovered once the training is already spent.
        probe(0, "begin", required=True)
        step, it = 0, iter(loader)
        while step < max_steps:
            try:
                q, pos, neg = next(it)
            except StopIteration:
                it = iter(loader)
                q, pos, neg = next(it)
            to = lambda enc: {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=bool(args['bf16']) and device.type == 'cuda'):
                loss = pairwise_nll(model(**to(q)), model(**to(pos)), model(**to(neg)))
            value, grad_norm = optimization_step(
                model, optimizer, scheduler, loss,
                max_grad_norm=float(args['max_grad_norm']), step=step)
            step += 1
            if step % log_every == 0:
                append_jsonl(log_path, {"global_step": step, "loss": value,
                                        "grad_norm": grad_norm,
                                        "lr": scheduler.get_last_lr()[0]})
                print(f"[warmup] step {step}/{max_steps} loss {value:.4f}", flush=True)
            # Rescue artifact, NOT a resumable checkpoint: there is no optimizer state
            # and no resume. Named interim-* rather than checkpoint-* so it stays out
            # of _newest_model_artifact and cannot trip the optimizer.pt check. Without
            # it a wall-clock kill at 4h59m leaves nothing at all.
            if save_steps and step % save_steps == 0 and step < max_steps:
                interim = output_dir / f"interim-{step}"
                model.save_pretrained(interim)
                tokenizer.save_pretrained(interim)
                probe(step, "interim")
                print(f"[warmup] wrote rescue checkpoint {interim}", flush=True)

        probe(step, "end")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        assert_training_succeeded(output_dir, manifest)
        # Exactly the call assert_permitted_init will make on this artifact.
        load_ance_encoder(str(output_dir))

        weights = next(output_dir / n for n in ('model.safetensors', 'pytorch_model.bin')
                       if (output_dir / n).is_file())
        if cli.preflight:
            print(f"[warmup] ✅ preflight passed — {len(ds):,} triplets resolve, the "
                  f"model builds, {PREFLIGHT_STEPS} steps run, and the saved checkpoint "
                  f"reloads through load_ance_encoder. Nothing was written to "
                  f"{real_output_dir}.", flush=True)
            return 0
        print(f"[warmup] ✅ {step} steps -> {output_dir}\n"
              f"[warmup] {weights.name} sha256 = {_sha256(weights)}\n"
              f"[warmup] Next: evaluate it (expect MRR@10 ~0.311), then record that "
              f"hash as training.ance_paper.expected_init_sha256.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
