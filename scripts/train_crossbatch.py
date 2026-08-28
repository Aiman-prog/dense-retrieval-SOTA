"""
Cross-batch negatives training (RocketQA strategy 1) via GradCache.

This is DISTRIBUTED LARGE-BATCH training. The negative pool is one optimizer
step's passages gathered across ranks -- 512 queries/device x 2 ranks = 1024
queries, x train_group_size 2 = 2048 passages, so 2047 effective negatives per
query. Negatives are NOT carried across optimizer steps, and gradient
accumulation does NOT enlarge the pool: GradCache pools within a single
training_step only. Launched without torchrun, is_ddp is False, the all-gather
disappears and the pool silently halves -- which is what check_batch_invariants
refuses.

Runs start FRESH by default; see train_inbatch.py for why that needs saying.
"""

import sys
import os
import argparse
from pathlib import Path

# Setup pathing
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
from utils.helpers import get_training_context, patch_tevatron_loss, load_config, set_seed, \
                          log_startup_config, build_run_manifest, prepare_output_dir, \
                          assert_training_succeeded, attach_training_diagnostics, \
                          probe_triples_from_mixture, ranking_probe, require_recipe_keys, \
                          count_jsonl_examples
from data.preprocessor import MIXTURE_FILES, require_mixture_files
from transformers import AutoTokenizer
from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.driver.train import main as tevatron_train_main

# 🩹 Same two Tevatron patches in-batch applies. Cross-batch carried neither, and
# without the second one gradient_checkpointing=True crashes on XLM-RoBERTa.
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)

def _tevatron_gc_enable(self, **kwargs):
    # NON-REENTRANT checkpointing, unlike in-batch's no-arg call. GradCache runs
    # several forward/backward passes per optimizer step; with reentrant checkpointing
    # DDP's autograd hooks fire twice for the same parameter and training dies with
    # "Expected to mark a variable ready only once ... marked as ready twice".
    # In-batch is single-process, so it never hits this and keeps the plain call.
    # Trainer passes gradient_checkpointing_kwargs={} -- an EMPTY DICT, not None
    # (transformers/trainer.py:1985-1990), so a setdefault on the outer kwargs is a
    # no-op and HF falls back to use_reentrant=True. Merge INTO the dict instead.
    gc_kwargs = dict(kwargs.pop('gradient_checkpointing_kwargs', None) or {})
    gc_kwargs.setdefault('use_reentrant', False)
    self.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=gc_kwargs, **kwargs)
DenseModel.gradient_checkpointing_enable = _tevatron_gc_enable

CONSUMED_KEYS = (
    'model_name', 'target_batch_size', 'per_device_batch_size',
    'gradient_accumulation_steps', 'gc_q_chunk_size', 'gc_p_chunk_size',
    'learning_rate', 'warmup_ratio', 'weight_decay', 'num_epochs',
    'train_group_size', 'grad_cache', 'bf16', 'max_grad_norm', 'logging_steps',
    'dataloader_num_workers', 'save_steps', 'save_total_limit',
    'gradient_checkpointing',
)

# Opt-in: the recipe declares these only when LoRA is used (see CLAUDE.md).
OPTIONAL_KEYS = ('lora', 'lora_r', 'lora_alpha', 'lora_dropout', 'lora_target_modules')


def check_batch_invariants(recipe, world_size):
    """Refuse to train unless the negative pool is the one the experiment claims.

    Every failure here is silent otherwise: a wrong world size still trains, just
    against half the pool, and the resulting NDCG is attributed to the method.
    """
    per_device = recipe['per_device_batch_size']
    target = recipe['target_batch_size']
    accum = recipe['gradient_accumulation_steps']
    group = recipe['train_group_size']

    expected_world = target // per_device
    if world_size != expected_world:
        raise ValueError(
            f"cross-batch expects world_size {expected_world} "
            f"(target_batch_size {target} / per_device_batch_size {per_device}) but "
            f"WORLD_SIZE={world_size}. Launch with "
            f"`torchrun --nproc_per_node={expected_world}`; a single process silently "
            f"drops the all-gather and halves the negative pool.")
    if per_device * world_size != target:
        raise ValueError(
            f"per_device_batch_size {per_device} x world_size {world_size} = "
            f"{per_device * world_size}, not target_batch_size {target}")
    if accum != 1:
        raise ValueError(
            f"gradient_accumulation_steps is {accum}; accumulation does NOT enlarge a "
            f"GradCache pool (it pools within one training_step), so any value other "
            f"than 1 changes the optimizer-step budget while leaving the pool at "
            f"{target}. Set it to 1.")
    if not recipe['grad_cache']:
        raise ValueError(
            "grad_cache is false; without GradCache Tevatron uses the plain trainer, "
            "patch_tevatron_loss (which rebinds names in gc_trainer only) does not "
            "apply, and the cross-batch pool this recipe exists to create is gone.")

    queries = target
    passages = queries * group
    return {"world_size": world_size, "queries": queries, "passages": passages,
            "negatives_per_query": passages - 1}


def main():
    parser = argparse.ArgumentParser(description="Cross-batch negatives training.")
    parser.add_argument('--resume', action='store_true',
                        help="Continue a run in the output dir whose manifest matches. "
                             "Without it the run starts fresh and stale checkpoints are removed.")
    parser.add_argument('--overwrite', action='store_true',
                        help="Discard an output dir produced by a DIFFERENT configuration.")
    args = parser.parse_args()

    # 1. Configuration & Paths via Centralized Context
    config = load_config()
    set_seed(config.get('seed', 42))
    ctx = get_training_context("crossbatch")
    recipe = ctx['args']
    require_recipe_keys("crossbatch", recipe, CONSUMED_KEYS, OPTIONAL_KEYS)
    log_startup_config("crossbatch", ctx, recipe)

    # Resolve the mixture directory path
    processed_dir = Path(ctx['processed_dir'])
    mixture_dir = processed_dir / "training_mixture"
    training_data_path = str(mixture_dir / "*.jsonl")

    mixture_files = require_mixture_files(mixture_dir, MIXTURE_FILES)

    # 2. Enforce the batch-size contract before a single GPU second is spent.
    # torchrun sets WORLD_SIZE; torch.distributed is not initialized yet here.
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    pool = check_batch_invariants(recipe, world_size)
    print(f"[crossbatch] world_size={pool['world_size']}  queries={pool['queries']}  "
          f"passages={pool['passages']}  negatives/query={pool['negatives_per_query']}  "
          f"grad_cache={recipe['grad_cache']}", flush=True)

    per_device_batch = recipe['per_device_batch_size']
    acc_steps = recipe['gradient_accumulation_steps']

    n_examples = count_jsonl_examples(training_data_path)
    total_steps = -(-n_examples // pool['queries']) * recipe['num_epochs']

    manifest = build_run_manifest(
        "crossbatch", ctx, recipe,
        data_files=mixture_files,
        world_size=world_size,
        negative_pool_size=pool['negatives_per_query'],
        optimizer_steps=total_steps,
        extra={"batch_invariants": pool},
    )
    # Deliberately NOT rank-gated. tevatron/retriever/driver/train.py:99-103 calls
    # get_last_checkpoint() independently on every rank, immediately before
    # trainer.train(), so each rank decides resume_from_checkpoint for itself. Gating
    # this to rank 0 would let rank 1 read the directory before rank 0's rmtree
    # finished and the two would disagree -- and torch.distributed is not initialized
    # here yet, so there is no barrier to close that window with. Both ranks running
    # it converges instead: rmtree is idempotent (ignore_errors), atomic_write uses a
    # unique temp file per call, and both ranks reach the same branch either way.
    output_dir = prepare_output_dir(ctx['output_dir'], manifest,
                                    resume=args.resume, overwrite=args.overwrite)

    # 3. Argument Construction
    args_list = [
        '--output_dir', str(output_dir),
        '--model_name_or_path', ctx['base_model'],
        '--dataset_name', 'json',
        '--dataset_path', training_data_path,
        '--dataset_split', 'train',
        '--do_train',

        # RocketQA Strategy 1: Cross-batch negatives via GradCache [cite: 10, 59]
        '--grad_cache', str(recipe['grad_cache']),
        '--gc_q_chunk_size', str(recipe['gc_q_chunk_size']),
        '--gc_p_chunk_size', str(recipe['gc_p_chunk_size']),
        '--per_device_train_batch_size', str(per_device_batch),
        '--gradient_accumulation_steps', str(acc_steps),

        # Precision: bf16 for A100 (better than fp16)
        '--fp16', 'False',
        '--bf16', str(recipe['bf16']),
        '--dtype', 'bfloat16' if recipe['bf16'] else 'float32',
        '--attn_implementation', 'eager',    # XLM-RoBERTa has no sdpa in this transformers ver
        '--optim', 'adamw_torch_fused',      # bitsandbytes not available; GradCache chunking handles memory
        '--gradient_checkpointing', str(recipe['gradient_checkpointing']),

        # Core Hyperparameters [cite: 244, 245]
        '--learning_rate', str(recipe['learning_rate']),
        '--num_train_epochs', str(recipe['num_epochs']),
        '--train_group_size', str(recipe['train_group_size']),       # 1 positive + 1 hard negative
        '--query_max_len', str(ctx['max_q']),
        '--passage_max_len', str(ctx['max_p']),

        # Stability & Debugging
        '--max_grad_norm', str(recipe['max_grad_norm']),        # Prevent gradient explosion → NaN/Inf → SIGFPE
        '--logging_steps', str(recipe['logging_steps']),
        '--overwrite_output_dir', 'True',
        '--save_strategy', 'steps',
        '--save_steps', str(recipe['save_steps']),
        '--save_total_limit', str(recipe['save_total_limit']),
        '--dataloader_num_workers', str(recipe['dataloader_num_workers']),
        '--pooling', ctx['pooling'],
        '--normalize', str(ctx['normalize']),
        '--temperature', str(ctx['temperature']),
        '--warmup_ratio', str(recipe['warmup_ratio']),
        '--weight_decay', str(recipe['weight_decay']),
        '--seed', str(config.get('seed', 42)),
    ]

    # LoRA: freeze base model, only train adapter matrices
    if recipe.get('lora', False):
        args_list += [
            '--lora', 'True',
            '--lora_r', str(recipe.get('lora_r', 16)),
            '--lora_alpha', str(recipe.get('lora_alpha', 32)),
            '--lora_dropout', str(recipe.get('lora_dropout', 0.1)),
            '--lora_target_modules', recipe.get('lora_target_modules', 'query,key,value'),
        ]

    sys.argv = ['train.py'] + args_list

    triples = probe_triples_from_mixture(mixture_files)
    # Tevatron constructs its Trainer without tokenizer=, so the callback receives
    # tokenizer=None. Load the one the run actually trains with and close over it.
    probe_tokenizer = AutoTokenizer.from_pretrained(ctx['base_model'])
    attach_training_diagnostics(
        output_dir,
        lambda model, tokenizer: ranking_probe(
            model, tokenizer or probe_tokenizer, triples,
            next(model.parameters()).device, ctx['max_q'], ctx['max_p']))
    # Must precede tevatron_train_main(): GradCacheTrainer.__init__ reads these
    # module-level names when the trainer is constructed.
    patch_tevatron_loss(ctx['temperature'])
    tevatron_train_main()

    # Rank 0 alone writes the checkpoint (Trainer.save_model) and training_log.jsonl
    # (the callback guards on is_world_process_zero), so rank 0 alone can validate them.
    # Every rank validating would race: rank 1 looks for artifacts it never wrote and
    # fails a run that actually succeeded.
    if int(os.environ.get('RANK', '0')) == 0:
        assert_training_succeeded(output_dir, manifest)
        print(f"✅ Cross-batch training completed. Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
