"""
Train In-Batch Negatives model on ReasonIR-HQ.
Refactored to use centralized context management from config.yaml.
MODIFIED: Now uses the training_mixture directory for data.

Runs start FRESH by default. Tevatron's driver resumes from whatever
``get_last_checkpoint(output_dir)`` finds, and ``--overwrite_output_dir`` does not
suppress that -- so a re-run into a finished directory used to resume, take zero
optimizer steps, re-save the old weights and print success. ``prepare_output_dir``
clears ``checkpoint-*`` unless ``--resume`` is given, which is what makes
``get_last_checkpoint`` return None.
"""

import sys
import os
import argparse
import math
from pathlib import Path

# Add src to path so we can import project utils
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_training_context, patch_tevatron_loss, load_config, set_seed, \
                          log_startup_config, build_run_manifest, prepare_output_dir, \
                          assert_training_succeeded, attach_training_diagnostics, \
                          probe_triples_from_mixture, ranking_probe, require_recipe_keys
from data.preprocessor import MIXTURE_FILES, require_mixture_files
from transformers import AutoTokenizer
from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.driver.train import main as tevatron_train_main

# 🩹 Tevatron Bug Patch
if not hasattr(DenseModel, "_keys_to_ignore_on_save"):
    setattr(DenseModel, "_keys_to_ignore_on_save", None)

# 🩹 Tevatron bug: EncoderModel.gradient_checkpointing_enable() calls
# `self.encoder.model.gradient_checkpointing_enable()`, but for a full fine-tune
# `self.encoder` IS the HF model (XLMRobertaModel — no `.model` attr), so HF Trainer's
# gradient_checkpointing=True crashes with "'XLMRobertaModel' object has no attribute 'model'".
# Forward straight to the encoder — mirrors exactly what run_fast_grass does (proven to fit
# batch 64 @ q1024/p512). Accept **kwargs so Trainer's gradient_checkpointing_kwargs= is
# absorbed; we intentionally don't forward it (matches the no-arg fast_grass call).
def _tevatron_gc_enable(self, **kwargs):
    self.encoder.gradient_checkpointing_enable()
DenseModel.gradient_checkpointing_enable = _tevatron_gc_enable

# Every training.inbatch key this script reads. require_recipe_keys fails on a
# declared-but-unread key, so config.yaml cannot drift back into being decorative.
CONSUMED_KEYS = (
    'model_name', 'batch_size', 'learning_rate', 'num_epochs', 'train_group_size',
    'bf16', 'dataloader_num_workers', 'warmup_ratio', 'weight_decay', 'max_grad_norm',
    'logging_steps', 'save_total_limit', 'save_fraction', 'gradient_checkpointing',
)


def main():
    parser = argparse.ArgumentParser(description="In-batch negatives training.")
    parser.add_argument('--resume', action='store_true',
                        help="Continue a run in the output dir whose manifest matches. "
                             "Without it the run starts fresh and stale checkpoints are removed.")
    parser.add_argument('--overwrite', action='store_true',
                        help="Discard an output dir produced by a DIFFERENT configuration.")
    args = parser.parse_args()

    # 1. Get unified context (Hyperparameters + Absolute Paths)
    config = load_config()
    set_seed(config.get('seed', 42))
    ctx = get_training_context("inbatch")
    recipe = ctx['args']
    require_recipe_keys("inbatch", recipe, CONSUMED_KEYS)
    log_startup_config("inbatch", ctx)

    # --- PATH MODIFICATION: Resolve the mixture directory ---
    # We look for the folder named 'training_mixture' inside the data directory
    processed_dir = Path(ctx['processed_dir'])
    mixture_dir = processed_dir / "training_mixture"

    # Tevatron accepts one glob; strict validation below guarantees it contains only
    # the three declared mixed-training components.
    training_data_path = str(mixture_dir / "*.jsonl")

    mixture_files = require_mixture_files(mixture_dir, MIXTURE_FILES)

    # --- Count training examples and calculate checkpoint intervals ---
    num_examples = 0
    for f in mixture_files:
        with open(f) as fh:
            num_examples += sum(1 for _ in fh)

    batch_size = recipe['batch_size']
    num_epochs = recipe['num_epochs']
    total_steps = math.ceil(num_examples / batch_size) * num_epochs
    save_steps = max(1, int(total_steps * recipe['save_fraction']))

    # --- CONFIG PARAMETER PRINTS ---
    print("\n" + "="*40)
    print("🛠️  VERIFYING CONFIGURATION PARAMETERS")
    print(f"▶️  Train Group Size:   {recipe.get('train_group_size')}")
    print(f"▶️  Batch Size:         {recipe.get('batch_size')}")
    print(f"▶️  Learning Rate:      {recipe.get('learning_rate')}")
    print(f"▶️  Num Epochs:         {recipe.get('num_epochs')}")
    print(f"📂 Training Data:      {training_data_path}")
    print(f"📊 Training examples:  {num_examples}")
    print(f"📊 Total steps:        {total_steps}, saving every {save_steps} steps")
    print("="*40 + "\n")

    # 1b. Run identity, then gate the output directory before anything is written.
    # In-batch is single-GPU, so the pool is one step's passages: batch * group - 1.
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    manifest = build_run_manifest(
        "inbatch", ctx, recipe,
        data_files=mixture_files,
        world_size=world_size,
        negative_pool_size=batch_size * world_size * recipe['train_group_size'] - 1,
        optimizer_steps=total_steps,
    )
    output_dir = prepare_output_dir(ctx['output_dir'], manifest,
                                    resume=args.resume, overwrite=args.overwrite)

    # 2. Map YAML/Context to Tevatron Arguments
    training_args = [
        '--output_dir', str(output_dir),
        '--model_name_or_path', ctx['base_model'],
        '--dataset_name', 'json',
        '--dataset_path', training_data_path,    # Updated to use the glob path
        '--dataset_split', 'train',
        '--do_train',
        '--per_device_train_batch_size', str(batch_size),
        '--learning_rate', str(recipe['learning_rate']),
        '--num_train_epochs', str(num_epochs),
        '--train_group_size', str(recipe['train_group_size']),
        '--query_max_len', str(ctx['max_q']),
        '--passage_max_len', str(ctx['max_p']),
        '--bf16', str(recipe['bf16']),
        '--fp16', 'False',
        '--dtype', 'bfloat16' if recipe['bf16'] else 'float32',
        '--logging_steps', str(recipe['logging_steps']),
        '--overwrite_output_dir', 'True',
        '--save_strategy', 'steps',
        '--save_steps', str(save_steps),
        '--save_total_limit', str(recipe['save_total_limit']),
        '--attn_implementation', 'eager',     # XLM-RoBERTa has no sdpa in this transformers ver
        '--gradient_checkpointing', str(recipe['gradient_checkpointing']),
                                              # THE memory fix: frees the eager-attention
                                              # activations (~50GB at q1024).
        '--dataloader_num_workers', str(recipe['dataloader_num_workers']),
        '--optim', 'adamw_torch_fused',       # bitsandbytes not available in this container
        '--warmup_ratio', str(recipe['warmup_ratio']),
        '--weight_decay', str(recipe['weight_decay']),
        '--max_grad_norm', str(recipe['max_grad_norm']),
        '--pooling', ctx['pooling'],
        '--normalize', str(ctx['normalize']),
        '--temperature', str(ctx['temperature']),
        '--seed', str(config.get('seed', 42)),
    ]

    # 3. Inject Arguments into sys.argv
    sys.argv = ['train.py'] + training_args

    # 4. Run Training Directly
    print(f"🚀 Starting In-Batch Training for model: {recipe['model_name']}")
    print(f"📂 Loading data from: {training_data_path}")

    triples = probe_triples_from_mixture(mixture_files)
    # Tevatron constructs its Trainer without tokenizer=, so the callback receives
    # tokenizer=None. Load the one the run actually trains with and close over it.
    probe_tokenizer = AutoTokenizer.from_pretrained(ctx['base_model'])
    attach_training_diagnostics(
        output_dir,
        lambda model, tokenizer: ranking_probe(
            model, tokenizer or probe_tokenizer, triples,
            next(model.parameters()).device, ctx['max_q'], ctx['max_p']))
    patch_tevatron_loss(ctx['temperature'])
    tevatron_train_main()

    # A clean return is not success: validate before claiming one.
    assert_training_succeeded(output_dir, manifest)
    print(f"✅ Training completed. Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
