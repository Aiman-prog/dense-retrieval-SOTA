"""
Train Cross-Batch (RocketQA) model on ReasonIR-HQ using Tevatron (GitHub version).
Technique: Gradient Caching on Single GPU.
IMPLEMENTATION: "Trojan Horse" DDP Wrapping with GC Reference Update.
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed

# Tevatron Imports (GitHub Tevatron uses tevatron.retriever.* structure)
from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.gc_trainer import GradCacheTrainer

# Try importing TrainCollator, fallback to None if missing
try:
    from tevatron.retriever.collator import TrainCollator
except ImportError:
    TrainCollator = None

# Project Imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))
from utils.helpers import load_config, get_data_base_dir

logger = logging.getLogger(__name__)

def main():
    # 1. Configuration & Paths
    config_path = project_root / 'config' / 'config.yaml'
    yaml_config = load_config(str(config_path))
    
    base_dir = get_data_base_dir()
    processed_dir = Path(os.environ.get('PROCESSED_DATA_DIR', f'{base_dir}/data/processed'))
    train_dir = processed_dir / 'reasonir_crossbatch_train'
    train_file = train_dir / 'train.jsonl'
    output_dir = Path(f'{base_dir}/models/crossbatch_reasonir')

    if not train_file.exists():
        print(f"❌ ERROR: Training file not found: {train_file}")
        sys.exit(1)

    # 2. Argument Construction
    target_batch_size = int(os.environ.get('TARGET_BATCH_SIZE', '1024'))  # Default 1024 for production (gpu-v100)
    physical_batch_size = int(os.environ.get('PHYSICAL_BATCH_SIZE', '64'))  # Default 64 for gpu-v100 (32GB GPU with FP32)
    chunk_size = int(os.environ.get('CHUNK_SIZE', '64'))  # Default 64 for gpu-v100 (accumulates 16 chunks to reach 1024)
    
    args_list = [
        '--output_dir', str(output_dir),
        '--model_name_or_path', yaml_config['model']['base_model'],
        '--dataset_name', 'json',
        '--dataset_path', str(train_file),
        '--dataset_split', 'train',
        '--do_train',
        '--per_device_train_batch_size', str(physical_batch_size),
        '--gc_q_chunk_size', str(chunk_size),
        '--gc_p_chunk_size', str(chunk_size),
        '--learning_rate', os.environ.get('LEARNING_RATE', '1e-5'),
        '--num_train_epochs', os.environ.get('NUM_EPOCHS', '3'),
        '--train_group_size', '1',
        '--query_max_len', '128',
        '--passage_max_len', '512',
        '--grad_cache', 'True',
        '--dataloader_num_workers', '4',  # gpu-v100: more CPUs available
        '--fp16', 'False',  # Disable FP16 (causes scaler errors)
        '--bf16', 'False',  # Disable BF16 (V100 doesn't support BF16, only A100 does)
        '--overwrite_output_dir', 'True',
        '--logging_steps', '10',
        '--save_steps', '500'  # Save checkpoint every 500 steps (allows resume if training crashes)
    ]

    # 3. Parse Arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TevatronTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args_list)
    
    # Validate mixed precision settings
    # NOTE: fp16 and bf16 are mutually exclusive - you can't enable both
    if training_args.fp16 and training_args.bf16:
        raise ValueError("Cannot enable both fp16 and bf16 simultaneously. Choose one or neither.")
    
    # BF16 is preferred for A100 GPUs (no gradient scaling needed, more stable than FP16)
    print(f"✅ Mixed precision: fp16={training_args.fp16}, bf16={training_args.bf16}")
    
    # 4. Initialize Distributed Environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if not dist.is_initialized():
        if not os.environ.get('MASTER_ADDR'):
            os.environ['MASTER_ADDR'] = 'localhost'
        if not os.environ.get('MASTER_PORT'):
            # Use SLURM_JOB_ID to generate unique port, or fallback to random port
            # Port range: 10000-65535 (avoiding system ports)
            slurm_job_id = os.environ.get('SLURM_JOB_ID')
            if slurm_job_id:
                # Use job ID to generate port: 10000 + (job_id % 55535)
                port = 10000 + (int(slurm_job_id) % 55535)
            else:
                # Random port if no SLURM_JOB_ID
                import random
                port = random.randint(10000, 65535)
            os.environ['MASTER_PORT'] = str(port)
            print(f"✅ Using MASTER_PORT={port} (from SLURM_JOB_ID={slurm_job_id if slurm_job_id else 'N/A'})")
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    
    set_seed(training_args.seed)

    # 5. Build Model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    print("DEBUG: Building DenseModel...")
    try:
        model = DenseModel.build(model_args, training_args)
    except TypeError:
        try:
            model = DenseModel.build(model_args, data_args, training_args)
        except TypeError:
            model = DenseModel.build(model_args, training_args, config=config, cache_dir=model_args.cache_dir)
    print("✅ DenseModel built.")

    # Move to device but DO NOT WRAP yet
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    model.to(device)

    # 6. Load Data (trainer=None fix)
    print("DEBUG: Loading training dataset...")
    try:
        train_dataset = TrainDataset(data_args, trainer=None)
    except TypeError:
        train_dataset = TrainDataset(data_args, tokenizer)

    # 7. Create Trainer
    # TrainCollator exists - use it directly
    try:
        data_collator = TrainCollator(
            tokenizer,
            max_p_len=data_args.passage_max_len,
            max_q_len=data_args.query_max_len
        )
    except (TypeError, AttributeError):
        # Fallback: try with data_args if signature is different
        if TrainCollator:
            data_collator = TrainCollator(data_args, tokenizer)
        else:
            # Last resort: create a simple collator
            from transformers import DataCollatorWithPadding
            data_collator = DataCollatorWithPadding(tokenizer)

    # Patch GradCacheTrainer for scaler bug
    # The bug: GradCacheTrainer accesses self.scaler unconditionally in __init__
    # before it's initialized by the parent Trainer class
    original_init = GradCacheTrainer.__init__
    def safe_init(self, *args, **kwargs):
        # Get training args from kwargs or use the one from outer scope
        trainer_args = kwargs.get('args', training_args)
        
        # Set a dummy scaler BEFORE calling original_init to avoid AttributeError
        # BF16 doesn't need scaler, FP16 would need one but we're using BF16
        self.scaler = None
        
        # Now call the original __init__ - parent Trainer will initialize scaler properly
        original_init(self, *args, **kwargs)
        
        # AFTER parent __init__, ensure scaler is None for BF16 (no scaling needed)
        if getattr(trainer_args, 'bf16', False):
            self.scaler = None
            print("✅ BF16 enabled (no scaler needed - wider dynamic range)")
        else:
            # No mixed precision
            self.scaler = None
            print("✅ No mixed precision (FP32)")
    GradCacheTrainer.__init__ = safe_init
    
    try:
        trainer = GradCacheTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Verify BF16 is enabled (no scaler needed)
        if training_args.bf16:
            print(f"✅ BF16 mixed precision enabled (no scaler needed)")
    finally:
        GradCacheTrainer.__init__ = original_init
    
    # 8. Wrap in DDP AFTER GradCacheTrainer creation
    # GradCacheTrainer rejects DDP-wrapped models during initialization,
    # but GradCache needs DDP-wrapped models for its assertion check
    if not isinstance(trainer.model, DDP):
        print("🔥 Wrapping trainer.model in DDP (after GradCacheTrainer init)...")
        trainer.model = DDP(
            trainer.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        print("✅ trainer.model wrapped in DDP")
        
        # CRITICAL: Update GradCache's model references to use DDP-wrapped model
        # GradCache needs DDP-wrapped models for its assertion check
        if hasattr(trainer, 'gc') and hasattr(trainer.gc, 'models'):
            try:
                num_models = len(trainer.gc.models)
                trainer.gc.models = [trainer.model] * num_models
                print(f"✅ Updated GradCache.models to use DDP-wrapped model ({num_models} references)")
            except (AttributeError, TypeError) as e:
                print(f"⚠️ WARNING: Could not update trainer.gc.models: {e}")
                if isinstance(trainer.gc.models, list):
                    for i in range(len(trainer.gc.models)):
                        trainer.gc.models[i] = trainer.model
                    print(f"✅ Updated {len(trainer.gc.models)} references in-place")
        
        # CRITICAL: Always add monkey-patch to ensure DDP check passes at runtime
        # GradCache checks DDP wrapping inside cache_step, so we need to patch it
        # This is a safety net in case trainer.gc.models gets reset during training
        if hasattr(trainer, 'gc') and hasattr(trainer.gc, 'cache_step'):
            original_cache_step = trainer.gc.cache_step.__func__
            def patched_cache_step(self_gc, *args, **kwargs):
                # Ensure models are DDP-wrapped before assertion check
                if hasattr(self_gc, 'models') and not all(isinstance(m, DDP) for m in self_gc.models):
                    self_gc.models = [trainer.model] * len(self_gc.models)
                return original_cache_step(self_gc, *args, **kwargs)
            trainer.gc.cache_step = patched_cache_step.__get__(trainer.gc, type(trainer.gc))
            print("✅ Added monkey-patch to ensure DDP check passes at runtime")
    
    # CRITICAL: Patch trainer._save() and save_model() to handle DDP-wrapped models
    # Tevatron's trainer._save() rejects DDP-wrapped models (line 30 in trainer.py)
    # We need to unwrap the model before saving - apply patches ALWAYS (even if model not DDP yet)
    # Extract the unbound method to avoid double-binding issues
    original_save_func = trainer._save.__func__ if hasattr(trainer._save, '__func__') else trainer._save
    def patched_save(self_trainer, output_dir, *args, **kwargs):
        # Temporarily unwrap DDP model for saving
        was_ddp = isinstance(self_trainer.model, DDP)
        if was_ddp:
            original_model = self_trainer.model
            self_trainer.model = original_model.module  # Unwrap for saving
        try:
            # Call original unbound method with self and all arguments
            result = original_save_func(self_trainer, output_dir, *args, **kwargs)
        finally:
            if was_ddp:
                self_trainer.model = original_model  # Restore DDP wrapper
        return result
    trainer._save = patched_save.__get__(trainer, type(trainer))
    print("✅ Patched trainer._save() to handle DDP models")
    
    # Also patch save_model() to handle DDP-wrapped models
    # save_model signature: save_model(output_dir=None, _internal_call=False)
    # Extract the unbound method to avoid double-binding issues
    original_save_model_func = trainer.save_model.__func__ if hasattr(trainer.save_model, '__func__') else trainer.save_model
    def patched_save_model(self_trainer, *args, **kwargs):
        # Temporarily unwrap DDP model for saving
        was_ddp = isinstance(self_trainer.model, DDP)
        if was_ddp:
            original_model = self_trainer.model
            self_trainer.model = original_model.module  # Unwrap for saving
        try:
            # Call original unbound method with self and all arguments
            result = original_save_model_func(self_trainer, *args, **kwargs)
            # Print confirmation when checkpoint saving succeeds
            output_dir = args[0] if args else kwargs.get('output_dir', 'checkpoint')
            print(f"✅ Checkpoint saved successfully to {output_dir}")
        finally:
            if was_ddp:
                self_trainer.model = original_model  # Restore DDP wrapper
        return result
    trainer.save_model = patched_save_model.__get__(trainer, type(trainer))
    print("✅ Patched trainer.save_model() to handle DDP models")

    # 9. Link Trainer to Dataset
    if hasattr(train_dataset, 'trainer'):
        train_dataset.trainer = trainer
        print("✅ Linked Trainer to TrainDataset")
    
    # 10. Train
    try:
        print(f"🚀 Starting Training with Virtual Batch {target_batch_size}...")
        if training_args.bf16:
            print("   Using BF16 mixed precision (no gradient scaling needed)")
        trainer.train()
        
        # 11. Save
        print(f"✅ Saving model...")
        if isinstance(trainer.model, DDP):
            original_model = trainer.model.module
            trainer.model = original_model
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            print("✅ Tokenizer saved.")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)