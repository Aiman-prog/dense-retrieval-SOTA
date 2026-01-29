"""
Train Cross-Batch (RocketQA) model on ReasonIR-HQ using Tevatron (GitHub version).
Technique: Gradient Caching on Single GPU.
Logic: Fixed "NoneType" Trainer error + Config.yaml + File Path Resolution.
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed

# Tevatron Imports
from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.gc_trainer import GradCacheTrainer

# Try importing TrainCollator
try:
    from tevatron.retriever.collator import TrainCollator
except ImportError:
    TrainCollator = None

# Project Imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))
from utils.helpers import load_config, get_training_context

logger = logging.getLogger(__name__)

def main():
    # 1. Configuration & Paths via Centralized Context
    ctx = get_training_context("crossbatch")
    recipe = ctx['args'] 
    config_yaml = load_config()
    
    # --- PATH FIX: Resolve the mixture directory ---
    processed_dir = Path(ctx['train_file']).parent.resolve()
    mixture_dir = processed_dir / "training_mixture"
    output_dir = str(ctx['output_dir'])

    # Use the glob pattern to load all files in the mixture
    training_data_path = str(mixture_dir / "*.jsonl")

    if not mixture_dir.exists():
        print(f"❌ ERROR: Training mixture directory not found: {mixture_dir}")
        sys.exit(1)

    # 2. Argument Construction
    args_list = [
        '--output_dir', output_dir,
        '--model_name_or_path', config_yaml['model']['base_model'],
        '--dataset_name', 'json',
        '--dataset_path', training_data_path,
        '--dataset_split', 'train',
        '--do_train',
        '--per_device_train_batch_size', str(recipe['batch_size']),
        '--gc_q_chunk_size', str(recipe['batch_size']),
        '--gc_p_chunk_size', str(recipe['batch_size']),
        '--learning_rate', str(recipe['learning_rate']),
        '--num_train_epochs', str(recipe['num_epochs']),
        '--train_group_size', str(recipe['train_group_size']),
        '--query_max_len', str(config_yaml['model']['query_max_len']),
        '--passage_max_len', str(config_yaml['model']['passage_max_len']),
        '--grad_cache', 'True',
        '--dataloader_num_workers', str(config_yaml['training'].get('dataloader_num_workers', 2)),
        '--fp16', 'False',
        '--bf16', str(recipe.get('bf16', False)),
        '--overwrite_output_dir', str(config_yaml['training'].get('overwrite_output_dir', True)),
        '--logging_steps', str(config_yaml['training'].get('logging_steps', 10)),
    ]

    # 3. Parse Arguments into Dataclasses
    parser = HfArgumentParser((ModelArguments, DataArguments, TevatronTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args_list)
    
    # 4. Initialize Distributed Environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    
    set_seed(training_args.seed)

    # 5. Build Model (3-Stage Fallback)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    print("DEBUG: Building DenseModel...")
    try:
        model = DenseModel.build(model_args, training_args)
    except TypeError:
        try:
            model = DenseModel.build(model_args, data_args, training_args)
        except TypeError:
            model = DenseModel.build(model_args, training_args, config=config)
    print("✅ DenseModel built.")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    model.to(device)

    # 6. Load Data
    print("DEBUG: Loading training dataset...")
    try:
        train_dataset = TrainDataset(data_args, trainer=None)
    except TypeError:
        train_dataset = TrainDataset(data_args, tokenizer)

    # 7. Create Collator
    try:
        data_collator = TrainCollator(
            tokenizer,
            max_p_len=data_args.passage_max_len,
            max_q_len=data_args.query_max_len
        )
    except (TypeError, AttributeError):
        if TrainCollator:
            data_collator = TrainCollator(data_args, tokenizer)
        else:
            from transformers import DataCollatorWithPadding
            data_collator = DataCollatorWithPadding(tokenizer)

    # 8. GradCacheTrainer Patch (scaler bug)
    original_init = GradCacheTrainer.__init__
    def safe_init(self, *args, **kwargs):
        self.scaler = None
        original_init(self, *args, **kwargs)
        self.scaler = None
    GradCacheTrainer.__init__ = safe_init
    
    try:
        trainer = GradCacheTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
    finally:
        GradCacheTrainer.__init__ = original_init
    
    # 9. --- CRITICAL: LINK TRAINER TO DATASET ---
    train_dataset.trainer = trainer
    if hasattr(train_dataset, 'set_epoch'):
        train_dataset.set_epoch(0)

    # 10. --- CRITICAL: DDP WRAPPING ---
    if not isinstance(trainer.model, DDP):
        trainer.model = DDP(
            trainer.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        if hasattr(trainer, 'gc') and hasattr(trainer.gc, 'models'):
            trainer.gc.models = [trainer.model] * len(trainer.gc.models)
        
        if hasattr(trainer, 'gc') and hasattr(trainer.gc, 'cache_step'):
            original_cache_step = trainer.gc.cache_step.__func__
            def patched_cache_step(self_gc, *args, **kwargs):
                if hasattr(self_gc, 'models') and not all(isinstance(m, DDP) for m in self_gc.models):
                    self_gc.models = [trainer.model] * len(self_gc.models)
                return original_cache_step(self_gc, *args, **kwargs)
            trainer.gc.cache_step = patched_cache_step.__get__(trainer.gc, type(trainer.gc))

    # 11. --- CRITICAL: SAVE MODEL PATCHES ---
    original_save_func = trainer._save.__func__ if hasattr(trainer._save, '__func__') else trainer._save
    def patched_save(self_trainer, output_dir, *args, **kwargs):
        was_ddp = isinstance(self_trainer.model, DDP)
        if was_ddp:
            original_model = self_trainer.model
            self_trainer.model = original_model.module
        try:
            return original_save_func(self_trainer, output_dir, *args, **kwargs)
        finally:
            if was_ddp:
                self_trainer.model = original_model
    trainer._save = patched_save.__get__(trainer, type(trainer))

    # 12. Train
    print(f"🚀 Starting Cross-Batch Training: {recipe['model_name']}")
    print(f"📦 Physical Batch: {recipe['batch_size']}")
    print(f"📦 Virtual Target: {recipe.get('target_batch_size', 'N/A')}")
    
    trainer.train()
    
    # 13. Final Save
    if dist.get_rank() == 0:
        trainer.save_model(str(ctx['output_dir']))
        tokenizer.save_pretrained(str(ctx['output_dir']))
        print(f"✅ Training completed. Saved to: {ctx['output_dir']}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()