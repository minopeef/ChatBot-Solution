"""
Optimized LLM QLoRA training script using utility modules.
This is an example of how to refactor existing training scripts.
"""

import os
import sys
import gc
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from datasets import Dataset
from transformers import AutoConfig, TrainingArguments
from peft import get_peft_model

from models.llm_model import Model
from utils.config import (
    load_config, parse_args, get_checkpoint_dir, get_model_output_path
)
from utils.data_utils import load_data, format_prompt, setup_tokenizer, get_fold_split
from utils.logging_utils import setup_logging, get_logger
from utils.trainer_base import BaseTrainer, cleanup_resources

# Setup logging
logger = setup_logging(level=20)  # INFO level


class CustomTrainer(BaseTrainer):
    """Custom trainer with differential learning rates."""
    
    def create_optimizer(self):
        """Setup optimizer with differential learning rates."""
        from transformers import Trainer
        
        opt_model = self.model
        
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n in decay_parameters and 
                            ('score' not in n and 'embed_tokens' not in n) and 
                            p.requires_grad)
                    ],
                    "learning_rate": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n in decay_parameters and 
                            ('score' in n or 'embed_tokens' in n) and 
                            p.requires_grad)
                    ],
                    "learning_rate": getattr(self.args, 'head_lr', 1e-5),
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() 
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
            
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")
            
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")
            
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, torch.nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
        
        return self.optimizer


def get_trainer(
    base_trainer: BaseTrainer,
    config: any,
    tokenizer: any,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    fold: int
):
    """
    Create trainer instance.
    
    Args:
        base_trainer: BaseTrainer instance
        config: Configuration object
        tokenizer: Tokenizer instance
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        fold: Fold number
        
    Returns:
        Trainer instance
    """
    from transformers import Trainer
    
    # Get training arguments
    training_args = base_trainer.get_training_arguments(
        fold=fold,
        learning_rate=config.lr,
        batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.accum,
        num_epochs=config.epochs,
        weight_decay=config.weight_decay,
        max_length=config.training.max_length,
        max_grad_norm=1000.0,
    )
    
    # Get quantization config
    quant_config = base_trainer.get_quantization_config()
    
    # Load model
    model_config = AutoConfig.from_pretrained(
        config.model_name, 
        trust_remote_code=True
    )
    model = Model(
        model_config,
        config.model_name,
        quant_config=quant_config,
        pad_token_id=tokenizer.pad_token_id
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if 'no_cap' in config.exp_name:
        model.config.attn_logit_softcapping = None
    
    # Setup LoRA
    lora_config = base_trainer.get_lora_config(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=getattr(config, 'lora_dropout', 0.0),
        target_modules=config.target_modules,
    )
    
    logger.info(f"LoRA modules to save: {lora_config.modules_to_save}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Get data collator
    collator = base_trainer.get_data_collator(tokenizer)
    
    # Create custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator
    )
    
    return trainer


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config, {'tta': args.tta})
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Override checkpoint/data dirs if provided
    checkpoint_dir = args.checkpoint_dir or get_checkpoint_dir()
    data_dir = args.data_dir or os.getenv('DATA_DIR', 'data')
    
    # Initialize base trainer
    base_trainer = BaseTrainer(
        model_name=config.model_name,
        exp_name=config.exp_name,
        checkpoint_dir=checkpoint_dir
    )
    
    # Load data
    try:
        logger.info("Loading training data...")
        df = load_data(
            data_dir=data_dir,
            include_lmsys_33k=True,
            include_orpo=False,
            tta=config.tta
        )
        logger.info(f"Loaded {len(df)} samples")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Setup tokenizer
    try:
        logger.info(f"Loading tokenizer for {config.model_name}")
        tokenizer = setup_tokenizer(config.model_name)
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        sys.exit(1)
    
    # Convert to dataset and preprocess
    logger.info("Preprocessing dataset...")
    ds = Dataset.from_pandas(df.to_pandas())
    
    # Format prompts
    def format_fn(row):
        return format_prompt(
            row,
            template="<PROMPT>{}</PROMPT><RESPONSE A>{}</RESPONSE A><RESPONSE B>{}</RESPONSE B>",
            swap_responses=config.tta
        )
    
    ds = ds.map(format_fn, num_proc=8, batched=False)
    
    # Tokenize
    def tokenize_fn(batch):
        return tokenizer(
            batch['formatted_prompt'],
            max_length=config.training.max_length,
            truncation=True,
            padding=False
        )
    
    tok_ds = ds.map(
        tokenize_fn,
        batched=False,
        num_proc=8,
        remove_columns=[c for c in ds.column_names if c not in ['labels', 'fold']],
    )
    
    # Train on folds
    num_folds = 1  # Can be configured
    for fold in range(num_folds):
        logger.info(f"Training fold {fold}/{num_folds-1}")
        
        try:
            # Get fold split
            dds = get_fold_split(tok_ds, fold, sort_by_length=True)
            
            # Create trainer
            trainer = get_trainer(
                base_trainer,
                config,
                tokenizer,
                dds['train'],
                dds['test'],
                fold
            )
            
            # Train
            logger.info("Starting training...")
            trainer.train()
            
            logger.info(f"Completed training for fold {fold}")
            
        except Exception as e:
            logger.error(f"Error during training fold {fold}: {e}", exc_info=True)
            continue
        finally:
            # Cleanup
            cleanup_resources(trainer)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

