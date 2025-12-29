"""
Base trainer class for common training functionality.
"""

from typing import Optional, Dict, Any
from pathlib import Path

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from utils.config import get_checkpoint_dir, get_model_output_path
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseTrainer:
    """Base class for training utilities."""
    
    def __init__(
        self,
        model_name: str,
        exp_name: str,
        checkpoint_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize base trainer.
        
        Args:
            model_name: Model name or path
            exp_name: Experiment name
            checkpoint_dir: Checkpoint directory (default: from config)
            **kwargs: Additional arguments
        """
        self.model_name = model_name
        self.exp_name = exp_name
        self.checkpoint_dir = checkpoint_dir or get_checkpoint_dir()
        
    def get_training_arguments(
        self,
        fold: int,
        learning_rate: float,
        batch_size: int,
        gradient_accumulation_steps: int,
        num_epochs: int,
        weight_decay: float = 0.01,
        warmup_steps: int = 10,
        max_length: int = 1800,
        **kwargs
    ) -> TrainingArguments:
        """
        Create training arguments.
        
        Args:
            fold: Fold number
            learning_rate: Learning rate
            batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            num_epochs: Number of training epochs
            weight_decay: Weight decay
            warmup_steps: Warmup steps
            max_length: Maximum sequence length
            **kwargs: Additional TrainingArguments
            
        Returns:
            TrainingArguments object
        """
        output_dir = get_model_output_path(
            self.checkpoint_dir,
            self.model_name,
            self.exp_name,
            fold
        )
        
        # Ensure output directory exists
        Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
        
        default_args = {
            'output_dir': output_dir,
            'learning_rate': learning_rate,
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'num_train_epochs': num_epochs,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'lr_scheduler_type': 'cosine',
            'bf16': True,
            'bf16_full_eval': True,
            'tf32': True,
            'optim': 'paged_adamw_8bit',
            'evaluation_strategy': 'epoch',
            'logging_steps': 1,
            'save_strategy': 'epoch',
            'save_total_limit': 1,
            'dataloader_num_workers': 4,
            'dataloader_pin_memory': True,
            'ddp_find_unused_parameters': False,
            'gradient_checkpointing': True,
            'group_by_length': True,
            'report_to': 'wandb',
            'run_name': f'{self.model_name}/{self.exp_name}/fold-{fold}'.replace('/', '_'),
        }
        
        # Update with any provided kwargs
        default_args.update(kwargs)
        
        return TrainingArguments(**default_args)
    
    def get_quantization_config(
        self,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype: Optional[torch.dtype] = None
    ) -> BitsAndBytesConfig:
        """
        Create quantization configuration.
        
        Args:
            load_in_4bit: Whether to load in 4-bit
            bnb_4bit_quant_type: Quantization type
            bnb_4bit_use_double_quant: Whether to use double quantization
            bnb_4bit_compute_dtype: Compute dtype
            
        Returns:
            BitsAndBytesConfig object
        """
        if bnb_4bit_compute_dtype is None:
            bnb_4bit_compute_dtype = torch.bfloat16
        
        return BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )
    
    def get_lora_config(
        self,
        r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[list] = None,
        modules_to_save: Optional[list] = None,
        task_type: TaskType = TaskType.SEQ_CLS
    ) -> LoraConfig:
        """
        Create LoRA configuration.
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: Target modules for LoRA
            modules_to_save: Modules to save (not quantized)
            task_type: Task type
            
        Returns:
            LoraConfig object
        """
        if target_modules is None:
            target_modules = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ]
        
        if modules_to_save is None:
            modules_to_save = ['score', 'lstm']
        
        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias='none',
            task_type=task_type,
            inference_mode=False,
            modules_to_save=modules_to_save,
        )
    
    def get_data_collator(
        self,
        tokenizer: PreTrainedTokenizer,
        pad_to_multiple_of: Optional[int] = None
    ) -> DataCollatorWithPadding:
        """
        Create data collator.
        
        Args:
            tokenizer: Tokenizer to use
            pad_to_multiple_of: Pad to multiple of this number
            
        Returns:
            DataCollatorWithPadding object
        """
        return DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=pad_to_multiple_of
        )


def cleanup_resources(trainer: Optional[Trainer] = None):
    """
    Clean up GPU memory and Python objects.
    
    Args:
        trainer: Optional trainer to delete
    """
    if trainer is not None:
        del trainer
    
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    logger.info("Resources cleaned up")

