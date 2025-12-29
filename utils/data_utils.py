"""
Data loading and preprocessing utilities.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import polars as pl
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np


def load_data(
    data_dir: str = "data",
    include_lmsys_33k: bool = True,
    include_orpo: bool = False,
    tta: bool = False
) -> pl.DataFrame:
    """
    Load and combine training datasets.
    
    Args:
        data_dir: Directory containing data files
        include_lmsys_33k: Whether to include lmsys-33k-deduplicated dataset
        include_orpo: Whether to include orpo-dpo-mix-40k dataset
        tta: Whether to swap labels for test-time augmentation
        
    Returns:
        Combined DataFrame with all datasets
        
    Raises:
        FileNotFoundError: If required data files don't exist
    """
    data_path = Path(data_dir)
    
    # Load main training data
    train_path = data_path / 'train.parquet'
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    dfs = [
        pl.read_parquet(train_path).select([
            pl.col('id'), 'model_a', 'model_b', 'prompt', 
            'response_a', 'response_b', 'labels', 'fold'
        ])
    ]
    
    # Load additional datasets
    if include_lmsys_33k:
        lmsys_path = data_path / 'lmsys-33k-deduplicated.parquet'
        if lmsys_path.exists():
            dfs.append(
                pl.read_parquet(lmsys_path).select([
                    pl.col('id').cast(pl.Int64), 'model_a', 'model_b', 
                    'prompt', 'response_a', 'response_b', 'labels', 'fold'
                ])
            )
    
    if include_orpo:
        orpo_path = data_path / 'orpo-dpo-mix-40k.parquet'
        if orpo_path.exists():
            dfs.append(
                pl.read_parquet(orpo_path).select([
                    pl.col('id').cast(pl.Int64), 'model_a', 'model_b',
                    'prompt', 'response_a', 'response_b', 'labels', 'fold'
                ])
            )
    
    df = pl.concat(dfs)
    
    # Swap labels for TTA
    if tta:
        df = df.with_columns(
            labels=pl.when(pl.col('labels') == 0)
                .then(1)
                .when(pl.col('labels') == 1)
                .then(0)
                .otherwise(pl.col('labels'))
        )
    
    return df


def format_prompt(
    row: Dict[str, Any],
    template: str = "<PROMPT>{}</PROMPT><RESPONSE A>{}</RESPONSE A><RESPONSE B>{}</RESPONSE B>",
    swap_responses: bool = False
) -> Dict[str, str]:
    """
    Format prompt from row data.
    
    Args:
        row: Dictionary containing 'prompt', 'response_a', 'response_b' lists
        template: Template string for formatting
        swap_responses: Whether to swap response_a and response_b
        
    Returns:
        Dictionary with 'prompt' or 'formatted_prompt' key
    """
    chat_list = zip(row['prompt'], row['response_a'], row['response_b'])
    
    if swap_responses:
        responses = [
            template.format(r[0], r[2], r[1]) 
            for r in chat_list
        ]
    else:
        responses = [
            template.format(r[0], r[1], r[2]) 
            for r in chat_list
        ]
    
    return {'formatted_prompt': ''.join(responses)}


def setup_tokenizer(
    model_name: str,
    trust_remote_code: bool = True
) -> PreTrainedTokenizer:
    """
    Setup and configure tokenizer for a model.
    
    Args:
        model_name: Name or path of the model
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # Model-specific tokenizer configurations
    model_lower = model_name.lower()
    
    if 'qwen' in model_lower:
        tokenizer.padding_side = 'left'
    elif 'mistral-nemo' in model_lower:
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def get_fold_split(
    dataset: Dataset,
    fold_num: int,
    sort_by_length: bool = True,
    length_column: str = 'prompt_length'
) -> DatasetDict:
    """
    Split dataset into train/test based on fold number.
    
    Args:
        dataset: HuggingFace Dataset
        fold_num: Fold number to use as test set
        sort_by_length: Whether to sort test set by length
        length_column: Column name for length calculation
        
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    train_ds = dataset.filter(lambda x: x['fold'] != fold_num)
    test_ds = dataset.filter(lambda x: x['fold'] == fold_num)
    
    if sort_by_length and 'input_ids' in test_ds.column_names:
        def calc_length(example):
            return {length_column: len(example['input_ids'])}
        
        test_ds = test_ds.map(calc_length, num_proc=8)
        test_ds = test_ds.sort(length_column, reverse=True)
        test_ds = test_ds.remove_columns(length_column)
    
    return DatasetDict({
        "train": train_ds,
        "test": test_ds
    })


def tokenize_batch(
    batch: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    truncation: bool = True,
    padding: bool = False
) -> Dict[str, List]:
    """
    Tokenize a batch of texts.
    
    Args:
        batch: Dictionary with text data
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        truncation: Whether to truncate
        padding: Whether to pad
        
    Returns:
        Tokenized batch
    """
    return tokenizer(
        batch['formatted_prompt'] if 'formatted_prompt' in batch else batch['prompt'],
        max_length=max_length,
        truncation=truncation,
        padding=padding
    )

