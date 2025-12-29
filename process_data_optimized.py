"""
Optimized data processing script with error handling and logging.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset

from utils.config import get_data_dir
from utils.logging_utils import setup_logging, get_logger

# Setup logging
logger = setup_logging(level=20)  # INFO level


def process_train_data(data_dir: str, n_splits: int = 4, random_state: int = 42) -> None:
    """
    Process training data and create folds.
    
    Args:
        data_dir: Directory containing train.csv
        n_splits: Number of folds for cross-validation
        random_state: Random state for reproducibility
    """
    data_path = Path(data_dir)
    train_csv = data_path / 'train.csv'
    
    if not train_csv.exists():
        raise FileNotFoundError(f"Training data not found: {train_csv}")
    
    logger.info(f"Loading training data from {train_csv}")
    train = (
        pl.read_csv(train_csv)
        .with_columns(
            prompt=pl.col('prompt').str.json_decode(),
            response_a=pl.col('response_a').str.json_decode(),
            response_b=pl.col('response_b').str.json_decode(),
            labels=pl.when(pl.col('winner_model_a') == 1)
                .then(0)
                .when(pl.col('winner_model_b') == 1)
                .then(1)
                .otherwise(2),
            fold=pl.lit(-1)
        )
    )
    
    logger.info(f"Creating {n_splits} stratified folds")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    train_pd = train.to_pandas()
    for i, (train_idx, test_idx) in enumerate(skf.split(train_pd, train_pd['labels'])):
        train_pd.loc[test_idx, 'fold'] = i
    
    logger.info("Fold distribution:")
    logger.info(train_pd['fold'].value_counts())
    
    train = pl.from_pandas(train_pd)
    output_path = data_path / 'train.parquet'
    train.write_parquet(output_path)
    logger.info(f"Saved processed training data to {output_path}")


def process_test_data(data_dir: str) -> None:
    """
    Process test data.
    
    Args:
        data_dir: Directory containing test.csv
    """
    data_path = Path(data_dir)
    test_csv = data_path / 'test.csv'
    
    if not test_csv.exists():
        logger.warning(f"Test data not found: {test_csv}, skipping...")
        return
    
    logger.info(f"Loading test data from {test_csv}")
    test = (
        pl.read_csv(test_csv)
        .with_columns(
            prompt=pl.col('prompt').str.json_decode(),
            response_a=pl.col('response_a').str.json_decode(),
            response_b=pl.col('response_b').str.json_decode(),
        )
    )
    
    output_path = data_path / 'test.parquet'
    test.write_parquet(output_path)
    logger.info(f"Saved processed test data to {output_path}")


def process_lmsys_33k(data_dir: str) -> None:
    """
    Process lmsys-33k-deduplicated dataset.
    
    Args:
        data_dir: Directory containing lmsys-33k-deduplicated.csv
    """
    data_path = Path(data_dir)
    csv_path = data_path / 'lmsys-33k-deduplicated.csv'
    
    if not csv_path.exists():
        logger.warning(f"lmsys-33k data not found: {csv_path}, skipping...")
        return
    
    logger.info(f"Processing lmsys-33k data from {csv_path}")
    ds = (
        pl.read_csv(csv_path)
        .with_row_index()
        .with_columns(
            prompt=pl.col('prompt').str.json_decode(),
            response_a=pl.col('response_a').str.json_decode(),
            response_b=pl.col('response_b').str.json_decode(),
            labels=pl.when(pl.col('winner_model_a') == 1)
                .then(0)
                .when(pl.col('winner_model_b') == 1)
                .then(1)
                .otherwise(2),
            fold=pl.lit(-1),
            id=pl.col('index')
        )
        .drop('index')
    )
    
    output_path = data_path / 'lmsys-33k-deduplicated.parquet'
    ds.write_parquet(output_path)
    logger.info(f"Saved processed lmsys-33k data to {output_path}")


def process_orpo_dataset(data_dir: str, num_proc: int = 12) -> None:
    """
    Process orpo-dpo-mix-40k dataset from Hugging Face.
    
    Args:
        data_dir: Directory to save processed data
        num_proc: Number of processes for mapping
    """
    data_path = Path(data_dir)
    
    try:
        logger.info("Loading orpo-dpo-mix-40k dataset from Hugging Face")
        ds = load_dataset('mlabonne/orpo-dpo-mix-40k', split='train')
        
        def split_columns(row):
            """Split conversation columns into prompt and responses."""
            conversation_a = row['chosen']
            conversation_b = row['rejected']
            
            prompt = []
            response_a = []
            for turn in conversation_a:
                if turn['role'] == 'user':
                    prompt.append(turn['content'])
                else:
                    response_a.append(turn['content'])
            
            response_b = []
            for turn in conversation_b:
                if turn['role'] == 'assistant':
                    response_b.append(turn['content'])
            
            return {
                'prompt': prompt,
                'response_a': response_a,
                'response_b': response_b
            }
        
        logger.info(f"Processing dataset with {num_proc} processes")
        ds = ds.map(split_columns, num_proc=num_proc)
        ds_df = pl.from_pandas(ds.to_pandas())
        ds_df = ds_df.with_columns(fold=pl.lit(-1))
        
        output_path = data_path / 'orpo-dpo-mix-40k.parquet'
        ds_df.write_parquet(output_path)
        logger.info(f"Saved processed orpo-dpo-mix-40k data to {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing orpo-dpo-mix-40k dataset: {e}")
        logger.warning("Continuing without orpo-dpo-mix-40k dataset")


def main():
    """Main processing function."""
    data_dir = get_data_dir()
    data_path = Path(data_dir)
    
    # Ensure data directory exists
    data_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing data in directory: {data_path}")
    
    try:
        # Process training data
        process_train_data(data_dir)
        
        # Process test data
        process_test_data(data_dir)
        
        # Process additional datasets
        process_lmsys_33k(data_dir)
        process_orpo_dataset(data_dir)
        
        logger.info("Data processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data processing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

