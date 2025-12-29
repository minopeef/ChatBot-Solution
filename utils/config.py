"""
Configuration management utilities.
"""

import os
import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Optional
import argparse


def get_checkpoint_dir() -> str:
    """Get checkpoint directory from environment variable or use default."""
    return os.getenv('CHECKPOINT_DIR', '/mnt/one/kaggle/lmsys-chatbot-arena')


def get_data_dir() -> str:
    """Get data directory from environment variable or use default."""
    return os.getenv('DATA_DIR', 'data')


def get_wandb_project() -> str:
    """Get Weights & Biases project name from environment variable or use default."""
    return os.getenv('WANDB_PROJECT', 'lmsys-chatbot-arena')


def load_config(config_path: str, additional_args: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        additional_args: Additional arguments to merge into config
        
    Returns:
        SimpleNamespace object containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)
    
    if args is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    # Merge additional arguments
    if additional_args:
        args.update(additional_args)
    
    # Convert nested dicts to SimpleNamespace
    for k, v in args.items():
        if isinstance(v, dict):
            args[k] = SimpleNamespace(**v)
    
    return SimpleNamespace(**args)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for training/validation scripts.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='ChatBot Solution Training/Validation')
    parser.add_argument(
        "-C", "--config",
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--tta",
        action='store_true',
        help="Whether to do test-time augmentation"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory (default: from env or config)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory (default: from env or config)"
    )
    
    return parser.parse_args()


def get_model_output_path(
    checkpoint_dir: str,
    model_name: str,
    exp_name: str,
    fold: int,
    suffix: str = ""
) -> str:
    """
    Generate standardized model output path.
    
    Args:
        checkpoint_dir: Base checkpoint directory
        model_name: Model name (will be sanitized)
        exp_name: Experiment name
        fold: Fold number
        suffix: Optional suffix to append
        
    Returns:
        Full path to model checkpoint directory
    """
    # Sanitize model name for filesystem
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    path = f"{checkpoint_dir}/{safe_model_name}-{exp_name}-fold-{fold}"
    if suffix:
        path = f"{path}-{suffix}"
    return path

