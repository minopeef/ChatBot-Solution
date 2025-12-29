# Project Optimization Guide

This document outlines the optimizations made to the ChatBot Solution project and how to use them.

## 🎯 Optimization Goals

1. **Reduce Code Duplication**: Extract common functionality into reusable utilities
2. **Improve Configuration Management**: Centralize configuration with environment variable support
3. **Add Error Handling**: Proper error handling and validation throughout
4. **Enhance Logging**: Replace global warning suppression with proper logging
5. **Better Code Organization**: Modular structure with clear separation of concerns
6. **Type Safety**: Add type hints for better IDE support and documentation

## 📁 New Structure

### Utility Modules (`utils/`)

#### `utils/config.py`
- Centralized configuration loading from YAML files
- Environment variable support for paths
- Standardized path generation
- Command-line argument parsing

**Usage:**
```python
from utils.config import load_config, get_checkpoint_dir, parse_args

# Load config from YAML
config = load_config('configs/gemma_rm.yaml')

# Get checkpoint directory (from env or default)
ckpt_dir = get_checkpoint_dir()

# Parse CLI arguments
args = parse_args()
```

#### `utils/data_utils.py`
- Unified data loading functions
- Reusable prompt formatting
- Tokenizer setup with model-specific configurations
- Fold splitting utilities

**Usage:**
```python
from utils.data_utils import load_data, format_prompt, setup_tokenizer

# Load data
df = load_data(data_dir='data', include_lmsys_33k=True)

# Setup tokenizer
tokenizer = setup_tokenizer('meta-llama/Meta-Llama-3-8B-Instruct')

# Format prompts
formatted = format_prompt(row, template="<PROMPT>{}</PROMPT>...")
```

#### `utils/logging_utils.py`
- Proper logging setup instead of global warning suppression
- Configurable log levels
- File logging support

**Usage:**
```python
from utils.logging_utils import setup_logging, get_logger

# Setup logging
logger = setup_logging(level=logging.INFO)

# Get logger for module
logger = get_logger(__name__)
logger.info("Training started")
```

#### `utils/trainer_base.py`
- Base trainer class with common functionality
- Standardized training arguments
- LoRA and quantization config helpers
- Resource cleanup utilities

**Usage:**
```python
from utils.trainer_base import BaseTrainer

base_trainer = BaseTrainer(
    model_name='meta-llama/Meta-Llama-3-8B-Instruct',
    exp_name='my_experiment'
)

# Get training arguments
training_args = base_trainer.get_training_arguments(
    fold=0,
    learning_rate=1e-4,
    batch_size=4,
    gradient_accumulation_steps=2,
    num_epochs=1
)
```

## 🔧 Migration Guide

### Step 1: Update Imports

**Before:**
```python
import yaml
from types import SimpleNamespace
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-C", "--config", help="config filename")
args = parser.parse_args()
config = yaml.safe_load(open(args.config).read())
```

**After:**
```python
from utils.config import load_config, parse_args

args = parse_args()
config = load_config(args.config)
```

### Step 2: Replace Hard-coded Paths

**Before:**
```python
ckpt_base_dir = '/mnt/one/kaggle/lmsys-chatbot-arena'
```

**After:**
```python
from utils.config import get_checkpoint_dir

ckpt_base_dir = get_checkpoint_dir()  # Or set CHECKPOINT_DIR env var
```

### Step 3: Use Data Utilities

**Before:**
```python
df = pl.concat([
    pl.read_parquet('data/train.parquet')
        .select([pl.col('id'), 'model_a', 'model_b', ...]),
    pl.read_parquet('data/lmsys-33k-deduplicated.parquet')
        .select([...])
])
```

**After:**
```python
from utils.data_utils import load_data

df = load_data(data_dir='data', include_lmsys_33k=True)
```

### Step 4: Replace Logging

**Before:**
```python
import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)
```

**After:**
```python
from utils.logging_utils import setup_logging, get_logger

logger = setup_logging(level=logging.INFO)
logger = get_logger(__name__)
```

## 🌍 Environment Variables

Create a `.env` file (or set environment variables) with:

```bash
# Checkpoint directory
CHECKPOINT_DIR=/path/to/checkpoints

# Data directory
DATA_DIR=data

# Weights & Biases project
WANDB_PROJECT=lmsys-chatbot-arena
```

## 📝 Optimized Scripts

### `process_data_optimized.py`
- Error handling for missing files
- Proper logging
- Modular functions
- Better error messages

### `llm_qlora_optimized.py`
- Uses all utility modules
- Proper error handling
- Better resource management
- Type hints

## 🚀 Benefits

1. **Maintainability**: Changes to common functionality only need to be made once
2. **Consistency**: All scripts use the same utilities, ensuring consistency
3. **Error Handling**: Proper error messages and validation
4. **Flexibility**: Easy to change paths and configurations via environment variables
5. **Debugging**: Better logging makes debugging easier
6. **Type Safety**: Type hints improve IDE support and catch errors early

## 📋 TODO: Migration Checklist

- [ ] Update all training scripts to use `utils/config.py`
- [ ] Replace hard-coded paths with environment variables
- [ ] Update data loading to use `utils/data_utils.py`
- [ ] Replace warning suppression with proper logging
- [ ] Add type hints to all functions
- [ ] Update documentation
- [ ] Add unit tests for utility functions

## 🔄 Backward Compatibility

The optimized scripts are provided as examples. Existing scripts continue to work as before. You can migrate gradually:

1. Start with new scripts using optimized utilities
2. Gradually migrate existing scripts
3. Keep old scripts until migration is complete

## 📚 Additional Resources

- See `llm_qlora_optimized.py` for a complete example
- See `process_data_optimized.py` for data processing example
- Check `utils/` directory for all available utilities

