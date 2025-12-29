# LMSYS Chatbot Arena - 3rd Place Solution

This repository contains the winning solution (3rd place) for the LMSYS Chatbot Arena competition. The solution implements a comprehensive two-stage training pipeline using multiple model architectures and training techniques for preference learning and response ranking.

## 🏆 Competition Overview

The LMSYS Chatbot Arena competition focuses on training models to predict which of two chatbot responses is preferred by human evaluators. The solution uses a combination of cross-encoder models, LLM-based reward models, and pseudo-labeling techniques to achieve high performance.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
- [Usage](#usage)
- [Model Configurations](#model-configurations)
- [Key Components](#key-components)

## ✨ Features

- **Multi-Model Ensemble**: Combines cross-encoders, LLM-based reward models, and bi-encoders
- **Two-Stage Training**: Stage 1 generates pseudo labels, Stage 2 trains on pseudo-labeled data
- **Pseudo-Labeling**: Leverages unlabeled data through iterative pseudo-labeling
- **Test-Time Augmentation (TTA)**: Improves robustness through response swapping
- **Multiple Training Methods**: Supports SFT, DPO, QLoRA, and reward model training
- **Efficient Training**: Uses QLoRA, gradient checkpointing, and mixed precision training
- **Out-of-Fold Optimization**: Ensemble optimization for better predictions

## 🏗️ Architecture

The solution employs multiple model architectures:

1. **Cross-Encoder Models**: DeBERTa-based models that encode prompt-response pairs together
2. **LLM-Based Reward Models**: Large language models (Gemma, Llama, Mistral, Qwen) fine-tuned with QLoRA for preference prediction
3. **Bi-Encoder Models**: Siamese networks for separate encoding of responses
4. **Sequence Models**: Transformer-based sequence classification models

### Supported Base Models

- **Gemma**: Gemma-2 variants (2B, 27B)
- **Llama**: Llama-3 variants (8B, 70B)
- **Mistral**: Mistral and Mistral-Nemo variants
- **Qwen**: Qwen2 variants
- **Others**: InternLM, Starling, Zephyr, Mathstral

## 🚀 Setup

### System Requirements

- **OS**: Ubuntu 22.04
- **GPUs**: 
  - Training: 2x4090s, 4-8x A100s/4090s/H100s (vast.ai), or 8xH100s (Lambda)
  - Minimum: 2x GPUs with 24GB+ VRAM for Stage 1
  - Recommended: 8x GPUs with 40GB+ VRAM for Stage 2
- **Python**: 3.10 (via conda)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ChatBot-Solution
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate unsloth
   ```

3. **Authenticate with Hugging Face** (required for downloading datasets):
   ```bash
   huggingface-cli login
   ```

4. **Prepare data directories**:
   ```bash
   mkdir -p data/preds data/pseudo
   ```

5. **Download competition data**:
   - Place `train.csv` and `test.csv` in the `data/` directory

## 📁 Project Structure

```
ChatBot-Solution/
├── configs/                 # YAML configuration files for different models
│   ├── gemma_rm.yaml
│   ├── llama_3.yaml
│   ├── pair_pref.yaml
│   └── ...
├── models/                  # Custom model implementations
│   ├── llm_model.py        # LLM-based reward model
│   ├── biencoder.py        # Bi-encoder architecture
│   ├── sw_transformer.py   # Sliding window transformer
│   └── ...
├── scripts/                 # Utility scripts
│   ├── generate.py         # Text generation script
│   ├── vllm_generate.py    # VLLM-based generation for pseudo-labeling
│   └── awq_quantize.py     # Model quantization
├── deepspeed/               # DeepSpeed configuration
│   └── zero2.json
├── ce_train.py             # Cross-encoder training
├── llm_qlora.py            # LLM QLoRA training
├── llm_validate.py         # Model validation/inference
├── llm_pseudo_label.py     # Pseudo-label generation
├── llm_train_pseudo.py     # Training on pseudo labels
├── dpo_train.py            # Direct Preference Optimization
├── sft_train.py            # Supervised Fine-Tuning
├── siamese_train.py        # Siamese network training
├── sequence_train.py       # Sequence model training
├── process_data.py         # Data preprocessing
├── oof_optimization.py     # Out-of-fold ensemble optimization
├── run_stage_1.sh          # Stage 1 training script
├── run_stage_2.sh          # Stage 2 training script
└── README.md
```

## 🔄 Training Pipeline

### Stage 1: Initial Training and Pseudo-Label Generation

1. **Process training data**:
   ```bash
   python process_data.py
   ```
   This script:
   - Processes competition data (`train.csv`, `test.csv`)
   - Downloads additional datasets from Hugging Face (lmsys-33k, orpo-dpo-mix-40k, etc.)
   - Creates stratified folds for cross-validation
   - Saves processed data as parquet files

2. **Generate paired completions for pseudo-labeling** (optional):
   ```bash
   python scripts/vllm_generate.py
   ```
   Generates paired completions for the lmsys-1m dataset.

3. **Run Stage 1 training**:
   ```bash
   bash run_stage_1.sh
   ```
   
   This script trains:
   - Pair preference models (`pair_pref.yaml`)
   - Gemma reward models (`gemma_rm.yaml`, `gemma_rm_no_cap.yaml`)
   - Generates pseudo labels using trained models
   - Performs out-of-fold optimization

### Stage 2: Training on Pseudo Labels

1. **Run Stage 2 training**:
   ```bash
   bash run_stage_2.sh
   ```
   
   This script:
   - Trains models on pseudo-labeled data
   - Uses configurations optimized for 8 GPUs
   - Generates final predictions

### GPU Configuration

**Stage 1** (2 GPUs):
- Effective batch size: 8
- Config: `batch_size: 4`, `accum: 2`

**Stage 2** (8 GPUs):
- Effective batch size: 8
- Config: `batch_size: 4`, `accum: 2` (per GPU)

To adjust for different GPU counts, modify the `batch_size` and `accum` parameters in the config files to maintain the same effective batch size.

## 💻 Usage

### Training a Single Model

Train a specific model using a configuration file:

```bash
accelerate launch llm_qlora.py -C configs/gemma_rm.yaml
```

### Validation/Inference

Run inference on a dataset:

```bash
# Standard inference
accelerate launch llm_validate.py -C configs/gemma_rm.yaml

# With test-time augmentation
accelerate launch llm_validate.py -C configs/gemma_rm.yaml --tta
```

### Pseudo-Label Generation

Generate pseudo labels using a trained model:

```bash
# Standard
accelerate launch llm_pseudo_label.py -C configs/gemma_rm.yaml

# With TTA
accelerate launch llm_pseudo_label.py -C configs/gemma_rm.yaml --tta
```

### Custom Dataset Inference

To run inference on your own dataset:

1. Replace `train.parquet` in `llm_validate.py` with your dataset
2. Ensure your dataset has the same structure (prompt, response_a, response_b columns)
3. Run validation:
   ```bash
   accelerate launch llm_validate.py -C configs/<your_config>.yaml
   ```

### Ensemble Predictions

For TTA-based ensemble:
1. Run validation with `--tta` flag
2. Run validation without `--tta` flag
3. Ensemble the saved prediction files using `oof_optimization.py`

## ⚙️ Model Configurations

Configuration files are located in `configs/` and specify:

- **Model**: Base model name from Hugging Face
- **Training parameters**: Learning rate, epochs, batch size, gradient accumulation
- **LoRA parameters**: Rank, alpha, target modules
- **Sequence length**: Max length for training and validation

Example configuration (`configs/gemma_rm.yaml`):

```yaml
model_name: sfairXC/FsfairX-Gemma2-RM-v0.1
exp_name: llm_surround_no_lstm

lr: 1.0e-4
epochs: 1
weight_decay: 0.01

lora_r: 64
lora_alpha: 16
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

training:
  batch_size: 4
  accum: 2
  max_length: 1800

validation:
  batch_size: 4
  accum: 1
  max_length: 8192
```

## 🔧 Key Components

### Training Scripts

- **`llm_qlora.py`**: QLoRA fine-tuning for LLM-based reward models
- **`ce_train.py`**: Cross-encoder training (DeBERTa-based)
- **`dpo_train.py`**: Direct Preference Optimization training
- **`sft_train.py`**: Supervised Fine-Tuning for language models
- **`siamese_train.py`**: Siamese network training for bi-encoders
- **`sequence_train.py`**: Sequence classification model training
- **`pseudo_ce_train.py`**: Cross-encoder training with pseudo labels

### Model Implementations

- **`models/llm_model.py`**: Custom LLM-based reward model wrapper
- **`models/biencoder.py`**: Bi-encoder architecture for separate encoding
- **`models/sw_transformer.py`**: Sliding window transformer for long sequences
- **`models/positional_embedding.py`**: Custom positional embeddings

### Data Processing

- **`process_data.py`**: Main data preprocessing pipeline
- **`process_ultrafeedback.py`**: UltraFeedback dataset processing
- **`pseudo_label.py`**: Pseudo-label generation utilities

### Optimization

- **`oof_optimization.py`**: Out-of-fold ensemble optimization
- **`generate_evaluation.py`**: Evaluation metric generation

## 📊 Training Techniques

### Pseudo-Labeling

The solution uses iterative pseudo-labeling:
1. Train initial models on labeled data
2. Generate predictions on unlabeled data
3. Use high-confidence predictions as pseudo labels
4. Retrain models on combined labeled + pseudo-labeled data

### Test-Time Augmentation (TTA)

TTA improves robustness by:
- Swapping response_a and response_b
- Averaging predictions from both configurations
- Reducing bias toward response ordering

### QLoRA Fine-Tuning

Efficient fine-tuning using:
- 4-bit quantization
- LoRA adapters (rank 64, alpha 16)
- Gradient checkpointing
- Mixed precision training (bf16)

## 📝 Notes

- All models use cross-validation with 4-5 folds
- Training uses Weights & Biases (wandb) for logging
- Checkpoints are saved to `/mnt/one/kaggle/lmsys-chatbot-arena/` (modify in scripts)
- The solution was trained on a combination of local and cloud GPUs

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This solution achieved 3rd place in the LMSYS Chatbot Arena competition. The implementation leverages multiple open-source models and datasets from the Hugging Face ecosystem.
