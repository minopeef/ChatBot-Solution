# 🏆 LMSYS Chatbot Arena - 3rd Place Solution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.43.1-green.svg)](https://huggingface.co/docs/transformers)

> 🎯 **Winning solution** (3rd place) for the [LMSYS Chatbot Arena](https://chat.lmsys.org/?arena) competition. This repository implements a comprehensive two-stage training pipeline using multiple model architectures and training techniques for preference learning and response ranking.

![Architecture Overview](docs/images/architecture.png)
*Figure 1: High-level architecture of the solution*

## 📋 Table of Contents

- [🏆 Competition Overview](#-competition-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Setup](#-setup)
- [📁 Project Structure](#-project-structure)
- [🔄 Training Pipeline](#-training-pipeline)
- [💻 Usage](#-usage)
- [⚙️ Model Configurations](#️-model-configurations)
- [🔧 Key Components](#-key-components)
- [📊 Training Techniques](#-training-techniques)
- [📝 Notes](#-notes)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

## 🏆 Competition Overview

The [LMSYS Chatbot Arena](https://chat.lmsys.org/?arena) competition focuses on training models to predict which of two chatbot responses is preferred by human evaluators. The solution uses a combination of cross-encoder models, LLM-based reward models, and pseudo-labeling techniques to achieve high performance.

**Competition Links:**
- 🌐 [LMSYS Chatbot Arena](https://chat.lmsys.org/?arena)
- 📊 [Competition Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
- 📚 [LMSYS Research](https://lmsys.org/)

## ✨ Features

- 🔀 **Multi-Model Ensemble**: Combines cross-encoders, LLM-based reward models, and bi-encoders
- 🔄 **Two-Stage Training**: Stage 1 generates pseudo labels, Stage 2 trains on pseudo-labeled data
- 🏷️ **Pseudo-Labeling**: Leverages unlabeled data through iterative pseudo-labeling
- 🔁 **Test-Time Augmentation (TTA)**: Improves robustness through response swapping
- 🎓 **Multiple Training Methods**: Supports SFT, DPO, QLoRA, and reward model training
- ⚡ **Efficient Training**: Uses QLoRA, gradient checkpointing, and mixed precision training
- 📈 **Out-of-Fold Optimization**: Ensemble optimization for better predictions
- 🎯 **High Performance**: Achieved 3rd place in the competition

## 🏗️ Architecture

![Training Pipeline](docs/images/training_pipeline.png)
*Figure 2: Two-stage training pipeline*

The solution employs multiple model architectures:

1. **🔗 Cross-Encoder Models**: [DeBERTa](https://huggingface.co/microsoft/deberta-v3-large)-based models that encode prompt-response pairs together
2. **🤖 LLM-Based Reward Models**: Large language models ([Gemma](https://huggingface.co/google/gemma-2-27b-it), [Llama](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [Qwen](https://huggingface.co/Qwen/Qwen2-7B-Instruct)) fine-tuned with QLoRA for preference prediction
3. **🔀 Bi-Encoder Models**: Siamese networks for separate encoding of responses
4. **📊 Sequence Models**: Transformer-based sequence classification models

### Supported Base Models

| Model Family | Variants | Hugging Face Links |
|-------------|----------|-------------------|
| **🟢 Gemma** | Gemma-2 (2B, 27B) | [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) |
| **🦙 Llama** | Llama-3 (8B, 70B) | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| **🌪️ Mistral** | Mistral, Mistral-Nemo | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| **🔷 Qwen** | Qwen2 variants | [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) |
| **🔵 Others** | InternLM, Starling, Zephyr, Mathstral | [InternLM](https://huggingface.co/internlm), [Starling](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha) |

## 🚀 Setup

### 📋 System Requirements

- **🖥️ OS**: Ubuntu 22.04
- **🎮 GPUs**: 
  - Training: 2x4090s, 4-8x A100s/4090s/H100s ([vast.ai](https://vast.ai)), or 8xH100s ([Lambda](https://lambdalabs.com))
  - Minimum: 2x GPUs with 24GB+ VRAM for Stage 1
  - Recommended: 8x GPUs with 40GB+ VRAM for Stage 2
- **🐍 Python**: 3.10 (via conda)
- **📦 Dependencies**: See [environment.yml](environment.yml)

### 🔧 Installation

1. **📥 Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ChatBot-Solution
   ```

2. **🌐 Create conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate unsloth
   ```

3. **🔐 Authenticate with Hugging Face** (required for downloading datasets):
   ```bash
   huggingface-cli login
   ```
   > 💡 Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)

4. **📁 Prepare data directories**:
   ```bash
   mkdir -p data/preds data/pseudo
   ```

5. **📊 Download competition data**:
   - Place `train.csv` and `test.csv` in the `data/` directory
   - Data format: CSV files with columns `prompt`, `response_a`, `response_b`, `winner_model_a`, `winner_model_b`

## 📁 Project Structure

```
ChatBot-Solution/
├── 📂 configs/                 # YAML configuration files for different models
│   ├── gemma_rm.yaml           # Gemma reward model config
│   ├── llama_3.yaml            # Llama-3 config
│   ├── pair_pref.yaml          # Pair preference model config
│   └── ...
├── 📂 models/                  # Custom model implementations
│   ├── llm_model.py           # LLM-based reward model
│   ├── biencoder.py           # Bi-encoder architecture
│   ├── sw_transformer.py      # Sliding window transformer
│   └── ...
├── 📂 scripts/                 # Utility scripts
│   ├── generate.py            # Text generation script
│   ├── vllm_generate.py       # VLLM-based generation for pseudo-labeling
│   └── awq_quantize.py        # Model quantization
├── 📂 deepspeed/               # DeepSpeed configuration
│   └── zero2.json
├── 📂 data/                    # Data directory
│   ├── train.csv              # Training data
│   ├── test.csv               # Test data
│   ├── preds/                 # Predictions output
│   └── pseudo/                # Pseudo labels
├── 🐍 ce_train.py             # Cross-encoder training
├── 🐍 llm_qlora.py            # LLM QLoRA training
├── 🐍 llm_validate.py         # Model validation/inference
├── 🐍 llm_pseudo_label.py     # Pseudo-label generation
├── 🐍 llm_train_pseudo.py     # Training on pseudo labels
├── 🐍 dpo_train.py            # Direct Preference Optimization
├── 🐍 sft_train.py            # Supervised Fine-Tuning
├── 🐍 siamese_train.py        # Siamese network training
├── 🐍 sequence_train.py       # Sequence model training
├── 🐍 process_data.py         # Data preprocessing
├── 🐍 oof_optimization.py     # Out-of-fold ensemble optimization
├── 📜 run_stage_1.sh          # Stage 1 training script
├── 📜 run_stage_2.sh          # Stage 2 training script
├── 📄 environment.yml         # Conda environment file
├── 📄 LICENSE                 # MIT License
└── 📖 README.md               # This file
```

## 🔄 Training Pipeline

![Training Flow](docs/images/training_flow.png)
*Figure 3: Detailed training flow diagram*

### 📍 Stage 1: Initial Training and Pseudo-Label Generation

1. **📊 Process training data**:
   ```bash
   python process_data.py
   ```
   This script:
   - ✅ Processes competition data (`train.csv`, `test.csv`)
   - 📥 Downloads additional datasets from Hugging Face:
     - [lmsys-33k-deduplicated](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
     - [orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)
   - 🔀 Creates stratified folds for cross-validation
   - 💾 Saves processed data as parquet files

2. **🎯 Generate paired completions for pseudo-labeling** (optional):
   ```bash
   python scripts/vllm_generate.py
   ```
   Generates paired completions for the [lmsys-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) dataset.

3. **🚀 Run Stage 1 training**:
   ```bash
   bash run_stage_1.sh
   ```
   
   This script trains:
   - 🔗 Pair preference models (`pair_pref.yaml`)
   - 🟢 Gemma reward models (`gemma_rm.yaml`, `gemma_rm_no_cap.yaml`)
   - 🏷️ Generates pseudo labels using trained models
   - 📈 Performs out-of-fold optimization

### 📍 Stage 2: Training on Pseudo Labels

1. **🚀 Run Stage 2 training**:
   ```bash
   bash run_stage_2.sh
   ```
   
   This script:
   - 🎓 Trains models on pseudo-labeled data
   - 🎮 Uses configurations optimized for 8 GPUs
   - 📊 Generates final predictions

### 🎮 GPU Configuration

| Stage | GPUs | Effective Batch Size | Config |
|-------|------|---------------------|--------|
| **Stage 1** | 2 GPUs | 8 | `batch_size: 4`, `accum: 2` |
| **Stage 2** | 8 GPUs | 8 | `batch_size: 4`, `accum: 2` (per GPU) |

> 💡 **Tip**: To adjust for different GPU counts, modify the `batch_size` and `accum` parameters in the config files to maintain the same effective batch size.

## 💻 Usage

### 🎓 Training a Single Model

Train a specific model using a configuration file:

```bash
accelerate launch llm_qlora.py -C configs/gemma_rm.yaml
```

> 📚 Learn more about [Accelerate](https://huggingface.co/docs/accelerate) for distributed training

### ✅ Validation/Inference

Run inference on a dataset:

```bash
# Standard inference
accelerate launch llm_validate.py -C configs/gemma_rm.yaml

# With test-time augmentation
accelerate launch llm_validate.py -C configs/gemma_rm.yaml --tta
```

### 🏷️ Pseudo-Label Generation

Generate pseudo labels using a trained model:

```bash
# Standard
accelerate launch llm_pseudo_label.py -C configs/gemma_rm.yaml

# With TTA
accelerate launch llm_pseudo_label.py -C configs/gemma_rm.yaml --tta
```

### 📊 Custom Dataset Inference

To run inference on your own dataset:

1. Replace `train.parquet` in `llm_validate.py` with your dataset
2. Ensure your dataset has the same structure (prompt, response_a, response_b columns)
3. Run validation:
   ```bash
   accelerate launch llm_validate.py -C configs/<your_config>.yaml
   ```

### 🎯 Ensemble Predictions

For TTA-based ensemble:
1. Run validation with `--tta` flag
2. Run validation without `--tta` flag
3. Ensemble the saved prediction files using `oof_optimization.py`

## ⚙️ Model Configurations

Configuration files are located in `configs/` and specify:

- **🤖 Model**: Base model name from Hugging Face
- **📊 Training parameters**: Learning rate, epochs, batch size, gradient accumulation
- **🔧 LoRA parameters**: Rank, alpha, target modules
- **📏 Sequence length**: Max length for training and validation

### 📝 Example Configuration

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

> 📖 See all available configurations in the [`configs/`](configs/) directory

## 🔧 Key Components

### 🎓 Training Scripts

| Script | Description | Documentation |
|--------|-------------|--------------|
| **`llm_qlora.py`** | QLoRA fine-tuning for LLM-based reward models | [QLoRA Paper](https://arxiv.org/abs/2305.14314) |
| **`ce_train.py`** | Cross-encoder training (DeBERTa-based) | [DeBERTa Paper](https://arxiv.org/abs/2006.03654) |
| **`dpo_train.py`** | Direct Preference Optimization training | [DPO Paper](https://arxiv.org/abs/2305.18290) |
| **`sft_train.py`** | Supervised Fine-Tuning for language models | - |
| **`siamese_train.py`** | Siamese network training for bi-encoders | - |
| **`sequence_train.py`** | Sequence classification model training | - |
| **`pseudo_ce_train.py`** | Cross-encoder training with pseudo labels | - |

### 🏗️ Model Implementations

- **`models/llm_model.py`**: Custom LLM-based reward model wrapper
- **`models/biencoder.py`**: Bi-encoder architecture for separate encoding
- **`models/sw_transformer.py`**: Sliding window transformer for long sequences
- **`models/positional_embedding.py`**: Custom positional embeddings

### 📊 Data Processing

- **`process_data.py`**: Main data preprocessing pipeline
- **`process_ultrafeedback.py`**: [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) dataset processing
- **`pseudo_label.py`**: Pseudo-label generation utilities

### 🎯 Optimization

- **`oof_optimization.py`**: Out-of-fold ensemble optimization
- **`generate_evaluation.py`**: Evaluation metric generation

## 📊 Training Techniques

### 🏷️ Pseudo-Labeling

![Pseudo-Labeling Process](docs/images/pseudo_labeling.png)
*Figure 4: Pseudo-labeling workflow*

The solution uses iterative pseudo-labeling:
1. 🎓 Train initial models on labeled data
2. 🔮 Generate predictions on unlabeled data
3. ✅ Use high-confidence predictions as pseudo labels
4. 🔄 Retrain models on combined labeled + pseudo-labeled data

### 🔁 Test-Time Augmentation (TTA)

TTA improves robustness by:
- 🔀 Swapping response_a and response_b
- 📊 Averaging predictions from both configurations
- 🎯 Reducing bias toward response ordering

### ⚡ QLoRA Fine-Tuning

Efficient fine-tuning using:
- 🔢 4-bit quantization ([BitsAndBytes](https://github.com/TimDettmers/bitsandbytes))
- 🔧 LoRA adapters (rank 64, alpha 16) ([PEFT](https://github.com/huggingface/peft))
- 💾 Gradient checkpointing
- 🎨 Mixed precision training (bf16)

> 📚 Learn more: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

## 📝 Notes

- ✅ All models use cross-validation with 4-5 folds
- 📊 Training uses [Weights & Biases](https://wandb.ai/) (wandb) for logging
- 💾 Checkpoints are saved to `/mnt/one/kaggle/lmsys-chatbot-arena/` (modify in scripts)
- ☁️ The solution was trained on a combination of local and cloud GPUs
- 🔧 Modify paths in scripts to match your setup

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🙏 Acknowledgments

- 🏆 This solution achieved **3rd place** in the [LMSYS Chatbot Arena](https://chat.lmsys.org/?arena) competition
- 🤗 The implementation leverages multiple open-source models and datasets from the [Hugging Face](https://huggingface.co/) ecosystem
- 📚 Special thanks to:
  - [LMSYS](https://lmsys.org/) for organizing the competition
  - [Hugging Face](https://huggingface.co/) for model hosting and datasets
  - [Unsloth](https://github.com/unslothai/unsloth) for efficient training utilities
  - [VLLM](https://github.com/vllm-project/vllm) for fast inference
  - [Accelerate](https://github.com/huggingface/accelerate) for distributed training

## 🔗 Useful Links

- 🌐 [LMSYS Chatbot Arena](https://chat.lmsys.org/?arena)
- 📊 [Competition Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
- 🤗 [Hugging Face Models](https://huggingface.co/models)
- 📚 [Transformers Documentation](https://huggingface.co/docs/transformers)
- 🚀 [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- 🔧 [PEFT Documentation](https://huggingface.co/docs/peft)
- ⚡ [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- 🎯 [DPO Paper](https://arxiv.org/abs/2305.18290)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for the ML community

</div>
