# Code Output Prediction Training System

A comprehensive system for training open-source Large Language Models (LLMs) to predict code execution outputs using **supervised fine-tuning** and **reinforcement learning with verification rewards**.

## 🎯 Overview

This system implements a complete pipeline for training models to predict the exact output of code execution:

1. **📊 Data Generation**: Generates diverse code examples with test inputs and verified outputs
2. **🎯 Supervised Fine-tuning**: Trains models using standard supervised learning on code-output pairs  
3. **🚀 Reinforcement Learning**: Uses verification-based rewards to improve prediction accuracy
4. **📈 Evaluation**: Comprehensive evaluation metrics and model comparison tools

## ✨ Key Features

- **🤖 Multiple Model Support**: Pre-configured support for popular open-source models (CodeLlama, DeepSeek Coder, StarCoder2, Phi-2, Mistral)
- **🔧 Custom Model Support**: Easy integration of any HuggingFace model
- **💾 Memory Efficient**: 4-bit quantization and LoRA fine-tuning for reduced memory usage
- **📊 Dataset Management**: Structured dataset creation, storage, and management
- **⚡ Parallel Training**: Optimized training pipelines with checkpointing and resumption
- **📈 Comprehensive Evaluation**: Multiple metrics including exact match, fuzzy match, and reward-based evaluation
- **🔄 Complete Pipeline**: End-to-end automation from data generation to trained models

## ��️ Installation

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU with 8GB+ VRAM (recommended)
- **Memory**: 16GB+ RAM recommended
- **Storage**: 10GB+ free space for models and data

### Quick Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd code-output-prediction
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up OpenAI API key**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

5. **Verify installation**:
```bash
python validate_installation.py
```

### Alternative Installation

For systems with specific requirements:

```bash
# For CPU-only systems
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. List Available Models
```bash
python train.py list-models
```

### 2. Generate Training Data
```bash
python train.py generate-data --languages python javascript --num-samples 100
```

### 3. Run Complete Training Pipeline
```bash
python train.py full-pipeline --model phi-2 --languages python --num-samples 50
```

### 4. Evaluate Trained Models
```bash
python train.py evaluate --rl-path checkpoints/rl/final_model --dataset-name code_prediction_dataset
```

## 📋 Available Models

| Model | Size | Description | Memory (4-bit) |
|-------|------|-------------|----------------|
| `phi-2` | 2.7B | Microsoft Phi-2 - compact reasoning model | ~1.5GB |
| `codellama-7b` | 7B | Meta's Code Llama - excellent for code | ~3.5GB |
| `deepseek-coder-6.7b` | 6.7B | DeepSeek Coder - specialized for code | ~3.4GB |
| `starcoder2-7b` | 7B | StarCoder2 - diverse programming languages | ~3.5GB |
| `mistral-7b` | 7B | Mistral - excellent general reasoning | ~3.5GB |

## 📖 Usage Guide

### Data Generation

Generate training datasets with diverse code examples:

```bash
# Generate Python data
python train.py generate-data --languages python --num-samples 200

# Generate multi-language data
python train.py generate-data --languages python javascript rust --num-samples 100

# Custom dataset name
python train.py generate-data --languages python --num-samples 50 --dataset-name my_custom_dataset
```

### Supervised Fine-tuning

Train models using supervised learning on code-output pairs:

```bash
# Basic supervised training
python train.py train-supervised --model phi-2 --dataset-name my_dataset

# Custom hyperparameters
python train.py train-supervised \
    --model codellama-7b \
    --dataset-name my_dataset \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-4

# With experiment tracking
python train.py train-supervised --model phi-2 --use-wandb --run-name "phi2-experiment-1"
```

### Reinforcement Learning

Improve models using verification-based rewards:

```bash
# RL training from supervised checkpoint
python train.py train-rl \
    --model phi-2 \
    --dataset-name my_dataset \
    --base-model-path checkpoints/supervised \
    --episodes 1000

# RL from base model
python train.py train-rl --model phi-2 --dataset-name my_dataset --episodes 500
```

### Complete Pipeline

Run the entire training pipeline in one command:

```bash
# Full pipeline with default settings
python train.py full-pipeline --model phi-2 --languages python --num-samples 100

# Advanced pipeline configuration
python train.py full-pipeline \
    --model codellama-7b \
    --languages python javascript \
    --num-samples 200 \
    --epochs 3 \
    --rl-episodes 1000 \
    --batch-size 4 \
    --output-dir my_training_run \
    --run-name "codellama-multi-lang"

# Skip specific stages
python train.py full-pipeline --model phi-2 --skip-rl  # Only supervised training
python train.py full-pipeline --model phi-2 --skip-supervised  # Only RL training
```

### Custom Models

Use any HuggingFace model:

```bash
# Use custom model for training
python train.py train-supervised \
    --model custom \
    --custom-model-id microsoft/DialoGPT-medium \
    --dataset-name my_dataset
```

### Dataset Management

```bash
# List available datasets
python train.py list-datasets

# Export dataset to CSV for analysis
python -c "
from src.core.dataset_manager import DatasetManager
dm = DatasetManager()
dm.export_to_csv('my_dataset', 'train')
"
```

## 🏗️ System Architecture

```
├── src/
│   ├── core/                 # Core system components
│   │   ├── model_manager.py  # HuggingFace model management
│   │   ├── dataset_manager.py # Dataset creation and management
│   │   ├── verifier.py       # Output verification logic
│   │   └── language_factory.py # Multi-language support
│   ├── training/             # Training implementations
│   │   ├── supervised_trainer.py # Supervised fine-tuning
│   │   ├── rl_trainer.py     # Reinforcement learning
│   │   └── training_pipeline.py # Complete pipeline orchestration
│   ├── generators/           # Code generation
│   └── executors/           # Code execution
├── main.py                  # Code generation and execution system
├── train.py                # Comprehensive training CLI
└── validate_installation.py # Installation validation
```

## ⚙️ Configuration

### Training Configuration

Key hyperparameters can be configured:

**Supervised Training**:
- `epochs`: Number of training epochs (default: 3)
- `batch_size`: Training batch size (default: 4) 
- `learning_rate`: Learning rate (default: 2e-4)
- `lora_rank`: LoRA rank for parameter-efficient training (default: 64)

**RL Training**:
- `episodes`: Number of RL episodes (default: 500)
- `batch_size`: RL batch size (default: 4)
- `learning_rate`: RL learning rate (default: 1.4e-5)
- `exact_match_reward`: Reward for exact output match (default: 10.0)
- `close_match_reward`: Reward for close output match (default: 5.0)

### Memory Optimization

The system automatically applies memory optimizations:

- **4-bit Quantization**: Reduces memory usage by ~75%
- **LoRA Fine-tuning**: Only trains 0.1-1% of parameters
- **Gradient Checkpointing**: Trades compute for memory
- **Dynamic Batching**: Adjusts batch size based on available memory

## 📊 Evaluation Metrics

The system provides comprehensive evaluation:

- **Exact Accuracy**: Percentage of exactly correct predictions
- **Close Accuracy**: Percentage of approximately correct predictions (fuzzy matching)
- **Average Reward**: Mean reward score from verification system
- **Loss Metrics**: Standard training/validation loss

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**:
```bash
# Reduce batch size
python train.py train-supervised --batch-size 1

# Use smaller model
python train.py train-supervised --model phi-2
```

**2. Missing Dependencies**:
```bash
# Install missing packages
pip install torch transformers datasets accelerate peft bitsandbytes trl
```

**3. OpenAI API Errors**:
```bash
# Verify API key
echo $OPENAI_API_KEY

# Set API key
export OPENAI_API_KEY="your-key-here"
```

### Performance Tips

1. **Use GPU**: Training is significantly faster with CUDA
2. **Batch Size**: Start with smaller batch sizes and increase gradually
3. **Model Size**: Begin with smaller models (phi-2) for experimentation
4. **LoRA Parameters**: Lower rank reduces memory but may impact quality

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional programming languages
- More sophisticated reward functions
- Better evaluation metrics
- Memory optimization techniques
- Model architectures

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HuggingFace**: For the transformers library and model hosting
- **OpenAI**: For the API used in data generation
- **Meta**: For Code Llama models
- **Microsoft**: For Phi-2 model
- **DeepSeek**: For specialized code models

## 📚 Citation

If you use this system in your research, please cite:

```bibtex
@software{code_output_prediction,
  title={Code Output Prediction Training System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/code-output-prediction}
}
```

---

**Ready to train your own code prediction models? Start with the Quick Start guide above!** 🚀