# Quick Start Guide

Get up and running with code output prediction training in under 10 minutes.

## Prerequisites

1. **Python 3.8+** with pip
2. **CUDA GPU** (recommended, 8GB+ VRAM)  
3. **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
4. **16GB+ RAM** and **10GB+ free storage**

## Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd code-output-prediction

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies  
pip install -r requirements.txt

# 4. Set API key
export OPENAI_API_KEY="your-api-key-here"

# 5. Verify installation
python validate_installation.py
```

## Essential Commands

### 1. See Available Models
```bash
python train.py list-models
```

### 2. Quick Full Pipeline (Recommended)
```bash
# Train Phi-2 model with Python code (small, fast)
python train.py full-pipeline --model phi-2 --languages python --num-samples 20

# Train CodeLlama with multiple languages (larger, better)
python train.py full-pipeline --model codellama-7b --languages python javascript --num-samples 50
```

### 3. Step-by-Step Training

**Generate Data:**
```bash
python train.py generate-data --languages python --num-samples 50
```

**Supervised Training:**
```bash  
python train.py train-supervised --model phi-2 --dataset-name code_prediction_dataset
```

**RL Training:**
```bash
python train.py train-rl --model phi-2 --base-model-path checkpoints/supervised --dataset-name code_prediction_dataset
```

### 4. Evaluate Models
```bash
python train.py evaluate --rl-path checkpoints/rl/final_model --dataset-name code_prediction_dataset
```

## Command Quick Reference

| Command | Purpose | Time | GPU Memory |
|---------|---------|------|------------|
| `list-models` | Show available models | Instant | None |
| `generate-data` | Create training data | 5-10 min | None |
| `train-supervised` | Train with supervision | 10-30 min | 4-8GB |
| `train-rl` | Train with RL | 30-60 min | 4-8GB |
| `full-pipeline` | Complete training | 45-90 min | 4-8GB |

## Configuration Tips

**For Limited GPU Memory (<8GB):**
```bash
python train.py train-supervised --model phi-2 --batch-size 1
```

**For CPU-Only Training:**
```bash
# Use smallest model and batch size
python train.py train-supervised --model phi-2 --batch-size 1 --epochs 1
```

**For Production Training:**
```bash
python train.py full-pipeline \
    --model codellama-7b \
    --languages python javascript \
    --num-samples 200 \
    --epochs 3 \
    --rl-episodes 1000 \
    --run-name "production-run"
```

## Expected Results

After training, you should see:

- **Supervised Model**: ~2-4 test loss (lower is better)
- **RL Model**: ~60-80% exact accuracy on test set
- **Training Time**: 30-90 minutes for full pipeline
- **Model Size**: ~3-7GB (4-bit quantized)

## ❓ Common Issues & Solutions

**CUDA Out of Memory:**
```bash
# Reduce batch size
--batch-size 1

# Use smaller model
--model phi-2
```

**OpenAI API Errors:**
```bash
# Check API key
echo $OPENAI_API_KEY

# Check usage limits at platform.openai.com
```

**Import Errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.8+
```

**Slow Training:**
```bash
# Verify GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi
```

## Next Steps

1. **Start Small**: Begin with `phi-2` model and 20 samples
2. **Scale Up**: Move to `codellama-7b` with more data  
3. **Experiment**: Try different languages and hyperparameters
4. **Custom Models**: Use `--model custom --custom-model-id YOUR_MODEL`

## 📚 Full Documentation

- **Complete README**: See [README.md](README.md) for detailed documentation
- **Examples**: Check [examples/](examples/) for code samples
- **Help**: Run `python train.py --help` for all options

---

**Ready to train your first model?** Start with:
```bash
python train.py full-pipeline --model phi-2 --languages python --num-samples 20
``` 