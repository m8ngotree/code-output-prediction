#!/usr/bin/env python3
"""
Installation Validation Script

Run this script to verify that all dependencies are properly installed
and the system is ready for training.
"""

import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("accelerate", "HuggingFace Accelerate"),
        ("peft", "Parameter-Efficient Fine-Tuning"),
        ("bitsandbytes", "BitsAndBytes"),
        ("trl", "Transformer Reinforcement Learning"),
        ("openai", "OpenAI API"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("tqdm", "Progress Bars"),
        ("yaml", "PyYAML")
    ]
    
    all_installed = True
    
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {description} - Installed")
        except ImportError:
            print(f"❌ {description} - Missing")
            all_installed = False
    
    return all_installed

def check_gpu():
    """Check GPU availability and CUDA support."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ CUDA GPU Available: {gpu_name} ({memory_gb:.1f}GB)")
            if memory_gb >= 8:
                print("✅ GPU Memory: Sufficient for training")
            else:
                print("⚠️  GPU Memory: Limited, recommend batch_size=1")
            return True
        else:
            print("⚠️  GPU: CUDA not available, will use CPU (slower)")
            return False
    except Exception as e:
        print(f"❌ GPU Check Failed: {e}")
        return False

def check_openai_key():
    """Check if OpenAI API key is set."""
    import os
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI API Key: Set")
        return True
    else:
        print("⚠️  OpenAI API Key: Not set (required for data generation)")
        return False

def check_project_structure():
    """Check if project structure is correct."""
    required_files = [
        "train.py",
        "main.py", 
        "requirements.txt",
        "src/core/model_manager.py",
        "src/training/supervised_trainer.py",
        "src/training/rl_trainer.py"
    ]
    
    all_present = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - Present")
        else:
            print(f"❌ {file_path} - Missing")
            all_present = False
    
    return all_present

def main():
    """Run all validation checks."""
    print("🔍 Code Output Prediction System - Installation Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("GPU Support", check_gpu()),
        ("OpenAI API Key", check_openai_key()),
        ("Project Structure", check_project_structure())
    ]
    
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 All checks passed! System is ready for training.")
        print("\nNext steps:")
        print("1. python train.py list-models")
        print("2. python train.py full-pipeline --model phi-2 --num-samples 10")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. pip install -r requirements.txt")
        print("2. export OPENAI_API_KEY='your-key-here'")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 