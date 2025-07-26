"""
HuggingFace Model Manager

Handles loading, configuring, and managing open-source LLMs from HuggingFace
for code output prediction training.
"""

from typing import Dict, List, Optional, Tuple
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages HuggingFace model selection, loading, and configuration."""
    
    # Curated list of proven open-source models for code understanding
    RECOMMENDED_MODELS = {
        "codellama-7b": {
            "model_id": "codellama/CodeLlama-7b-Instruct-hf",
            "description": "Meta's Code Llama 7B - excellent for code understanding",
            "context_length": 4096,
            "recommended_batch_size": 4
        },
        "deepseek-coder-6.7b": {
            "model_id": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "description": "DeepSeek Coder 6.7B - specialized for code tasks",
            "context_length": 4096,
            "recommended_batch_size": 4
        },
        "starcoder2-7b": {
            "model_id": "bigcode/starcoder2-7b",
            "description": "StarCoder2 7B - trained on diverse programming languages",
            "context_length": 4096,
            "recommended_batch_size": 4
        },
        "phi-2": {
            "model_id": "microsoft/phi-2",
            "description": "Microsoft Phi-2 - compact but capable reasoning model",
            "context_length": 2048,
            "recommended_batch_size": 8
        },
        "mistral-7b": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "description": "Mistral 7B - excellent general reasoning capabilities",
            "context_length": 4096,
            "recommended_batch_size": 4
        }
    }
    
    def __init__(self, use_quantization: bool = True):
        """
        Initialize model manager.
        
        Args:
            use_quantization: Whether to use 4-bit quantization for memory efficiency
        """
        self.use_quantization = use_quantization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_models = {}  # Cache for loaded models
        
    def list_available_models(self) -> Dict[str, Dict]:
        """Get list of recommended models with descriptions."""
        return self.RECOMMENDED_MODELS.copy()
    
    def load_model_and_tokenizer(
        self, 
        model_key: str, 
        custom_model_id: Optional[str] = None
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer with optimal configuration.
        
        Args:
            model_key: Key from RECOMMENDED_MODELS or 'custom'
            custom_model_id: HuggingFace model ID if using custom model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_key == "custom" and custom_model_id:
            model_id = custom_model_id
            logger.info(f"Loading custom model: {model_id}")
        elif model_key in self.RECOMMENDED_MODELS:
            model_id = self.RECOMMENDED_MODELS[model_key]["model_id"]
            logger.info(f"Loading recommended model: {model_key}")
        else:
            raise ValueError(f"Unknown model key: {model_key}")
        
        # Check cache first
        cache_key = f"{model_key}_{model_id}"
        if cache_key in self.loaded_models:
            logger.info("Using cached model")
            return self.loaded_models[cache_key]
        
        # Configure quantization for memory efficiency
        quantization_config = None
        if self.use_quantization and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        
        # Cache the loaded model
        self.loaded_models[cache_key] = (model, tokenizer)
        
        logger.info(f"Successfully loaded model: {model_id}")
        return model, tokenizer
    
    def prepare_model_for_training(
        self, 
        model: AutoModelForCausalLM,
        use_lora: bool = True,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1
    ) -> AutoModelForCausalLM:
        """
        Prepare model for efficient fine-tuning using LoRA.
        
        Args:
            model: Base model to prepare
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_rank: LoRA rank (lower = fewer parameters)
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            
        Returns:
            Model prepared for training
        """
        if not use_lora:
            return model
        
        # Prepare model for k-bit training if quantized
        if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
        return model
    
    def get_model_info(self, model_key: str) -> Dict:
        """Get detailed information about a model."""
        if model_key in self.RECOMMENDED_MODELS:
            return self.RECOMMENDED_MODELS[model_key].copy()
        else:
            return {"model_id": model_key, "description": "Custom model"}
    
    def estimate_memory_usage(self, model_key: str) -> Dict[str, str]:
        """Estimate memory requirements for model."""
        if model_key not in self.RECOMMENDED_MODELS:
            return {"error": "Cannot estimate memory for custom model"}
        
        # Rough estimates based on model size and quantization
        base_size_gb = 7 if "7b" in model_key or "6.7b" in model_key else 3
        
        estimates = {
            "full_precision": f"~{base_size_gb * 4:.1f}GB",
            "fp16": f"~{base_size_gb * 2:.1f}GB", 
            "4bit_quantized": f"~{base_size_gb * 0.5:.1f}GB",
            "recommended": "4bit_quantized for training, fp16 for inference"
        }
        
        return estimates 