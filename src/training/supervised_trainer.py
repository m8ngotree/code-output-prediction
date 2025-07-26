"""
Supervised Fine-tuning Trainer for Code Output Prediction

Trains models to predict code execution outputs using supervised learning
on generated datasets.
"""

import os
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import DatasetDict
from peft import PeftModel
import wandb

from ..core.model_manager import ModelManager
from ..core.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


@dataclass
class SupervisedTrainingConfig:
    """Configuration for supervised fine-tuning."""
    
    # Model configuration
    model_key: str = "phi-2"  # Default to smaller model
    custom_model_id: Optional[str] = None
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Training hyperparameters
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Training settings
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    logging_steps: int = 100
    max_length: int = 512
    
    # Output and monitoring
    output_dir: str = "checkpoints/supervised"
    run_name: Optional[str] = None
    use_wandb: bool = False
    early_stopping_patience: int = 3
    
    # Data configuration
    dataset_name: str = "code_prediction_dataset"
    test_size: float = 0.1
    dataset_strategy: str = "all"  # "all", "errors", "stratified"
    error_ratio: float = 0.7  # For stratified strategy
    evaluate_base_model: bool = False  # Enable error-focused training


class SupervisedTrainer:
    """Trainer for supervised fine-tuning on code output prediction."""
    
    def __init__(self, config: SupervisedTrainingConfig):
        """
        Initialize supervised trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model_manager = ModelManager(use_quantization=True)
        self.dataset_manager = DatasetManager()
        
        # Set up output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        
    def prepare_dataset_for_training(self, dataset: DatasetDict, tokenizer) -> DatasetDict:
        """
        Prepare dataset for training by tokenizing and formatting.
        
        Args:
            dataset: Raw dataset from DatasetManager
            tokenizer: Model tokenizer
            
        Returns:
            Tokenized dataset ready for training
        """
        logger.info("Preparing dataset for training...")
        
        def format_instruction(example):
            """Format example as instruction-following conversation."""
            # Create input-output pair
            conversation = example["conversation"]
            
            # Format as conversation
            formatted_text = ""
            for turn in conversation:
                if turn["role"] == "user":
                    formatted_text += f"### Instruction:\n{turn['content']}\n\n"
                elif turn["role"] == "assistant":
                    formatted_text += f"### Response:\n{turn['content']}\n"
            
            # Add end token
            formatted_text += tokenizer.eos_token
            
            return {"text": formatted_text}
        
        def tokenize_function(examples):
            """Tokenize examples for training."""
            # Tokenize the formatted text
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Format examples
        formatted_dataset = dataset.map(
            format_instruction,
            remove_columns=dataset["train"].column_names,
            desc="Formatting examples"
        )
        
        # Tokenize examples
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset["train"].column_names,
            desc="Tokenizing examples"
        )
        
        logger.info(f"Dataset prepared: {len(tokenized_dataset['train'])} train, {len(tokenized_dataset['validation'])} val examples")
        
        return tokenized_dataset
    
    def train(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run supervised fine-tuning.
        
        Args:
            dataset_name: Name of dataset to use (optional, uses config default)
            
        Returns:
            Training results and metrics
        """
        # Use provided dataset name or config default
        dataset_name = dataset_name or self.config.dataset_name
        
        logger.info(f"Starting supervised fine-tuning with dataset: {dataset_name}")
        
        # Load dataset
        try:
            dataset = self.dataset_manager.load_dataset(dataset_name)
            logger.info(f"Loaded dataset with {len(dataset['train'])} training examples")
        except FileNotFoundError:
            raise ValueError(f"Dataset '{dataset_name}' not found. Create it first using the dataset manager.")
        
        # Load model and tokenizer
        logger.info(f"Loading model: {self.config.model_key}")
        model, tokenizer = self.model_manager.load_model_and_tokenizer(
            self.config.model_key,
            self.config.custom_model_id
        )
        
        # Prepare model for training
        model = self.model_manager.prepare_model_for_training(
            model,
            use_lora=self.config.use_lora,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout
        )
        
        # Prepare dataset
        tokenized_dataset = self.prepare_dataset_for_training(dataset, tokenizer)
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8 if tokenizer.pad_token_id is not None else None
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            warmup_steps=self.config.warmup_steps,
            
            # Evaluation and saving
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Logging
            logging_steps=self.config.logging_steps,
            report_to="wandb" if self.config.use_wandb else None,
            run_name=self.config.run_name or f"supervised_{self.config.model_key}",
            
            # Memory optimization
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available(),
            
            # Misc
            remove_unused_columns=False,
            seed=42
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        )
        
        # Initialize wandb if requested
        if self.config.use_wandb:
            wandb.init(
                project="code-output-prediction",
                name=training_args.run_name,
                config=self.config.__dict__
            )
        
        logger.info("Starting training...")
        
        # Train the model
        train_result = trainer.train()
        
        # Save the final model
        trainer.save_model()
        trainer.save_state()
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
        
        # Prepare results summary
        results = {
            "train_results": train_result,
            "test_results": test_results,
            "model_path": str(self.output_dir),
            "config": self.config.__dict__,
            "dataset_info": self.dataset_manager.get_dataset_stats(dataset_name)
        }
        
        # Save results
        results_file = self.output_dir / "training_results.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training completed! Model saved to {self.output_dir}")
        logger.info(f"Final test loss: {test_results['eval_loss']:.4f}")
        
        # Clean up wandb
        if self.config.use_wandb:
            wandb.finish()
        
        return results
    
    def evaluate_model(self, model_path: str, dataset_name: str) -> Dict[str, float]:
        """
        Evaluate a trained model on a dataset.
        
        Args:
            model_path: Path to trained model
            dataset_name: Name of dataset to evaluate on
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating model from {model_path} on dataset {dataset_name}")
        
        # Load dataset
        dataset = self.dataset_manager.load_dataset(dataset_name)
        
        # Load base model and tokenizer
        model, tokenizer = self.model_manager.load_model_and_tokenizer(
            self.config.model_key,
            self.config.custom_model_id
        )
        
        # Load fine-tuned weights
        if self.config.use_lora:
            model = PeftModel.from_pretrained(model, model_path)
        else:
            # Load full model weights
            model.load_state_dict(torch.load(Path(model_path) / "pytorch_model.bin"))
        
        model.eval()
        
        # Prepare dataset
        tokenized_dataset = self.prepare_dataset_for_training(dataset, tokenizer)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create trainer for evaluation
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        
        # Evaluate on test set
        results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def create_error_focused_dataset(self, samples: List[Dict[str, Any]], 
                                   dataset_name: str,
                                   description: str = "") -> Dict[str, Any]:
        """
        Create error-focused dataset from samples using base model evaluation.
        
        Args:
            samples: Raw samples from code generation
            dataset_name: Name for the created dataset
            description: Dataset description
            
        Returns:
            Dataset creation results
        """
        logger.info(f"Creating error-focused dataset: {dataset_name}")
        
        base_model_predictions = None
        if self.config.dataset_strategy != "all" and self.config.evaluate_base_model:
            logger.info("Evaluating base model to identify errors...")
            
            # Load base model for evaluation
            model, tokenizer = self.model_manager.load_model_and_tokenizer(
                self.config.model_key,
                self.config.custom_model_id
            )
            
            # Evaluate base model on samples
            base_model_predictions = self.dataset_manager.evaluate_base_model(
                model, tokenizer, samples
            )
            
            logger.info(f"Base model evaluation complete")
        
        # Create dataset with specified strategy
        dataset = self.dataset_manager.create_dataset_from_samples(
            samples=samples,
            dataset_name=dataset_name,
            description=description,
            strategy=self.config.dataset_strategy,
            base_model_predictions=base_model_predictions,
            error_ratio=self.config.error_ratio
        )
        
        return {
            "dataset_name": dataset_name,
            "strategy": self.config.dataset_strategy,
            "total_examples": len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"]),
            "train_examples": len(dataset["train"])
        } 