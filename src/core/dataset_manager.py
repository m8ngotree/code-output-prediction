"""
Dataset Manager for Code Output Prediction

Handles storage, loading, and management of training datasets generated
from code execution examples.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
import logging

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages code-output prediction datasets."""
    
    def __init__(self, data_dir: str = "data/datasets"):
        """
        Initialize dataset manager.
        
        Args:
            data_dir: Directory to store datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset metadata tracking
        self.metadata_file = self.data_dir / "dataset_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"datasets": {}, "last_updated": None}
    
    def _save_metadata(self):
        """Save dataset metadata to file."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def create_dataset_from_samples(
        self, 
        samples: List[Dict[str, Any]], 
        dataset_name: str,
        description: str = "",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        strategy: str = "all",
        base_model_predictions: Optional[List[Dict[str, Any]]] = None,
        error_ratio: float = 0.7
    ) -> DatasetDict:
        """
        Convert code execution samples into a structured training dataset.
        
        Args:
            samples: List of code execution samples from CodePredictionSystem
            dataset_name: Name for the dataset
            description: Description of the dataset
            split_ratios: Train/validation/test split ratios
            strategy: Dataset creation strategy ("all", "errors", "stratified")
            base_model_predictions: Predictions from base model for error filtering
            error_ratio: Ratio of error examples in stratified strategy
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        logger.info(f"Creating dataset '{dataset_name}' from {len(samples)} samples")
        
        # Extract training examples from samples
        training_examples = []
        
        for sample in samples:
            if not sample.get("success", False):
                continue
                
            generation = sample["generation"]
            execution_results = sample["execution_results"]
            
            for exec_result in execution_results:
                if not exec_result.get("success", False):
                    continue
                
                # Create training example
                example = {
                    "code": generation["code"],
                    "language": generation["language"],
                    "input": exec_result.get("input_used", ""),
                    "expected_output": exec_result["output"],
                    "concept": generation.get("concept", ""),
                    "application": generation.get("application", ""),
                    "test_description": exec_result.get("test_description", ""),
                    
                    # Create instruction prompt for training
                    "instruction": self._create_instruction_prompt(
                        generation["code"],
                        generation["language"], 
                        exec_result.get("input_used", "")
                    ),
                    
                    # Full conversation format for training
                    "conversation": [
                        {
                            "role": "user",
                            "content": self._create_instruction_prompt(
                                generation["code"],
                                generation["language"],
                                exec_result.get("input_used", "")
                            )
                        },
                        {
                            "role": "assistant", 
                            "content": exec_result["output"]
                        }
                    ],
                    
                    # Metadata
                    "sample_id": self._generate_sample_id(generation["code"], exec_result.get("input_used", "")),
                    "created_at": datetime.now().isoformat()
                }
                
                training_examples.append(example)
        
        logger.info(f"Extracted {len(training_examples)} training examples")
        
        if not training_examples:
            raise ValueError("No valid training examples found in samples")
        
        # Apply dataset creation strategy
        if strategy != "all" and base_model_predictions:
            training_examples = self._apply_strategy(training_examples, base_model_predictions, strategy, error_ratio)
            logger.info(f"After {strategy} strategy: {len(training_examples)} examples")
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_list(training_examples)
        
        # Split into train/val/test
        train_size = int(len(training_examples) * split_ratios[0])
        val_size = int(len(training_examples) * split_ratios[1])
        
        # Shuffle and split
        dataset = dataset.shuffle(seed=42)
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, len(training_examples)))
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
        # Save dataset
        dataset_path = self.data_dir / dataset_name
        dataset_dict.save_to_disk(str(dataset_path))
        
        # Update metadata
        self.metadata["datasets"][dataset_name] = {
            "description": description,
            "num_examples": len(training_examples),
            "splits": {
                "train": len(train_dataset),
                "validation": len(val_dataset),
                "test": len(test_dataset)
            },
            "languages": list(set(ex["language"] for ex in training_examples)),
            "concepts": list(set(ex["concept"] for ex in training_examples)),
            "created_at": datetime.now().isoformat(),
            "path": str(dataset_path)
        }
        self._save_metadata()
        
        logger.info(f"Dataset saved to {dataset_path}")
        return dataset_dict
    
    def _create_instruction_prompt(self, code: str, language: str, input_data: str) -> str:
        """Create standardized instruction prompt for training."""
        prompt = f"""Predict the exact output of this {language} code.

CODE:
```{language}
{code}
```

INPUT: {input_data if input_data else "No input provided"}

Output:"""
        return prompt
    
    def _generate_sample_id(self, code: str, input_data: str) -> str:
        """Generate unique ID for a code-input pair."""
        content = f"{code}|{input_data}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def load_dataset(self, dataset_name: str) -> DatasetDict:
        """Load an existing dataset."""
        dataset_path = self.data_dir / dataset_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found at {dataset_path}")
        
        logger.info(f"Loading dataset '{dataset_name}' from {dataset_path}")
        return load_from_disk(str(dataset_path))
    
    def list_datasets(self) -> Dict[str, Dict]:
        """List all available datasets with metadata."""
        return self.metadata["datasets"].copy()
    
    def combine_datasets(
        self, 
        dataset_names: List[str], 
        new_dataset_name: str,
        description: str = "Combined dataset"
    ) -> DatasetDict:
        """
        Combine multiple datasets into one.
        
        Args:
            dataset_names: List of dataset names to combine
            new_dataset_name: Name for the combined dataset
            description: Description for the combined dataset
            
        Returns:
            Combined DatasetDict
        """
        logger.info(f"Combining datasets: {dataset_names}")
        
        all_datasets = []
        total_examples = 0
        
        for name in dataset_names:
            dataset = self.load_dataset(name)
            # Combine all splits
            combined = concatenate_datasets([
                dataset["train"], 
                dataset["validation"], 
                dataset["test"]
            ])
            all_datasets.append(combined)
            total_examples += len(combined)
        
        # Combine all datasets
        combined_dataset = concatenate_datasets(all_datasets)
        
        # Re-split the combined dataset
        train_size = int(total_examples * 0.8)
        val_size = int(total_examples * 0.1)
        
        combined_dataset = combined_dataset.shuffle(seed=42)
        train_dataset = combined_dataset.select(range(train_size))
        val_dataset = combined_dataset.select(range(train_size, train_size + val_size))
        test_dataset = combined_dataset.select(range(train_size + val_size, total_examples))
        
        new_dataset_dict = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
        # Save combined dataset
        dataset_path = self.data_dir / new_dataset_name
        new_dataset_dict.save_to_disk(str(dataset_path))
        
        # Update metadata
        self.metadata["datasets"][new_dataset_name] = {
            "description": f"{description} (Combined from: {', '.join(dataset_names)})",
            "num_examples": total_examples,
            "splits": {
                "train": len(train_dataset),
                "validation": len(val_dataset),
                "test": len(test_dataset)
            },
            "combined_from": dataset_names,
            "created_at": datetime.now().isoformat(),
            "path": str(dataset_path)
        }
        self._save_metadata()
        
        logger.info(f"Combined dataset saved as '{new_dataset_name}' with {total_examples} examples")
        return new_dataset_dict
    
    def get_dataset_stats(self, dataset_name: str) -> Dict:
        """Get detailed statistics for a dataset."""
        dataset = self.load_dataset(dataset_name)
        
        # Analyze the training split
        train_data = dataset["train"]
        
        stats = {
            "total_examples": len(train_data),
            "languages": {},
            "concepts": {},
            "avg_code_length": 0,
            "avg_output_length": 0,
        }
        
        # Collect statistics
        code_lengths = []
        output_lengths = []
        
        for example in train_data:
            # Language distribution
            lang = example["language"]
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
            
            # Concept distribution
            concept = example["concept"]
            stats["concepts"][concept] = stats["concepts"].get(concept, 0) + 1
            
            # Length statistics
            code_lengths.append(len(example["code"]))
            output_lengths.append(len(example["expected_output"]))
        
        stats["avg_code_length"] = sum(code_lengths) / len(code_lengths)
        stats["avg_output_length"] = sum(output_lengths) / len(output_lengths)
        
        return stats
    
    def export_to_csv(self, dataset_name: str, split: str = "train") -> str:
        """Export dataset split to CSV for analysis."""
        dataset = self.load_dataset(dataset_name)
        
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in dataset")
        
        # Convert to pandas DataFrame
        df = dataset[split].to_pandas()
        
        # Save to CSV
        csv_path = self.data_dir / f"{dataset_name}_{split}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Exported {split} split to {csv_path}")
        return str(csv_path)
    
    def _apply_strategy(self, training_examples: List[Dict[str, Any]], 
                       base_model_predictions: List[Dict[str, Any]], 
                       strategy: str, error_ratio: float) -> List[Dict[str, Any]]:
        """Apply dataset creation strategy to filter training examples."""
        from .verifier import OutputVerifier
        verifier = OutputVerifier()
        
        # Match examples with predictions
        error_examples = []
        correct_examples = []
        
        for example, prediction in zip(training_examples, base_model_predictions):
            is_correct = verifier.verify(
                example["expected_output"], 
                prediction.get("predicted_output", ""), 
                "exact"
            )["is_correct"]
            
            if is_correct:
                correct_examples.append(example)
            else:
                error_examples.append(example)
        
        if strategy == "errors":
            return error_examples
        elif strategy == "stratified":
            return self._create_stratified_dataset(error_examples, correct_examples, error_ratio)
        else:
            return training_examples
    
    def _create_stratified_dataset(self, error_examples: List[Dict[str, Any]], 
                                  correct_examples: List[Dict[str, Any]], 
                                  error_ratio: float) -> List[Dict[str, Any]]:
        """Create stratified dataset with specified error ratio."""
        import random
        
        total_errors = len(error_examples)
        if total_errors == 0:
            return correct_examples
        
        # Calculate how many correct examples to include
        target_correct = int(total_errors * (1 - error_ratio) / error_ratio)
        
        # Sample correct examples if we have more than needed
        if len(correct_examples) > target_correct:
            sampled_correct = random.sample(correct_examples, target_correct)
        else:
            sampled_correct = correct_examples
        
        return error_examples + sampled_correct
    
    def evaluate_base_model(self, model, tokenizer, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate base model on samples to identify errors."""
        import torch
        from tqdm import tqdm
        
        predictions = []
        
        for sample in tqdm(samples, desc="Evaluating base model"):
            # Create instruction prompt
            prompt = self._create_instruction_prompt(
                sample["generation"]["code"],
                sample["generation"]["language"],
                sample["execution_results"][0].get("input_used", "")
            )
            
            # Generate prediction
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode prediction
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                predicted_output = response[len(prompt):].strip()
            else:
                predicted_output = response.strip()
            
            predictions.append({
                "sample_id": self._generate_sample_id(
                    sample["generation"]["code"], 
                    sample["execution_results"][0].get("input_used", "")
                ),
                "predicted_output": predicted_output
            })
        
        return predictions 