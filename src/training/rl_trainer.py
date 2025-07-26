"""
Reinforcement Learning Trainer for Code Output Prediction

Uses reinforcement learning with verification-based rewards to train models
to predict code execution outputs more accurately.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import numpy as np
from tqdm import tqdm

from ..core.model_manager import ModelManager
from ..core.dataset_manager import DatasetManager
from ..core.verifier import OutputVerifier

logger = logging.getLogger(__name__)


@dataclass
class RLTrainingConfig:
    """Configuration for RL training with verification rewards."""
    
    # Model configuration
    model_key: str = "phi-2"
    custom_model_id: Optional[str] = None
    base_model_path: Optional[str] = None  # Path to supervised fine-tuned model
    
    # RL hyperparameters
    learning_rate: float = 1.4e-5
    batch_size: int = 4
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    ppo_epochs: int = 4
    max_length: int = 512
    
    # PPO specific
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    adap_kl_ctrl: bool = True
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    
    # Training settings
    total_episodes: int = 1000
    max_new_tokens: int = 100
    temperature: float = 0.7
    do_sample: bool = True
    
    # Reward configuration
    exact_match_reward: float = 10.0
    close_match_reward: float = 5.0
    wrong_format_penalty: float = -2.0
    execution_error_penalty: float = -5.0
    
    # Output and monitoring
    output_dir: str = "checkpoints/rl"
    save_freq: int = 100
    log_freq: int = 10
    
    # Data configuration
    dataset_name: str = "code_prediction_dataset"


class CodeOutputRLTrainer:
    """RL trainer for code output prediction with verification rewards."""
    
    def __init__(self, config: RLTrainingConfig):
        """
        Initialize RL trainer.
        
        Args:
            config: RL training configuration
        """
        self.config = config
        self.model_manager = ModelManager(use_quantization=False)  # RL needs full precision
        self.dataset_manager = DatasetManager()
        self.verifier = OutputVerifier()
        
        # Set up output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_stats = []
        
    def load_or_create_model(self) -> Tuple[AutoModelForCausalLMWithValueHead, AutoTokenizer]:
        """
        Load model for RL training. Uses supervised fine-tuned model if available.
        
        Returns:
            Tuple of (model_with_value_head, tokenizer)
        """
        logger.info("Loading model for RL training...")
        
        if self.config.base_model_path and Path(self.config.base_model_path).exists():
            # Load supervised fine-tuned model
            logger.info(f"Loading fine-tuned model from {self.config.base_model_path}")
            
            # Load base model and tokenizer
            base_model, tokenizer = self.model_manager.load_model_and_tokenizer(
                self.config.model_key,
                self.config.custom_model_id
            )
            
            # Load fine-tuned weights
            from peft import PeftModel
            if (Path(self.config.base_model_path) / "adapter_config.json").exists():
                # LoRA weights
                base_model = PeftModel.from_pretrained(base_model, self.config.base_model_path)
                base_model = base_model.merge_and_unload()  # Merge for RL training
            else:
                # Full model weights
                base_model.load_state_dict(
                    torch.load(Path(self.config.base_model_path) / "pytorch_model.bin")
                )
        else:
            # Load base model
            logger.info("Loading base model (no fine-tuning found)")
            base_model, tokenizer = self.model_manager.load_model_and_tokenizer(
                self.config.model_key,
                self.config.custom_model_id
            )
        
        # Create model with value head for PPO
        model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
        
        logger.info("Model loaded successfully for RL training")
        return model, tokenizer
    
    def prepare_rl_dataset(self, dataset_name: str) -> Dataset:
        """
        Prepare dataset for RL training.
        
        Args:
            dataset_name: Name of dataset to load
            
        Returns:
            Dataset formatted for RL training
        """
        logger.info(f"Preparing RL dataset: {dataset_name}")
        
        # Load dataset
        dataset_dict = self.dataset_manager.load_dataset(dataset_name)
        
        # Use training split for RL
        dataset = dataset_dict["train"]
        
        # Format for RL: we need prompts that the model will complete
        def format_for_rl(example):
            """Format example for RL training - prompt only, no target."""
            return {
                "query": example["instruction"],  # The prompt
                "code": example["code"],
                "language": example["language"],
                "input": example["input"],
                "expected_output": example["expected_output"],  # For reward calculation
                "concept": example["concept"],
                "application": example["application"]
            }
        
        rl_dataset = dataset.map(format_for_rl, remove_columns=dataset.column_names)
        
        logger.info(f"Prepared {len(rl_dataset)} examples for RL training")
        return rl_dataset
    
    def compute_reward(self, predicted_output: str, expected_output: str, 
                      code: str, input_data: str) -> float:
        """
        Compute reward based on prediction quality.
        
        Args:
            predicted_output: Model's predicted output
            expected_output: Correct output
            code: Source code
            input_data: Input data
            
        Returns:
            Reward score
        """
        # Verify prediction accuracy
        verification = self.verifier.verify(expected_output, predicted_output, "exact")
        
        if verification["is_correct"]:
            # Exact match - highest reward
            return self.config.exact_match_reward
        
        # Check for close matches
        close_verification = self.verifier.verify(expected_output, predicted_output, "fuzzy")
        if close_verification["is_correct"]:
            # Close match - partial reward
            return self.config.close_match_reward
        
        # Check if output format is reasonable (not complete garbage)
        if self._is_reasonable_output(predicted_output, expected_output):
            # Wrong but reasonable format - small penalty
            return -1.0
        else:
            # Completely wrong format or execution error
            return self.config.wrong_format_penalty
    
    def _is_reasonable_output(self, predicted: str, expected: str) -> bool:
        """Check if predicted output has reasonable format."""
        # Basic heuristics for reasonable output
        if len(predicted.strip()) == 0:
            return False
        
        # Check if it's in a similar format (numbers, text, etc.)
        expected_type = self._infer_output_type(expected)
        predicted_type = self._infer_output_type(predicted)
        
        return expected_type == predicted_type
    
    def _infer_output_type(self, output: str) -> str:
        """Infer the type of output (number, text, list, etc.)."""
        output = output.strip()
        
        if output.isdigit() or output.replace('.', '').replace('-', '').isdigit():
            return "number"
        elif output.startswith('[') and output.endswith(']'):
            return "list"
        elif output.startswith('{') and output.endswith('}'):
            return "dict"
        elif output.lower() in ['true', 'false']:
            return "boolean"
        else:
            return "text"
    
    def train(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run RL training with verification rewards.
        
        Args:
            dataset_name: Name of dataset to use
            
        Returns:
            Training results and metrics
        """
        dataset_name = dataset_name or self.config.dataset_name
        
        logger.info(f"Starting RL training with dataset: {dataset_name}")
        
        # Load model and tokenizer
        model, tokenizer = self.load_or_create_model()
        
        # Prepare dataset
        rl_dataset = self.prepare_rl_dataset(dataset_name)
        
        # PPO configuration
        ppo_config = PPOConfig(
            model_name=self.config.model_key,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            ppo_epochs=self.config.ppo_epochs,
            init_kl_coef=self.config.init_kl_coef,
            target=self.config.target_kl,
            adap_kl_ctrl=self.config.adap_kl_ctrl,
            gamma=self.config.gamma,
            lam=self.config.lam,
            cliprange=self.config.cliprange,
            cliprange_value=self.config.cliprange_value,
            vf_coef=self.config.vf_coef,
        )
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=tokenizer,
            dataset=rl_dataset
        )
        
        # Training loop
        logger.info(f"Starting PPO training for {self.config.total_episodes} episodes...")
        
        for episode in tqdm(range(self.config.total_episodes), desc="RL Training"):
            # Sample batch from dataset
            batch_size = min(self.config.batch_size, len(rl_dataset))
            batch_indices = np.random.choice(len(rl_dataset), batch_size, replace=False)
            batch = [rl_dataset[i] for i in batch_indices]
            
            # Prepare queries (prompts)
            queries = [example["query"] for example in batch]
            query_tensors = [tokenizer.encode(q, return_tensors="pt")[0] for q in queries]
            
            # Generate responses
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode responses
            responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            
            # Extract only the new generated part (remove prompt)
            generated_outputs = []
            for query, response in zip(queries, responses):
                if response.startswith(query):
                    generated_output = response[len(query):].strip()
                else:
                    generated_output = response.strip()
                generated_outputs.append(generated_output)
            
            # Compute rewards
            rewards = []
            for i, output in enumerate(generated_outputs):
                expected = batch[i]["expected_output"]
                code = batch[i]["code"]
                input_data = batch[i]["input"]
                
                reward = self.compute_reward(output, expected, code, input_data)
                rewards.append(torch.tensor(reward))
            
            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Log progress
            if episode % self.config.log_freq == 0:
                avg_reward = np.mean([r.item() for r in rewards])
                logger.info(f"Episode {episode}: avg_reward={avg_reward:.2f}, "
                           f"kl_div={stats['objective/kl']:.4f}")
                
                # Store episode stats
                self.episode_rewards.append(avg_reward)
                self.episode_stats.append({
                    "episode": episode,
                    "avg_reward": avg_reward,
                    "kl_div": stats['objective/kl'],
                    "policy_loss": stats['ppo/loss/policy'],
                    "value_loss": stats['ppo/loss/value']
                })
            
            # Save checkpoint
            if episode % self.config.save_freq == 0 and episode > 0:
                checkpoint_dir = self.output_dir / f"checkpoint_{episode}"
                ppo_trainer.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint at episode {episode}")
        
        # Save final model
        final_model_dir = self.output_dir / "final_model"
        ppo_trainer.save_pretrained(final_model_dir)
        
        # Prepare results
        results = {
            "config": self.config.__dict__,
            "episode_rewards": self.episode_rewards,
            "episode_stats": self.episode_stats,
            "final_avg_reward": np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            "model_path": str(final_model_dir)
        }
        
        # Save results
        results_file = self.output_dir / "rl_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"RL training completed! Final model saved to {final_model_dir}")
        logger.info(f"Final average reward: {results['final_avg_reward']:.2f}")
        
        return results
    
    def evaluate_rl_model(self, model_path: str, dataset_name: str, num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate RL-trained model on prediction accuracy.
        
        Args:
            model_path: Path to RL-trained model
            dataset_name: Dataset to evaluate on
            num_samples: Number of samples to evaluate
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating RL model from {model_path}")
        
        # Load model and tokenizer
        model, tokenizer = self.load_or_create_model()
        
        # Load RL-trained weights
        model.load_state_dict(torch.load(Path(model_path) / "pytorch_model.bin"))
        model.eval()
        
        # Load dataset
        dataset = self.prepare_rl_dataset(dataset_name)
        
        # Sample examples for evaluation
        eval_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        correct_predictions = 0
        close_predictions = 0
        total_reward = 0
        
        for idx in tqdm(eval_indices, desc="Evaluating"):
            example = dataset[int(idx)]
            query = example["query"]
            expected_output = example["expected_output"]
            
            # Generate prediction
            inputs = tokenizer.encode(query, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode prediction
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(query):
                predicted_output = response[len(query):].strip()
            else:
                predicted_output = response.strip()
            
            # Check accuracy
            exact_match = self.verifier.verify(expected_output, predicted_output, "exact")
            close_match = self.verifier.verify(expected_output, predicted_output, "fuzzy")
            
            if exact_match["is_correct"]:
                correct_predictions += 1
            elif close_match["is_correct"]:
                close_predictions += 1
            
            # Compute reward
            reward = self.compute_reward(predicted_output, expected_output, 
                                       example["code"], example["input"])
            total_reward += reward
        
        # Calculate metrics
        accuracy = correct_predictions / len(eval_indices)
        close_accuracy = (correct_predictions + close_predictions) / len(eval_indices)
        avg_reward = total_reward / len(eval_indices)
        
        results = {
            "exact_accuracy": accuracy,
            "close_accuracy": close_accuracy,
            "avg_reward": avg_reward,
            "num_samples": len(eval_indices)
        }
        
        logger.info(f"RL Model Evaluation Results:")
        logger.info(f"  Exact Accuracy: {accuracy:.2%}")
        logger.info(f"  Close Accuracy: {close_accuracy:.2%}")
        logger.info(f"  Average Reward: {avg_reward:.2f}")
        
        return results 