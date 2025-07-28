#!/usr/bin/env python3
"""
Code Output Prediction Training System

A comprehensive system for training open-source LLMs to predict code execution outputs
using supervised fine-tuning and reinforcement learning with verification rewards.

Usage:
    python train.py --help                    # Show this help
    python train.py list-models               # List available models
    python train.py generate-data             # Generate training data only
    python train.py train-supervised          # Run supervised training only
    python train.py train-rl                  # Run RL training only  
    python train.py full-pipeline             # Run complete pipeline
    python train.py evaluate                  # Evaluate trained models
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.model_manager import ModelManager
from src.core.dataset_manager import DatasetManager
from src.training.training_pipeline import TrainingPipeline, PipelineConfig
from src.training.supervised_trainer import SupervisedTrainer, SupervisedTrainingConfig
from src.training.rl_trainer import CodeOutputRLTrainer, RLTrainingConfig


class TrainingCLI:
    """Command-line interface for the training system."""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.dataset_manager = DatasetManager()
    
    def list_models(self):
        """Display available models and their information."""
        print("Available Models for Training:")
        print("=" * 60)
        
        models = self.model_manager.list_available_models()
        
        for key, info in models.items():
            print(f"\n{key}")
            print(f"   Model ID: {info['model_id']}")
            print(f"   Description: {info['description']}")
            print(f"   Context Length: {info['context_length']:,} tokens")
            print(f"   Recommended Batch Size: {info['recommended_batch_size']}")
            
            # Show memory estimates
            memory = self.model_manager.estimate_memory_usage(key)
            if "error" not in memory:
                print(f"   Memory Requirements:")
                print(f"     • Full Precision: {memory['full_precision']}")
                print(f"     • FP16: {memory['fp16']}")
                print(f"     • 4-bit Quantized: {memory['4bit_quantized']}")
                print(f"     • Recommended: {memory['recommended']}")
        
        print(f"\nTo use a custom model, specify --model custom --custom-model-id YOUR_MODEL_ID")
        print(f"All models support LoRA fine-tuning for memory efficiency")
    
    def list_datasets(self):
        """Display available datasets."""
        print("Available Training Datasets:")
        print("=" * 50)
        
        datasets = self.dataset_manager.list_datasets()
        
        if not datasets:
            print("No datasets found. Generate some training data first!")
            return
        
        for name, info in datasets.items():
            print(f"\n{name}")
            print(f"   Description: {info['description']}")
            print(f"   Examples: {info['num_examples']:,}")
            print(f"   Languages: {', '.join(info['languages'])}")
            print(f"   Created: {info['created_at'][:19]}")
            
            splits = info['splits']
            print(f"   Splits: Train={splits['train']}, Val={splits['validation']}, Test={splits['test']}")
    
    def generate_data_command(self, args):
        """Generate training data."""
        print("🔄 Generating Training Data")
        print("=" * 40)
        
        # Validate API key
        api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key required")
            print("   Set OPENAI_API_KEY environment variable or use --openai-api-key")
            return
        
        # Import here to avoid early dependency issues
        from main import CodePredictionSystem
        
        all_samples = []
        
        for language in args.languages:
            print(f"\n📝 Generating {args.num_samples} {language} samples...")
            
            system = CodePredictionSystem(api_key, language)
            samples = system.generate_batch(args.num_samples)
            successful = [s for s in samples if s.get("success", False)]
            
            print(f"   {len(successful)}/{len(samples)} samples successful")
            all_samples.extend(successful)
        
        if not all_samples:
            print("No successful samples generated")
            return
        
        # Create dataset
        print(f"\nCreating dataset '{args.dataset_name}'...")
        dataset = self.dataset_manager.create_dataset_from_samples(
            samples=all_samples,
            dataset_name=args.dataset_name,
            description=f"Generated dataset with {len(all_samples)} samples"
        )
        
        # Show statistics
        stats = self.dataset_manager.get_dataset_stats(args.dataset_name)
        print(f"\nDataset created successfully")
        print(f"   Name: {args.dataset_name}")
        print(f"   Total examples: {stats['total_examples']:,}")
        print(f"   Languages: {list(stats['languages'].keys())}")
        print(f"   Average code length: {stats['avg_code_length']:.0f} chars")
    
    def train_supervised_command(self, args):
        """Run supervised fine-tuning."""
        print("Supervised Fine-tuning")
        print("=" * 35)
        
        # Configure training
        config = SupervisedTrainingConfig(
            model_key=args.model,
            custom_model_id=args.custom_model_id,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            use_wandb=args.use_wandb,
            run_name=args.run_name,
            dataset_strategy=args.dataset_strategy,
            error_ratio=args.error_ratio,
            evaluate_base_model=args.evaluate_base_model
        )
        
        print(f"Configuration:")
        print(f"   Model: {config.model_key}")
        print(f"   Dataset: {config.dataset_name}")
        print(f"   Dataset Strategy: {config.dataset_strategy}")
        if config.dataset_strategy == "stratified":
            print(f"   Error Ratio: {config.error_ratio}")
        print(f"   Evaluate Base Model: {config.evaluate_base_model}")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Batch Size: {config.batch_size}")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   Output: {config.output_dir}")
        
        # Run training
        trainer = SupervisedTrainer(config)
        results = trainer.train()
        
        print(f"\nTraining completed")
        print(f"   Model saved to: {results['model_path']}")
        print(f"   Final test loss: {results['test_results']['eval_loss']:.4f}")
    
    def train_rl_command(self, args):
        """Run RL training."""
        print("Reinforcement Learning Training")
        print("=" * 45)
        
        # Configure training
        config = RLTrainingConfig(
            model_key=args.model,
            custom_model_id=args.custom_model_id,
            base_model_path=args.base_model_path,
            total_episodes=args.episodes,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name
        )
        
        print(f"Configuration:")
        print(f"   Model: {config.model_key}")
        print(f"   Base Model: {config.base_model_path or 'None (using base model)'}")
        print(f"   Dataset: {config.dataset_name}")
        print(f"   Episodes: {config.total_episodes}")
        print(f"   Batch Size: {config.batch_size}")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   Output: {config.output_dir}")
        
        # Run training
        trainer = CodeOutputRLTrainer(config)
        results = trainer.train()
        
        print(f"\nRL training completed")
        print(f"   Model saved to: {results['model_path']}")
        print(f"   Final average reward: {results['final_avg_reward']:.2f}")
    
    def full_pipeline_command(self, args):
        """Run the complete training pipeline."""
        print("Complete Training Pipeline")
        print("=" * 40)
        
        # Validate API key for data generation
        api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key required for data generation")
            return
        
        # Configure pipeline
        config = PipelineConfig(
            num_samples_per_language=args.num_samples,
            languages=args.languages,
            model_key=args.model,
            custom_model_id=args.custom_model_id,
            dataset_name=args.dataset_name,
            run_supervised=not args.skip_supervised,
            run_rl=not args.skip_rl,
            supervised_epochs=args.epochs,
            supervised_batch_size=args.batch_size,
            supervised_learning_rate=args.learning_rate,
            rl_episodes=args.rl_episodes,
            rl_batch_size=args.batch_size,
            rl_learning_rate=args.rl_learning_rate,
            output_dir=args.output_dir,
            run_name=args.run_name,
            openai_api_key=api_key
        )
        
        print(f"📋 Pipeline Configuration:")
        print(f"   Model: {config.model_key}")
        print(f"   Languages: {', '.join(config.languages)}")
        print(f"   Samples per language: {config.num_samples_per_language}")
        print(f"   Dataset: {config.dataset_name}")
        print(f"   Supervised training: {'Yes' if config.run_supervised else 'No'}")
        print(f"   RL training: {'Yes' if config.run_rl else 'No'}")
        print(f"   Output directory: {config.output_dir}")
        
        # Run pipeline
        pipeline = TrainingPipeline(config)
        results = pipeline.run_complete_pipeline()
        
        print(f"\nPipeline completed")
        print(f"   Status: {results['status']}")
        print(f"   Stages completed: {len(results['stages_completed'])}")
        if results.get('supervised_model_path'):
            print(f"   Supervised model: {results['supervised_model_path']}")
        if results.get('rl_model_path'):
            print(f"   RL model: {results['rl_model_path']}")
    
    def evaluate_command(self, args):
        """Evaluate trained models."""
        print("Model Evaluation")
        print("=" * 25)
        
        if args.supervised_path:
            print(f"Evaluating supervised model: {args.supervised_path}")
            config = SupervisedTrainingConfig(
                model_key=args.model,
                custom_model_id=args.custom_model_id
            )
            trainer = SupervisedTrainer(config)
            results = trainer.evaluate_model(args.supervised_path, args.dataset_name)
            print(f"   Test Loss: {results['eval_loss']:.4f}")
        
        if args.rl_path:
            print(f"Evaluating RL model: {args.rl_path}")
            config = RLTrainingConfig(
                model_key=args.model,
                custom_model_id=args.custom_model_id
            )
            trainer = CodeOutputRLTrainer(config)
            results = trainer.evaluate_rl_model(args.rl_path, args.dataset_name, args.num_samples)
            print(f"   Exact Accuracy: {results['exact_accuracy']:.2%}")
            print(f"   Close Accuracy: {results['close_accuracy']:.2%}")
            print(f"   Average Reward: {results['avg_reward']:.2f}")


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Code Output Prediction Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python train.py list-models
  
  # Generate training data for Python and JavaScript
  python train.py generate-data --languages python javascript --num-samples 100
  
  # Run supervised fine-tuning with Phi-2 model
  python train.py train-supervised --model phi-2 --dataset my_dataset
  
  # Run complete pipeline with CodeLlama
  python train.py full-pipeline --model codellama-7b --languages python --num-samples 50
  
  # Use custom model from HuggingFace
  python train.py train-supervised --model custom --custom-model-id microsoft/DialoGPT-medium
  
  # Error-focused supervised training
  python train.py train-supervised --model phi-2 --dataset-strategy errors --evaluate-base-model
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List models command
    subparsers.add_parser('list-models', help='List available models')
    
    # List datasets command  
    subparsers.add_parser('list-datasets', help='List available datasets')
    
    # Generate data command
    gen_parser = subparsers.add_parser('generate-data', help='Generate training data')
    gen_parser.add_argument('--languages', nargs='+', default=['python'], 
                           choices=['python', 'javascript', 'rust', 'cpp'],
                           help='Programming languages to generate data for')
    gen_parser.add_argument('--num-samples', type=int, default=50,
                           help='Number of samples per language')
    gen_parser.add_argument('--dataset-name', default='code_prediction_dataset',
                           help='Name for the created dataset')
    gen_parser.add_argument('--openai-api-key', help='OpenAI API key')
    
    # Common model arguments
    def add_model_args(parser):
        parser.add_argument('--model', default='phi-2',
                           help='Model to use (see list-models for options)')
        parser.add_argument('--custom-model-id', 
                           help='Custom HuggingFace model ID (use with --model custom)')
    
    # Common training arguments
    def add_training_args(parser):
        parser.add_argument('--dataset-name', default='code_prediction_dataset',
                           help='Dataset to train on')
        parser.add_argument('--batch-size', type=int, default=4,
                           help='Training batch size')
        parser.add_argument('--learning-rate', type=float, default=2e-4,
                           help='Learning rate')
        parser.add_argument('--output-dir', default='checkpoints',
                           help='Output directory for models')
        parser.add_argument('--run-name', help='Custom run name')
    
    # Supervised training command
    sup_parser = subparsers.add_parser('train-supervised', help='Run supervised fine-tuning')
    add_model_args(sup_parser)
    add_training_args(sup_parser)
    sup_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    sup_parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')
    sup_parser.add_argument('--dataset-strategy', choices=['all', 'errors', 'stratified'], 
                           default='all', help='Dataset creation strategy')
    sup_parser.add_argument('--error-ratio', type=float, default=0.7,
                           help='Ratio of error examples in stratified strategy')
    sup_parser.add_argument('--evaluate-base-model', action='store_true',
                           help='Evaluate base model to identify errors for focused training')
    
    # RL training command
    rl_parser = subparsers.add_parser('train-rl', help='Run RL training')
    add_model_args(rl_parser)
    add_training_args(rl_parser)
    rl_parser.add_argument('--episodes', type=int, default=500, help='Number of RL episodes')
    rl_parser.add_argument('--base-model-path', help='Path to supervised model (optional)')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full-pipeline', help='Run complete training pipeline')
    add_model_args(full_parser)
    add_training_args(full_parser)
    full_parser.add_argument('--languages', nargs='+', default=['python'],
                            choices=['python', 'javascript', 'rust', 'cpp'],
                            help='Languages for data generation')
    full_parser.add_argument('--num-samples', type=int, default=50,
                            help='Samples per language')
    full_parser.add_argument('--epochs', type=int, default=3,
                            help='Supervised training epochs')
    full_parser.add_argument('--rl-episodes', type=int, default=500,
                            help='RL training episodes')
    full_parser.add_argument('--rl-learning-rate', type=float, default=1.4e-5,
                            help='RL learning rate')
    full_parser.add_argument('--skip-supervised', action='store_true',
                            help='Skip supervised training phase')
    full_parser.add_argument('--skip-rl', action='store_true',
                            help='Skip RL training phase')
    full_parser.add_argument('--openai-api-key', help='OpenAI API key')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    add_model_args(eval_parser)
    eval_parser.add_argument('--dataset-name', default='code_prediction_dataset',
                            help='Dataset to evaluate on')
    eval_parser.add_argument('--supervised-path', help='Path to supervised model')
    eval_parser.add_argument('--rl-path', help='Path to RL model')
    eval_parser.add_argument('--num-samples', type=int, default=100,
                            help='Number of samples to evaluate')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = TrainingCLI()
    
    try:
        if args.command == 'list-models':
            cli.list_models()
        elif args.command == 'list-datasets':
            cli.list_datasets()
        elif args.command == 'generate-data':
            cli.generate_data_command(args)
        elif args.command == 'train-supervised':
            cli.train_supervised_command(args)
        elif args.command == 'train-rl':
            cli.train_rl_command(args)
        elif args.command == 'full-pipeline':
            cli.full_pipeline_command(args)
        elif args.command == 'evaluate':
            cli.evaluate_command(args)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 