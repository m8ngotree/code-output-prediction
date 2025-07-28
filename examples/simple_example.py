#!/usr/bin/env python3
"""
Simple Example: Code Output Prediction Training

This example demonstrates how to use the training system programmatically
to train a model to predict code execution outputs.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.training.training_pipeline import TrainingPipeline, PipelineConfig
from main import CodePredictionSystem
from src.core.dataset_manager import DatasetManager

def simple_training_example():
    """Run a simple training example with minimal configuration."""
    
    print("Simple Code Output Prediction Training Example")
    print("=" * 60)
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Step 1: Generate a small dataset
    print("\nStep 1: Generating training data...")
    
    # Create code prediction system
    system = CodePredictionSystem(api_key, "python")
    
    # Generate some samples
    print("   Generating 10 Python code samples...")
    samples = system.generate_batch(10)
    successful_samples = [s for s in samples if s.get("success", False)]
    
    print(f"   Generated {len(successful_samples)}/{len(samples)} successful samples")
    
    if len(successful_samples) < 3:
        print("Error: Not enough successful samples for training")
        return
    
    # Step 2: Create dataset
    print("\nStep 2: Creating training dataset...")
    
    dataset_manager = DatasetManager()
    dataset_name = "simple_example_dataset"
    
    dataset = dataset_manager.create_dataset_from_samples(
        samples=successful_samples,
        dataset_name=dataset_name,
        description="Simple example dataset for code output prediction"
    )
    
    print(f"   Dataset created: {dataset_name}")
    
    # Step 3: Run training pipeline
    print("\nStep 3: Running training pipeline...")
    
    # Configure pipeline for quick training
    config = PipelineConfig(
        num_samples_per_language=0,  # Skip data generation (we already have data)
        model_key="phi-2",  # Use smallest model for quick example
        dataset_name=dataset_name,
        run_supervised=True,
        run_rl=False,  # Skip RL for simplicity
        supervised_epochs=1,  # Just 1 epoch for quick demo
        supervised_batch_size=2,
        output_dir="examples/simple_training_output",
        run_name="simple_example",
        openai_api_key=api_key
    )
    
    # Create pipeline
    pipeline = TrainingPipeline(config)
    
    # Skip data generation since we already have data
    print("   Skipping data generation (using existing samples)")
    
    # Run supervised training only
    print("   Running supervised fine-tuning...")
    supervised_path = pipeline.run_supervised_training(dataset_name)
    
    if supervised_path:
        print(f"   Model trained successfully")
        print(f"   Model saved to: {supervised_path}")
        
        # Step 4: Simple evaluation
        print("\nStep 4: Evaluating the model...")
        
        evaluation_results = pipeline.evaluate_models(
            dataset_name=dataset_name,
            supervised_path=supervised_path
        )
        
        if "supervised" in evaluation_results:
            loss = evaluation_results["supervised"]["eval_loss"]
            print(f"   Test Loss: {loss:.4f}")
        
        print("\nTraining example completed successfully")
        print(f"   Results saved to: {pipeline.output_dir}")
        
    else:
        print("Error: Training failed")


def quick_evaluation_example():
    """Quick example of evaluating a pre-trained model."""
    
    print("\nQuick Evaluation Example")
    print("=" * 40)
    
    # This would evaluate a model if one exists
    dataset_manager = DatasetManager()
    datasets = dataset_manager.list_datasets()
    
    if not datasets:
        print("No datasets available for evaluation")
        return
    
    print("Available datasets:")
    for name, info in datasets.items():
        print(f"  • {name}: {info['num_examples']} examples")


if __name__ == "__main__":
    try:
        simple_training_example()
        quick_evaluation_example()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nFor help, run: python train.py --help") 