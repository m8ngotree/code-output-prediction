"""
Complete Training Pipeline for Code Output Prediction

Orchestrates the entire training process:
1. Generate code execution samples
2. Create training datasets  
3. Supervised fine-tuning
4. Reinforcement learning with verification rewards
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime

from ..core.model_manager import ModelManager
from ..core.dataset_manager import DatasetManager
from .supervised_trainer import SupervisedTrainer, SupervisedTrainingConfig
from .rl_trainer import CodeOutputRLTrainer, RLTrainingConfig
from main import CodePredictionSystem

logger = logging.getLogger(__name__)


@dataclass 
class PipelineConfig:
    """Configuration for the complete training pipeline."""
    
    # Data generation
    num_samples_per_language: int = 50
    languages: List[str] = None  # Default: ["python", "javascript"] 
    
    # Model selection
    model_key: str = "phi-2"
    custom_model_id: Optional[str] = None
    
    # Dataset configuration
    dataset_name: str = "code_prediction_pipeline"
    dataset_description: str = "Generated code execution dataset for training"
    
    # Training stages
    run_supervised: bool = True
    run_rl: bool = True
    
    # Supervised training config
    supervised_epochs: int = 3
    supervised_batch_size: int = 4
    supervised_learning_rate: float = 2e-4
    
    # RL training config  
    rl_episodes: int = 500
    rl_batch_size: int = 4
    rl_learning_rate: float = 1.4e-5
    
    # Output configuration
    output_dir: str = "training_runs"
    run_name: Optional[str] = None
    save_intermediate: bool = True
    
    # OpenAI API for data generation
    openai_api_key: Optional[str] = None


class TrainingPipeline:
    """Complete training pipeline for code output prediction."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize training pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Set default languages if not provided
        if config.languages is None:
            config.languages = ["python", "javascript"]
        
        # Set up output directory
        self.output_dir = Path(config.output_dir)
        if config.run_name:
            self.output_dir = self.output_dir / config.run_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.output_dir / f"run_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_manager = ModelManager()
        self.dataset_manager = DatasetManager()
        
        # Save config
        config_file = self.output_dir / "pipeline_config.json"
        with open(config_file, 'w') as f:
            json.dump(config.__dict__, f, indent=2, default=str)
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for the pipeline."""
        log_file = self.output_dir / "training_pipeline.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def generate_training_data(self) -> List[Dict[str, Any]]:
        """
        Generate code execution samples for training.
        
        Returns:
            List of generated samples
        """
        logger.info("Starting data generation phase...")
        
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key required for data generation")
        
        all_samples = []
        
        for language in self.config.languages:
            logger.info(f"Generating {self.config.num_samples_per_language} {language} samples...")
            
            # Create code prediction system for this language
            system = CodePredictionSystem(self.config.openai_api_key, language)
            
            # Generate samples
            samples = system.generate_batch(self.config.num_samples_per_language)
            successful_samples = [s for s in samples if s.get("success", False)]
            
            logger.info(f"Generated {len(successful_samples)}/{len(samples)} successful {language} samples")
            all_samples.extend(successful_samples)
        
        logger.info(f"Data generation complete: {len(all_samples)} total samples")
        
        # Save raw samples if requested
        if self.config.save_intermediate:
            samples_file = self.output_dir / "raw_samples.json"
            with open(samples_file, 'w') as f:
                json.dump(all_samples, f, indent=2, default=str)
            logger.info(f"Raw samples saved to {samples_file}")
        
        return all_samples
    
    def create_training_dataset(self, samples: List[Dict[str, Any]]) -> str:
        """
        Create training dataset from generated samples.
        
        Args:
            samples: List of code execution samples
            
        Returns:
            Dataset name
        """
        logger.info("Creating training dataset...")
        
        # Create dataset
        dataset = self.dataset_manager.create_dataset_from_samples(
            samples=samples,
            dataset_name=self.config.dataset_name,
            description=self.config.dataset_description
        )
        
        # Get dataset statistics
        stats = self.dataset_manager.get_dataset_stats(self.config.dataset_name)
        
        logger.info(f"Dataset created: {self.config.dataset_name}")
        logger.info(f"   Total examples: {stats['total_examples']}")
        logger.info(f"   Languages: {list(stats['languages'].keys())}")
        logger.info(f"   Concepts: {len(stats['concepts'])}")
        
        # Save dataset info
        dataset_info_file = self.output_dir / "dataset_info.json"
        with open(dataset_info_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        return self.config.dataset_name
    
    def run_supervised_training(self, dataset_name: str) -> Optional[str]:
        """
        Run supervised fine-tuning phase.
        
        Args:
            dataset_name: Name of dataset to train on
            
        Returns:
            Path to trained model if successful
        """
        if not self.config.run_supervised:
            logger.info("Skipping supervised training (disabled in config)")
            return None
            
        logger.info("Starting supervised fine-tuning phase...")
        
        # Configure supervised training
        supervised_config = SupervisedTrainingConfig(
            model_key=self.config.model_key,
            custom_model_id=self.config.custom_model_id,
            num_epochs=self.config.supervised_epochs,
            batch_size=self.config.supervised_batch_size,
            learning_rate=self.config.supervised_learning_rate,
            output_dir=str(self.output_dir / "supervised"),
            dataset_name=dataset_name,
            run_name=f"supervised_{self.config.model_key}",
            use_wandb=False  # Keep simple for now
        )
        
        # Run supervised training
        trainer = SupervisedTrainer(supervised_config)
        results = trainer.train(dataset_name)
        
        logger.info("Supervised training completed")
        logger.info(f"   Model saved to: {results['model_path']}")
        logger.info(f"   Final test loss: {results['test_results']['eval_loss']:.4f}")
        
        return results['model_path']
    
    def run_rl_training(self, dataset_name: str, base_model_path: Optional[str] = None) -> Optional[str]:
        """
        Run reinforcement learning phase.
        
        Args:
            dataset_name: Name of dataset to train on
            base_model_path: Path to supervised model (if available)
            
        Returns:
            Path to RL-trained model if successful
        """
        if not self.config.run_rl:
            logger.info("Skipping RL training (disabled in config)")
            return None
            
        logger.info("Starting reinforcement learning phase...")
        
        # Configure RL training
        rl_config = RLTrainingConfig(
            model_key=self.config.model_key,
            custom_model_id=self.config.custom_model_id,
            base_model_path=base_model_path,
            total_episodes=self.config.rl_episodes,
            batch_size=self.config.rl_batch_size,
            learning_rate=self.config.rl_learning_rate,
            output_dir=str(self.output_dir / "rl"),
            dataset_name=dataset_name
        )
        
        # Run RL training
        trainer = CodeOutputRLTrainer(rl_config)
        results = trainer.train(dataset_name)
        
        logger.info("RL training completed")
        logger.info(f"   Model saved to: {results['model_path']}")
        logger.info(f"   Final average reward: {results['final_avg_reward']:.2f}")
        
        return results['model_path']
    
    def evaluate_models(self, dataset_name: str, supervised_path: Optional[str] = None, 
                       rl_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate trained models on the dataset.
        
        Args:
            dataset_name: Dataset to evaluate on
            supervised_path: Path to supervised model
            rl_path: Path to RL model
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating trained models...")
        
        results = {"evaluation_timestamp": datetime.now().isoformat()}
        
        # Evaluate supervised model
        if supervised_path and self.config.run_supervised:
            logger.info("Evaluating supervised model...")
            supervised_config = SupervisedTrainingConfig(
                model_key=self.config.model_key,
                custom_model_id=self.config.custom_model_id
            )
            trainer = SupervisedTrainer(supervised_config)
            supervised_results = trainer.evaluate_model(supervised_path, dataset_name)
            results["supervised"] = supervised_results
            logger.info(f"Supervised model loss: {supervised_results['eval_loss']:.4f}")
        
        # Evaluate RL model
        if rl_path and self.config.run_rl:
            logger.info("Evaluating RL model...")
            rl_config = RLTrainingConfig(
                model_key=self.config.model_key,
                custom_model_id=self.config.custom_model_id
            )
            trainer = CodeOutputRLTrainer(rl_config)
            rl_results = trainer.evaluate_rl_model(rl_path, dataset_name, num_samples=50)
            results["rl"] = rl_results
            logger.info(f"RL model accuracy: {rl_results['exact_accuracy']:.2%}")
        
        # Save evaluation results
        eval_file = self.output_dir / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline from start to finish.
        
        Returns:
            Pipeline results including model paths and evaluation metrics
        """
        logger.info("Starting complete training pipeline...")
        logger.info(f"Output directory: {self.output_dir}")
        
        pipeline_results = {
            "config": self.config.__dict__,
            "start_time": datetime.now().isoformat(),
            "stages_completed": []
        }
        
        try:
            # Stage 1: Generate training data
            logger.info("\n" + "="*60)
            logger.info("STAGE 1: DATA GENERATION")
            logger.info("="*60)
            
            samples = self.generate_training_data()
            pipeline_results["num_samples"] = len(samples)
            pipeline_results["stages_completed"].append("data_generation")
            
            # Stage 2: Create dataset
            logger.info("\n" + "="*60) 
            logger.info("STAGE 2: DATASET CREATION")
            logger.info("="*60)
            
            dataset_name = self.create_training_dataset(samples)
            pipeline_results["dataset_name"] = dataset_name
            pipeline_results["stages_completed"].append("dataset_creation")
            
            # Stage 3: Supervised fine-tuning
            logger.info("\n" + "="*60)
            logger.info("STAGE 3: SUPERVISED FINE-TUNING")
            logger.info("="*60)
            
            supervised_path = self.run_supervised_training(dataset_name)
            pipeline_results["supervised_model_path"] = supervised_path
            if supervised_path:
                pipeline_results["stages_completed"].append("supervised_training")
            
            # Stage 4: RL training
            logger.info("\n" + "="*60)
            logger.info("STAGE 4: REINFORCEMENT LEARNING")
            logger.info("="*60)
            
            rl_path = self.run_rl_training(dataset_name, supervised_path)
            pipeline_results["rl_model_path"] = rl_path
            if rl_path:
                pipeline_results["stages_completed"].append("rl_training")
            
            # Stage 5: Evaluation
            logger.info("\n" + "="*60)
            logger.info("STAGE 5: MODEL EVALUATION")
            logger.info("="*60)
            
            evaluation_results = self.evaluate_models(dataset_name, supervised_path, rl_path)
            pipeline_results["evaluation"] = evaluation_results
            pipeline_results["stages_completed"].append("evaluation")
            
            # Pipeline completion
            pipeline_results["end_time"] = datetime.now().isoformat()
            pipeline_results["status"] = "completed"
            
            logger.info("\n" + "="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Total stages completed: {len(pipeline_results['stages_completed'])}")
            logger.info(f"Dataset: {dataset_name} ({pipeline_results['num_samples']} samples)")
            if supervised_path:
                logger.info(f"Supervised model: {supervised_path}")
            if rl_path:
                logger.info(f"RL model: {rl_path}")
            logger.info(f"Results saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            pipeline_results["status"] = "failed"
            pipeline_results["error"] = str(e)
            pipeline_results["end_time"] = datetime.now().isoformat()
            raise
        
        finally:
            # Save pipeline results
            results_file = self.output_dir / "pipeline_results.json"
            with open(results_file, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
        
        return pipeline_results 