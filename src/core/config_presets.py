"""
Configuration Presets for Training

Provides pre-configured settings for different use cases and hardware configurations.
"""

from dataclasses import dataclass
from typing import List, Optional
from ..training.training_pipeline import PipelineConfig


class ConfigPresets:
    """Pre-configured training settings for different scenarios."""
    
    @staticmethod
    def quick_test(model_key: str = "phi-2") -> PipelineConfig:
        """Quick test configuration for validating the pipeline."""
        return PipelineConfig(
            num_samples_per_language=5,
            languages=["python"],
            model_key=model_key,
            supervised_epochs=1,
            supervised_batch_size=2,
            rl_episodes=50,
            rl_batch_size=2,
            dataset_name="quick_test",
            run_name="quick_test"
        )
    
    @staticmethod 
    def development(model_key: str = "phi-2", languages: List[str] = None) -> PipelineConfig:
        """Development configuration for iterating on the training pipeline."""
        if languages is None:
            languages = ["python", "javascript"]
            
        return PipelineConfig(
            num_samples_per_language=25,
            languages=languages,
            model_key=model_key,
            supervised_epochs=2,
            supervised_batch_size=4,
            rl_episodes=200,
            rl_batch_size=4,
            dataset_name="development_dataset",
            run_name="dev_run"
        )
    
    @staticmethod
    def production(model_key: str = "codellama-7b", languages: List[str] = None) -> PipelineConfig:
        """Production configuration for high-quality model training."""
        if languages is None:
            languages = ["python", "javascript", "rust"]
            
        return PipelineConfig(
            num_samples_per_language=100,
            languages=languages,
            model_key=model_key,
            supervised_epochs=5,
            supervised_batch_size=8,
            rl_episodes=1000,
            rl_batch_size=6,
            dataset_name="production_dataset",
            run_name="production_run"
        )
    
    @staticmethod
    def low_memory(model_key: str = "phi-2") -> PipelineConfig:
        """Configuration optimized for low-memory systems."""
        return PipelineConfig(
            num_samples_per_language=20,
            languages=["python"],
            model_key=model_key,
            supervised_epochs=3,
            supervised_batch_size=1,
            rl_episodes=300,
            rl_batch_size=1,
            dataset_name="low_memory_dataset",
            run_name="low_memory_run"
        )
    
    @staticmethod
    def research(model_key: str = "deepseek-coder-6.7b", languages: List[str] = None) -> PipelineConfig:
        """Research configuration for comprehensive evaluation."""
        if languages is None:
            languages = ["python", "javascript", "rust", "cpp"]
            
        return PipelineConfig(
            num_samples_per_language=75,
            languages=languages,
            model_key=model_key,
            supervised_epochs=4,
            supervised_batch_size=6,
            rl_episodes=800,
            rl_batch_size=4,
            dataset_name="research_dataset", 
            run_name="research_run"
        )
    
    @staticmethod
    def get_preset_info() -> dict:
        """Get information about available presets."""
        return {
            "quick_test": {
                "description": "Fast validation of pipeline (5 samples, 1 epoch)",
                "time_estimate": "~5 minutes",
                "memory_usage": "Low"
            },
            "development": {
                "description": "Development iteration (25 samples, 2 epochs)",
                "time_estimate": "~30 minutes", 
                "memory_usage": "Medium"
            },
            "production": {
                "description": "High-quality training (100 samples, 5 epochs)",
                "time_estimate": "~3 hours",
                "memory_usage": "High"
            },
            "low_memory": {
                "description": "Optimized for <8GB GPU memory",
                "time_estimate": "~45 minutes",
                "memory_usage": "Very Low"
            },
            "research": {
                "description": "Comprehensive multi-language evaluation",
                "time_estimate": "~2.5 hours", 
                "memory_usage": "High"
            }
        }