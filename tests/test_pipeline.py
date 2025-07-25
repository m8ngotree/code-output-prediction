"""
Unit tests for the Pipeline class.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.pipeline import (
    PipelineConfig, PipelineStatistics, PipelineState,
    CodeOutputPredictionPipeline
)


class TestPipelineConfig(unittest.TestCase):
    """Test PipelineConfig class."""
    
    def test_default_config(self):
        """Test default configuration loading."""
        config = PipelineConfig()
        
        self.assertIsInstance(config.config, dict)
        self.assertIn("pipeline", config.config)
        self.assertIn("code_generation", config.config)
        self.assertIn("execution", config.config)
        self.assertIn("logging", config.config)
    
    def test_get_config_value(self):
        """Test getting configuration values."""
        config = PipelineConfig()
        
        # Test existing value
        num_samples = config.get("pipeline.num_samples")
        self.assertEqual(num_samples, 10)
        
        # Test default value
        non_existent = config.get("nonexistent.key", "default")
        self.assertEqual(non_existent, "default")
    
    def test_config_merge(self):
        """Test configuration merging."""
        # Create temporary config file
        temp_config = {
            "pipeline": {
                "num_samples": 20
            },
            "new_section": {
                "new_value": 42
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(temp_config, f)
            config_path = f.name
        
        try:
            config = PipelineConfig(config_path)
            
            # Check merged value
            self.assertEqual(config.get("pipeline.num_samples"), 20)
            
            # Check new section
            self.assertEqual(config.get("new_section.new_value"), 42)
            
            # Check existing default values are preserved
            self.assertIsNotNone(config.get("logging.level"))
            
        finally:
            Path(config_path).unlink()


class TestPipelineStatistics(unittest.TestCase):
    """Test PipelineStatistics class."""
    
    def test_initialization(self):
        """Test statistics initialization."""
        stats = PipelineStatistics()
        
        self.assertEqual(stats.codes_requested, 0)
        self.assertEqual(stats.codes_generated, 0)
        self.assertEqual(stats.codes_failed, 0)
        self.assertEqual(stats.executions_attempted, 0)
        self.assertEqual(stats.executions_successful, 0)
        self.assertEqual(len(stats.errors), 0)
        self.assertIsNotNone(stats.start_time)
        self.assertIsNone(stats.end_time)
    
    def test_add_error(self):
        """Test error tracking."""
        stats = PipelineStatistics()
        
        stats.add_error("test_component", "Test error", {"detail": "value"})
        
        self.assertEqual(len(stats.errors), 1)
        error = stats.errors[0]
        self.assertEqual(error["component"], "test_component")
        self.assertEqual(error["error"], "Test error")
        self.assertEqual(error["details"]["detail"], "value")
        self.assertIn("timestamp", error)
    
    def test_finalize_and_to_dict(self):
        """Test statistics finalization and conversion."""
        stats = PipelineStatistics()
        
        # Add some test data
        stats.codes_requested = 10
        stats.codes_generated = 8
        stats.codes_failed = 2
        stats.generation_times = [1.0, 2.0, 1.5]
        
        stats.finalize()
        
        self.assertIsNotNone(stats.end_time)
        
        # Test conversion to dict
        stats_dict = stats.to_dict()
        
        self.assertIn("run_info", stats_dict)
        self.assertIn("code_generation", stats_dict)
        self.assertIn("execution", stats_dict)
        self.assertIn("performance", stats_dict)
        
        # Check calculated values
        self.assertEqual(stats_dict["code_generation"]["success_rate"], 80.0)
        self.assertEqual(stats_dict["code_generation"]["avg_generation_time"], 1.5)


class TestPipelineState(unittest.TestCase):
    """Test PipelineState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = Path(self.temp_dir) / "test_state.json"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test state initialization."""
        state = PipelineState(str(self.state_file))
        
        self.assertEqual(state.state["completed_samples"], [])
        self.assertEqual(state.state["current_sample"], 0)
        self.assertEqual(state.state["failed_samples"], [])
        self.assertEqual(state.state["intermediate_results"], {})
    
    def test_mark_sample_completed(self):
        """Test marking samples as completed."""
        state = PipelineState(str(self.state_file))
        
        result = {"sample_id": 1, "success": True}
        state.mark_sample_completed(1, result)
        
        self.assertIn(1, state.state["completed_samples"])
        self.assertEqual(state.state["intermediate_results"]["1"], result)
        self.assertEqual(state.state["current_sample"], 2)
    
    def test_mark_sample_failed(self):
        """Test marking samples as failed."""
        state = PipelineState(str(self.state_file))
        
        state.mark_sample_failed(1, "Test error")
        
        self.assertIn(1, state.state["failed_samples"])
        self.assertNotIn(1, state.state["completed_samples"])
    
    def test_save_and_load_state(self):
        """Test saving and loading state."""
        state1 = PipelineState(str(self.state_file))
        
        # Modify state
        state1.mark_sample_completed(1, {"test": "data"})
        state1.save_state()
        
        # Create new instance to test loading
        state2 = PipelineState(str(self.state_file))
        
        self.assertIn(1, state2.state["completed_samples"])
        self.assertEqual(state2.state["intermediate_results"]["1"]["test"], "data")
    
    def test_resume_point(self):
        """Test getting resume point."""
        state = PipelineState(str(self.state_file))
        
        self.assertEqual(state.get_resume_point(), 0)
        
        state.mark_sample_completed(2, {"test": "data"})
        self.assertEqual(state.get_resume_point(), 3)
    
    def test_is_sample_completed(self):
        """Test checking if sample is completed."""
        state = PipelineState(str(self.state_file))
        
        self.assertFalse(state.is_sample_completed(1))
        
        state.mark_sample_completed(1, {"test": "data"})
        self.assertTrue(state.is_sample_completed(1))


class TestCodeOutputPredictionPipeline(unittest.TestCase):
    """Test main pipeline class."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            config = {
                "pipeline": {"num_samples": 5, "output_dir": tempfile.mkdtemp()},
                "logging": {"level": "WARNING", "console": True}
            }
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            pipeline = CodeOutputPredictionPipeline(config_path)
            
            self.assertIsNotNone(pipeline.config)
            self.assertIsNotNone(pipeline.logger) 
            self.assertIsNotNone(pipeline.stats)
            self.assertIsNotNone(pipeline.output_dir)
            
        finally:
            Path(config_path).unlink()
    
    def test_logging_setup(self):
        """Test logging configuration."""
        pipeline = CodeOutputPredictionPipeline()
        
        self.assertIsNotNone(pipeline.logger)
        self.assertEqual(pipeline.logger.name, "pipeline")
    
    @patch.dict('os.environ', {})  # Clear environment
    def test_initialization_no_api_key(self):
        """Test initialization without API key."""
        pipeline = CodeOutputPredictionPipeline()
        
        # Should fail to initialize components without API key
        success = pipeline.initialize_components()
        self.assertFalse(success)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('src.pipeline.CodeGenerator')
    @patch('src.pipeline.SeedManager')
    @patch('src.pipeline.InputGenerator')
    @patch('src.pipeline.PythonExecutor')
    def test_component_initialization_success(self, mock_executor, mock_input_gen, 
                                            mock_seed_manager, mock_code_gen):
        """Test successful component initialization."""
        # Mock all components
        mock_seed_manager.return_value = Mock()
        mock_code_gen.return_value = Mock()
        mock_input_gen.return_value = Mock()
        mock_executor.return_value = Mock()
        
        pipeline = CodeOutputPredictionPipeline()
        success = pipeline.initialize_components()
        
        self.assertTrue(success)
        self.assertIsNotNone(pipeline.seed_manager)
        self.assertIsNotNone(pipeline.code_generator)
        self.assertIsNotNone(pipeline.input_generator)
        self.assertIsNotNone(pipeline.executor)
        self.assertIsNotNone(pipeline.state)


if __name__ == "__main__":
    unittest.main()