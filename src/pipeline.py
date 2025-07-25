"""
Main Pipeline Controller for Code Output Prediction System.

This module orchestrates the complete pipeline from code generation to execution
and result collection, with comprehensive error handling and progress tracking.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from .generators.code_generator import CodeGenerator, CodeGenerationError
from .generators.input_generator import InputGenerator
from .executors.python_executor import PythonExecutor, SecurityLimits, ExecutionResult
from .seeds.seed_manager import SeedManager


class PipelineConfig:
    """Configuration management for the pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults."""
        self.config = self._load_default_config()
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self._merge_config(user_config)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "pipeline": {
                "num_samples": 10,
                "resume_from": None,
                "output_dir": "data/pipeline_results",
                "intermediate_save": True,
                "save_interval": 5
            },
            "code_generation": {
                "config_path": "configs/code_generator.yaml",
                "retry_attempts": 3,
                "timeout_seconds": 60
            },
            "input_generation": {
                "inputs_per_code": 8,
                "max_input_variants": 10,
                "include_edge_cases": True
            },
            "execution": {
                "timeout_seconds": 30,
                "max_memory_mb": 256,
                "max_concurrent": 1,
                "retry_failures": True
            },
            "logging": {
                "level": "INFO",
                "file": None,
                "console": True,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "progress": {
                "show_progress_bars": True,
                "detailed_logging": True,
                "save_statistics": True
            }
        }
    
    def _merge_config(self, user_config: Dict[str, Any]):
        """Merge user configuration with defaults."""
        def deep_merge(default: Dict, user: Dict):
            for key, value in user.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    deep_merge(default[key], value)
                else:
                    default[key] = value
        
        deep_merge(self.config, user_config)
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value


class PipelineStatistics:
    """Statistics collection and reporting for pipeline runs."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.end_time = None
        
        # Code generation stats
        self.codes_requested = 0
        self.codes_generated = 0
        self.codes_failed = 0
        self.generation_times = []
        
        # Input generation stats
        self.inputs_generated = 0
        self.input_generation_times = []
        
        # Execution stats
        self.executions_attempted = 0
        self.executions_successful = 0
        self.executions_failed = 0
        self.executions_timeout = 0
        self.execution_times = []
        
        # Error tracking
        self.errors = []
        
        # Component performance
        self.component_times = {
            "code_generation": 0.0,
            "input_generation": 0.0,
            "code_execution": 0.0,
            "data_persistence": 0.0
        }
    
    def add_error(self, component: str, error: str, details: Dict[str, Any] = None):
        """Add an error to the statistics."""
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "error": error,
            "details": details or {}
        })
    
    def finalize(self):
        """Finalize statistics collection."""
        self.end_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        total_time = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        return {
            "run_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "total_duration_seconds": total_time,
                "pipeline_version": "1.0.0"
            },
            "code_generation": {
                "requested": self.codes_requested,
                "successful": self.codes_generated,
                "failed": self.codes_failed,
                "success_rate": self.codes_generated / max(self.codes_requested, 1) * 100,
                "avg_generation_time": sum(self.generation_times) / max(len(self.generation_times), 1),
                "total_time": self.component_times["code_generation"]
            },
            "input_generation": {
                "total_inputs": self.inputs_generated,
                "avg_generation_time": sum(self.input_generation_times) / max(len(self.input_generation_times), 1),
                "total_time": self.component_times["input_generation"]
            },
            "execution": {
                "attempted": self.executions_attempted,
                "successful": self.executions_successful,
                "failed": self.executions_failed,
                "timeout": self.executions_timeout,
                "success_rate": self.executions_successful / max(self.executions_attempted, 1) * 100,
                "avg_execution_time": sum(self.execution_times) / max(len(self.execution_times), 1),
                "total_time": self.component_times["code_execution"]
            },
            "performance": {
                "component_times": self.component_times,
                "total_errors": len(self.errors),
                "throughput_codes_per_minute": self.codes_generated / max(total_time / 60, 1),
                "throughput_executions_per_minute": self.executions_successful / max(total_time / 60, 1)
            },
            "errors": self.errors
        }


class PipelineState:
    """Manages pipeline state for resuming interrupted runs."""
    
    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.state = {
            "completed_samples": [],
            "current_sample": 0,
            "failed_samples": [],
            "intermediate_results": {},
            "last_save": None
        }
        self.load_state()
    
    def load_state(self):
        """Load state from file if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    saved_state = json.load(f)
                    self.state.update(saved_state)
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not load state file: {e}")
    
    def save_state(self):
        """Save current state to file."""
        self.state["last_save"] = datetime.now().isoformat()
        
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except IOError as e:
            logging.error(f"Could not save state file: {e}")
    
    def mark_sample_completed(self, sample_id: int, result: Dict[str, Any]):
        """Mark a sample as completed."""
        if sample_id not in self.state["completed_samples"]:
            self.state["completed_samples"].append(sample_id)
        
        if sample_id in self.state["failed_samples"]:
            self.state["failed_samples"].remove(sample_id)
        
        self.state["intermediate_results"][str(sample_id)] = result
        self.state["current_sample"] = max(self.state["current_sample"], sample_id + 1)
    
    def mark_sample_failed(self, sample_id: int, error: str):
        """Mark a sample as failed."""
        if sample_id not in self.state["failed_samples"]:
            self.state["failed_samples"].append(sample_id)
        
        if sample_id in self.state["completed_samples"]:
            self.state["completed_samples"].remove(sample_id)
    
    def get_resume_point(self) -> int:
        """Get the sample index to resume from."""
        return self.state["current_sample"]
    
    def is_sample_completed(self, sample_id: int) -> bool:
        """Check if a sample is already completed."""
        return sample_id in self.state["completed_samples"]
    
    def get_completed_results(self) -> Dict[str, Any]:
        """Get all completed results."""
        return self.state["intermediate_results"]


class CodeOutputPredictionPipeline:
    """
    Main pipeline controller that orchestrates the entire process.
    
    Manages code generation, input generation, execution, and result collection
    with comprehensive error handling and progress tracking.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = PipelineConfig(config_path)
        self.logger = self._setup_logging()
        self.stats = PipelineStatistics()
        
        # Initialize components
        self.seed_manager: Optional[SeedManager] = None
        self.code_generator: Optional[CodeGenerator] = None
        self.input_generator: Optional[InputGenerator] = None
        self.executor: Optional[PythonExecutor] = None
        
        # Pipeline state
        self.state: Optional[PipelineState] = None
        self.output_dir = Path(self.config.get("pipeline.output_dir"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Pipeline initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("pipeline")
        logger.setLevel(getattr(logging, self.config.get("logging.level", "INFO")))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        formatter = logging.Formatter(self.config.get("logging.format"))
        
        # Console handler
        if self.config.get("logging.console", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        log_file = self.config.get("logging.file")
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def initialize_components(self) -> bool:
        """
        Initialize all pipeline components.
        
        Returns:
            True if all components initialized successfully, False otherwise
        """
        try:
            self.logger.info("Initializing pipeline components...")
            
            # Initialize seed manager
            self.seed_manager = SeedManager()
            self.logger.info("✅ Seed manager initialized")
            
            # Initialize code generator
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.error("❌ OPENAI_API_KEY environment variable not set")
                return False
            
            config_path = self.config.get("code_generation.config_path")
            self.code_generator = CodeGenerator(config_path=config_path, api_key=api_key)
            self.logger.info("✅ Code generator initialized")
            
            # Initialize input generator
            self.input_generator = InputGenerator()
            self.logger.info("✅ Input generator initialized")
            
            # Initialize executor
            security_limits = SecurityLimits()
            security_limits.timeout_seconds = self.config.get("execution.timeout_seconds", 30)
            security_limits.max_memory_mb = self.config.get("execution.max_memory_mb", 256)
            
            self.executor = PythonExecutor(limits=security_limits)
            self.logger.info("✅ Code executor initialized")
            
            # Initialize state management
            state_file = self.output_dir / "pipeline_state.json"
            self.state = PipelineState(str(state_file))
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            self.stats.add_error("initialization", str(e))
            return False
    
    def generate_code_sample(self, sample_id: int) -> Optional[Dict[str, Any]]:
        """
        Generate a single code sample.
        
        Args:
            sample_id: Unique identifier for this sample
            
        Returns:
            Generated code data or None if failed
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Generating code sample {sample_id}")
            
            # Generate code using random seeds
            result = self.code_generator.generate_code()
            
            if not result["success"]:
                raise CodeGenerationError(f"Code generation failed: {result.get('error', 'Unknown error')}")
            
            generation_time = time.time() - start_time
            self.stats.generation_times.append(generation_time)
            self.stats.codes_generated += 1
            
            code_data = {
                "sample_id": sample_id,
                "code": result["code"],
                "metadata": result["metadata"],
                "seeds_used": result["seeds_used"],
                "generation_time": generation_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Code sample {sample_id} generated in {generation_time:.2f}s")
            return code_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate code sample {sample_id}: {e}")
            self.stats.add_error("code_generation", str(e), {"sample_id": sample_id})
            self.stats.codes_failed += 1
            return None
        
        finally:
            self.stats.component_times["code_generation"] += time.time() - start_time
    
    def generate_inputs_for_code(self, code_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate inputs for a code sample.
        
        Args:
            code_data: Code sample data
            
        Returns:
            Input data or None if failed
        """
        start_time = time.time()
        
        try:
            sample_id = code_data["sample_id"]
            self.logger.info(f"Generating inputs for sample {sample_id}")
            
            # Generate inputs
            inputs_result = self.input_generator.generate_inputs(code_data["code"])
            
            input_count = inputs_result["total_inputs"]
            self.stats.inputs_generated += input_count
            
            generation_time = time.time() - start_time
            self.stats.input_generation_times.append(generation_time)
            
            inputs_data = {
                "sample_id": sample_id,
                "inputs": inputs_result,
                "generation_time": generation_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Generated {input_count} inputs for sample {sample_id} in {generation_time:.2f}s")
            return inputs_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate inputs for sample {code_data['sample_id']}: {e}")
            self.stats.add_error("input_generation", str(e), {"sample_id": code_data["sample_id"]})
            return None
        
        finally:
            self.stats.component_times["input_generation"] += time.time() - start_time
    
    def execute_code_with_inputs(self, code_data: Dict[str, Any], 
                               inputs_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute code with all generated inputs.
        
        Args:
            code_data: Code sample data
            inputs_data: Input data
            
        Returns:
            List of execution results
        """
        start_time = time.time()
        execution_results = []
        
        try:
            sample_id = code_data["sample_id"]
            code = code_data["code"]
            inputs = inputs_data["inputs"]["inputs"]
            
            # Collect all inputs from different categories
            all_inputs = []
            for category, input_list in inputs.items():
                all_inputs.extend(input_list)
            
            self.logger.info(f"Executing code sample {sample_id} with {len(all_inputs)} inputs")
            
            # Progress bar for executions if enabled
            input_iterator = all_inputs
            if self.config.get("progress.show_progress_bars"):
                input_iterator = tqdm(all_inputs, desc=f"Executing sample {sample_id}", 
                                    leave=False, disable=not self.config.get("progress.show_progress_bars"))
            
            for i, input_data in enumerate(input_iterator):
                exec_start = time.time()
                
                try:
                    # Determine execution mode based on input data
                    if input_data.get("data_type") == "file":
                        # File-based execution
                        file_inputs = {f"input_file_{i}.txt": str(input_data["data"])}
                        result = self.executor.execute_with_files(code, file_inputs)
                    elif isinstance(input_data["data"], list):
                        # Script execution with CLI args
                        result = self.executor.execute_script(code, input_data["data"])
                    else:
                        # Try to execute as script first
                        result = self.executor.execute_script(code)
                    
                    exec_time = time.time() - exec_start
                    self.stats.execution_times.append(exec_time)
                    
                    if result.success:
                        self.stats.executions_successful += 1
                    else:
                        self.stats.executions_failed += 1
                        if result.timeout_occurred:
                            self.stats.executions_timeout += 1
                    
                    execution_results.append({
                        "input_id": input_data["input_id"],
                        "input_data": input_data,
                        "result": result.to_dict(),
                        "execution_time": exec_time,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    self.stats.executions_attempted += 1
                    
                except Exception as e:
                    self.logger.warning(f"Execution failed for input {i}: {e}")
                    self.stats.executions_failed += 1
                    self.stats.executions_attempted += 1
                    
                    execution_results.append({
                        "input_id": input_data["input_id"], 
                        "input_data": input_data,
                        "result": {"success": False, "error": str(e)},
                        "execution_time": time.time() - exec_start,
                        "timestamp": datetime.now().isoformat()
                    })
            
            successful_executions = sum(1 for r in execution_results if r["result"].get("success", False))
            self.logger.info(f"✅ Executed sample {sample_id}: {successful_executions}/{len(execution_results)} successful")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute sample {code_data['sample_id']}: {e}")
            self.stats.add_error("code_execution", str(e), {"sample_id": code_data["sample_id"]})
        
        finally:
            self.stats.component_times["code_execution"] += time.time() - start_time
        
        return execution_results
    
    def save_sample_results(self, sample_id: int, code_data: Dict[str, Any], 
                          inputs_data: Dict[str, Any], execution_results: List[Dict[str, Any]]):
        """
        Save results for a completed sample.
        
        Args:
            sample_id: Sample identifier
            code_data: Code generation data
            inputs_data: Input generation data  
            execution_results: Execution results
        """
        start_time = time.time()
        
        try:
            sample_result = {
                "sample_id": sample_id,
                "code_data": code_data,
                "inputs_data": inputs_data,
                "execution_results": execution_results,
                "summary": {
                    "total_executions": len(execution_results),
                    "successful_executions": sum(1 for r in execution_results if r["result"].get("success", False)),
                    "failed_executions": sum(1 for r in execution_results if not r["result"].get("success", False)),
                    "total_execution_time": sum(r.get("execution_time", 0) for r in execution_results)
                },
                "completed_at": datetime.now().isoformat()
            }
            
            # Save individual sample result
            sample_file = self.output_dir / f"sample_{sample_id:04d}.json"
            with open(sample_file, 'w') as f:
                json.dump(sample_result, f, indent=2, default=str)
            
            # Update state
            self.state.mark_sample_completed(sample_id, sample_result)
            
            self.logger.info(f"✅ Saved results for sample {sample_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results for sample {sample_id}: {e}")
            self.stats.add_error("data_persistence", str(e), {"sample_id": sample_id})
        
        finally:
            self.stats.component_times["data_persistence"] += time.time() - start_time
    
    def run_pipeline(self, num_samples: Optional[int] = None) -> bool:
        """
        Run the complete pipeline.
        
        Args:
            num_samples: Number of samples to generate (overrides config)
            
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        if not self.initialize_components():
            return False
        
        num_samples = num_samples or self.config.get("pipeline.num_samples", 10)
        save_interval = self.config.get("pipeline.save_interval", 5)
        
        self.logger.info(f"🚀 Starting pipeline run with {num_samples} samples")
        self.stats.codes_requested = num_samples
        
        # Determine starting point for resume
        start_sample = self.state.get_resume_point()
        if start_sample > 0:
            self.logger.info(f"📝 Resuming from sample {start_sample}")
        
        # Progress tracking
        if self.config.get("progress.show_progress_bars"):
            sample_iterator = tqdm(range(start_sample, num_samples), 
                                 desc="Processing samples",
                                 initial=start_sample, 
                                 total=num_samples)
        else:
            sample_iterator = range(start_sample, num_samples)
        
        try:
            for sample_id in sample_iterator:
                # Skip if already completed
                if self.state.is_sample_completed(sample_id):
                    self.logger.info(f"⏭️  Skipping completed sample {sample_id}")
                    continue
                
                self.logger.info(f"🔄 Processing sample {sample_id}/{num_samples}")
                
                # Step 1: Generate code
                code_data = self.generate_code_sample(sample_id)
                if not code_data:
                    self.state.mark_sample_failed(sample_id, "Code generation failed")
                    continue
                
                # Step 2: Generate inputs
                inputs_data = self.generate_inputs_for_code(code_data)
                if not inputs_data:
                    self.state.mark_sample_failed(sample_id, "Input generation failed")
                    continue
                
                # Step 3: Execute code with inputs
                execution_results = self.execute_code_with_inputs(code_data, inputs_data)
                
                # Step 4: Save results
                self.save_sample_results(sample_id, code_data, inputs_data, execution_results)
                
                # Periodic state save
                if (sample_id + 1) % save_interval == 0:
                    self.state.save_state()
                    self.logger.info(f"💾 State saved at sample {sample_id}")
            
            # Final state save
            self.state.save_state()
            
            # Generate final report
            self.generate_final_report()
            
            self.logger.info("🎉 Pipeline completed successfully!")
            return True
            
        except KeyboardInterrupt:
            self.logger.info("⏹️  Pipeline interrupted by user")
            self.state.save_state()
            return False
        
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed with error: {e}")
            self.stats.add_error("pipeline", str(e))
            self.state.save_state()
            return False
        
        finally:
            self.cleanup()
    
    def generate_final_report(self):
        """Generate final pipeline report and dataset."""
        self.logger.info("📊 Generating final report...")
        
        self.stats.finalize()
        
        # Save statistics
        stats_file = self.output_dir / "pipeline_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2, default=str)
        
        # Create consolidated dataset
        completed_results = self.state.get_completed_results()
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "pipeline_version": "1.0.0",
                "total_samples": len(completed_results),
                "statistics": self.stats.to_dict()
            },
            "samples": list(completed_results.values())
        }
        
        dataset_file = self.output_dir / "complete_dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        self.logger.info(f"📁 Final dataset saved to {dataset_file}")
        self.logger.info(f"📈 Statistics saved to {stats_file}")
    
    def cleanup(self):
        """Clean up pipeline resources."""
        self.logger.info("🧹 Cleaning up pipeline resources...")
        
        if self.executor:
            self.executor.cleanup()
        
        if self.input_generator:
            self.input_generator.cleanup_temp_files()


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line interface parser."""
    parser = argparse.ArgumentParser(
        description="Code Output Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py --samples 20 --config configs/pipeline.yaml
  python src/pipeline.py --resume --output-dir results/experiment1
  python src/pipeline.py --samples 5 --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--samples", "-n",
        type=int,
        help="Number of code samples to generate"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from previous interrupted run"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    
    return parser


def main():
    """Main entry point for the pipeline."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = CodeOutputPredictionPipeline(config_path=args.config)
        
        # Override config with CLI arguments
        if args.samples:
            pipeline.config.config["pipeline"]["num_samples"] = args.samples
        
        if args.output_dir:
            pipeline.config.config["pipeline"]["output_dir"] = args.output_dir
            pipeline.output_dir = Path(args.output_dir)
            pipeline.output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.log_level:
            pipeline.config.config["logging"]["level"] = args.log_level
            pipeline.logger.setLevel(getattr(logging, args.log_level))
        
        if args.no_progress:
            pipeline.config.config["progress"]["show_progress_bars"] = False
        
        # Run pipeline
        success = pipeline.run_pipeline()
        
        if success:
            print("✅ Pipeline completed successfully!")
            sys.exit(0)
        else:
            print("❌ Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()