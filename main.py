#!/usr/bin/env python3
"""
Code Output Prediction System - Main Entry Point

A streamlined system for generating Python code using OpenAI's API,
executing it safely, and verifying outputs.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import pickle
from datetime import datetime

from tqdm import tqdm
from src.core.language_factory import LanguageFactory
from src.core.verifier import OutputVerifier


class CodePredictionSystem:
    """
    Main system orchestrating code generation, execution, and verification.
    
    This class provides a complete pipeline for:
    - Generating code samples using language-specific generators
    - Executing code safely in isolated environments
    - Verifying outputs and managing results
    - Supporting checkpointing for long-running batch operations
    """
    
    def __init__(self, api_key: str, language: str = "python"):
        """
        Initialize the system components.
        
        Args:
            api_key (str): OpenAI API key for code generation
            language (str): Programming language to generate (default: "python")
        """
        self.language = language
        self.generator, self.executor = LanguageFactory.create_generator_and_executor(language, api_key)
        self.verifier = OutputVerifier()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def generate_single_sample(self) -> Dict[str, Any]:
        """
        Generate, execute, and verify a single code sample with multiple test inputs.
        
        Returns:
            Dict[str, Any]: Complete sample result including generation, execution, and verification
        """
        # Generate code with test inputs
        result = self.generator.generate_code()
        if not result["success"]:
            return result
        
        # Execute code with each test input
        execution_results = []
        test_inputs = result.get("test_inputs", [])
        
        if test_inputs:
            # Use progress bar for test execution
            test_bar = tqdm(test_inputs, desc="Running tests", leave=False)
            for test_input in test_bar:
                test_bar.set_postfix_str(test_input['description'][:30])
                execution_result = self.executor.execute(result["code"], test_input["value"])
                execution_result["test_description"] = test_input["description"]
                execution_result["input_used"] = test_input["value"]
                execution_results.append(execution_result)
        else:
            # Fallback: execute without input
            execution_result = self.executor.execute(result["code"])
            execution_result["test_description"] = "No input test"
            execution_result["input_used"] = None
            execution_results.append(execution_result)
        
        # Check if any execution was successful
        any_success = any(exec_result["success"] for exec_result in execution_results)
        
        # Combine results
        sample = {
            "generation": result,
            "execution_results": execution_results,
            "success": result["success"] and any_success
        }
        
        return sample
    
    def generate_batch(self, count: int, checkpoint_interval: int = 10, resume: bool = False) -> List[Dict[str, Any]]:
        """
        Generate multiple code samples with progress tracking and checkpointing.
        
        Args:
            count (int): Total number of samples to generate
            checkpoint_interval (int): Save checkpoint every N samples (default: 10)
            resume (bool): Whether to resume from existing checkpoint (default: False)
        
        Returns:
            List[Dict[str, Any]]: List of generated samples
        """
        samples = []
        successful = 0
        start_index = 0
        
        # Load checkpoint if resuming
        checkpoint_file = self.checkpoint_dir / f"batch_{self.language}_{count}.pkl"
        if resume and checkpoint_file.exists():
            checkpoint_data = self._load_checkpoint(checkpoint_file)
            samples = checkpoint_data["samples"]
            successful = checkpoint_data["successful"]
            start_index = len(samples)
            print(f"Resuming from checkpoint: {start_index}/{count} samples completed")
        
        # Generate remaining samples with progress bar
        remaining = count - start_index
        if remaining > 0:
            progress_bar = tqdm(range(start_index, count), 
                              desc=f"Generating {self.language} samples",
                              initial=start_index, 
                              total=count)
            
            for i in progress_bar:
                sample = self.generate_single_sample()
                samples.append(sample)
                
                if sample["success"]:
                    successful += 1
                    # Save successful sample
                    self._save_sample(sample, i)
                
                # Update progress bar with current stats
                progress_bar.set_postfix({
                    'success': f"{successful}/{len(samples)}",
                    'rate': f"{(successful/len(samples)*100):.1f}%"
                })
                
                # Save checkpoint periodically
                if (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(checkpoint_file, samples, successful)
        
        # Save final checkpoint and cleanup
        if checkpoint_file.exists():
            checkpoint_file.unlink()  # Remove checkpoint when complete
        
        return samples
    
    def _save_checkpoint(self, checkpoint_file: Path, samples: List[Dict[str, Any]], successful: int):
        """
        Save current progress to checkpoint file.
        
        Args:
            checkpoint_file (Path): Path to checkpoint file
            samples (List[Dict[str, Any]]): Current samples
            successful (int): Number of successful samples
        """
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "language": self.language,
            "samples": samples,
            "successful": successful,
            "total_attempted": len(samples)
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def _load_checkpoint(self, checkpoint_file: Path) -> Dict[str, Any]:
        """
        Load checkpoint data from file.
        
        Args:
            checkpoint_file (Path): Path to checkpoint file
            
        Returns:
            Dict[str, Any]: Checkpoint data
        """
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    
    def _save_sample(self, sample: Dict[str, Any], index: int):
        """
        Save a sample to the output directory with comprehensive metadata.
        
        Args:
            sample (Dict[str, Any]): Sample data to save
            index (int): Sample index for filename generation
        """
        filename = f"sample_{index:04d}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(sample, f, indent=2)
        
        # Also save the code separately for easy viewing
        language = sample["generation"]["language"]
        extension = self._get_file_extension(language)
        code_filename = f"sample_{index:04d}.{extension}"
        code_filepath = self.output_dir / code_filename
        
        with open(code_filepath, 'w') as f:
            f.write(sample["generation"]["code"])
        
        # Also save a summary of execution results for easy viewing
        summary_filename = f"sample_{index:04d}_summary.txt"
        summary_filepath = self.output_dir / summary_filename
        
        with open(summary_filepath, 'w') as f:
            f.write(f"Language: {sample['generation']['language']}\n")
            f.write(f"Application: {sample['generation']['application']}\n")
            f.write(f"Concept: {sample['generation']['concept']}\n")
            f.write(f"Success: {sample['success']}\n\n")
            
            f.write("EXECUTION RESULTS:\n")
            f.write("=" * 50 + "\n")
            
            for i, exec_result in enumerate(sample["execution_results"], 1):
                f.write(f"\nTest {i}: {exec_result['test_description']}\n")
                f.write(f"Input: {exec_result['input_used']}\n")
                f.write(f"Success: {exec_result['success']}\n")
                if exec_result['success']:
                    f.write(f"Output: {exec_result['output']}\n")
                else:
                    f.write(f"Error: {exec_result['error']}\n")
    
    def _get_file_extension(self, language: str) -> str:
        """
        Get file extension for programming language.
        
        Args:
            language (str): Programming language name
            
        Returns:
            str: File extension for the language
        """
        extensions = {
            "python": "py",
            "javascript": "js", 
            "rust": "rs",
            "cpp": "cpp"
        }
        return extensions.get(language, "txt")


def main():
    """
    Main entry point with enhanced argument parsing and checkpoint support.
    
    Usage: python main.py [api_key] [language] [count] [--resume] [--checkpoint-interval N]
    """
    # Parse command line arguments
    api_key = os.getenv("OPENAI_API_KEY")
    language = "python"  # default
    count = 5  # default
    resume = False
    checkpoint_interval = 10
    
    # Parse arguments
    args = sys.argv[1:]
    
    # Handle flags
    if "--resume" in args:
        resume = True
        args.remove("--resume")
    
    if "--checkpoint-interval" in args:
        idx = args.index("--checkpoint-interval")
        if idx + 1 < len(args):
            try:
                checkpoint_interval = int(args[idx + 1])
                args.pop(idx)  # Remove flag
                args.pop(idx)  # Remove value
            except ValueError:
                print("Error: Checkpoint interval must be an integer")
                sys.exit(1)
        else:
            print("Error: --checkpoint-interval requires a value")
            sys.exit(1)
    
    # Parse positional arguments
    if not api_key and args:
        api_key = args.pop(0)
    
    if args:
        potential_lang = args[0].lower()
        if potential_lang in LanguageFactory.get_supported_languages():
            language = args.pop(0)
    
    if args:
        try:
            count = int(args[0])
        except ValueError:
            print("Error: Count must be an integer")
            sys.exit(1)
    
    if not api_key:
        print("Error: OpenAI API key required")
        print("Usage: python main.py [api_key] [language] [count] [--resume] [--checkpoint-interval N]")
        print(f"Supported languages: {', '.join(LanguageFactory.get_supported_languages())}")
        print("Set OPENAI_API_KEY environment variable or pass as argument")
        sys.exit(1)
    
    print(f"Code Output Prediction System")
    print(f"Language: {language}")
    print(f"Generating {count} samples...")
    if resume:
        print("Resume mode enabled - will continue from checkpoint if available")
    print(f"Checkpoint interval: {checkpoint_interval} samples")
    
    try:
        # Initialize system
        system = CodePredictionSystem(api_key, language)
        
        # Generate samples with progress tracking and checkpointing
        samples = system.generate_batch(count, checkpoint_interval, resume)
        
        # Print summary
        successful = sum(1 for s in samples if s["success"])
        print(f"\nGeneration Complete!")
        print(f"═" * 50)
        print(f"Language: {language}")
        print(f"Total samples: {len(samples)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(samples) - successful}")
        print(f"Success rate: {successful/len(samples)*100:.1f}%")
        print(f"Output saved to: {system.output_dir}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user - checkpoint saved for resume")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()