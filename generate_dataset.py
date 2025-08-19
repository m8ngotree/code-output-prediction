#!/usr/bin/env python3
"""
Synthetic Code Dataset Generator

Generates a JSONL dataset of code examples with verified execution outputs
for training language models on code understanding and output prediction.

Usage: python generate_dataset.py [--samples N] [--output filename.jsonl]
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import argparse
from tqdm import tqdm
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.language_factory import LanguageFactory


class SyntheticDatasetGenerator:
    """
    Generates synthetic code datasets with verified execution outputs.
    
    Creates code examples using Python libraries (pandas, numpy, requests, etc.)
    and captures their actual execution outputs for training language models.
    """
    
    def __init__(self, api_key: str):
        """Initialize the dataset generator."""
        self.api_key = api_key
        self.supported_languages = ["python-library"]
        
    def generate_single_example(self, language: str) -> Dict[str, Any]:
        """
        Generate a single Python library code example with verified output.
        
        Args:
            language: Should be "python-library"
            
        Returns:
            Dict with code, input, output, and metadata
        """
        try:
            # Create generator and executor for python-library
            generator, executor = LanguageFactory.create_generator_and_executor(language, self.api_key)
            
            # Generate and execute the library task
            result = generator.create_single_task()
            if not result["success"]:
                return {"success": False, "error": result.get("error", "Generation failed")}
            
            generation = result["generation"]
            execution = result["execution"]
            
            # Find the first successful execution
            successful_execution = None
            for exec_result in execution["results"]:
                if exec_result["success"]:
                    successful_execution = exec_result
                    break
            
            if not successful_execution:
                return {"success": False, "error": "No successful execution found"}
            
            return {
                "success": True,
                "language": "python",  # Use "python" for consistency
                "library": generation.get("library"),
                "difficulty": generation.get("difficulty"),
                "task_type": generation.get("task_type"),
                "code": generation["code"],
                "input": successful_execution.get("input_used"),
                "output": successful_execution["output"],
                "reasoning_traps": generation.get("reasoning_traps", []),
                "application": generation.get("application"),
                "concept": generation.get("concept"),
                "execution_time": successful_execution.get("execution_time")
            }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_dataset(self, total_samples: int = 1000, 
                        output_file: str = "synthetic_code_dataset.jsonl") -> Dict[str, Any]:
        """
        Generate a complete dataset with Python library examples.
        
        Args:
            total_samples: Total number of examples to generate
            output_file: Output JSONL filename
            
        Returns:
            Generation statistics
        """
        generated_examples = []
        stats = {
            "total_attempted": 0,
            "total_successful": 0,
            "start_time": time.time()
        }
        
        print(f"Generating {total_samples} Python library code examples")
        print(f"Output file: {output_file}")
        print("=" * 60)
        
        # Generate examples
        with tqdm(total=total_samples, desc="Generating Python library examples") as pbar:
            for i in range(total_samples):
                stats["total_attempted"] += 1
                
                example = self.generate_single_example("python-library")
                
                if example["success"]:
                    # Clean up the example for JSONL output
                    clean_example = {
                        "language": example["language"],
                        "code": example["code"],
                        "input": example.get("input"),
                        "output": example["output"]
                    }
                    
                    # Add optional fields if present
                    for optional_field in ["library", "difficulty", "task_type", "application", 
                                         "concept", "reasoning_traps", "execution_time"]:
                        if optional_field in example and example[optional_field]:
                            clean_example[optional_field] = example[optional_field]
                    
                    generated_examples.append(clean_example)
                    stats["total_successful"] += 1
                else:
                    # Log first few errors for debugging
                    failed_count = stats["total_attempted"] - stats["total_successful"]
                    if failed_count <= 5:
                        error_msg = example.get("error", "Unknown error")
                        tqdm.write(f"Failed example {i+1}: {error_msg}")
                
                pbar.update(1)
                if stats["total_attempted"] > 0:
                    success_rate = stats["total_successful"] / stats["total_attempted"] * 100
                    pbar.set_postfix({'success_rate': f"{success_rate:.1f}%"})
        
        # Save to JSONL file
        print(f"\nSaving {len(generated_examples)} examples to {output_file}...")
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in generated_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Calculate final statistics
        stats["end_time"] = time.time()
        stats["duration_minutes"] = (stats["end_time"] - stats["start_time"]) / 60
        stats["success_rate"] = stats["total_successful"] / stats["total_attempted"] * 100
        stats["output_file"] = str(output_path.absolute())
        stats["file_size_mb"] = output_path.stat().st_size / (1024 * 1024)
        
        return stats
    
    def validate_dataset(self, jsonl_file: str) -> Dict[str, Any]:
        """
        Validate a generated JSONL dataset.
        
        Args:
            jsonl_file: Path to JSONL file to validate
            
        Returns:
            Validation results
        """
        print(f"Validating dataset: {jsonl_file}")
        
        validation = {
            "total_examples": 0,
            "valid_examples": 0,
            "invalid_examples": 0,
            "languages": {},
            "errors": []
        }
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    validation["total_examples"] += 1
                    
                    try:
                        example = json.loads(line.strip())
                        
                        # Check required fields
                        required_fields = ["language", "code", "output"]
                        missing_fields = [field for field in required_fields if field not in example]
                        
                        if missing_fields:
                            validation["invalid_examples"] += 1
                            validation["errors"].append(f"Line {line_num}: Missing fields: {missing_fields}")
                        else:
                            validation["valid_examples"] += 1
                            
                            # Count by language
                            lang = example["language"]
                            if lang not in validation["languages"]:
                                validation["languages"][lang] = 0
                            validation["languages"][lang] += 1
                    
                    except json.JSONDecodeError as e:
                        validation["invalid_examples"] += 1
                        validation["errors"].append(f"Line {line_num}: JSON decode error: {e}")
        
        except FileNotFoundError:
            validation["errors"].append(f"File not found: {jsonl_file}")
            return validation
        
        validation["is_valid"] = validation["invalid_examples"] == 0
        validation["success_rate"] = validation["valid_examples"] / validation["total_examples"] * 100
        
        return validation


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic code dataset for HuggingFace")
    parser.add_argument("--samples", type=int, default=1000, 
                       help="Total number of samples to generate (default: 1000)")
    parser.add_argument("--output", type=str, default="synthetic_code_dataset.jsonl",
                       help="Output JSONL filename (default: synthetic_code_dataset.jsonl)")
    parser.add_argument("--validate", type=str, 
                       help="Validate an existing JSONL file instead of generating")
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not args.validate:
        print("Error: OPENAI_API_KEY environment variable is required for generation")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    if args.validate:
        # Validation mode
        generator = SyntheticDatasetGenerator(api_key or "dummy")
        validation = generator.validate_dataset(args.validate)
        
        print("\nValidation Results:")
        print("=" * 40)
        print(f"Total examples: {validation['total_examples']}")
        print(f"Valid examples: {validation['valid_examples']}")
        print(f"Invalid examples: {validation['invalid_examples']}")
        print(f"Success rate: {validation['success_rate']:.1f}%")
        print(f"Dataset valid: {'✓' if validation['is_valid'] else '✗'}")
        
        print("\nLanguage distribution:")
        for lang, count in validation['languages'].items():
            print(f"  {lang}: {count} examples")
        
        if validation['errors']:
            print(f"\nErrors ({len(validation['errors'])}):")
            for error in validation['errors'][:10]:  # Show first 10 errors
                print(f"  {error}")
            if len(validation['errors']) > 10:
                print(f"  ... and {len(validation['errors']) - 10} more errors")
        
    else:
        # Generation mode
        generator = SyntheticDatasetGenerator(api_key)
        
        print("Synthetic Code Dataset Generator")
        print("=" * 40)
        print("Generating dataset for HuggingFace upload...")
        
        try:
            stats = generator.generate_dataset(args.samples, args.output)
            
            # Print final statistics
            print("\n" + "=" * 60)
            print("DATASET GENERATION COMPLETE")
            print("=" * 60)
            print(f"Total examples generated: {stats['total_successful']}/{stats['total_attempted']}")
            print(f"Overall success rate: {stats['success_rate']:.1f}%")
            print(f"Generation time: {stats['duration_minutes']:.1f} minutes")
            print(f"Output file: {stats['output_file']}")
            print(f"File size: {stats['file_size_mb']:.1f} MB")
            
            print(f"\nPython library examples generated successfully!")
            
            print(f"\nDataset ready for HuggingFace upload!")
            print(f"Upload command: huggingface-cli upload your-username/synthetic-code-dataset {args.output}")
            
        except KeyboardInterrupt:
            print("\nGeneration interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\nError during generation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
