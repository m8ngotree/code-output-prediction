#!/usr/bin/env python3
"""
Library Code Generation Demo

This script demonstrates how to generate library-focused code examples
with verified execution outputs using the streamlined system.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.generators.library_task_factory import LibraryTaskFactory, DifficultyLevel


def demo_library_generation():
    """Demonstrate library code generation with different difficulty levels."""
    print("Library Code Generation Demo")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create factory
    factory = LibraryTaskFactory(api_key)
    
    # Check available libraries
    available_libs = factory.get_available_libraries()
    print(f"Available libraries: {available_libs}")
    
    if not available_libs:
        print("No libraries available. Please install: pip install pandas numpy requests")
        return
    
    print("\nGenerating examples with different difficulty levels...")
    print("-" * 50)
    
    # Generate examples for each difficulty level
    difficulty_levels = [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE, DifficultyLevel.ADVANCED]
    
    for difficulty in difficulty_levels:
        print(f"\n{difficulty.name} Level Example:")
        print("-" * 30)
        
        try:
            # Generate a task
            task = factory.create_single_task("pandas", difficulty)
            
            if task["success"]:
                generation = task["generation"]
                execution = task["execution"]
                
                print(f"Task Type: {generation.get('task_type', 'Unknown')}")
                print(f"Reasoning Traps: {generation.get('reasoning_traps', [])}")
                print(f"\nGenerated Code:")
                print("```python")
                print(generation["code"])
                print("```")
                
                print(f"\nExecution: {'✓ Success' if execution['success'] else '✗ Failed'}")
                
                if execution["success"]:
                    for i, result in enumerate(execution["results"]):
                        if result["success"]:
                            print(f"Output {i+1}: {result['output'][:100]}...")
                            break
                else:
                    print(f"Error: {execution['results'][0].get('error', 'Unknown error')}")
                    
            else:
                print(f"Generation failed: {task.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error generating {difficulty.name} example: {e}")
    
    print("\n" + "=" * 50)
    print("Demo complete! This shows how the system generates")
    print("library-focused code with varying complexity levels.")
    print("\nTo generate a full dataset:")
    print("python generate_dataset.py --samples 250 --output dataset.jsonl")


if __name__ == "__main__":
    demo_library_generation()
