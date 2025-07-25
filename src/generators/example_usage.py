"""
Example usage of the CodeGenerator class.

This script demonstrates how to use the CodeGenerator to create
synthetic Python programs based on application and concept seeds.
"""

import os
from pathlib import Path

from code_generator import CodeGenerator


def main():
    """Demonstrate CodeGenerator usage."""
    
    # Set up API key (you'll need to provide your own)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Initialize generator
        print("Initializing CodeGenerator...")
        generator = CodeGenerator(api_key=api_key)
        
        # Generate single code example
        print("\n=== Generating single code example ===")
        result = generator.generate_code(
            application="web scraping",
            concept="recursion"
        )
        
        if result["success"]:
            print(f"✅ Generated code successfully!")
            print(f"📁 Code saved to: {result['files']['code']}")
            print(f"📄 Metadata saved to: {result['files']['metadata']}")
            print(f"🌱 Seeds used: {result['seeds_used']}")
            print("\n--- Generated Code Preview ---")
            print(result["code"][:500] + "..." if len(result["code"]) > 500 else result["code"])
        else:
            print(f"❌ Generation failed: {result['error']}")
        
        # Generate random example
        print("\n=== Generating random example ===")
        result = generator.generate_code()  # Random seeds
        
        if result["success"]:
            print(f"✅ Generated code successfully!")
            print(f"🌱 Random seeds used: {result['seeds_used']}")
            print(f"📊 Code stats: {result['metadata']['code_stats']}")
        
        # Generate batch
        print("\n=== Generating batch of examples ===")
        results = generator.generate_batch(count=3)
        
        successful = sum(1 for r in results if r.get("success", False))
        print(f"✅ Generated {successful}/{len(results)} examples successfully")
        
        for i, result in enumerate(results, 1):
            if result.get("success"):
                seeds = result["seeds_used"]
                print(f"  {i}. {seeds['application']} + {seeds['concept']}")
        
        # Show generation statistics
        print("\n=== Generation Statistics ===")
        stats = generator.get_generation_stats()
        print(f"📁 Total files generated: {stats['total_files']}")
        print(f"📝 Total lines of code: {stats['total_lines']}")
        print(f"🔧 Total functions: {stats['total_functions']}")
        print(f"🌱 Unique applications used: {stats['unique_applications']}")
        print(f"🧠 Unique concepts used: {stats['unique_concepts']}")
        
        if stats['applications_used']:
            print(f"\n📋 Applications: {', '.join(stats['applications_used'][:5])}")
        if stats['concepts_used']:
            print(f"💡 Concepts: {', '.join(stats['concepts_used'][:5])}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()