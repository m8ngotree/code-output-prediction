#!/usr/bin/env python3
"""
Main entry point for the code-output-prediction project.
"""

import argparse
import os
import sys
from pathlib import Path

# Try to load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def generate_mode(args):
    """Handle generate mode - create synthetic code examples."""
    try:
        from src.generators.code_generator import CodeGenerator
        from src.seeds.seed_manager import SeedManager
        
        print("🚀 Starting code generation...")
        
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Error: OPENAI_API_KEY environment variable not set")
            print("Please set your OpenAI API key:")
            print("  export OPENAI_API_KEY='your-api-key-here'")
            return 1
        
        # Initialize generator
        generator = CodeGenerator(config_path=args.config, api_key=api_key)
        print("✅ CodeGenerator initialized successfully")
        
        if args.count and args.count > 1:
            # Batch generation
            print(f"📝 Generating {args.count} code examples...")
            results = generator.generate_batch(args.count)
            
            successful = sum(1 for r in results if r.get("success", False))
            print(f"✅ Generated {successful}/{len(results)} examples successfully")
            
            for i, result in enumerate(results, 1):
                if result.get("success"):
                    seeds = result["seeds_used"]
                    print(f"  {i}. {seeds['application']} + {seeds['concept']}")
                else:
                    print(f"  {i}. ❌ Failed: {result.get('error', 'Unknown error')}")
        
        else:
            # Single generation
            print("📝 Generating single code example...")
            result = generator.generate_code(
                application=args.application,
                concept=args.concept
            )
            
            if result["success"]:
                print("✅ Code generated successfully!")
                print(f"📁 Code saved to: {result['files']['code']}")
                print(f"📄 Metadata saved to: {result['files']['metadata']}")
                print(f"🌱 Seeds used: {result['seeds_used']}")
                print(f"📊 Code stats: {result['metadata']['code_stats']}")
                
                if args.preview:
                    print("\n--- Generated Code Preview ---")
                    code = result["code"]
                    preview = code[:500] + "..." if len(code) > 500 else code
                    print(preview)
            else:
                print(f"❌ Generation failed: {result['error']}")
                return 1
        
        # Show statistics
        if args.stats:
            print("\n📊 Generation Statistics:")
            stats = generator.get_generation_stats()
            print(f"  Total files: {stats['total_files']}")
            print(f"  Total lines: {stats['total_lines']}")
            print(f"  Total functions: {stats['total_functions']}")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"❌ Error during code generation: {e}")
        return 1


def execute_mode(args):
    """Handle execute mode - run generated code and capture outputs."""
    print("⚙️ Execute mode not yet implemented")
    print("This will run generated code and capture outputs for dataset creation")
    return 0


def verify_mode(args):
    """Handle verify mode - validate generated outputs."""
    print("🔍 Verify mode not yet implemented") 
    print("This will validate generated code outputs and create verification datasets")
    return 0


def list_seeds():
    """List available seeds."""
    try:
        from src.seeds.seed_manager import SeedManager
        
        manager = SeedManager()
        print("🌱 Available Seeds:")
        
        apps = manager.get_all_seeds("applications")
        concepts = manager.get_all_seeds("concepts")
        
        print(f"\n📱 Applications ({len(apps)}):")
        for i, app in enumerate(apps, 1):
            print(f"  {i:2d}. {app}")
        
        print(f"\n🧠 Concepts ({len(concepts)}):")
        for i, concept in enumerate(concepts, 1):
            print(f"  {i:2d}. {concept}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error listing seeds: {e}")


def pipeline_mode(args):
    """Handle pipeline mode - run complete end-to-end pipeline."""
    try:
        from src.pipeline import CodeOutputPredictionPipeline
        
        print("🚀 Starting complete pipeline...")
        
        # Initialize pipeline
        pipeline = CodeOutputPredictionPipeline(config_path=args.config)
        
        # Override config with CLI arguments
        if args.samples:
            pipeline.config.config["pipeline"]["num_samples"] = args.samples
        
        if args.output:
            pipeline.config.config["pipeline"]["output_dir"] = args.output
            pipeline.output_dir = Path(args.output)
            pipeline.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run complete pipeline
        success = pipeline.run_pipeline()
        
        if success:
            print("🎉 Pipeline completed successfully!")
            return 0
        else:
            print("❌ Pipeline failed!")
            return 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        return 1


def main():
    """Main function to run the code output prediction system."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for LLM code execution prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --mode pipeline --samples 10 --config configs/pipeline.yaml
  
  # Individual components
  python main.py --mode generate --count 5
  python main.py --mode generate --application "web scraping" --concept "recursion"
  python main.py --list-seeds
  python main.py --mode generate --preview --stats
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["generate", "execute", "verify", "pipeline"],
        default="pipeline",
        help="Operation mode (default: pipeline)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated",
        help="Output directory (default: data/generated)"
    )
    
    # Pipeline-specific arguments
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples to generate (pipeline mode)"
    )
    
    # Generation-specific arguments
    parser.add_argument(
        "--application",
        type=str,
        help="Specific application seed to use"
    )
    
    parser.add_argument(
        "--concept", 
        type=str,
        help="Specific concept seed to use"
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of examples to generate (default: 1)"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview of generated code"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true", 
        help="Show generation statistics"
    )
    
    parser.add_argument(
        "--list-seeds",
        action="store_true",
        help="List available seeds and exit"
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_seeds:
        list_seeds()
        return 0
    
    print("🤖 Code Output Prediction System")
    print("=" * 40)
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    
    if args.config:
        print(f"Config: {args.config}")
    
    # Route to appropriate handler
    if args.mode == "generate":
        return generate_mode(args)
    elif args.mode == "execute":
        return execute_mode(args)  
    elif args.mode == "verify":
        return verify_mode(args)
    elif args.mode == "pipeline":
        return pipeline_mode(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())