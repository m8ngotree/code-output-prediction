#!/usr/bin/env python3
"""
Example script demonstrating the complete pipeline.

This script shows how to run the code-output-prediction pipeline
without requiring API access for demonstration purposes.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def demo_pipeline_structure():
    """Demonstrate the pipeline structure and components."""
    print("🔧 Code Output Prediction Pipeline Demo")
    print("=" * 50)
    
    try:
        from pipeline import (
            PipelineConfig, PipelineStatistics, PipelineState,
            CodeOutputPredictionPipeline
        )
        
        print("✅ Successfully imported pipeline components")
        
        # Test configuration
        print("\n📋 Testing Configuration System:")
        config = PipelineConfig()
        print(f"  Default samples: {config.get('pipeline.num_samples')}")
        print(f"  Timeout: {config.get('execution.timeout_seconds')}s")
        print(f"  Output dir: {config.get('pipeline.output_dir')}")
        
        # Test statistics
        print("\n📊 Testing Statistics System:")
        stats = PipelineStatistics()
        stats.codes_requested = 5
        stats.codes_generated = 4
        stats.codes_failed = 1
        stats.finalize()
        
        stats_dict = stats.to_dict()
        print(f"  Success rate: {stats_dict['code_generation']['success_rate']:.1f}%")
        print(f"  Total errors: {stats_dict['performance']['total_errors']}")
        
        # Test state management
        print("\n💾 Testing State Management:")
        state = PipelineState("demo_state.json")
        state.mark_sample_completed(1, {"demo": "data"})
        print(f"  Resume point: {state.get_resume_point()}")
        print(f"  Sample 1 completed: {state.is_sample_completed(1)}")
        
        # Cleanup demo state
        Path("demo_state.json").unlink(missing_ok=True)
        
        # Test pipeline initialization (without API key)
        print("\n🏗️  Testing Pipeline Initialization:")
        pipeline = CodeOutputPredictionPipeline()
        print(f"  Logger initialized: {pipeline.logger.name}")
        print(f"  Output directory: {pipeline.output_dir}")
        print("  ⚠️  Note: Full initialization requires OPENAI_API_KEY")
        
        print("\n✅ All pipeline components are working correctly!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 This is expected if dependencies are not installed")
        print("   Run: pip install -r requirements.txt")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def show_pipeline_workflow():
    """Show the complete pipeline workflow."""
    print("\n🔄 Complete Pipeline Workflow:")
    print("=" * 30)
    
    workflow_steps = [
        "1. 🌱 Initialize Components",
        "   - Seed Manager (loads application/concept seeds)",
        "   - Code Generator (connects to OpenAI GPT-4)",
        "   - Input Generator (analyzes code for inputs)",
        "   - Python Executor (secure code execution)",
        "",
        "2. 🔄 For Each Sample:",
        "   a) 📝 Generate Code",
        "      - Sample random application + concept seeds",
        "      - Generate Python code via GPT-4",
        "      - Post-process and validate code",
        "",
        "   b) 🎯 Generate Inputs",
        "      - Parse code with AST analysis",
        "      - Identify input requirements",
        "      - Generate diverse test cases (normal/edge/boundary)",
        "",
        "   c) ⚡ Execute Code",
        "      - Run code with all generated inputs",
        "      - Capture outputs, errors, and metadata",
        "      - Apply security limits and timeouts",
        "",
        "   d) 💾 Save Results",
        "      - Store code, inputs, and execution results",
        "      - Update pipeline state for resume capability",
        "",
        "3. 📊 Generate Report",
        "   - Compile statistics and performance metrics",
        "   - Create consolidated dataset",
        "   - Export results in JSON format"
    ]
    
    for step in workflow_steps:
        print(step)


def show_usage_examples():
    """Show usage examples for the pipeline."""
    print("\n🚀 Usage Examples:")
    print("=" * 20)
    
    examples = [
        "# Run complete end-to-end pipeline",
        "python main.py --mode pipeline --samples 10",
        "",
        "# Use custom configuration",
        "python main.py --mode pipeline --config configs/pipeline.yaml",
        "",
        "# Generate more samples with custom output directory",
        "python main.py --mode pipeline --samples 50 --output results/experiment1",
        "",
        "# Direct pipeline execution",
        "python src/pipeline.py --samples 5 --log-level DEBUG",
        "",
        "# Individual components (legacy mode)",
        "python main.py --mode generate --count 5",
        "python main.py --list-seeds",
    ]
    
    for example in examples:
        if example.startswith("#"):
            print(f"\n💡 {example}")
        elif example == "":
            print()
        else:
            print(f"   {example}")


def show_expected_outputs():
    """Show what outputs the pipeline generates."""
    print("\n📁 Expected Output Structure:")
    print("=" * 30)
    
    structure = """
data/pipeline_results/
├── pipeline_state.json          # Resume state
├── pipeline_statistics.json     # Performance metrics
├── complete_dataset.json        # Final consolidated dataset
├── sample_0001.json             # Individual sample results
├── sample_0002.json
├── sample_0003.json
└── ...

Each sample_XXXX.json contains:
├── code_data                    # Generated Python code + metadata
├── inputs_data                  # Generated test inputs
├── execution_results            # All execution outputs
└── summary                      # Success/failure statistics
"""
    
    print(structure)


def main():
    """Run the complete pipeline demonstration."""
    demo_pipeline_structure()
    show_pipeline_workflow()
    show_usage_examples()
    show_expected_outputs()
    
    print("\n🎯 Ready to run the pipeline!")
    print("=" * 30)
    print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run pipeline: python main.py --mode pipeline --samples 5")
    print("")
    print("📚 For more options: python main.py --help")


if __name__ == "__main__":
    main()