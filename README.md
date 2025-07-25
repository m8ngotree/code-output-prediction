# Code Output Prediction

A Python project for generating synthetic datasets where LLMs predict code execution outputs using OpenAI's GPT-4.

## Features

- **🔄 Complete End-to-End Pipeline**: Orchestrates code generation, input generation, and execution
- **🌱 Seed Management System**: Manages application and programming concept seeds for code generation
- **🤖 AI-Powered Code Generation**: Uses OpenAI's GPT-4 to generate complex Python programs
- **🎯 Smart Input Generation**: AST-based analysis to generate diverse test inputs
- **🔒 Secure Code Execution**: Safe execution with timeouts and resource limits
- **📊 Progress Tracking**: Real-time progress bars and comprehensive statistics
- **💾 Resume Capability**: Resume interrupted runs from any point
- **⚙️ Configurable**: Fully customizable via YAML configuration files
- **📈 Rich Metadata**: Comprehensive tracking of generation parameters, timing, and statistics

## Project Structure

```
code-output-prediction/
├── src/
│   ├── pipeline.py              # 🔄 Main pipeline controller
│   ├── generators/
│   │   ├── code_generator.py    # 🤖 AI code generation engine
│   │   ├── input_generator.py   # 🎯 Smart input generation
│   │   └── example_usage.py     # Usage examples
│   ├── executors/
│   │   ├── python_executor.py   # 🔒 Secure code execution
│   │   └── executor_example.py  # Executor examples
│   ├── seeds/
│   │   └── seed_manager.py      # 🌱 Seed data management
│   ├── verifiers/               # Output verification (future)
│   └── utils/                   # Utility functions
├── data/
│   ├── seeds/                   # Seed data (applications, concepts)
│   ├── generated/               # Generated code examples
│   ├── inputs/                  # Generated test inputs
│   ├── pipeline_results/        # Complete pipeline outputs
│   └── datasets/                # Final datasets
├── configs/
│   ├── pipeline.yaml            # 🔄 Pipeline configuration
│   ├── code_generator.yaml      # Code generation settings
│   └── python_executor.yaml     # Execution settings
├── tests/                       # Comprehensive unit tests
├── requirements.txt             # Python dependencies
├── main.py                     # CLI entry point
└── pipeline_example.py         # Pipeline demonstration
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd code-output-prediction
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Quick Start

### 🚀 Complete Pipeline (Recommended)

Run the end-to-end pipeline to generate synthetic datasets:

```bash
# Generate 10 code samples with inputs and execution results
python main.py --mode pipeline --samples 10

# Use custom configuration and output directory
python main.py --mode pipeline --samples 20 --config configs/pipeline.yaml --output results/experiment1

# Resume an interrupted run
python main.py --mode pipeline --samples 50 --output results/experiment1
```

**Pipeline Output:**
- `data/pipeline_results/complete_dataset.json` - Final consolidated dataset
- `data/pipeline_results/sample_XXXX.json` - Individual sample results
- `data/pipeline_results/pipeline_statistics.json` - Performance metrics
- `data/pipeline_results/pipeline_state.json` - Resume state

### 📊 What the Pipeline Does

1. **🤖 Code Generation**: Uses GPT-4 to generate Python programs from random seed combinations
2. **🎯 Input Generation**: Analyzes code to generate diverse test inputs (normal, edge, boundary cases)
3. **⚡ Secure Execution**: Runs code with all inputs in isolated processes with timeouts
4. **💾 Data Collection**: Stores code, inputs, outputs, and metadata in structured format

### 🔧 Individual Components

### Using the SeedManager

```python
from src.seeds.seed_manager import SeedManager

# Initialize seed manager
manager = SeedManager()

# Get available seeds
apps = manager.get_all_seeds("applications")
concepts = manager.get_all_seeds("concepts")

# Sample random seeds
sample_apps = manager.sample_seeds("applications", 3)
sample_concepts = manager.sample_seeds("concepts", 3)
```

### Generating Code

```python
from src.generators.code_generator import CodeGenerator

# Initialize generator (requires OpenAI API key)
generator = CodeGenerator()

# Generate single example
result = generator.generate_code(
    application="web scraping",
    concept="recursion"
)

if result["success"]:
    print(f"Generated code saved to: {result['files']['code']}")
    print(result["code"])

# Generate batch
results = generator.generate_batch(count=5)
```

## Configuration

The system is configured via `configs/code_generator.yaml`:

- **OpenAI Settings**: Model, temperature, token limits
- **Generation Parameters**: Function count, code structure preferences
- **Output Settings**: File naming, metadata inclusion
- **Retry Logic**: API failure handling

## Current Implementation Status

### ✅ Completed Components

1. **Seed Management System** (`src/seeds/seed_manager.py`)
   - Load/save seed data from JSON files
   - Sample seeds with/without replacement
   - Add/remove seeds dynamically
   - Full test coverage

2. **Code Generator** (`src/generators/code_generator.py`)
   - OpenAI GPT-4 integration
   - Template-based prompt system
   - Code post-processing and validation
   - Metadata tracking and file persistence
   - Retry logic for API failures
   - Batch generation capabilities

3. **Configuration System**
   - YAML-based configuration
   - Customizable prompts and parameters

4. **Test Suite**
   - Comprehensive unit tests
   - Mock-based testing for API components

### 🚧 Future Components

- **Code Executors**: Run generated code and capture outputs
- **Output Verifiers**: Validate execution results
- **Dataset Builders**: Create training datasets from generated examples

## Usage Examples

### Command Line Interface

```bash
# Generate code with specific seeds
python main.py --mode generate --application "web scraping" --concept "recursion"

# Generate batch of examples
python main.py --mode generate --count 10

# Use custom configuration
python main.py --config custom_config.yaml --mode generate
```

### Python API

```python
# See src/generators/example_usage.py for detailed examples
python src/generators/example_usage.py
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m unittest discover tests -v

# Run specific test modules
python -m unittest tests.test_seed_manager -v
python -m unittest tests.test_code_generator -v
```

## Data

### Seed Data

The system includes predefined seeds in `data/seeds/`:

- **Applications** (10 types): web scraping, data analysis, game development, cryptography, etc.
- **Concepts** (10 types): recursion, dynamic programming, graph algorithms, etc.

### Generated Examples

Generated code is saved to `data/generated/` with:
- Python source files
- JSON metadata files with generation details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license here]

## Requirements

- Python 3.8+
- OpenAI API key
- See `requirements.txt` for full dependency list