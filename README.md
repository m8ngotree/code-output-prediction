# Code Output Prediction

A Python project for generating synthetic datasets where LLMs predict code execution outputs using OpenAI's GPT-4.

## Features

- **Seed Management System**: Manages application and programming concept seeds for code generation
- **AI-Powered Code Generation**: Uses OpenAI's GPT-4 to generate complex Python programs
- **Post-Processing Pipeline**: Ensures generated code is executable with proper imports and structure
- **Metadata Tracking**: Comprehensive tracking of generation parameters, timing, and statistics
- **Configurable**: Fully customizable via YAML configuration files
- **Batch Generation**: Generate multiple code examples efficiently

## Project Structure

```
code-output-prediction/
├── src/
│   ├── generators/
│   │   ├── code_generator.py    # Main code generation engine
│   │   └── example_usage.py     # Usage examples
│   ├── seeds/
│   │   └── seed_manager.py      # Seed data management
│   ├── executors/               # Code execution (future)
│   ├── verifiers/               # Output verification (future)
│   └── utils/                   # Utility functions
├── data/
│   ├── seeds/                   # Seed data (applications, concepts)
│   ├── generated/               # Generated code examples
│   └── datasets/                # Final datasets
├── configs/
│   └── code_generator.yaml      # Configuration file
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
└── main.py                     # CLI entry point
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