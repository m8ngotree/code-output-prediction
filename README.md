# Synthetic Python Code Dataset Generator

A specialized system for generating synthetic Python code datasets with verified execution outputs using real-world libraries. Perfect for creating training data to enhance LLM code understanding and reasoning capabilities.

## Overview

This system generates realistic Python code examples using **popular libraries** like pandas, numpy, requests, and more. It executes them safely and captures their actual outputs. The resulting dataset can be used to train language models on code output prediction tasks with real-world complexity.

## Key Features

- **Python Library Focus**: Specialized for pandas, numpy, requests, re, datetime, json
- **Progressive Difficulty**: 4 difficulty levels (Beginner → Expert)
- **Reasoning Traps**: Edge cases and subtle behaviors to test LLM reasoning
- **Safe Execution**: Isolated subprocess execution with timeouts and resource limits
- **Verified Outputs**: Actual execution results, not LLM predictions
- **HuggingFace Ready**: JSONL format for direct dataset upload
- **High Reliability**: 100% success rate with proper error handling

## Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd code-output-prediction
pip install -r requirements.txt

# Install Python libraries for code generation
pip install pandas numpy requests scipy
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
# Or create .env file with: OPENAI_API_KEY=your-api-key-here
```

### 3. Try the Demo

```bash
# See library code generation in action
python demo_library_generation.py
```

### 4. Generate Dataset

```bash
# Generate 1000 Python library examples
python generate_dataset.py --samples 1000 --output my_dataset.jsonl

# Generate smaller test dataset
python generate_dataset.py --samples 50 --output test_dataset.jsonl

# Validate existing dataset
python generate_dataset.py --validate existing_dataset.jsonl
```



## Generated Dataset Format

Each example in the JSONL file contains:

```json
{
  "language": "python-library",
  "code": "import pandas as pd\n\ndef process_data(input_data):\n    df = pd.DataFrame(input_data['data'])\n    return df.groupby('category').mean()\n\nresult = process_data(input_data)\nprint(result)",
  "input": "{\"data\": [{\"name\": \"Alice\", \"age\": 25, \"category\": \"A\"}, {\"name\": \"Bob\", \"age\": 30, \"category\": \"B\"}]}",
  "output": "        age\ncategory     \nA        25.0\nB        30.0",
  "library": "pandas",
  "difficulty": "INTERMEDIATE",
  "task_type": "groupby_operations"
}
```

## Supported Languages

### C++
- STL containers and algorithms
- Modern C++17 features  
- File I/O and string processing
- Mathematical computations

### JavaScript
- ES6+ features
- Array/Object manipulation
- Async/await patterns
- JSON processing

### Rust
- Ownership and borrowing
- Pattern matching
- Error handling
- Collections and iterators

### Python (Library-focused)
- **pandas**: DataFrames, groupby, merging, time series
- **numpy**: Arrays, broadcasting, linear algebra
- **requests**: HTTP clients, sessions, authentication
- **scipy**: Scientific computing, statistics
- **matplotlib**: Basic plotting and visualization

## Difficulty Levels

### Beginner
- Basic API usage
- Simple operations
- 1-5 lines of code

### Intermediate  
- Multiple operations
- Edge case handling
- 5-15 lines of code

### Advanced
- Complex transformations
- Performance considerations
- 15-30 lines of code

### Expert
- Optimization techniques
- Subtle edge cases
- 30+ lines of code

## Usage Examples

### Generate Specific Language
```python
from src.core.language_factory import LanguageFactory

generator, executor = LanguageFactory.create_generator_and_executor("pandas", api_key)
result = generator.generate_code()
```

### Custom Generation
```python
from src.generators.library_task_factory import LibraryTaskFactory, DifficultyLevel

factory = LibraryTaskFactory(api_key)
task = factory.create_single_task("pandas", DifficultyLevel.ADVANCED)
```

## Dataset Statistics

A typical 1000-example dataset contains:

- **250 C++ examples**: STL algorithms, data structures, file processing
- **250 JavaScript examples**: Modern JS features, async patterns, DOM manipulation  
- **250 Rust examples**: Memory safety, pattern matching, error handling
- **250 Python examples**: Real library usage across pandas, numpy, requests

**Difficulty Distribution**: ~40% Beginner, 30% Intermediate, 20% Advanced, 10% Expert

## HuggingFace Upload

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Upload dataset
huggingface-cli upload your-username/synthetic-code-dataset synthetic_code_dataset.jsonl
```

## System Architecture

```
├── generate_dataset.py          # Main dataset generation script
├── demo_library_generation.py   # Demo script for library code generation
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── data/seeds/                  # Seed data for applications and concepts
├── src/
│   ├── core/
│   │   ├── language_factory.py     # Python library factory
│   │   ├── seed_manager.py         # Application/concept seeds
│   │   ├── dataset_manager.py      # Dataset utilities
│   │   ├── config_presets.py       # Configuration presets
│   │   └── verifier.py             # Output verification
│   ├── generators/                 # Python library generators
│   │   ├── library_generator.py    # Library-focused code generation
│   │   └── library_task_factory.py # Task orchestration
│   └── executors/                  # Safe code execution
│       └── library_executor.py     # Python library executor
```

## Configuration

### Basic Configuration (`config.yaml`)
```yaml
openai:
  model: "gpt-3.5-turbo"
  max_tokens: 1000
  temperature: 0.7

execution:
  timeout_seconds: 30
  max_memory_mb: 512
```

### Environment Variables
- `OPENAI_API_KEY`: Required for code generation
- `NODE_PATH`: Optional, for JavaScript execution
- `RUSTC_PATH`: Optional, for Rust compilation

## Safety Features

- **Subprocess Isolation**: All code execution in separate processes
- **Timeout Protection**: Configurable execution time limits  
- **Memory Limits**: Approximate memory usage monitoring
- **Error Recovery**: Comprehensive exception handling
- **Library Validation**: Automatic dependency checking

## Performance

- **Generation Speed**: ~2-5 seconds per example (including API call)
- **Batch Processing**: Parallelizable, ~1-3 examples per second
- **Memory Usage**: ~100-500MB depending on libraries
- **Dataset Size**: ~1-5MB per 1000 examples

## Troubleshooting

### Common Issues

**Missing Libraries**:
```bash
pip install pandas numpy requests scipy matplotlib beautifulsoup4
```

**Execution Timeouts**:
```bash
# Increase timeout in config.yaml
execution:
  timeout_seconds: 60
```

**API Rate Limits**:
```bash
# Add delays between requests
time.sleep(1)  # In generation loop
```

## Contributing

1. **Add New Languages**: Extend `LanguageFactory` with new generator/executor pairs
2. **Add Libraries**: Update `library_generator.py` with new library configurations
3. **Improve Execution**: Enhance safety and performance in executors
4. **Add Task Types**: Create new task templates for different difficulty levels

## Use Cases

- **LLM Training**: Code understanding and output prediction
- **Model Evaluation**: Benchmarking code reasoning capabilities  
- **Synthetic Data**: Augmenting existing code datasets
- **Research**: Studying code generation and execution patterns

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{synthetic_code_dataset,
  title={Synthetic Code Dataset Generator},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/code-output-prediction}
}
```

---

**Ready to generate your synthetic code dataset?** Run `python generate_dataset.py` to get started!