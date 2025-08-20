# Synthetic Python Library Code Dataset Generator

A specialized system for generating synthetic Python code datasets with verified execution outputs using real-world libraries. Designed specifically for creating training data to enhance LLM code understanding and output prediction capabilities with popular Python libraries.

## Overview

This system generates realistic Python code examples using **popular libraries** like pandas, numpy, requests, datetime, json, and regex. Each example is executed safely in an isolated environment to capture actual outputs. The resulting dataset can be used to train language models on code output prediction tasks with real-world library complexity and edge cases.

## Key Features

- **Python Library Focus**: Specialized for pandas, numpy, requests, re, datetime, json
- **Progressive Difficulty**: 4 difficulty levels (Beginner → Expert) 
- **Reasoning Traps**: Library-specific edge cases and subtle behaviors to test LLM reasoning
- **Safe Execution**: Isolated subprocess execution with timeouts and memory limits
- **Verified Outputs**: Actual execution results from real library code, not predictions
- **HuggingFace Ready**: JSONL format for direct dataset upload
- **Real Library Usage**: Focuses on practical library APIs and common patterns

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
# Generate 1000 Python library examples (default)
python generate_dataset.py --samples 1000 --output my_dataset.jsonl

# Generate smaller test dataset
python generate_dataset.py --samples 50 --output test_dataset.jsonl

# Generate with default settings (1000 samples)
python generate_dataset.py

# Validate existing dataset
python generate_dataset.py --validate existing_dataset.jsonl
```



## Generated Dataset Format

Each example in the JSONL file contains:

```json
{
  "language": "python",
  "code": "import pandas as pd\n\ndef missing_data_handling(input_data):\n    df = pd.DataFrame(input_data['data'])\n    df['salary'] = df['salary'].fillna(df['salary'].mean())\n    df['age'] = df['age'].astype(int)\n    return \"success\" if df['salary'].isnull().sum() == 0 else \"error\"\n\nresult = missing_data_handling({\"data\": [{\"name\": \"Alice\", \"age\": 25, \"salary\": 50000}, {\"name\": \"Bob\", \"age\": 30, \"salary\": null}]})\nprint(result)",
  "input": null,
  "output": "success",
  "library": "pandas",
  "difficulty": "INTERMEDIATE", 
  "task_type": "missing_data_handling",
  "reasoning_traps": ["parameter_interactions", "data_type_conversions", "null_handling", "index_alignment", "copy_vs_view", "chained_assignment"],
  "execution_time": 0.37
}
```

## Supported Python Libraries

### pandas
- **Beginner**: Basic DataFrame operations, column selection, simple filtering, basic aggregation
- **Intermediate**: GroupBy operations, merge operations, pivot tables, datetime operations, missing data handling
- **Advanced**: Multi-index operations, complex transformations, performance optimization, custom aggregations
- **Expert**: Memory-efficient processing, complex window functions, advanced indexing tricks, categorical data optimization

### numpy
- **Beginner**: Array creation, basic operations, simple indexing, shape manipulation
- **Intermediate**: Broadcasting operations, advanced indexing, statistical operations, linear algebra
- **Advanced**: Vectorization tricks, memory views, custom dtypes, performance optimization
- **Expert**: Advanced broadcasting, structured arrays, memory mapping, C integration patterns

### requests
- **Beginner**: Simple GET/POST requests, header handling, status code checking
- **Intermediate**: Session management, authentication handling, timeout and retries, file uploads
- **Advanced**: Custom adapters, streaming requests, certificate handling, proxy configuration
- **Expert**: Connection pooling, custom authentication, performance optimization

### Other Libraries
- **regex (re)**: Pattern matching, lookahead/lookbehind, recursive patterns, Unicode handling
- **datetime**: Basic creation, timezone handling, leap seconds, calendar edge cases
- **json**: Parsing, incremental parsing, precision handling, circular references

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
from src.generators.library_task_factory import LibraryTaskFactory
from src.generators.library_generator import DifficultyLevel

factory = LibraryTaskFactory(api_key)
result = factory.create_single_task()  # Generates random library task
```

## Dataset Statistics

A typical 1000-example dataset contains:

- **Python library examples**: Real library usage across pandas, numpy, requests, datetime, json, regex
- **Library Distribution**: Balanced across all supported libraries
- **Difficulty Distribution**: ~40% Beginner, 30% Intermediate, 20% Advanced, 10% Expert
- **Execution Times**: Range from 0.05s to 2s depending on complexity
- **Success Rate**: High reliability with comprehensive error handling

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
│   │   ├── language_factory.py     # Factory for generator/executor pairs
│   │   ├── seed_manager.py         # Application/concept seeds
│   │   └── verifier.py             # Output verification
│   ├── generators/                 # Code generation components
│   │   ├── library_generator.py    # Library-focused code generation
│   │   └── library_task_factory.py # Task orchestration
│   └── executors/                  # Safe code execution
│       └── library_executor.py     # Python library executor
```

## Configuration

### Basic Configuration (`config.yaml`)
```yaml
openai:
  model: "gpt-4o-mini-2024-07-18"
  max_tokens: 1000
  temperature: 0.7

execution:
  timeout_seconds: 10
  max_memory_mb: 128
```

### Environment Variables
- `OPENAI_API_KEY`: Required for code generation

## Safety Features

- **Subprocess Isolation**: All code execution in separate processes
- **Timeout Protection**: Configurable execution time limits  
- **Memory Limits**: Approximate memory usage monitoring
- **Error Recovery**: Comprehensive exception handling
- **Library Validation**: Automatic dependency checking

## Performance

- **Generation Speed**: ~2-5 seconds per example (including OpenAI API call and execution)
- **Memory Usage**: ~100-200MB for generation and execution
- **Dataset Size**: ~1-3MB per 1000 examples
- **Execution Time**: Individual examples range from 0.05s to 2s

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
  timeout_seconds: 30
```

**API Rate Limits**:
```bash
# Add delays between requests
time.sleep(1)  # In generation loop
```

## Contributing

1. **Add New Libraries**: Update `library_generator.py` with new library configurations and task types
2. **Improve Execution**: Enhance safety and performance in executors 
3. **Add Task Types**: Create new task templates for different difficulty levels
4. **Enhance Reasoning Traps**: Add more subtle edge cases and library-specific behaviors

## Use Cases

- **LLM Training**: Train models on Python library code understanding and output prediction
- **Model Evaluation**: Benchmark code reasoning capabilities with real library edge cases
- **Synthetic Data**: Augment existing code datasets with verified execution outputs
- **Research**: Study library API usage patterns and LLM reasoning on complex code

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