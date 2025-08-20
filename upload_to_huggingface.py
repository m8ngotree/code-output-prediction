#!/usr/bin/env python3
"""
Script to upload the synthetic code dataset to Hugging Face Hub.
"""

import json
import os
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
import pandas as pd

def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_dataset_card(dataset_name, num_examples, languages, libraries, difficulties):
    """Create a dataset card for the repository."""
    card_content = f"""---
license: mit
task_categories:
- text-generation
- code-generation
language:
- code
tags:
- code
- python
- synthetic
- execution
- output-prediction
size_categories:
- 1K<n<10K
---

# {dataset_name}

## Dataset Description

This is a synthetic dataset for code output prediction, containing {num_examples:,} examples of Python code snippets with their corresponding execution outputs.

### Dataset Summary

The dataset consists of Python code snippets from various libraries with their execution results. Each example includes:
- **code**: Python code snippet
- **output**: Expected execution output
- **library**: Python library used (e.g., {', '.join(sorted(libraries)[:5])})
- **difficulty**: Task difficulty level ({', '.join(sorted(difficulties))})
- **task_type**: Specific type of coding task
- **reasoning_traps**: Common pitfalls and edge cases
- **execution_time**: Time taken to execute the code

### Languages
- {', '.join(languages)}

### Libraries Covered
{chr(10).join([f'- {lib}' for lib in sorted(libraries)])}

### Difficulty Levels
{chr(10).join([f'- {diff}' for diff in sorted(difficulties)])}

## Dataset Structure

### Data Fields

- `language` (string): Programming language (currently only "python")
- `code` (string): Python code snippet
- `input` (string or null): Input data for the code (if applicable)
- `output` (string): Expected execution output
- `library` (string): Primary Python library used
- `difficulty` (string): Difficulty level (BEGINNER, INTERMEDIATE, ADVANCED, EXPERT)
- `task_type` (string): Specific type of coding task
- `reasoning_traps` (list of strings): Common pitfalls and edge cases
- `execution_time` (float): Time taken to execute the code in seconds

### Data Splits

The dataset contains a single split:
- **train**: {num_examples:,} examples

## Dataset Creation

This dataset was synthetically generated to provide diverse examples of Python code execution scenarios across multiple libraries and difficulty levels.

### Source Data

The dataset was created using automated generation techniques to produce realistic code snippets and their corresponding outputs.

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/{dataset_name}")

# Access examples
for example in dataset['train']:
    print(f"Code: {{example['code']}}")
    print(f"Output: {{example['output']}}")
    print(f"Library: {{example['library']}}")
    print("---")
```

## Considerations for Using the Data

### Social Impact of Dataset

This dataset is intended for educational and research purposes in code understanding and execution prediction.

### Limitations

- Currently only covers Python programming language
- Synthetic data may not capture all real-world coding scenarios
- Limited to specific libraries and task types

## Additional Information

### Licensing Information

This dataset is released under the MIT License.

### Citation Information

If you use this dataset in your research, please cite:

```bibtex
@dataset{{synthetic_code_dataset,
  title={{Synthetic Code Output Prediction Dataset}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/your-username/{dataset_name}}}
}}
```
"""
    return card_content

def main():
    # Configuration
    DATASET_FILE = "synthetic_code_dataset.jsonl"
    DATASET_NAME = "synthetic-code-output-prediction"  # Change this to your desired dataset name
    HF_USERNAME = None  # Will be set after login
    
    print("🚀 Starting Hugging Face dataset upload process...")
    
    # Check if dataset file exists
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Error: {DATASET_FILE} not found!")
        return
    
    # Login to Hugging Face
    print("🔐 Please login to Hugging Face...")
    try:
        login()  # This will prompt for your token if not already logged in
        api = HfApi()
        user_info = api.whoami()
        HF_USERNAME = user_info['name']
        print(f"✅ Logged in as: {HF_USERNAME}")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print("Please make sure you have a Hugging Face account and token.")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return
    
    # Load the data
    print(f"📂 Loading data from {DATASET_FILE}...")
    try:
        data = load_jsonl(DATASET_FILE)
        print(f"✅ Loaded {len(data):,} examples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Analyze the data
    print("🔍 Analyzing dataset...")
    languages = set(item.get('language', 'unknown') for item in data)
    libraries = set(item.get('library', 'unknown') for item in data)
    difficulties = set(item.get('difficulty', 'unknown') for item in data)
    
    print(f"   Languages: {sorted(languages)}")
    print(f"   Libraries: {sorted(libraries)}")
    print(f"   Difficulties: {sorted(difficulties)}")
    
    # Create the dataset
    print("📊 Creating Hugging Face dataset...")
    try:
        # Convert to pandas DataFrame first for easier handling
        df = pd.DataFrame(data)
        
        # Create the dataset
        dataset = Dataset.from_pandas(df)
        
        # Create a DatasetDict with a single split
        dataset_dict = DatasetDict({
            'train': dataset
        })
        
        print(f"✅ Dataset created with {len(dataset):,} examples")
        
        # Print dataset info
        print("📋 Dataset info:")
        print(dataset_dict)
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return
    
    # Create repository name
    repo_name = f"{HF_USERNAME}/{DATASET_NAME}"
    
    # Create dataset card
    print("📝 Creating dataset card...")
    card_content = create_dataset_card(
        DATASET_NAME, 
        len(data), 
        languages, 
        libraries, 
        difficulties
    )
    
    # Upload to Hugging Face
    print(f"☁️  Uploading dataset to {repo_name}...")
    try:
        dataset_dict.push_to_hub(
            repo_name,
            commit_message="Initial dataset upload",
            private=False  # Set to True if you want a private dataset
        )
        
        # Upload the dataset card
        api.upload_file(
            path_or_fileobj=card_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Add dataset card"
        )
        
        print(f"🎉 Successfully uploaded dataset!")
        print(f"📍 Dataset URL: https://huggingface.co/datasets/{repo_name}")
        
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        return
    
    # Test loading the uploaded dataset
    print("🧪 Testing dataset loading...")
    try:
        from datasets import load_dataset
        test_dataset = load_dataset(repo_name)
        print(f"✅ Successfully loaded dataset with {len(test_dataset['train']):,} examples")
        
        # Show a sample
        print("📄 Sample example:")
        sample = test_dataset['train'][0]
        print(f"   Language: {sample['language']}")
        print(f"   Library: {sample['library']}")
        print(f"   Difficulty: {sample['difficulty']}")
        print(f"   Code: {sample['code'][:100]}...")
        print(f"   Output: {sample['output']}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not test dataset loading: {e}")
    
    print("✨ Upload process completed!")

if __name__ == "__main__":
    main()

