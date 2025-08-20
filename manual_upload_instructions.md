# Manual Upload Instructions for Hugging Face Dataset

## Prerequisites

1. **Install required packages:**
   ```bash
   pip install huggingface_hub datasets pandas
   ```

2. **Get a Hugging Face token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "Write" permissions
   - Keep it safe, you'll need it for authentication

## Option 1: Using the Automated Script

1. **Run the upload script:**
   ```bash
   python upload_to_huggingface.py
   ```

2. **Follow the prompts:**
   - Login with your Hugging Face token when prompted
   - The script will automatically upload your dataset

## Option 2: Manual Step-by-Step Upload

### Step 1: Login to Hugging Face

```python
from huggingface_hub import login
login()  # Enter your token when prompted
```

### Step 2: Load and Prepare Dataset

```python
import json
from datasets import Dataset, DatasetDict
import pandas as pd

# Load the JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Load your data
data = load_jsonl('synthetic_code_dataset.jsonl')
print(f"Loaded {len(data)} examples")

# Convert to Hugging Face dataset
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)
dataset_dict = DatasetDict({'train': dataset})
```

### Step 3: Upload Dataset

```python
# Upload to Hugging Face Hub
# Replace 'your-username' with your actual Hugging Face username
# Replace 'dataset-name' with your desired dataset name
repo_name = "your-username/synthetic-code-output-prediction"

dataset_dict.push_to_hub(
    repo_name,
    commit_message="Initial dataset upload",
    private=False  # Set to True for private dataset
)

print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_name}")
```

### Step 4: Verify Upload

```python
from datasets import load_dataset

# Test loading the uploaded dataset
test_dataset = load_dataset(repo_name)
print(f"Successfully loaded {len(test_dataset['train'])} examples")

# Show sample
sample = test_dataset['train'][0]
print("Sample example:")
print(f"Code: {sample['code'][:100]}...")
print(f"Output: {sample['output']}")
print(f"Library: {sample['library']}")
```

## Option 3: Using Hugging Face Hub Web Interface

1. **Create a new dataset repository:**
   - Go to https://huggingface.co/new-dataset
   - Choose a name like "synthetic-code-output-prediction"
   - Make it public or private as desired

2. **Upload files:**
   - Click "Upload files"
   - Upload your `synthetic_code_dataset.jsonl` file
   - Add a README.md with dataset description

3. **Configure dataset:**
   - The platform will automatically detect the JSONL format
   - Add appropriate tags and metadata

## Dataset Information

Your dataset contains:
- **Format**: JSONL (JSON Lines)
- **Size**: ~874 examples (based on the sample)
- **Content**: Python code snippets with execution outputs
- **Libraries**: numpy, pandas, requests, re, datetime, etc.
- **Difficulty levels**: BEGINNER, INTERMEDIATE, ADVANCED, EXPERT

## Important Notes

1. **Dataset Name**: Choose a descriptive name like "synthetic-code-output-prediction"
2. **License**: Consider adding an appropriate license (MIT, Apache 2.0, etc.)
3. **Tags**: Add relevant tags like "code", "python", "synthetic", "execution"
4. **Description**: Include a comprehensive README explaining the dataset structure and usage

## Troubleshooting

- **Authentication Error**: Make sure your token has write permissions
- **Large File**: If the file is too large, consider splitting it or using Git LFS
- **Format Issues**: Ensure your JSONL file is properly formatted with one JSON object per line

## After Upload

Once uploaded, your dataset will be available at:
`https://huggingface.co/datasets/your-username/your-dataset-name`

You can then load it anywhere using:
```python
from datasets import load_dataset
dataset = load_dataset("your-username/your-dataset-name")
```
