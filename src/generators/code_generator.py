"""
Advanced code generator using OpenAI API with comprehensive error handling and validation.

This module provides the CodeGenerator class which interfaces with OpenAI's API
to generate Python code samples with test inputs, designed specifically to create
challenging code that tests the reasoning capabilities of language models.
"""

import ast
import re
import time
from typing import Dict, Any, Optional

import openai
import yaml
from tqdm import tqdm
from ..core.seed_manager import SeedManager


class CodeGenerator:
    """
    Generates Python code using OpenAI API with intelligent prompt engineering.
    
    This class creates challenging Python programs designed to test language model
    reasoning capabilities through complex logic, edge cases, and tricky scenarios.
    It includes comprehensive validation, error handling, and progress tracking.
    
    Attributes:
        client (openai.OpenAI): OpenAI API client instance
        seed_manager (SeedManager): Manager for application and concept seeds
        config (Dict[str, Any]): Configuration loaded from YAML file
    """
    
    def __init__(self, api_key: str, config_path: str = "config.yaml"):
        """
        Initialize the code generator with API client and configuration.
        
        Args:
            api_key (str): OpenAI API key for authentication
            config_path (str): Path to YAML configuration file (default: "config.yaml")
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
            openai.AuthenticationError: If API key is invalid
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.seed_manager = SeedManager()
        
        # Load and validate config
        try:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML configuration: {e}")
    
    def generate_code(self) -> Dict[str, Any]:
        """
        Generate a single challenging Python code sample with test inputs.
        
        Creates a complex program designed to test language model reasoning through
        edge cases, tricky logic flows, and subtle implementation details.
        
        Returns:
            Dict[str, Any]: Generation result containing:
                - success (bool): Whether generation succeeded
                - code (str): Generated Python code (if successful)
                - test_inputs (List[Dict]): Test cases with descriptions and values
                - language (str): Programming language ("python")
                - application (str): Application domain used
                - concept (str): Programming concept implemented
                - generation_time (float): Time taken for API call
                - tokens_used (int): Total tokens consumed
                - error (str): Error message (if failed)
        
        Raises:
            openai.OpenAIError: If API call fails
            ValueError: If generated code has syntax errors
        """
        # Get random seeds for code generation
        application = self.seed_manager.get_random_application()
        concept = self.seed_manager.get_random_concept()
        
        # Create challenging prompt specifically designed to trip up smaller LLMs
        prompt = self._build_generation_prompt(application, concept)
        
        try:
            # Call OpenAI API with progress indication for longer requests
            start_time = time.time()
            
            # Create a simple progress indicator for API calls
            with tqdm(total=1, desc="Generating code", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
                response = self.client.chat.completions.create(
                    model=self.config['openai']['model'],
                    messages=[
                        {"role": "system", "content": "You are a Python programming expert. Write clean, well-documented code."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config['openai']['max_tokens'],
                    temperature=self.config['openai']['temperature']
                )
                pbar.update(1)
            
            generation_time = time.time() - start_time
            
            # Extract and validate code
            raw_response = response.choices[0].message.content
            code, test_inputs = self._extract_code_and_inputs(raw_response)
            
            # Validate syntax
            if not self._validate_syntax(code):
                raise ValueError("Generated code has syntax errors")
            
            # Add main block if configured
            if self.config['generation']['include_main'] and 'if __name__ == "__main__"' not in code:
                code += '\n\nif __name__ == "__main__":\n    main()'
            
            return {
                "success": True,
                "code": code,
                "test_inputs": test_inputs,
                "language": "python",
                "application": application,
                "concept": concept,
                "generation_time": generation_time,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "language": "python",
                "application": application,
                "concept": concept
            }
    
    def _build_generation_prompt(self, application: str, concept: str) -> str:
        """
        Build the generation prompt for creating challenging code.
        
        Args:
            application (str): Application domain for the code
            concept (str): Programming concept to implement
            
        Returns:
            str: Formatted prompt for code generation
        """
        return f"""Create a challenging Python program for {application} using {concept}. 

Requirements:
- Include {self.config['generation']['min_functions']}-{self.config['generation']['max_functions']} functions
- Add subtle edge cases, off-by-one errors, or counter-intuitive behavior
- Include multiple variables with similar names or confusing logic
- Use nested loops, complex conditionals, or tricky indexing
- Add boundary conditions that might cause errors
- Make the logic flow non-obvious to test reasoning abilities

Focus on creating code that would challenge a smaller language model's ability to trace execution correctly.

Also provide 3-5 test inputs that will exercise different edge cases and tricky scenarios.

Format your response as:
```python
# Your code here
```

TEST_INPUTS:
1. input1_description: actual_input_value
2. input2_description: actual_input_value
3. input3_description: actual_input_value"""
    
    def _extract_code_and_inputs(self, text: str) -> tuple:
        """
        Extract Python code and test inputs from OpenAI response text.
        
        Parses the generated response to separate the Python code block from
        the test input specifications using regex patterns.
        
        Args:
            text (str): Raw response text from OpenAI API
            
        Returns:
            tuple: (code, test_inputs) where:
                - code (str): Extracted Python code
                - test_inputs (List[Dict]): List of test cases with descriptions and values
        """
        # Extract code block using regex
        code_pattern = r'```python\n(.*?)\n```'
        code_match = re.search(code_pattern, text, re.DOTALL)
        
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Fallback: assume text before TEST_INPUTS is code
            if "TEST_INPUTS:" in text:
                code = text.split("TEST_INPUTS:")[0].strip()
            else:
                code = text.strip()
        
        # Extract test inputs
        test_inputs = []
        inputs_pattern = r'TEST_INPUTS:\s*(.*?)(?:\n\n|\Z)'
        inputs_match = re.search(inputs_pattern, text, re.DOTALL)
        
        if inputs_match:
            inputs_text = inputs_match.group(1)
            # Parse individual inputs (format: "1. description: value")
            input_lines = re.findall(r'\d+\.\s*(.*?):\s*(.*?)(?:\n|$)', inputs_text)
            
            for description, value in input_lines:
                test_inputs.append({
                    "description": description.strip(),
                    "value": value.strip()
                })
        
        return code, test_inputs
    
    def _validate_syntax(self, code: str) -> bool:
        """
        Validate that generated code has correct Python syntax.
        
        Uses the ast module to parse the code and detect syntax errors
        before execution, helping catch malformed generated code early.
        
        Args:
            code (str): Python code to validate
            
        Returns:
            bool: True if syntax is valid, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False