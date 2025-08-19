"""
Library-focused Code Generator for Output Prediction Tasks

This module generates Python code that uses real-world libraries with varying
difficulty levels, designed to test LLM reasoning capabilities through
library API usage, edge cases, and complex transformations.
"""

import ast
import json
import random
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import openai
import yaml
from ..core.seed_manager import SeedManager


class DifficultyLevel(Enum):
    """Difficulty levels for generated tasks."""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


@dataclass
class LibraryTask:
    """Represents a library-specific coding task."""
    library: str
    difficulty: DifficultyLevel
    task_type: str
    description: str
    input_data: Any
    expected_complexity: str
    reasoning_traps: List[str]


class LibraryCodeGenerator:
    """
    Generates library-focused code tasks with varying difficulty levels.
    
    Creates tasks that test LLM reasoning through:
    - Real-world library API usage
    - Edge case handling
    - Complex data transformations
    - Performance considerations
    - Error handling scenarios
    """
    
    def __init__(self, api_key: str, config_path: str = "config.yaml"):
        """Initialize the library-focused code generator."""
        self.client = openai.OpenAI(api_key=api_key)
        self.seed_manager = SeedManager()
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Define library configurations with difficulty progression
        self.library_configs = self._initialize_library_configs()
        
        # Input generators for realistic test data
        self.input_generators = self._initialize_input_generators()
    
    def _initialize_library_configs(self) -> Dict[str, Dict]:
        """Initialize library-specific task configurations."""
        return {
            "pandas": {
                "import_statement": "import pandas as pd\nimport numpy as np",
                "beginner_tasks": [
                    "basic_dataframe_operations",
                    "simple_filtering",
                    "column_selection",
                    "basic_aggregation"
                ],
                "intermediate_tasks": [
                    "groupby_operations",
                    "merge_operations",
                    "pivot_tables",
                    "datetime_operations",
                    "missing_data_handling"
                ],
                "advanced_tasks": [
                    "multi_index_operations",
                    "complex_transformations",
                    "performance_optimization",
                    "custom_aggregations"
                ],
                "expert_tasks": [
                    "memory_efficient_processing",
                    "complex_window_functions",
                    "advanced_indexing_tricks",
                    "categorical_data_optimization"
                ]
            },
            "numpy": {
                "import_statement": "import numpy as np",
                "beginner_tasks": [
                    "array_creation",
                    "basic_operations",
                    "simple_indexing",
                    "shape_manipulation"
                ],
                "intermediate_tasks": [
                    "broadcasting_operations",
                    "advanced_indexing",
                    "statistical_operations",
                    "linear_algebra"
                ],
                "advanced_tasks": [
                    "vectorization_tricks",
                    "memory_views",
                    "custom_dtypes",
                    "performance_optimization"
                ],
                "expert_tasks": [
                    "advanced_broadcasting",
                    "structured_arrays",
                    "memory_mapping",
                    "c_integration_patterns"
                ]
            },
            "requests": {
                "import_statement": "import requests\nimport json",
                "beginner_tasks": [
                    "simple_get_request",
                    "basic_post_request",
                    "header_handling",
                    "status_code_checking"
                ],
                "intermediate_tasks": [
                    "session_management",
                    "authentication_handling",
                    "timeout_and_retries",
                    "file_uploads"
                ],
                "advanced_tasks": [
                    "custom_adapters",
                    "streaming_requests",
                    "certificate_handling",
                    "proxy_configuration"
                ],
                "expert_tasks": [
                    "connection_pooling",
                    "custom_authentication",
                    "advanced_streaming",
                    "performance_monitoring"
                ]
            },
            "re": {
                "import_statement": "import re",
                "beginner_tasks": [
                    "simple_pattern_matching",
                    "basic_substitution",
                    "literal_matching",
                    "case_insensitive_matching"
                ],
                "intermediate_tasks": [
                    "character_classes",
                    "quantifiers",
                    "grouping_and_capturing",
                    "anchors_and_boundaries"
                ],
                "advanced_tasks": [
                    "lookahead_lookbehind",
                    "non_capturing_groups",
                    "conditional_patterns",
                    "recursive_patterns"
                ],
                "expert_tasks": [
                    "advanced_backreferences",
                    "atomic_groups",
                    "possessive_quantifiers",
                    "unicode_categories"
                ]
            },
            "datetime": {
                "import_statement": "from datetime import datetime, timedelta, timezone\nimport calendar",
                "beginner_tasks": [
                    "basic_datetime_creation",
                    "simple_formatting",
                    "date_arithmetic",
                    "current_time_operations"
                ],
                "intermediate_tasks": [
                    "timezone_handling",
                    "custom_formatting",
                    "date_parsing",
                    "business_day_calculations"
                ],
                "advanced_tasks": [
                    "complex_timezone_conversions",
                    "custom_calendar_operations",
                    "performance_optimized_parsing",
                    "leap_year_edge_cases"
                ],
                "expert_tasks": [
                    "microsecond_precision_handling",
                    "historical_timezone_data",
                    "calendar_system_conversions",
                    "astronomical_calculations"
                ]
            },
            "json": {
                "import_statement": "import json\nfrom collections import OrderedDict",
                "beginner_tasks": [
                    "basic_serialization",
                    "simple_deserialization",
                    "pretty_printing",
                    "basic_error_handling"
                ],
                "intermediate_tasks": [
                    "custom_encoders",
                    "streaming_parsing",
                    "nested_object_handling",
                    "unicode_handling"
                ],
                "advanced_tasks": [
                    "custom_object_serialization",
                    "memory_efficient_parsing",
                    "schema_validation",
                    "circular_reference_handling"
                ],
                "expert_tasks": [
                    "incremental_parsing",
                    "custom_json_extensions",
                    "performance_optimization",
                    "security_considerations"
                ]
            }
        }
    
    def _initialize_input_generators(self) -> Dict[str, callable]:
        """Initialize input data generators for different libraries."""
        return {
            "pandas": self._generate_dataframe_input,
            "numpy": self._generate_array_input,
            "requests": self._generate_request_input,
            "re": self._generate_text_input,
            "datetime": self._generate_datetime_input,
            "json": self._generate_json_input
        }
    
    def generate_code(self, library: str = None, difficulty: DifficultyLevel = None) -> Dict[str, Any]:
        """
        Generate a library-focused code task with specified difficulty.
        
        Args:
            library: Specific library to use (random if None)
            difficulty: Difficulty level (random if None)
            
        Returns:
            Dict containing generated code, test inputs, and metadata
        """
        # Select library and difficulty
        if library is None:
            library = random.choice(list(self.library_configs.keys()))
        if difficulty is None:
            difficulty = random.choice(list(DifficultyLevel))
        
        try:
            # Generate task specification
            task = self._create_library_task(library, difficulty)
            
            # Generate realistic input data
            input_data = self._generate_task_input(task)
            
            # Create the code generation prompt
            prompt = self._build_library_prompt(task, input_data)
            
            # Call OpenAI API
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.config['openai']['model'],
                messages=[
                    {"role": "system", "content": self._get_system_prompt(library, difficulty)},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config['openai']['max_tokens'],
                temperature=self.config['openai']['temperature']
            )
            generation_time = time.time() - start_time
            
            # Extract and validate code
            raw_response = response.choices[0].message.content
            code, expected_output = self._extract_code_and_expected_output(raw_response)
            
            # Validate syntax
            if not self._validate_syntax(code):
                raise ValueError("Generated code has syntax errors")
            
            # Create test inputs with varying complexity
            test_inputs = self._create_test_inputs(task, input_data)
            
            return {
                "success": True,
                "code": code,
                "test_inputs": test_inputs,
                "expected_output": expected_output,
                "language": "python",
                "library": library,
                "difficulty": difficulty.name,
                "task_type": task.task_type,
                "reasoning_traps": task.reasoning_traps,
                "generation_time": generation_time,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "language": "python",
                "library": library,
                "difficulty": difficulty.name if difficulty else "unknown"
            }
    
    def _create_library_task(self, library: str, difficulty: DifficultyLevel) -> LibraryTask:
        """Create a specific task for the given library and difficulty."""
        config = self.library_configs[library]
        
        # Select appropriate task type based on difficulty
        difficulty_key = f"{difficulty.name.lower()}_tasks"
        available_tasks = config[difficulty_key]
        task_type = random.choice(available_tasks)
        
        # Define reasoning traps based on difficulty
        reasoning_traps = self._get_reasoning_traps(library, difficulty, task_type)
        
        return LibraryTask(
            library=library,
            difficulty=difficulty,
            task_type=task_type,
            description=self._get_task_description(library, task_type, difficulty),
            input_data=None,  # Will be generated later
            expected_complexity=self._get_expected_complexity(difficulty),
            reasoning_traps=reasoning_traps
        )
    
    def _get_reasoning_traps(self, library: str, difficulty: DifficultyLevel, task_type: str) -> List[str]:
        """Define specific reasoning traps for each library and difficulty."""
        traps = []
        
        if difficulty == DifficultyLevel.BEGINNER:
            traps = ["basic_api_usage", "simple_edge_cases"]
        elif difficulty == DifficultyLevel.INTERMEDIATE:
            traps = ["parameter_interactions", "data_type_conversions", "null_handling"]
        elif difficulty == DifficultyLevel.ADVANCED:
            traps = ["performance_pitfalls", "memory_implications", "subtle_api_behaviors"]
        elif difficulty == DifficultyLevel.EXPERT:
            traps = ["optimization_tricks", "undocumented_behaviors", "complex_interactions"]
        
        # Add library-specific traps
        library_specific_traps = {
            "pandas": ["index_alignment", "copy_vs_view", "chained_assignment"],
            "numpy": ["broadcasting_rules", "memory_layout", "dtype_promotion"],
            "requests": ["session_persistence", "connection_pooling", "timeout_behavior"],
            "re": ["greedy_vs_lazy", "backtracking", "unicode_handling"],
            "datetime": ["timezone_naive_aware", "leap_second_handling", "calendar_edge_cases"],
            "json": ["precision_loss", "object_ordering", "circular_references"]
        }
        
        if library in library_specific_traps:
            traps.extend(library_specific_traps[library])
        
        return traps
    
    def _get_task_description(self, library: str, task_type: str, difficulty: DifficultyLevel) -> str:
        """Generate a description for the specific task."""
        descriptions = {
            "pandas": {
                "basic_dataframe_operations": "Count rows or columns in a DataFrame",
                "groupby_operations": "Calculate a single aggregate value from grouped data",
                "multi_index_operations": "Extract a simple value from hierarchical data",
                "memory_efficient_processing": "Return a performance metric as a number"
            },
            "numpy": {
                "array_creation": "Calculate the sum or mean of a created array",
                "broadcasting_operations": "Return a single result from broadcasted operations",
                "vectorization_tricks": "Measure performance improvement as a ratio",
                "advanced_broadcasting": "Return True/False for broadcasting compatibility"
            },
            "requests": {
                "simple_get_request": "Return the HTTP status code from a request",
                "session_management": "Count successful requests in a session",
                "custom_adapters": "Return True/False for adapter functionality",
                "connection_pooling": "Measure response time as a number"
            },
            "re": {
                "simple_pattern_matching": "Count matches found in text",
                "lookahead_lookbehind": "Return True/False for pattern existence",
                "recursive_patterns": "Count nested pattern occurrences",
                "unicode_categories": "Return the number of Unicode matches"
            },
            "datetime": {
                "basic_datetime_creation": "Return the day of week as a number",
                "timezone_handling": "Calculate time difference in hours",
                "complex_timezone_conversions": "Return True/False for DST status",
                "astronomical_calculations": "Calculate days between dates"
            },
            "json": {
                "basic_serialization": "Count keys in a JSON object",
                "custom_encoders": "Return True/False for encoding success",
                "incremental_parsing": "Count parsed JSON objects",
                "security_considerations": "Return validation status as boolean"
            }
        }
        
        return descriptions.get(library, {}).get(task_type, f"{task_type} using {library}")
    
    def _get_expected_complexity(self, difficulty: DifficultyLevel) -> str:
        """Get expected code complexity for the difficulty level."""
        complexity_map = {
            DifficultyLevel.BEGINNER: "1-5 lines, single operation",
            DifficultyLevel.INTERMEDIATE: "5-15 lines, multiple operations with conditions",
            DifficultyLevel.ADVANCED: "15-30 lines, complex logic with error handling",
            DifficultyLevel.EXPERT: "30+ lines, optimization and edge case handling"
        }
        return complexity_map[difficulty]
    
    def _build_library_prompt(self, task: LibraryTask, input_data: Dict[str, Any]) -> str:
        """Build the code generation prompt for the specific library task."""
        difficulty_instructions = {
            DifficultyLevel.BEGINNER: "Create straightforward code that demonstrates basic usage.",
            DifficultyLevel.INTERMEDIATE: "Include edge case handling and multiple operations.",
            DifficultyLevel.ADVANCED: "Add performance considerations and complex transformations.",
            DifficultyLevel.EXPERT: "Include optimization tricks and handle subtle edge cases."
        }
        
        reasoning_trap_instructions = ""
        if task.reasoning_traps:
            reasoning_trap_instructions = f"\nSpecifically test these concepts: {', '.join(task.reasoning_traps)}"
        
        return f"""Create a Python function that {task.description} using the {task.library} library.

Difficulty Level: {task.difficulty.name}
{difficulty_instructions[task.difficulty]}

Requirements:
- Use the {task.library} library appropriately
- Include the necessary import statements
- Create a function that takes the provided input and returns a simple output
- The output must be a single number, boolean, or short string (max 50 characters)
- NO comments in the code
- NO complex outputs like dataframes, lists, or multi-line strings
- Output examples: 42, True, "success", 3.14, "error", 0, etc.
- {task.expected_complexity}
- At the end, call the function with the input data and print the result
{reasoning_trap_instructions}

Input data will be: {json.dumps(input_data, default=str)}

Format your response as clean Python code without any comments:
```python
import library_name

def function_name(input_data):
    return simple_result

result = function_name(input_data)
print(result)
```

The output must be simple and easily verifiable.
"""
    
    def _get_system_prompt(self, library: str, difficulty: DifficultyLevel) -> str:
        """Get system prompt tailored to library and difficulty."""
        base_prompt = f"You are a Python expert specializing in the {library} library. "
        output_requirement = "CRITICAL: Always return simple outputs (numbers, booleans, short strings). Never return complex objects, dataframes, or multi-line outputs. Write clean code without any comments. "
        
        if difficulty == DifficultyLevel.BEGINNER:
            base_prompt += output_requirement + "Write clear, simple code that demonstrates basic library usage."
        elif difficulty == DifficultyLevel.INTERMEDIATE:
            base_prompt += output_requirement + "Write code that handles common edge cases and uses intermediate features."
        elif difficulty == DifficultyLevel.ADVANCED:
            base_prompt += output_requirement + "Write sophisticated code that demonstrates advanced library features and optimizations."
        else:  # EXPERT
            base_prompt += output_requirement + "Write expert-level code that showcases deep library knowledge and handles complex edge cases."
        
        return base_prompt + " Focus on creating code that tests reasoning abilities through subtle behaviors and tricky scenarios."
    
    # Input generation methods for different libraries
    def _generate_dataframe_input(self, task: LibraryTask) -> Dict[str, Any]:
        """Generate realistic DataFrame input data."""
        if task.difficulty == DifficultyLevel.BEGINNER:
            return {
                "data": [
                    {"name": "Alice", "age": 25, "city": "New York"},
                    {"name": "Bob", "age": 30, "city": "San Francisco"},
                    {"name": "Charlie", "age": 35, "city": "New York"}
                ]
            }
        elif task.difficulty == DifficultyLevel.INTERMEDIATE:
            return {
                "data": [
                    {"name": "Alice", "age": 25, "city": "New York", "salary": 50000, "date": "2023-01-15"},
                    {"name": "Bob", "age": 30, "city": "San Francisco", "salary": None, "date": "2023-02-20"},
                    {"name": "Charlie", "age": 35, "city": "New York", "salary": 75000, "date": "2023-01-15"},
                    {"name": "Diana", "age": 28, "city": "Chicago", "salary": 60000, "date": "2023-03-10"}
                ]
            }
        else:  # Advanced/Expert
            return {
                "data": [
                    {"name": "Alice", "age": 25, "city": "New York", "salary": 50000, "date": "2023-01-15", "department": "Engineering", "performance": 4.2},
                    {"name": "Bob", "age": 30, "city": "San Francisco", "salary": None, "date": "2023-02-20", "department": "Sales", "performance": 3.8},
                    {"name": "Charlie", "age": 35, "city": "New York", "salary": 75000, "date": "2023-01-15", "department": "Engineering", "performance": 4.5},
                    {"name": "Diana", "age": 28, "city": "Chicago", "salary": 60000, "date": "2023-03-10", "department": "Marketing", "performance": 4.0},
                    {"name": "Eve", "age": 32, "city": "San Francisco", "salary": 80000, "date": "2023-01-01", "department": "Engineering", "performance": 4.7}
                ],
                "edge_cases": ["missing_values", "duplicate_dates", "unicode_names"]
            }
    
    def _generate_array_input(self, task: LibraryTask) -> Dict[str, Any]:
        """Generate NumPy array input data."""
        if task.difficulty == DifficultyLevel.BEGINNER:
            return {"array": [1, 2, 3, 4, 5], "shape": (5,)}
        elif task.difficulty == DifficultyLevel.INTERMEDIATE:
            return {"array": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "shape": (3, 3)}
        else:
            return {
                "array": [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
                "shape": (3, 2, 2),
                "dtype": "float64",
                "edge_cases": ["broadcasting", "memory_layout"]
            }
    
    def _generate_request_input(self, task: LibraryTask) -> Dict[str, Any]:
        """Generate HTTP request input data."""
        if task.difficulty == DifficultyLevel.BEGINNER:
            return {"url": "https://httpbin.org/get", "method": "GET"}
        elif task.difficulty == DifficultyLevel.INTERMEDIATE:
            return {
                "url": "https://httpbin.org/post",
                "method": "POST",
                "data": {"key": "value"},
                "headers": {"Content-Type": "application/json"}
            }
        else:
            return {
                "url": "https://httpbin.org/delay/1",
                "method": "GET",
                "timeout": 2,
                "retries": 3,
                "session_config": {"verify": True, "stream": False}
            }
    
    def _generate_text_input(self, task: LibraryTask) -> Dict[str, Any]:
        """Generate text input for regex tasks."""
        texts = {
            DifficultyLevel.BEGINNER: "The quick brown fox jumps over the lazy dog.",
            DifficultyLevel.INTERMEDIATE: "Contact us at support@example.com or call (555) 123-4567 for assistance.",
            DifficultyLevel.ADVANCED: "Parse this: {'name': 'John', 'age': 30} and [1, 2, 3] with 'quotes' and \"double quotes\".",
            DifficultyLevel.EXPERT: "Complex text with Unicode: café, naïve, résumé, and nested structures like ((()))"
        }
        
        return {"text": texts[task.difficulty]}
    
    def _generate_datetime_input(self, task: LibraryTask) -> Dict[str, Any]:
        """Generate datetime input data."""
        if task.difficulty == DifficultyLevel.BEGINNER:
            return {"date_string": "2023-12-25", "format": "%Y-%m-%d"}
        elif task.difficulty == DifficultyLevel.INTERMEDIATE:
            return {
                "date_string": "2023-12-25 14:30:00 EST",
                "format": "%Y-%m-%d %H:%M:%S %Z",
                "target_timezone": "UTC"
            }
        else:
            return {
                "date_string": "2023-03-12 02:30:00",  # DST transition
                "timezone": "US/Eastern",
                "operations": ["add_days", "convert_timezone", "handle_ambiguous"]
            }
    
    def _generate_json_input(self, task: LibraryTask) -> Dict[str, Any]:
        """Generate JSON input data."""
        if task.difficulty == DifficultyLevel.BEGINNER:
            return {"data": {"name": "John", "age": 30}}
        elif task.difficulty == DifficultyLevel.INTERMEDIATE:
            return {
                "data": {
                    "users": [
                        {"name": "John", "age": 30, "active": True},
                        {"name": "Jane", "age": 25, "active": False}
                    ],
                    "metadata": {"version": "1.0", "timestamp": "2023-12-25T10:00:00Z"}
                }
            }
        else:
            return {
                "data": {
                    "nested": {"deep": {"very": {"deep": {"value": 42}}}},
                    "array": [1, 2, {"nested": "value"}],
                    "special_chars": "unicode: café, emoji: 🎉",
                    "numbers": {"int": 42, "float": 3.14159, "big": 1234567890123456789}
                },
                "edge_cases": ["circular_refs", "precision_loss", "unicode_handling"]
            }
    
    def _generate_task_input(self, task: LibraryTask) -> Dict[str, Any]:
        """Generate appropriate input data for the task."""
        generator = self.input_generators.get(task.library)
        if generator:
            return generator(task)
        else:
            return {"generic_input": "test_data"}
    
    def _create_test_inputs(self, task: LibraryTask, base_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create multiple test inputs with varying complexity."""
        test_inputs = []
        
        # Base test case
        test_inputs.append({
            "description": f"Basic {task.library} {task.task_type} test",
            "value": json.dumps(base_input, default=str)
        })
        
        # Add edge cases based on difficulty
        if task.difficulty.value >= 2:  # Intermediate and above
            edge_case_input = self._create_edge_case_input(task, base_input)
            test_inputs.append({
                "description": f"Edge case for {task.task_type}",
                "value": json.dumps(edge_case_input, default=str)
            })
        
        if task.difficulty.value >= 3:  # Advanced and above
            stress_test_input = self._create_stress_test_input(task, base_input)
            test_inputs.append({
                "description": f"Stress test for {task.task_type}",
                "value": json.dumps(stress_test_input, default=str)
            })
        
        return test_inputs
    
    def _create_edge_case_input(self, task: LibraryTask, base_input: Dict[str, Any]) -> Dict[str, Any]:
        """Create edge case input based on the task."""
        # This would create edge cases specific to each library
        edge_input = base_input.copy()
        
        if task.library == "pandas":
            edge_input["edge_case"] = "empty_dataframe"
        elif task.library == "numpy":
            edge_input["edge_case"] = "zero_dimensional_array"
        elif task.library == "requests":
            edge_input["edge_case"] = "network_timeout"
        elif task.library == "re":
            edge_input["edge_case"] = "no_matches"
        elif task.library == "datetime":
            edge_input["edge_case"] = "invalid_date"
        elif task.library == "json":
            edge_input["edge_case"] = "malformed_json"
        
        return edge_input
    
    def _create_stress_test_input(self, task: LibraryTask, base_input: Dict[str, Any]) -> Dict[str, Any]:
        """Create stress test input for advanced testing."""
        stress_input = base_input.copy()
        stress_input["stress_test"] = True
        stress_input["large_data"] = True
        return stress_input
    
    def _extract_code_and_expected_output(self, text: str) -> Tuple[str, str]:
        """Extract code and expected output from the response."""
        # Extract code block
        code_pattern = r'```python\n(.*?)\n```'
        code_match = re.search(code_pattern, text, re.DOTALL)
        
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Fallback
            if "EXPECTED_OUTPUT:" in text:
                code = text.split("EXPECTED_OUTPUT:")[0].strip()
            else:
                code = text.strip()
        
        # Extract expected output
        expected_output = ""
        output_pattern = r'EXPECTED_OUTPUT:\s*(.*?)(?:\n\n|\Z)'
        output_match = re.search(output_pattern, text, re.DOTALL)
        
        if output_match:
            expected_output = output_match.group(1).strip()
        
        return code, expected_output
    
    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def get_available_libraries(self) -> List[str]:
        """Get list of available libraries."""
        return list(self.library_configs.keys())
    
    def get_difficulty_levels(self) -> List[str]:
        """Get list of available difficulty levels."""
        return [level.name for level in DifficultyLevel]
    
    def generate_batch(self, count: int, library: str = None, 
                      difficulty: DifficultyLevel = None) -> List[Dict[str, Any]]:
        """Generate a batch of library-focused tasks."""
        results = []
        for i in range(count):
            result = self.generate_code(library, difficulty)
            results.append(result)
        return results
