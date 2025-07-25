"""
Input Generator for analyzing Python code and generating diverse test inputs.

This module provides functionality to parse Python code, identify input requirements,
and generate comprehensive test cases with various data types and edge cases.
"""

import ast
import json
import random
import string
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

class InputRequirement:
    """Represents an input requirement identified from code analysis."""
    
    def __init__(self, name: str, param_type: str, inferred_type: Optional[str] = None,
                 constraints: Optional[Dict[str, Any]] = None, description: str = ""):
        self.name = name
        self.param_type = param_type  # 'function_param', 'cli_arg', 'stdin', 'file_input'
        self.inferred_type = inferred_type  # int, str, list, dict, etc.
        self.constraints = constraints or {}
        self.description = description


class TestInput:
    """Represents a generated test input with metadata."""
    
    def __init__(self, input_id: str, category: str, data: Any, 
                 description: str, data_type: str):
        self.input_id = input_id
        self.category = category  # 'normal', 'edge', 'boundary', 'random'
        self.data = data
        self.description = description
        self.data_type = data_type
        self.created_at = datetime.now().isoformat()


class CodeAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze Python code and identify input requirements."""
    
    def __init__(self):
        self.requirements: List[InputRequirement] = []
        self.function_params: Dict[str, List[str]] = {}
        self.imports: Set[str] = set()
        self.file_operations: List[str] = []
        self.cli_patterns: List[str] = []
        self.stdin_usage: bool = False
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions to extract parameters."""
        params = []
        for arg in node.args.args:
            param_name = arg.arg
            params.append(param_name)
            
            # Try to infer type from annotations
            inferred_type = None
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    inferred_type = arg.annotation.id
                elif isinstance(arg.annotation, ast.Constant):
                    inferred_type = str(arg.annotation.value)
            
            # Analyze parameter name for type hints
            if not inferred_type:
                inferred_type = self._infer_type_from_name(param_name)
            
            requirement = InputRequirement(
                name=param_name,
                param_type="function_param",
                inferred_type=inferred_type,
                description=f"Parameter for function {node.name}"
            )
            self.requirements.append(requirement)
        
        self.function_params[node.name] = params
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Visit function calls to identify input operations."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Check for input() calls
            if func_name == 'input':
                self.stdin_usage = True
                prompt = ""
                if node.args and isinstance(node.args[0], ast.Constant):
                    prompt = node.args[0].value
                
                requirement = InputRequirement(
                    name="stdin_input",
                    param_type="stdin",
                    inferred_type=self._infer_type_from_prompt(prompt),
                    description=f"Standard input: {prompt}"
                )
                self.requirements.append(requirement)
            
            # Check for file operations
            elif func_name in ['open', 'read', 'readlines', 'readline']:
                if node.args and isinstance(node.args[0], ast.Constant):
                    filename = node.args[0].value
                    self.file_operations.append(filename)
                    
                    requirement = InputRequirement(
                        name=f"file_{len(self.file_operations)}",
                        param_type="file_input",
                        inferred_type="file",
                        constraints={"filename": filename},
                        description=f"File input: {filename}"
                    )
                    self.requirements.append(requirement)
        
        elif isinstance(node.func, ast.Attribute):
            # Check for sys.argv usage
            if (isinstance(node.func.value, ast.Attribute) and 
                isinstance(node.func.value.value, ast.Name) and
                node.func.value.value.id == 'sys' and
                node.func.value.attr == 'argv'):
                
                requirement = InputRequirement(
                    name="cli_args",
                    param_type="cli_arg",
                    inferred_type="list",
                    description="Command line arguments"
                )
                self.requirements.append(requirement)
        
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import):
        """Track imports for context."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track from imports for context."""
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)
    
    def _infer_type_from_name(self, name: str) -> str:
        """Infer data type from parameter name patterns."""
        name_lower = name.lower()
        
        # Integer patterns
        if any(pattern in name_lower for pattern in 
               ['count', 'size', 'length', 'num', 'index', 'id', 'age', 'year']):
            return 'int'
        
        # String patterns
        if any(pattern in name_lower for pattern in 
               ['name', 'text', 'message', 'string', 'word', 'title', 'description']):
            return 'str'
        
        # List patterns
        if any(pattern in name_lower for pattern in 
               ['list', 'items', 'elements', 'array', 'data']):
            return 'list'
        
        # Dictionary patterns
        if any(pattern in name_lower for pattern in 
               ['dict', 'map', 'config', 'settings', 'params']):
            return 'dict'
        
        # Boolean patterns
        if any(pattern in name_lower for pattern in 
               ['is_', 'has_', 'can_', 'should_', 'enabled', 'flag']):
            return 'bool'
        
        # Float patterns
        if any(pattern in name_lower for pattern in 
               ['rate', 'score', 'value', 'price', 'weight', 'height']):
            return 'float'
        
        return 'str'  # Default to string
    
    def _infer_type_from_prompt(self, prompt: str) -> str:
        """Infer expected input type from input prompt."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['number', 'count', 'age', 'year']):
            return 'int'
        elif any(word in prompt_lower for word in ['price', 'rate', 'score', 'value']):
            return 'float'
        elif any(word in prompt_lower for word in ['yes', 'no', 'true', 'false']):
            return 'bool'
        else:
            return 'str'


class DataGenerator:
    """Generates test data for different Python data types."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
    
    def generate_int_values(self, constraints: Dict[str, Any] = None) -> List[Tuple[int, str, str]]:
        """Generate integer test values."""
        constraints = constraints or {}
        min_val = constraints.get('min', -1000)
        max_val = constraints.get('max', 1000)
        
        values = [
            # Normal cases
            (0, "normal", "zero value"),
            (1, "normal", "positive small integer"),
            (10, "normal", "positive medium integer"),
            (-1, "normal", "negative small integer"),
            (-10, "normal", "negative medium integer"),
            
            # Edge cases
            (min_val, "edge", "minimum value"),
            (max_val, "edge", "maximum value"),
            (min_val + 1, "boundary", "minimum boundary"),
            (max_val - 1, "boundary", "maximum boundary"),
            
            # Random cases
            (random.randint(min_val // 2, max_val // 2), "random", "random integer"),
        ]
        
        return values[:8]  # Limit to 8 values
    
    def generate_str_values(self, constraints: Dict[str, Any] = None) -> List[Tuple[str, str, str]]:
        """Generate string test values."""
        constraints = constraints or {}
        max_len = constraints.get('max_length', 100)
        
        values = [
            # Normal cases
            ("hello", "normal", "simple string"),
            ("Hello World", "normal", "string with space"),
            ("test123", "normal", "alphanumeric string"),
            
            # Edge cases
            ("", "edge", "empty string"),
            ("a" * max_len, "edge", "maximum length string"),
            ("!@#$%^&*()", "edge", "special characters"),
            ("unicode: ñáéíóú", "edge", "unicode characters"),
            
            # Boundary cases
            ("a", "boundary", "single character"),
            ("a" * (max_len - 1), "boundary", "near maximum length"),
            
            # Random cases
            (''.join(random.choices(string.ascii_letters, k=random.randint(5, 20))), 
             "random", "random string"),
        ]
        
        return values[:8]
    
    def generate_list_values(self, constraints: Dict[str, Any] = None) -> List[Tuple[List, str, str]]:
        """Generate list test values."""
        constraints = constraints or {}
        element_type = constraints.get('element_type', 'int')
        max_len = constraints.get('max_length', 10)
        
        if element_type == 'int':
            elements = [1, 2, 3, 4, 5]
        elif element_type == 'str':
            elements = ['a', 'b', 'c', 'd', 'e']
        else:
            elements = [1, 'a', 2, 'b', 3]
        
        values = [
            # Normal cases
            (elements[:3], "normal", "small list"),
            (elements, "normal", "medium list"),
            ([elements[0]] * 3, "normal", "repeated elements"),
            
            # Edge cases
            ([], "edge", "empty list"),
            (elements * (max_len // len(elements)), "edge", "large list"),
            ([None], "edge", "list with None"),
            
            # Boundary cases
            ([elements[0]], "boundary", "single element"),
            (elements[:max_len-1], "boundary", "near maximum length"),
            
            # Random cases
            (random.sample(elements * 3, min(len(elements * 3), random.randint(2, 6))), 
             "random", "random list"),
        ]
        
        return values[:8]
    
    def generate_dict_values(self, constraints: Dict[str, Any] = None) -> List[Tuple[Dict, str, str]]:
        """Generate dictionary test values."""
        constraints = constraints or {}
        
        values = [
            # Normal cases
            ({"key": "value"}, "normal", "simple dictionary"),
            ({"name": "John", "age": 30}, "normal", "mixed types dictionary"),
            ({"a": 1, "b": 2, "c": 3}, "normal", "numeric values dictionary"),
            
            # Edge cases
            ({}, "edge", "empty dictionary"),
            ({str(i): i for i in range(10)}, "edge", "large dictionary"),
            ({"nested": {"key": "value"}}, "edge", "nested dictionary"),
            
            # Boundary cases
            ({"single": "key"}, "boundary", "single key dictionary"),
            ({None: "null_key", "null_value": None}, "edge", "dictionary with None"),
            
            # Random cases
            ({f"key_{i}": random.randint(1, 100) for i in range(random.randint(2, 5))}, 
             "random", "random dictionary"),
        ]
        
        return values[:8]
    
    def generate_bool_values(self) -> List[Tuple[bool, str, str]]:
        """Generate boolean test values."""
        return [
            (True, "normal", "true value"),
            (False, "normal", "false value"),
        ]
    
    def generate_float_values(self, constraints: Dict[str, Any] = None) -> List[Tuple[float, str, str]]:
        """Generate float test values."""
        constraints = constraints or {}
        min_val = constraints.get('min', -100.0)
        max_val = constraints.get('max', 100.0)
        
        values = [
            # Normal cases
            (0.0, "normal", "zero float"),
            (1.5, "normal", "positive float"),
            (-1.5, "normal", "negative float"),
            (3.14159, "normal", "pi approximation"),
            
            # Edge cases
            (min_val, "edge", "minimum float"),
            (max_val, "edge", "maximum float"),
            (float('inf'), "edge", "positive infinity"),
            (float('-inf'), "edge", "negative infinity"),
            
            # Random cases
            (random.uniform(min_val / 2, max_val / 2), "random", "random float"),
        ]
        
        return values[:8]


class InputGenerator:
    """Main class for analyzing code and generating test inputs."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("data/inputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_generator = DataGenerator()
        self.temp_files: List[str] = []
    
    def analyze_code(self, code: str) -> List[InputRequirement]:
        """Analyze Python code to identify input requirements."""
        try:
            tree = ast.parse(code)
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            return analyzer.requirements
        except SyntaxError as e:
            print(f"Warning: Could not parse code due to syntax error: {e}")
            return []
    
    def generate_inputs_for_requirement(self, requirement: InputRequirement) -> List[TestInput]:
        """Generate test inputs for a specific requirement."""
        inputs = []
        data_type = requirement.inferred_type or 'str'
        
        if data_type == 'int':
            values = self.data_generator.generate_int_values(requirement.constraints)
            for i, (value, category, desc) in enumerate(values):
                inputs.append(TestInput(
                    input_id=f"{requirement.name}_{i}",
                    category=category,
                    data=value,
                    description=f"{desc} for {requirement.name}",
                    data_type='int'
                ))
        
        elif data_type == 'str':
            values = self.data_generator.generate_str_values(requirement.constraints)
            for i, (value, category, desc) in enumerate(values):
                inputs.append(TestInput(
                    input_id=f"{requirement.name}_{i}",
                    category=category,
                    data=value,
                    description=f"{desc} for {requirement.name}",
                    data_type='str'
                ))
        
        elif data_type == 'list':
            values = self.data_generator.generate_list_values(requirement.constraints)
            for i, (value, category, desc) in enumerate(values):
                inputs.append(TestInput(
                    input_id=f"{requirement.name}_{i}",
                    category=category,
                    data=value,
                    description=f"{desc} for {requirement.name}",
                    data_type='list'
                ))
        
        elif data_type == 'dict':
            values = self.data_generator.generate_dict_values(requirement.constraints)
            for i, (value, category, desc) in enumerate(values):
                inputs.append(TestInput(
                    input_id=f"{requirement.name}_{i}",
                    category=category,
                    data=value,
                    description=f"{desc} for {requirement.name}",
                    data_type='dict'
                ))
        
        elif data_type == 'bool':
            values = self.data_generator.generate_bool_values()
            for i, (value, category, desc) in enumerate(values):
                inputs.append(TestInput(
                    input_id=f"{requirement.name}_{i}",
                    category=category,
                    data=value,
                    description=f"{desc} for {requirement.name}",
                    data_type='bool'
                ))
        
        elif data_type == 'float':
            values = self.data_generator.generate_float_values(requirement.constraints)
            for i, (value, category, desc) in enumerate(values):
                inputs.append(TestInput(
                    input_id=f"{requirement.name}_{i}",
                    category=category,
                    data=value,
                    description=f"{desc} for {requirement.name}",
                    data_type='float'
                ))
        
        elif data_type == 'file':
            inputs.extend(self._generate_file_inputs(requirement))
        
        return inputs[:10]  # Limit to 10 inputs per requirement
    
    def _generate_file_inputs(self, requirement: InputRequirement) -> List[TestInput]:
        """Generate temporary test files for file input requirements."""
        inputs = []
        filename = requirement.constraints.get('filename', 'test_file.txt')
        file_ext = Path(filename).suffix
        
        # Different file content scenarios
        file_scenarios = [
            ("", "edge", "empty file"),
            ("single line", "normal", "single line file"),
            ("line 1\nline 2\nline 3", "normal", "multi-line file"),
            ("a" * 1000, "edge", "large content file"),
            ("special chars: !@#$%^&*()", "edge", "special characters file"),
            ("1\n2\n3\n4\n5", "normal", "numeric content"),
            ("word1 word2 word3", "normal", "space-separated words"),
            ("key=value\nname=test", "normal", "key-value pairs"),
        ]
        
        for i, (content, category, desc) in enumerate(file_scenarios):
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix=file_ext, delete=False
            )
            temp_file.write(content)
            temp_file.close()
            
            self.temp_files.append(temp_file.name)
            
            inputs.append(TestInput(
                input_id=f"{requirement.name}_{i}",
                category=category,
                data=temp_file.name,
                description=f"{desc} - {temp_file.name}",
                data_type='file'
            ))
        
        return inputs[:6]  # Limit file inputs
    
    def generate_inputs(self, code: str) -> Dict[str, Any]:
        """Generate comprehensive test inputs for Python code."""
        # Analyze code to identify requirements
        requirements = self.analyze_code(code)
        
        if not requirements:
            # If no specific requirements found, create generic inputs
            requirements = [
                InputRequirement("generic_input", "function_param", "str", 
                               description="Generic string input")
            ]
        
        # Generate inputs for each requirement
        all_inputs = []
        requirements_data = []
        
        for requirement in requirements:
            inputs = self.generate_inputs_for_requirement(requirement)
            all_inputs.extend(inputs)
            
            requirements_data.append({
                "name": requirement.name,
                "param_type": requirement.param_type,
                "inferred_type": requirement.inferred_type,
                "constraints": requirement.constraints,
                "description": requirement.description,
                "input_count": len(inputs)
            })
        
        # Organize inputs by category
        inputs_by_category = {
            "normal": [],
            "edge": [],
            "boundary": [],
            "random": []
        }
        
        for test_input in all_inputs:
            inputs_by_category[test_input.category].append({
                "input_id": test_input.input_id,
                "data": test_input.data,
                "description": test_input.description,
                "data_type": test_input.data_type,
                "created_at": test_input.created_at
            })
        
        return {
            "requirements": requirements_data,
            "inputs": inputs_by_category,
            "total_inputs": len(all_inputs),
            "analysis_timestamp": datetime.now().isoformat(),
            "temp_files": self.temp_files.copy()
        }
    
    def save_inputs(self, inputs_data: Dict[str, Any], filename: str) -> str:
        """Save generated inputs to JSON file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(inputs_data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(output_path)
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during input generation."""
        for temp_file in self.temp_files:
            try:
                Path(temp_file).unlink()
            except FileNotFoundError:
                pass
        self.temp_files.clear()
    
    def generate_and_save(self, code: str, output_filename: Optional[str] = None) -> Dict[str, Any]:
        """Generate inputs for code and save to file."""
        inputs_data = self.generate_inputs(code)
        
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"inputs_{timestamp}.json"
        
        output_path = self.save_inputs(inputs_data, output_filename)
        
        result = {
            "success": True,
            "inputs_data": inputs_data,
            "output_file": output_path,
            "summary": {
                "requirements_found": len(inputs_data["requirements"]),
                "total_inputs": inputs_data["total_inputs"],
                "normal_cases": len(inputs_data["inputs"]["normal"]),
                "edge_cases": len(inputs_data["inputs"]["edge"]),
                "boundary_cases": len(inputs_data["inputs"]["boundary"]),
                "random_cases": len(inputs_data["inputs"]["random"])
            }
        }
        
        return result
    
    def __del__(self):
        """Cleanup temporary files when object is destroyed."""
        self.cleanup_temp_files()