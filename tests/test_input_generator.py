"""
Unit tests for the InputGenerator class.
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path

from src.generators.input_generator import (
    InputGenerator, CodeAnalyzer, DataGenerator, 
    InputRequirement, TestInput
)


class TestInputRequirement(unittest.TestCase):
    """Test InputRequirement class."""
    
    def test_init(self):
        """Test InputRequirement initialization."""
        req = InputRequirement(
            name="test_param",
            param_type="function_param",
            inferred_type="int",
            constraints={"min": 0, "max": 100},
            description="Test parameter"
        )
        
        self.assertEqual(req.name, "test_param")
        self.assertEqual(req.param_type, "function_param")
        self.assertEqual(req.inferred_type, "int")
        self.assertEqual(req.constraints, {"min": 0, "max": 100})
        self.assertEqual(req.description, "Test parameter")


class TestTestInput(unittest.TestCase):
    """Test TestInput class."""
    
    def test_init(self):
        """Test TestInput initialization."""
        test_input = TestInput(
            input_id="test_1",
            category="normal",
            data=42,
            description="Test integer",
            data_type="int"
        )
        
        self.assertEqual(test_input.input_id, "test_1")
        self.assertEqual(test_input.category, "normal")
        self.assertEqual(test_input.data, 42)
        self.assertEqual(test_input.description, "Test integer")
        self.assertEqual(test_input.data_type, "int")
        self.assertIsNotNone(test_input.created_at)


class TestCodeAnalyzer(unittest.TestCase):
    """Test CodeAnalyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = CodeAnalyzer()
    
    def test_function_analysis(self):
        """Test analysis of function parameters."""
        code = """
def test_function(name: str, age: int, items: list):
    return f"{name} is {age} years old"
"""
        tree = ast.parse(code)
        self.analyzer.visit(tree)
        
        # Should find 3 function parameters
        func_params = [req for req in self.analyzer.requirements 
                      if req.param_type == "function_param"]
        self.assertEqual(len(func_params), 3)
        
        # Check parameter names and types
        param_names = [req.name for req in func_params]
        self.assertIn("name", param_names)
        self.assertIn("age", param_names)
        self.assertIn("items", param_names)
    
    def test_stdin_analysis(self):
        """Test analysis of stdin input."""
        code = """
name = input("Enter your name: ")
age = input("Enter your age: ")
"""
        tree = ast.parse(code)
        self.analyzer.visit(tree)
        
        stdin_reqs = [req for req in self.analyzer.requirements 
                     if req.param_type == "stdin"]
        self.assertTrue(len(stdin_reqs) >= 1)
        self.assertTrue(self.analyzer.stdin_usage)
    
    def test_file_analysis(self):
        """Test analysis of file operations."""
        code = """
with open("test.txt", "r") as f:
    content = f.read()
"""
        tree = ast.parse(code)
        self.analyzer.visit(tree)
        
        file_reqs = [req for req in self.analyzer.requirements 
                    if req.param_type == "file_input"]
        self.assertTrue(len(file_reqs) >= 1)
        self.assertIn("test.txt", self.analyzer.file_operations)
    
    def test_type_inference_from_name(self):
        """Test type inference from parameter names."""
        analyzer = CodeAnalyzer()
        
        # Test integer patterns
        self.assertEqual(analyzer._infer_type_from_name("count"), "int")
        self.assertEqual(analyzer._infer_type_from_name("user_age"), "int")
        
        # Test string patterns
        self.assertEqual(analyzer._infer_type_from_name("name"), "str")
        self.assertEqual(analyzer._infer_type_from_name("description"), "str")
        
        # Test list patterns
        self.assertEqual(analyzer._infer_type_from_name("items"), "list")
        self.assertEqual(analyzer._infer_type_from_name("data_list"), "list")
        
        # Test boolean patterns
        self.assertEqual(analyzer._infer_type_from_name("enabled"), "bool")
        self.assertEqual(analyzer._infer_type_from_name("flag"), "bool")


class TestDataGenerator(unittest.TestCase):
    """Test DataGenerator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = DataGenerator(seed=42)  # Fixed seed for reproducible tests
    
    def test_generate_int_values(self):
        """Test integer value generation."""
        values = self.generator.generate_int_values()
        
        self.assertTrue(len(values) > 0)
        
        # Check that we have different categories
        categories = [category for _, category, _ in values]
        self.assertIn("normal", categories)
        self.assertIn("edge", categories)
        
        # Check that all values are integers
        for value, _, _ in values:
            self.assertIsInstance(value, int)
    
    def test_generate_str_values(self):
        """Test string value generation."""
        values = self.generator.generate_str_values()
        
        self.assertTrue(len(values) > 0)
        
        # Check for empty string (edge case)
        value_data = [value for value, _, _ in values]
        self.assertIn("", value_data)
        
        # Check that all values are strings
        for value, _, _ in values:
            self.assertIsInstance(value, str)
    
    def test_generate_list_values(self):
        """Test list value generation."""
        values = self.generator.generate_list_values()
        
        self.assertTrue(len(values) > 0)
        
        # Check for empty list (edge case)
        value_data = [value for value, _, _ in values]
        self.assertIn([], value_data)
        
        # Check that all values are lists
        for value, _, _ in values:
            self.assertIsInstance(value, list)
    
    def test_generate_dict_values(self):
        """Test dictionary value generation."""
        values = self.generator.generate_dict_values()
        
        self.assertTrue(len(values) > 0)
        
        # Check for empty dict (edge case)
        value_data = [value for value, _, _ in values]
        self.assertIn({}, value_data)
        
        # Check that all values are dictionaries
        for value, _, _ in values:
            self.assertIsInstance(value, dict)
    
    def test_generate_bool_values(self):
        """Test boolean value generation."""
        values = self.generator.generate_bool_values()
        
        self.assertEqual(len(values), 2)  # Should have True and False
        
        value_data = [value for value, _, _ in values]
        self.assertIn(True, value_data)
        self.assertIn(False, value_data)
    
    def test_generate_float_values(self):
        """Test float value generation."""
        values = self.generator.generate_float_values()
        
        self.assertTrue(len(values) > 0)
        
        # Check that all values are floats
        for value, _, _ in values:
            self.assertIsInstance(value, float)


class TestInputGenerator(unittest.TestCase):
    """Test InputGenerator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = InputGenerator(output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.generator.cleanup_temp_files()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_simple_function(self):
        """Test analysis of a simple function."""
        code = """
def add_numbers(a: int, b: int) -> int:
    return a + b
"""
        requirements = self.generator.analyze_code(code)
        
        self.assertEqual(len(requirements), 2)
        
        param_names = [req.name for req in requirements]
        self.assertIn("a", param_names)
        self.assertIn("b", param_names)
        
        # Check inferred types
        for req in requirements:
            self.assertEqual(req.inferred_type, "int")
    
    def test_analyze_input_function(self):
        """Test analysis of code with input() calls."""
        code = """
name = input("Enter your name: ")
age = int(input("Enter your age: "))
print(f"Hello {name}, you are {age} years old")
"""
        requirements = self.generator.analyze_code(code)
        
        stdin_reqs = [req for req in requirements if req.param_type == "stdin"]
        self.assertTrue(len(stdin_reqs) > 0)
    
    def test_generate_inputs_for_int_requirement(self):
        """Test input generation for integer requirement."""
        requirement = InputRequirement(
            name="test_int",
            param_type="function_param",
            inferred_type="int"
        )
        
        inputs = self.generator.generate_inputs_for_requirement(requirement)
        
        self.assertTrue(len(inputs) > 0)
        
        # Check that we have different categories
        categories = [inp.category for inp in inputs]
        self.assertIn("normal", categories)
        self.assertIn("edge", categories)
        
        # Check that all inputs are for integers
        for inp in inputs:
            self.assertEqual(inp.data_type, "int")
            self.assertIsInstance(inp.data, int)
    
    def test_generate_inputs_for_str_requirement(self):
        """Test input generation for string requirement."""
        requirement = InputRequirement(
            name="test_str",
            param_type="function_param",
            inferred_type="str"
        )
        
        inputs = self.generator.generate_inputs_for_requirement(requirement)
        
        self.assertTrue(len(inputs) > 0)
        
        # Check for empty string
        input_data = [inp.data for inp in inputs]
        self.assertIn("", input_data)
        
        # Check that all inputs are strings
        for inp in inputs:
            self.assertEqual(inp.data_type, "str")
            self.assertIsInstance(inp.data, str)
    
    def test_generate_inputs_comprehensive(self):
        """Test comprehensive input generation."""
        code = """
def process_data(name: str, count: int, items: list):
    result = []
    for i in range(count):
        result.append(f"{name}_{i}")
    return result + items
"""
        inputs_data = self.generator.generate_inputs(code)
        
        self.assertIn("requirements", inputs_data)
        self.assertIn("inputs", inputs_data)
        self.assertIn("total_inputs", inputs_data)
        
        # Should have found 3 requirements
        self.assertEqual(len(inputs_data["requirements"]), 3)
        
        # Should have inputs in different categories
        self.assertIn("normal", inputs_data["inputs"])
        self.assertIn("edge", inputs_data["inputs"])
        self.assertIn("boundary", inputs_data["inputs"])
        self.assertIn("random", inputs_data["inputs"])
    
    def test_save_inputs(self):
        """Test saving inputs to file."""
        inputs_data = {
            "requirements": [],
            "inputs": {"normal": [], "edge": [], "boundary": [], "random": []},
            "total_inputs": 0,
            "analysis_timestamp": "2023-01-01T00:00:00"
        }
        
        filename = "test_inputs.json"
        output_path = self.generator.save_inputs(inputs_data, filename)
        
        self.assertTrue(Path(output_path).exists())
        
        # Verify content
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data["total_inputs"], 0)
    
    def test_generate_and_save(self):
        """Test complete workflow of generating and saving inputs."""
        code = """
def greet(name: str, age: int):
    return f"Hello {name}, you are {age} years old"
"""
        
        result = self.generator.generate_and_save(code)
        
        self.assertTrue(result["success"])
        self.assertIn("inputs_data", result)
        self.assertIn("output_file", result)
        self.assertIn("summary", result)
        
        # Check that file was created
        output_file = result["output_file"]
        self.assertTrue(Path(output_file).exists())
        
        # Check summary
        summary = result["summary"]
        self.assertGreater(summary["requirements_found"], 0)
        self.assertGreater(summary["total_inputs"], 0)
    
    def test_file_input_generation(self):
        """Test generation of file inputs."""
        requirement = InputRequirement(
            name="test_file",
            param_type="file_input",
            inferred_type="file",
            constraints={"filename": "test.txt"}
        )
        
        inputs = self.generator.generate_inputs_for_requirement(requirement)
        
        self.assertTrue(len(inputs) > 0)
        
        # Check that files were created
        for inp in inputs:
            if inp.data_type == "file":
                self.assertTrue(Path(inp.data).exists())
    
    def test_syntax_error_handling(self):
        """Test handling of code with syntax errors."""
        invalid_code = """
def broken_function(
    # Missing closing parenthesis and colon
"""
        requirements = self.generator.analyze_code(invalid_code)
        
        # Should return empty list for invalid code
        self.assertEqual(len(requirements), 0)


if __name__ == "__main__":
    unittest.main()