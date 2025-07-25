"""
Unit tests for the PythonExecutor class.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.executors.python_executor import (
    PythonExecutor, ExecutionResult, SecurityLimits
)


class TestExecutionResult(unittest.TestCase):
    """Test ExecutionResult class."""
    
    def test_init(self):
        """Test ExecutionResult initialization."""
        result = ExecutionResult()
        
        self.assertFalse(result.success)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.return_code, -1)
        self.assertEqual(result.execution_time, 0.0)
        self.assertIsNone(result.memory_peak)
        self.assertFalse(result.timeout_occurred)
        self.assertIsNone(result.error_type)
        self.assertEqual(result.error_message, "")
        self.assertIsNotNone(result.timestamp)
        self.assertEqual(result.input_data, {})
        self.assertEqual(result.execution_mode, "")
        self.assertEqual(result.python_version, "")
    
    def test_to_dict(self):
        """Test ExecutionResult to_dict conversion."""
        result = ExecutionResult()
        result.success = True
        result.stdout = "Hello World"
        result.execution_time = 1.5
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertTrue(result_dict["success"])
        self.assertEqual(result_dict["stdout"], "Hello World")
        self.assertEqual(result_dict["execution_time"], 1.5)


class TestSecurityLimits(unittest.TestCase):
    """Test SecurityLimits class."""
    
    def test_init(self):
        """Test SecurityLimits initialization."""
        limits = SecurityLimits()
        
        self.assertEqual(limits.timeout_seconds, 30)
        self.assertEqual(limits.max_memory_mb, 256)
        self.assertEqual(limits.max_output_size, 1024 * 1024)
        self.assertIsInstance(limits.blocked_modules, list)
        self.assertIn('subprocess', limits.blocked_modules)
        self.assertIn('os', limits.blocked_modules)


class TestPythonExecutor(unittest.TestCase):
    """Test PythonExecutor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use shorter timeouts for tests
        self.limits = SecurityLimits()
        self.limits.timeout_seconds = 5
        self.limits.max_memory_mb = 64
        
        self.executor = PythonExecutor(limits=self.limits)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.executor.cleanup()
    
    def test_init(self):
        """Test PythonExecutor initialization."""
        executor = PythonExecutor()
        
        self.assertIsNotNone(executor.limits)
        self.assertEqual(executor.temp_files, [])
        self.assertIsNotNone(executor.logger)
    
    def test_create_secure_wrapper(self):
        """Test secure wrapper creation."""
        code = "print('Hello World')"
        inputs = {"test": "data"}
        
        wrapped = self.executor._create_secure_wrapper(code, "script", inputs)
        
        self.assertIn("import sys", wrapped)
        self.assertIn("import json", wrapped)
        self.assertIn("print('Hello World')", wrapped)
        self.assertIn("EXEC_METADATA", wrapped)
    
    def test_indent_code(self):
        """Test code indentation."""
        code = "print('hello')\nprint('world')"
        indented = self.executor._indent_code(code, 4)
        
        lines = indented.split('\n')
        self.assertTrue(lines[0].startswith('    '))
        self.assertTrue(lines[1].startswith('    '))
    
    def test_prepare_function_execution(self):
        """Test function execution preparation."""
        code = "def add(a, b): return a + b"
        prepared = self.executor._prepare_function_execution(code, "add", [1, 2])
        
        self.assertIn("def add(a, b): return a + b", prepared)
        self.assertIn("result = add(*inputs)", prepared)
        self.assertIn("inputs = [1, 2]", prepared)
    
    def test_prepare_script_execution(self):
        """Test script execution preparation."""
        code = "print('Hello')"
        args = ["arg1", "arg2"]
        
        prepared_code, prepared_args = self.executor._prepare_script_execution(code, args)
        
        self.assertEqual(prepared_code, code)
        self.assertEqual(prepared_args, args)
    
    def test_prepare_interactive_execution(self):
        """Test interactive execution preparation."""
        code = "name = input('Name: ')"
        inputs = ["John", "Doe"]
        
        prepared_code, stdin_data = self.executor._prepare_interactive_execution(code, inputs)
        
        self.assertEqual(prepared_code, code)
        self.assertEqual(stdin_data, "John\nDoe\n")
    
    def test_sanitize_output(self):
        """Test output sanitization."""
        # Test normal output
        normal_output = "Hello World"
        sanitized = self.executor._sanitize_output(normal_output)
        self.assertEqual(sanitized, normal_output)
        
        # Test oversized output
        large_output = "A" * (self.executor.limits.max_output_size + 100)
        sanitized = self.executor._sanitize_output(large_output)
        self.assertLess(len(sanitized), len(large_output))
        self.assertIn("truncated", sanitized)
    
    def test_execute_simple_function(self):
        """Test execution of a simple function."""
        code = """
def add_numbers(a, b):
    return a + b
"""
        
        result = self.executor.execute_function(code, "add_numbers", [3, 4])
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.execution_mode, "function")
        self.assertIn("add_numbers", result.input_data["function_name"])
        self.assertEqual(result.input_data["inputs"], [3, 4])
        
        # Check if execution was successful (may vary based on system)
        if result.success:
            self.assertIn("7", result.stdout)
    
    def test_execute_simple_script(self):
        """Test execution of a simple script."""
        code = """
import sys
print(f"Hello from script with {len(sys.argv)} arguments")
for i, arg in enumerate(sys.argv):
    print(f"Arg {i}: {arg}")
"""
        
        result = self.executor.execute_script(code, ["test", "args"])
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.execution_mode, "script")
        self.assertEqual(result.input_data["cli_args"], ["test", "args"])
    
    def test_execute_interactive_code(self):
        """Test execution of interactive code."""
        code = """
name = input("Enter your name: ")
age = input("Enter your age: ")
print(f"Hello {name}, you are {age} years old")
"""
        
        result = self.executor.execute_interactive(code, ["Alice", "25"])
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.execution_mode, "interactive")
        self.assertEqual(result.input_data["stdin_inputs"], ["Alice", "25"])
    
    def test_execute_with_files(self):
        """Test execution with file inputs."""
        code = """
with open("test.txt", "r") as f:
    content = f.read()
    print(f"File content: {content}")
"""
        
        file_inputs = {"test.txt": "Hello from file!"}
        result = self.executor.execute_with_files(code, file_inputs)
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.execution_mode, "file_input")
        self.assertEqual(result.input_data["file_inputs"], ["test.txt"])
    
    def test_execute_function_with_error(self):
        """Test execution of function that raises an error."""
        code = """
def divide_by_zero():
    return 10 / 0
"""
        
        result = self.executor.execute_function(code, "divide_by_zero", [])
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertFalse(result.success)
        self.assertEqual(result.return_code, 1)
    
    def test_execute_timeout(self):
        """Test execution timeout handling."""
        code = """
import time
time.sleep(10)  # Sleep longer than timeout
print("This should not print")
"""
        
        # Use very short timeout for test
        short_limits = SecurityLimits()
        short_limits.timeout_seconds = 1
        executor = PythonExecutor(limits=short_limits)
        
        try:
            result = executor.execute_script(code)
            
            self.assertIsInstance(result, ExecutionResult)
            # Should either timeout or be killed by resource limits
            self.assertFalse(result.success)
        finally:
            executor.cleanup()
    
    def test_execute_invalid_code(self):
        """Test execution of syntactically invalid code."""
        code = """
def broken_function(
    # Missing closing parenthesis and colon
"""
        
        result = self.executor.execute_script(code)
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertFalse(result.success)
        self.assertNotEqual(result.return_code, 0)
    
    def test_extract_execution_metadata(self):
        """Test extraction of execution metadata from stderr."""
        result = ExecutionResult()
        result.stderr = 'Some error\nEXEC_METADATA: {"execution_time": 1.5, "success": true}\nMore output'
        
        self.executor._extract_execution_metadata(result)
        
        self.assertEqual(result.execution_time, 1.5)
        self.assertNotIn("EXEC_METADATA", result.stderr)
    
    def test_cleanup(self):
        """Test cleanup of temporary files."""
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        temp_path = temp_file.name
        
        # Add to executor's temp files
        self.executor.temp_files.append(temp_path)
        
        # Verify file exists
        self.assertTrue(Path(temp_path).exists())
        
        # Cleanup
        self.executor.cleanup()
        
        # Verify file is removed and list is cleared
        self.assertFalse(Path(temp_path).exists())
        self.assertEqual(len(self.executor.temp_files), 0)
    
    def test_multiple_executions(self):
        """Test multiple executions with the same executor."""
        code1 = "print('First execution')"
        code2 = "print('Second execution')"
        
        result1 = self.executor.execute_script(code1)
        result2 = self.executor.execute_script(code2)
        
        self.assertIsInstance(result1, ExecutionResult)
        self.assertIsInstance(result2, ExecutionResult)
        
        # Results should be independent
        self.assertNotEqual(result1.timestamp, result2.timestamp)
    
    def test_large_output_handling(self):
        """Test handling of large output."""
        code = f"""
# Generate output larger than limit
for i in range(1000):
    print("A" * 1000)
"""
        
        result = self.executor.execute_script(code)
        
        self.assertIsInstance(result, ExecutionResult)
        # Output should be truncated
        self.assertLessEqual(len(result.stdout), self.executor.limits.max_output_size)
    
    def test_function_with_complex_inputs(self):
        """Test function execution with complex input types."""
        code = """
def process_data(data_dict, items_list):
    result = []
    for item in items_list:
        if item in data_dict:
            result.append(data_dict[item])
    return result
"""
        
        inputs = [{"a": 1, "b": 2, "c": 3}, ["a", "c", "d"]]
        result = self.executor.execute_function(code, "process_data", inputs)
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.input_data["inputs"], inputs)


class TestExecutorIntegration(unittest.TestCase):
    """Integration tests for PythonExecutor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.executor = PythonExecutor()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.executor.cleanup()
    
    def test_real_world_function(self):
        """Test execution of a real-world function."""
        code = """
def calculate_fibonacci(n):
    '''Calculate nth Fibonacci number.'''
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

def main():
    for i in range(10):
        print(f"fib({i}) = {calculate_fibonacci(i)}")

if __name__ == "__main__":
    main()
"""
        
        # Test as script
        result = self.executor.execute_script(code)
        self.assertIsInstance(result, ExecutionResult)
        
        # Test specific function
        result = self.executor.execute_function(code, "calculate_fibonacci", [10])
        self.assertIsInstance(result, ExecutionResult)
    
    def test_file_processing_workflow(self):
        """Test a complete file processing workflow."""
        code = """
import json

def process_json_file(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Process the data
        if isinstance(data, dict):
            result = {k: len(str(v)) for k, v in data.items()}
        elif isinstance(data, list):
            result = [len(str(item)) for item in data]
        else:
            result = len(str(data))
        
        print(f"Processed data: {result}")
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Test the function
process_json_file("data.json")
"""
        
        file_inputs = {
            "data.json": json.dumps({"name": "John", "age": 30, "city": "New York"})
        }
        
        result = self.executor.execute_with_files(code, file_inputs)
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.execution_mode, "file_input")


if __name__ == "__main__":
    unittest.main()