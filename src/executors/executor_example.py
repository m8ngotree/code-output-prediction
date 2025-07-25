"""
Example usage of the PythonExecutor class.

This script demonstrates how to use the PythonExecutor to safely execute
Python code with different input methods and execution modes.
"""

import json
from python_executor import PythonExecutor, SecurityLimits


def demonstrate_function_execution():
    """Demonstrate function execution with various inputs."""
    print("=" * 60)
    print("Function Execution Examples")
    print("=" * 60)
    
    executor = PythonExecutor()
    
    # Example 1: Simple mathematical function
    print("\n1. Simple Mathematical Function:")
    math_code = """
def calculate_area(length, width):
    '''Calculate area of a rectangle.'''
    if length <= 0 or width <= 0:
        raise ValueError("Dimensions must be positive")
    return length * width
"""
    
    result = executor.execute_function(math_code, "calculate_area", [5, 3])
    print(f"   Success: {result.success}")
    print(f"   Output: {result.stdout.strip()}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    
    # Example 2: Function with error
    print("\n2. Function with Error Handling:")
    result = executor.execute_function(math_code, "calculate_area", [-1, 5])
    print(f"   Success: {result.success}")
    print(f"   Error: {result.error_message}")
    print(f"   Return code: {result.return_code}")
    
    # Example 3: Complex data processing function
    print("\n3. Complex Data Processing:")
    data_code = """
def analyze_scores(scores, threshold=70):
    '''Analyze student scores.'''
    if not scores:
        return {"error": "No scores provided"}
    
    passed = [s for s in scores if s >= threshold]
    failed = [s for s in scores if s < threshold]
    
    result = {
        "total_students": len(scores),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": len(passed) / len(scores) * 100,
        "average": sum(scores) / len(scores),
        "highest": max(scores),
        "lowest": min(scores)
    }
    
    print(f"Analysis Results: {result}")
    return result
"""
    
    scores = [85, 92, 67, 78, 95, 45, 88, 73, 91, 82]
    result = executor.execute_function(data_code, "analyze_scores", [scores, 75])
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Output: {result.stdout.strip()}")
    
    executor.cleanup()


def demonstrate_script_execution():
    """Demonstrate script execution with command line arguments."""
    print("\n" + "=" * 60)
    print("Script Execution Examples")
    print("=" * 60)
    
    executor = PythonExecutor()
    
    # Example 1: Simple script with arguments
    print("\n1. Script with Command Line Arguments:")
    script_code = """
import sys

def main():
    print(f"Script called with {len(sys.argv)} arguments:")
    for i, arg in enumerate(sys.argv):
        print(f"  argv[{i}]: {arg}")
    
    if len(sys.argv) > 1:
        try:
            numbers = [float(arg) for arg in sys.argv[1:]]
            total = sum(numbers)
            print(f"Sum of numbers: {total}")
        except ValueError:
            print("Error: All arguments must be numbers")

if __name__ == "__main__":
    main()
"""
    
    result = executor.execute_script(script_code, ["10", "20", "30.5"])
    print(f"   Success: {result.success}")
    print(f"   Output: {result.stdout.strip()}")
    
    # Example 2: File processing script
    print("\n2. File Processing Script:")
    file_script = """
import sys
import os

def process_file(filename):
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        return False
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        print(f"File: {filename}")
        print(f"Lines: {len(lines)}")
        print(f"Characters: {sum(len(line) for line in lines)}")
        print(f"Words: {sum(len(line.split()) for line in lines)}")
        
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if len(sys.argv) != 2:
    print("Usage: script.py <filename>")
    sys.exit(1)

process_file(sys.argv[1])
"""
    
    # Create a temporary file and execute with it
    file_inputs = {"sample.txt": "Hello world!\\nThis is a test file.\\nWith multiple lines."}
    result = executor.execute_with_files(file_script, file_inputs)
    
    # Update the script to use the temporary file
    result = executor.execute_script(file_script, ["sample.txt"])
    print(f"   Success: {result.success}")
    print(f"   Output: {result.stdout.strip()}")
    
    executor.cleanup()


def demonstrate_interactive_execution():
    """Demonstrate interactive execution with stdin simulation."""
    print("\n" + "=" * 60)
    print("Interactive Execution Examples")
    print("=" * 60)
    
    executor = PythonExecutor()
    
    # Example 1: Simple interactive program
    print("\n1. Simple Interactive Program:")
    interactive_code = """
def main():
    name = input("Enter your name: ")
    age = input("Enter your age: ")
    
    try:
        age = int(age)
        print(f"Hello {name}! You are {age} years old.")
        
        if age >= 18:
            print("You are an adult.")
        else:
            print("You are a minor.")
            
    except ValueError:
        print("Invalid age entered.")

main()
"""
    
    result = executor.execute_interactive(interactive_code, ["Alice", "25"])
    print(f"   Success: {result.success}")
    print(f"   Output: {result.stdout.strip()}")
    
    # Example 2: Interactive calculator
    print("\n2. Interactive Calculator:")
    calculator_code = """
def calculator():
    while True:
        operation = input("Enter operation (+, -, *, /) or 'quit': ")
        
        if operation.lower() == 'quit':
            print("Goodbye!")
            break
        
        if operation not in ['+', '-', '*', '/']:
            print("Invalid operation!")
            continue
        
        try:
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))
            
            if operation == '+':
                result = num1 + num2
            elif operation == '-':
                result = num1 - num2
            elif operation == '*':
                result = num1 * num2
            elif operation == '/':
                if num2 == 0:
                    print("Error: Division by zero!")
                    continue
                result = num1 / num2
            
            print(f"Result: {result}")
            
        except ValueError:
            print("Invalid number!")

calculator()
"""
    
    inputs = ["+", "10", "5", "*", "3", "4", "quit"]
    result = executor.execute_interactive(calculator_code, inputs)
    print(f"   Success: {result.success}")
    print(f"   Output: {result.stdout.strip()}")
    
    executor.cleanup()


def demonstrate_file_execution():
    """Demonstrate execution with file inputs."""
    print("\n" + "=" * 60)
    print("File Input Execution Examples")
    print("=" * 60)
    
    executor = PythonExecutor()
    
    # Example 1: CSV processing
    print("\n1. CSV Data Processing:")
    csv_code = """
def process_csv(filename):
    data = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split(',')
    print(f"Columns: {header}")
    
    # Parse data
    for line in lines[1:]:
        values = line.strip().split(',')
        row = dict(zip(header, values))
        data.append(row)
    
    print(f"Loaded {len(data)} records")
    
    # Calculate some statistics
    if 'age' in header:
        ages = [int(row['age']) for row in data if row['age'].isdigit()]
        if ages:
            print(f"Average age: {sum(ages) / len(ages):.1f}")
            print(f"Age range: {min(ages)} - {max(ages)}")
    
    return data

# Process the CSV file
data = process_csv("data.csv")
print(f"Sample record: {data[0] if data else 'No data'}")
"""
    
    csv_content = """name,age,city
John,25,New York
Alice,30,Los Angeles
Bob,35,Chicago
Carol,28,Houston"""
    
    result = executor.execute_with_files(csv_code, {"data.csv": csv_content})
    print(f"   Success: {result.success}")
    print(f"   Output: {result.stdout.strip()}")
    
    # Example 2: JSON configuration processing
    print("\n2. JSON Configuration Processing:")
    json_code = """
import json

def process_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("Configuration loaded:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Validate required settings
    required = ['database_url', 'api_key', 'debug']
    missing = [key for key in required if key not in config]
    
    if missing:
        print(f"Missing required settings: {missing}")
        return False
    
    print("Configuration is valid!")
    return True

process_config("config.json")
"""
    
    config_content = json.dumps({
        "database_url": "postgresql://localhost:5432/mydb",
        "api_key": "secret-key-123",
        "debug": True,
        "port": 8080,
        "workers": 4
    }, indent=2)
    
    result = executor.execute_with_files(json_code, {"config.json": config_content})
    print(f"   Success: {result.success}")
    print(f"   Output: {result.stdout.strip()}")
    
    executor.cleanup()


def demonstrate_security_features():
    """Demonstrate security features and limits."""
    print("\n" + "=" * 60)
    print("Security Features Demonstration")
    print("=" * 60)
    
    # Create executor with strict security limits
    strict_limits = SecurityLimits()
    strict_limits.timeout_seconds = 2
    strict_limits.max_memory_mb = 32
    strict_limits.max_output_size = 1000
    
    executor = PythonExecutor(limits=strict_limits)
    
    # Example 1: Timeout handling
    print("\n1. Timeout Handling (should timeout):")
    timeout_code = """
import time
print("Starting long operation...")
time.sleep(5)  # This will timeout
print("This shouldn't print")
"""
    
    result = executor.execute_script(timeout_code)
    print(f"   Success: {result.success}")
    print(f"   Timeout occurred: {result.timeout_occurred}")
    print(f"   Error: {result.error_message}")
    
    # Example 2: Large output handling
    print("\n2. Large Output Handling (should be truncated):")
    large_output_code = """
for i in range(100):
    print("A" * 100)  # Generate large output
"""
    
    result = executor.execute_script(large_output_code)
    print(f"   Success: {result.success}")
    print(f"   Output length: {len(result.stdout)} characters")
    print(f"   Truncated: {'truncated' in result.stdout.lower()}")
    
    # Example 3: Error handling
    print("\n3. Error Handling:")
    error_code = """
def divide_by_zero():
    return 10 / 0

result = divide_by_zero()
print(f"Result: {result}")
"""
    
    result = executor.execute_script(error_code)
    print(f"   Success: {result.success}")
    print(f"   Return code: {result.return_code}")
    print(f"   Error type: {result.error_type}")
    
    executor.cleanup()


def main():
    """Run all demonstration examples."""
    print("🔧 Python Executor Demonstration")
    print("This script shows various ways to safely execute Python code")
    
    try:
        demonstrate_function_execution()
        demonstrate_script_execution()
        demonstrate_interactive_execution()
        demonstrate_file_execution()
        demonstrate_security_features()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()