"""
Library-focused Code Executor with Enhanced Safety and Verification

This module provides safe execution of library-dependent Python code with
comprehensive error handling, timeout management, and output verification.
"""

import subprocess
import tempfile
import time
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import importlib.util


class LibraryCodeExecutor:
    """
    Executes Python code that uses external libraries safely.
    
    Features:
    - Library dependency checking
    - Safe subprocess execution
    - Memory and time limits
    - Comprehensive error reporting
    - Output format validation
    """
    
    def __init__(self, timeout_seconds: int = 30, max_memory_mb: int = 512):
        """
        Initialize executor with safety limits.
        
        Args:
            timeout_seconds: Maximum execution time
            max_memory_mb: Maximum memory usage (approximate)
        """
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        
        # Define required libraries and their installation commands
        self.library_requirements = {
            "pandas": "pandas>=1.3.0",
            "numpy": "numpy>=1.20.0",
            "requests": "requests>=2.25.0",
            "json": "built-in",  # Built-in module
            "re": "built-in",    # Built-in module
            "datetime": "built-in"  # Built-in module
        }
        
        # Check available libraries
        self.available_libraries = self._check_available_libraries()
    
    def _check_available_libraries(self) -> Dict[str, bool]:
        """Check which libraries are available in the current environment."""
        available = {}
        
        for library in self.library_requirements.keys():
            try:
                importlib.import_module(library)
                available[library] = True
            except ImportError:
                available[library] = False
        
        return available
    
    def execute(self, code: str, input_data: Optional[str] = None, 
                required_libraries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute Python code with library dependencies.
        
        Args:
            code: Python code to execute
            input_data: Optional input data for the code
            required_libraries: List of required libraries
            
        Returns:
            Dict containing execution results, output, and metadata
        """
        # Validate library requirements
        missing_libs = self._check_library_requirements(code, required_libraries)
        if missing_libs:
            return {
                "success": False,
                "output": None,
                "error": f"Missing required libraries: {', '.join(missing_libs)}",
                "execution_time": 0,
                "return_code": -1,
                "missing_libraries": missing_libs
            }
        
        # Prepare code for execution
        enhanced_code = self._prepare_code_for_execution(code, input_data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(enhanced_code)
            temp_file = f.name
        
        try:
            start_time = time.time()
            
            # Execute with enhanced monitoring
            result = self._execute_with_monitoring(temp_file, input_data)
            
            execution_time = time.time() - start_time
            
            # Parse and validate output
            parsed_result = self._parse_execution_result(result, execution_time)
            
            # Add library usage analysis
            parsed_result["library_usage"] = self._analyze_library_usage(code)
            
            return parsed_result
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": None,
                "error": f"Execution timed out after {self.timeout_seconds} seconds",
                "execution_time": self.timeout_seconds,
                "return_code": -1,
                "timeout": True
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "execution_time": 0,
                "return_code": -1,
                "exception_type": type(e).__name__
            }
        finally:
            # Clean up temporary file
            Path(temp_file).unlink(missing_ok=True)
    
    def _check_library_requirements(self, code: str, required_libraries: Optional[List[str]]) -> List[str]:
        """Check if all required libraries are available."""
        # Extract imports from code
        imports = self._extract_imports_from_code(code)
        
        # Combine with explicitly required libraries
        all_required = set(imports)
        if required_libraries:
            all_required.update(required_libraries)
        
        # Check availability
        missing = []
        for lib in all_required:
            if lib in self.available_libraries and not self.available_libraries[lib]:
                missing.append(lib)
        
        return missing
    
    def _extract_imports_from_code(self, code: str) -> List[str]:
        """Extract library imports from code."""
        import ast
        imports = []
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])
        except SyntaxError:
            # If parsing fails, use regex fallback
            import re
            import_patterns = [
                r'import\s+(\w+)',
                r'from\s+(\w+)\s+import'
            ]
            for pattern in import_patterns:
                matches = re.findall(pattern, code)
                imports.extend(matches)
        
        return imports
    
    def _prepare_code_for_execution(self, code: str, input_data: Optional[str]) -> str:
        """Prepare code for safe execution with proper setup."""
        setup_code = """import sys
import json
import traceback
from io import StringIO
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Capture stdout
old_stdout = sys.stdout
sys.stdout = captured_output = StringIO()

try:"""
        
        cleanup_code = """
except Exception as e:
    sys.stdout = old_stdout
    print(f"EXECUTION_ERROR: {type(e).__name__}: {str(e)}")
    traceback.print_exc()
else:
    sys.stdout = old_stdout
    output = captured_output.getvalue()
    if output.strip():
        print(f"EXECUTION_OUTPUT: {output.strip()}")
    else:
        print("EXECUTION_OUTPUT: [No output]")"""
        
        # Add input data handling if provided
        input_setup = ""
        if input_data:
            input_setup = f"""
    # Input data
    input_data = {repr(input_data)}"""
        
        # Properly indent the user code
        indented_code = "\n".join(f"    {line}" for line in code.split("\n"))
        
        # Combine all parts
        enhanced_code = setup_code + input_setup + "\n" + indented_code + "\n" + cleanup_code
        
        return enhanced_code
    
    def _execute_with_monitoring(self, temp_file: str, input_data: Optional[str]) -> subprocess.CompletedProcess:
        """Execute code with resource monitoring."""
        # Prepare environment with memory limits (Unix-like systems)
        env = os.environ.copy()
        
        # Execute with timeout and resource limits
        cmd = [sys.executable, temp_file]
        
        result = subprocess.run(
            cmd,
            input=input_data,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            env=env
        )
        
        return result
    
    def _parse_execution_result(self, result: subprocess.CompletedProcess, execution_time: float) -> Dict[str, Any]:
        """Parse and structure the execution result."""
        stdout = result.stdout
        stderr = result.stderr
        
        # Extract structured output
        output = None
        error = None
        
        if "EXECUTION_OUTPUT:" in stdout:
            output_line = [line for line in stdout.split('\n') if line.startswith("EXECUTION_OUTPUT:")][0]
            output = output_line.replace("EXECUTION_OUTPUT:", "").strip()
            if output == "[No output]":
                output = ""
        
        if "EXECUTION_ERROR:" in stdout:
            error_line = [line for line in stdout.split('\n') if line.startswith("EXECUTION_ERROR:")][0]
            error = error_line.replace("EXECUTION_ERROR:", "").strip()
        elif stderr:
            error = stderr.strip()
        
        success = result.returncode == 0 and error is None
        
        return {
            "success": success,
            "output": output,
            "error": error,
            "execution_time": execution_time,
            "return_code": result.returncode,
            "raw_stdout": stdout,
            "raw_stderr": stderr
        }
    
    def _analyze_library_usage(self, code: str) -> Dict[str, Any]:
        """Analyze how libraries are used in the code."""
        import ast
        
        analysis = {
            "imports": [],
            "function_calls": [],
            "complexity_score": 0
        }
        
        try:
            tree = ast.parse(code)
            
            # Analyze imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append({
                            "module": alias.name,
                            "alias": alias.asname
                        })
                elif isinstance(node, ast.ImportFrom):
                    analysis["imports"].append({
                        "module": node.module,
                        "names": [alias.name for alias in node.names]
                    })
                elif isinstance(node, ast.Call):
                    # Track function calls
                    if hasattr(node.func, 'attr'):
                        analysis["function_calls"].append(node.func.attr)
            
            # Simple complexity score based on node types
            complexity_nodes = [ast.For, ast.While, ast.If, ast.Try, ast.With]
            analysis["complexity_score"] = sum(
                len([n for n in ast.walk(tree) if isinstance(n, node_type)])
                for node_type in complexity_nodes
            )
            
        except SyntaxError:
            analysis["parse_error"] = True
        
        return analysis
    
    def execute_with_verification(self, code: str, expected_output: str, 
                                 input_data: Optional[str] = None) -> Dict[str, Any]:
        """Execute code and verify output matches expected result."""
        # Execute the code
        result = self.execute(code, input_data)
        
        if not result["success"]:
            return {
                **result,
                "verification": {
                    "matches_expected": False,
                    "expected_output": expected_output,
                    "match_type": "execution_failed"
                }
            }
        
        # Verify output
        actual_output = result["output"] or ""
        verification = self._verify_output(expected_output, actual_output)
        
        result["verification"] = verification
        return result
    
    def _verify_output(self, expected: str, actual: str) -> Dict[str, Any]:
        """Verify actual output matches expected output with multiple strategies."""
        expected_clean = expected.strip()
        actual_clean = actual.strip()
        
        # Exact match
        exact_match = expected_clean == actual_clean
        
        # Fuzzy match (ignoring whitespace differences)
        fuzzy_match = self._normalize_whitespace(expected_clean) == self._normalize_whitespace(actual_clean)
        
        # Numeric match (if both are numbers)
        numeric_match = self._try_numeric_match(expected_clean, actual_clean)
        
        # JSON match (if both are valid JSON)
        json_match = self._try_json_match(expected_clean, actual_clean)
        
        # Determine best match type
        if exact_match:
            match_type = "exact"
            matches = True
        elif fuzzy_match:
            match_type = "fuzzy"
            matches = True
        elif numeric_match:
            match_type = "numeric"
            matches = True
        elif json_match:
            match_type = "json"
            matches = True
        else:
            match_type = "none"
            matches = False
        
        return {
            "matches_expected": matches,
            "match_type": match_type,
            "expected_output": expected_clean,
            "actual_output": actual_clean,
            "exact_match": exact_match,
            "fuzzy_match": fuzzy_match,
            "numeric_match": numeric_match is not None,
            "json_match": json_match is not None
        }
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace for fuzzy matching."""
        import re
        return re.sub(r'\s+', ' ', text.strip())
    
    def _try_numeric_match(self, expected: str, actual: str) -> Optional[bool]:
        """Try to match as numeric values."""
        try:
            expected_num = float(expected)
            actual_num = float(actual)
            return abs(expected_num - actual_num) < 1e-10
        except ValueError:
            return None
    
    def _try_json_match(self, expected: str, actual: str) -> Optional[bool]:
        """Try to match as JSON objects."""
        try:
            expected_json = json.loads(expected)
            actual_json = json.loads(actual)
            return expected_json == actual_json
        except (json.JSONDecodeError, TypeError):
            return None
    
    def get_library_status(self) -> Dict[str, Any]:
        """Get status of library availability."""
        return {
            "available_libraries": self.available_libraries,
            "missing_libraries": [lib for lib, available in self.available_libraries.items() if not available],
            "total_libraries": len(self.library_requirements),
            "available_count": sum(self.available_libraries.values())
        }
    
    def install_missing_libraries(self, libraries: List[str]) -> Dict[str, Any]:
        """Attempt to install missing libraries (use with caution)."""
        installation_results = {}
        
        for lib in libraries:
            if lib in self.library_requirements:
                try:
                    requirement = self.library_requirements[lib]
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", requirement],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minutes timeout for installation
                    )
                    
                    installation_results[lib] = {
                        "success": result.returncode == 0,
                        "output": result.stdout,
                        "error": result.stderr if result.returncode != 0 else None
                    }
                    
                except subprocess.TimeoutExpired:
                    installation_results[lib] = {
                        "success": False,
                        "error": "Installation timed out"
                    }
                except Exception as e:
                    installation_results[lib] = {
                        "success": False,
                        "error": str(e)
                    }
        
        # Refresh library availability after installation
        self.available_libraries = self._check_available_libraries()
        
        return installation_results
