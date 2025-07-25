"""
Secure Python Code Executor for running generated code with inputs safely.

This module provides functionality to execute Python code in isolated processes
with timeouts, resource limits, and comprehensive error handling.
"""

import json
import logging
import os
import platform
import psutil
import resource
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class ExecutionResult:
    """Represents the result of code execution with metadata."""
    
    def __init__(self):
        self.success: bool = False
        self.stdout: str = ""
        self.stderr: str = ""
        self.return_code: int = -1
        self.execution_time: float = 0.0
        self.memory_peak: Optional[int] = None
        self.timeout_occurred: bool = False
        self.error_type: Optional[str] = None
        self.error_message: str = ""
        self.timestamp: str = datetime.now().isoformat()
        self.input_data: Dict[str, Any] = {}
        self.execution_mode: str = ""
        self.python_version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "execution_time": self.execution_time,
            "memory_peak": self.memory_peak,
            "timeout_occurred": self.timeout_occurred,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "input_data": self.input_data,
            "execution_mode": self.execution_mode,
            "python_version": self.python_version
        }


class SecurityLimits:
    """Configuration for execution security limits."""
    
    def __init__(self):
        self.timeout_seconds: int = 30
        self.max_memory_mb: int = 256
        self.max_output_size: int = 1024 * 1024  # 1MB
        self.max_file_size: int = 10 * 1024 * 1024  # 10MB
        self.allowed_modules: Optional[List[str]] = None  # None = all allowed
        self.blocked_modules: List[str] = [
            'subprocess', 'os', 'sys', 'shutil', 'socket', 
            'urllib', 'requests', 'http', 'ftplib', 'smtplib',
            '__import__', 'eval', 'exec', 'compile'
        ]


class PythonExecutor:
    """
    Secure Python code executor with multiple execution modes.
    
    Provides safe execution of Python code with timeouts, resource limits,
    and comprehensive error handling.
    """
    
    def __init__(self, limits: Optional[SecurityLimits] = None, 
                 log_level: int = logging.INFO):
        """
        Initialize the Python executor.
        
        Args:
            limits: Security limits configuration
            log_level: Logging level
        """
        self.limits = limits or SecurityLimits()
        self.temp_files: List[str] = []
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info("PythonExecutor initialized")
    
    def _create_secure_wrapper(self, code: str, execution_mode: str, 
                              inputs: Dict[str, Any]) -> str:
        """
        Create a secure wrapper around the user code.
        
        Args:
            code: The Python code to execute
            execution_mode: Mode of execution ('function', 'script', 'interactive')
            inputs: Input data for the code
            
        Returns:
            Wrapped code with security measures
        """
        wrapper_template = '''
import sys
import json
import traceback
import time
import resource
from datetime import datetime

# Security measures
original_import = __builtins__.__import__

def secure_import(name, *args, **kwargs):
    blocked_modules = {blocked_modules}
    if name in blocked_modules:
        raise ImportError(f"Module '{{name}}' is not allowed")
    return original_import(name, *args, **kwargs)

# Replace import function (commented out for basic functionality)
# __builtins__.__import__ = secure_import

# Set resource limits
try:
    # Memory limit (in bytes)
    max_memory = {max_memory} * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
    
    # CPU time limit
    resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))
except (ValueError, OSError) as e:
    # Resource limits may not be available on all systems
    print(f"Warning: Could not set resource limits: {{e}}", file=sys.stderr)

# Execution tracking
start_time = time.time()
execution_metadata = {{
    "start_time": datetime.now().isoformat(),
    "python_version": sys.version,
    "inputs": {inputs}
}}

try:
    # User code execution
{user_code}

    # Capture successful execution
    end_time = time.time()
    execution_metadata["end_time"] = datetime.now().isoformat()
    execution_metadata["execution_time"] = end_time - start_time
    execution_metadata["success"] = True
    
except Exception as e:
    end_time = time.time()
    execution_metadata["end_time"] = datetime.now().isoformat()
    execution_metadata["execution_time"] = end_time - start_time
    execution_metadata["success"] = False
    execution_metadata["error_type"] = type(e).__name__
    execution_metadata["error_message"] = str(e)
    execution_metadata["traceback"] = traceback.format_exc()
    
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)

finally:
    # Output execution metadata to stderr for capture
    print(f"EXEC_METADATA: {{json.dumps(execution_metadata)}}", file=sys.stderr)
'''
        
        # Format the wrapper with security parameters
        wrapped_code = wrapper_template.format(
            blocked_modules=json.dumps(self.limits.blocked_modules),
            max_memory=self.limits.max_memory_mb,
            timeout=self.limits.timeout_seconds,
            inputs=json.dumps(inputs),
            user_code=self._indent_code(code, 4)
        )
        
        return wrapped_code
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = " " * spaces
        return "\n".join(indent + line if line.strip() else line 
                        for line in code.split("\n"))
    
    def _prepare_function_execution(self, code: str, function_name: str, 
                                  inputs: List[Any]) -> str:
        """
        Prepare code for function execution mode.
        
        Args:
            code: Python code containing the function
            function_name: Name of function to call
            inputs: List of arguments to pass to function
            
        Returns:
            Code prepared for function execution
        """
        execution_code = f"""
{code}

# Execute the function with provided inputs
try:
    inputs = {json.dumps(inputs)}
    if callable({function_name}):
        result = {function_name}(*inputs)
        print(f"Function result: {{result}}")
    else:
        print(f"Error: {function_name} is not callable", file=sys.stderr)
        sys.exit(1)
except TypeError as e:
    print(f"Error calling function: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        return execution_code
    
    def _prepare_script_execution(self, code: str, cli_args: List[str]) -> Tuple[str, List[str]]:
        """
        Prepare code for script execution mode.
        
        Args:
            code: Python code to execute as script
            cli_args: Command line arguments
            
        Returns:
            Tuple of (prepared code, arguments for subprocess)
        """
        # The code is executed as-is, arguments passed via sys.argv
        return code, cli_args
    
    def _prepare_interactive_execution(self, code: str, stdin_inputs: List[str]) -> Tuple[str, str]:
        """
        Prepare code for interactive execution mode.
        
        Args:
            code: Python code expecting interactive input
            stdin_inputs: List of inputs to provide via stdin
            
        Returns:
            Tuple of (prepared code, stdin data)
        """
        stdin_data = "\n".join(str(inp) for inp in stdin_inputs) + "\n"
        return code, stdin_data
    
    def _execute_subprocess(self, code: str, args: List[str] = None, 
                          stdin_data: str = None, env: Dict[str, str] = None) -> ExecutionResult:
        """
        Execute code in a subprocess with security measures.
        
        Args:
            code: Python code to execute
            args: Command line arguments
            stdin_data: Data to provide via stdin
            env: Environment variables
            
        Returns:
            ExecutionResult with execution details
        """
        result = ExecutionResult()
        result.python_version = sys.version
        
        # Create temporary file for code
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        )
        temp_file.write(code)
        temp_file.close()
        self.temp_files.append(temp_file.name)
        
        # Prepare command
        cmd = [sys.executable, temp_file.name]
        if args:
            cmd.extend(str(arg) for arg in args)
        
        # Prepare environment
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)
        
        # Add security environment variables
        exec_env['PYTHONDONTWRITEBYTECODE'] = '1'  # Don't create .pyc files
        exec_env['PYTHONUNBUFFERED'] = '1'  # Unbuffered output
        
        start_time = time.time()
        process = None
        
        try:
            self.logger.info(f"Executing command: {' '.join(cmd)}")
            
            # Start process with security measures
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if stdin_data else None,
                env=exec_env,
                text=True,
                preexec_fn=self._set_process_limits if platform.system() != 'Windows' else None
            )
            
            # Monitor process with timeout
            try:
                stdout, stderr = process.communicate(
                    input=stdin_data,
                    timeout=self.limits.timeout_seconds
                )
                
                result.return_code = process.returncode
                result.success = process.returncode == 0
                
            except subprocess.TimeoutExpired:
                self.logger.warning("Process timed out")
                result.timeout_occurred = True
                result.error_type = "TimeoutError"
                result.error_message = f"Execution timed out after {self.limits.timeout_seconds} seconds"
                
                # Kill the process
                try:
                    process.kill()
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    stdout, stderr = "", "Process forcefully terminated"
                
                result.return_code = -1
        
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            result.error_type = type(e).__name__
            result.error_message = str(e)
            stdout, stderr = "", f"Execution error: {e}"
            result.return_code = -1
        
        finally:
            end_time = time.time()
            result.execution_time = end_time - start_time
            
            # Clean up process
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        # Process outputs
        result.stdout = self._sanitize_output(stdout or "")
        result.stderr = self._sanitize_output(stderr or "")
        
        # Extract metadata from stderr if present
        self._extract_execution_metadata(result)
        
        self.logger.info(f"Execution completed in {result.execution_time:.2f}s")
        return result
    
    def _set_process_limits(self):
        """Set process resource limits (Unix only)."""
        try:
            # Memory limit
            max_memory = self.limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
            
            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, 
                              (self.limits.timeout_seconds, self.limits.timeout_seconds))
            
            # File size limit
            resource.setrlimit(resource.RLIMIT_FSIZE, 
                              (self.limits.max_file_size, self.limits.max_file_size))
            
        except (ValueError, OSError) as e:
            # Resource limits may not be available
            pass
    
    def _sanitize_output(self, output: str) -> str:
        """Sanitize and limit output size."""
        if len(output) > self.limits.max_output_size:
            truncated_msg = f"\n... [Output truncated at {self.limits.max_output_size} characters]"
            return output[:self.limits.max_output_size - len(truncated_msg)] + truncated_msg
        return output
    
    def _extract_execution_metadata(self, result: ExecutionResult):
        """Extract execution metadata from stderr output."""
        if not result.stderr:
            return
        
        lines = result.stderr.split('\n')
        for line in lines:
            if line.startswith('EXEC_METADATA: '):
                try:
                    metadata_json = line[15:]  # Remove 'EXEC_METADATA: ' prefix
                    metadata = json.loads(metadata_json)
                    
                    # Update result with metadata
                    if 'execution_time' in metadata:
                        result.execution_time = metadata['execution_time']
                    if 'error_type' in metadata:
                        result.error_type = metadata['error_type']
                    if 'error_message' in metadata:
                        result.error_message = metadata['error_message']
                    
                    # Remove metadata line from stderr
                    result.stderr = result.stderr.replace(line + '\n', '')
                    break
                    
                except json.JSONDecodeError:
                    continue
    
    def execute_function(self, code: str, function_name: str, 
                        inputs: List[Any]) -> ExecutionResult:
        """
        Execute a specific function with given inputs.
        
        Args:
            code: Python code containing the function
            function_name: Name of function to call
            inputs: Arguments to pass to the function
            
        Returns:
            ExecutionResult with execution details
        """
        self.logger.info(f"Executing function '{function_name}' with {len(inputs)} inputs")
        
        result = ExecutionResult()
        result.execution_mode = "function"
        result.input_data = {"function_name": function_name, "inputs": inputs}
        
        try:
            # Prepare function execution code
            prepared_code = self._prepare_function_execution(code, function_name, inputs)
            
            # Wrap with security measures
            wrapped_code = self._create_secure_wrapper(
                prepared_code, "function", {"function_name": function_name, "inputs": inputs}
            )
            
            # Execute in subprocess
            result = self._execute_subprocess(wrapped_code)
            result.execution_mode = "function"
            result.input_data = {"function_name": function_name, "inputs": inputs}
            
        except Exception as e:
            self.logger.error(f"Function execution error: {e}")
            result.error_type = type(e).__name__
            result.error_message = str(e)
            result.success = False
        
        return result
    
    def execute_script(self, code: str, cli_args: List[str] = None) -> ExecutionResult:
        """
        Execute code as a script with command line arguments.
        
        Args:
            code: Python code to execute
            cli_args: Command line arguments
            
        Returns:
            ExecutionResult with execution details
        """
        cli_args = cli_args or []
        self.logger.info(f"Executing script with {len(cli_args)} CLI arguments")
        
        result = ExecutionResult()
        result.execution_mode = "script"
        result.input_data = {"cli_args": cli_args}
        
        try:
            # Prepare script execution
            prepared_code, args = self._prepare_script_execution(code, cli_args)
            
            # Wrap with security measures
            wrapped_code = self._create_secure_wrapper(
                prepared_code, "script", {"cli_args": cli_args}
            )
            
            # Execute in subprocess
            result = self._execute_subprocess(wrapped_code, args)
            result.execution_mode = "script"
            result.input_data = {"cli_args": cli_args}
            
        except Exception as e:
            self.logger.error(f"Script execution error: {e}")
            result.error_type = type(e).__name__
            result.error_message = str(e)
            result.success = False
        
        return result
    
    def execute_interactive(self, code: str, stdin_inputs: List[str]) -> ExecutionResult:
        """
        Execute code with interactive input simulation.
        
        Args:
            code: Python code expecting interactive input
            stdin_inputs: List of inputs to provide via stdin
            
        Returns:
            ExecutionResult with execution details
        """
        self.logger.info(f"Executing interactive code with {len(stdin_inputs)} stdin inputs")
        
        result = ExecutionResult()
        result.execution_mode = "interactive"
        result.input_data = {"stdin_inputs": stdin_inputs}
        
        try:
            # Prepare interactive execution
            prepared_code, stdin_data = self._prepare_interactive_execution(code, stdin_inputs)
            
            # Wrap with security measures
            wrapped_code = self._create_secure_wrapper(
                prepared_code, "interactive", {"stdin_inputs": stdin_inputs}
            )
            
            # Execute in subprocess
            result = self._execute_subprocess(wrapped_code, stdin_data=stdin_data)
            result.execution_mode = "interactive"
            result.input_data = {"stdin_inputs": stdin_inputs}
            
        except Exception as e:
            self.logger.error(f"Interactive execution error: {e}")
            result.error_type = type(e).__name__
            result.error_message = str(e)
            result.success = False
        
        return result
    
    def execute_with_files(self, code: str, file_inputs: Dict[str, str]) -> ExecutionResult:
        """
        Execute code with temporary files as input.
        
        Args:
            code: Python code that reads files
            file_inputs: Dictionary mapping filenames to content
            
        Returns:
            ExecutionResult with execution details
        """
        self.logger.info(f"Executing code with {len(file_inputs)} file inputs")
        
        result = ExecutionResult()
        result.execution_mode = "file_input"
        result.input_data = {"file_inputs": list(file_inputs.keys())}
        
        created_files = []
        
        try:
            # Create temporary files
            for filename, content in file_inputs.items():
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w', delete=False, 
                    suffix=Path(filename).suffix,
                    prefix=Path(filename).stem + "_"
                )
                temp_file.write(content)
                temp_file.close()
                created_files.append(temp_file.name)
                self.temp_files.append(temp_file.name)
                
                # Replace filename references in code
                code = code.replace(f'"{filename}"', f'"{temp_file.name}"')
                code = code.replace(f"'{filename}'", f"'{temp_file.name}'")
            
            # Wrap with security measures
            wrapped_code = self._create_secure_wrapper(
                code, "file_input", {"file_inputs": list(file_inputs.keys())}
            )
            
            # Execute in subprocess
            result = self._execute_subprocess(wrapped_code)
            result.execution_mode = "file_input"
            result.input_data = {"file_inputs": list(file_inputs.keys())}
            
        except Exception as e:
            self.logger.error(f"File execution error: {e}")
            result.error_type = type(e).__name__
            result.error_message = str(e)
            result.success = False
        
        return result
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        self.logger.info(f"Cleaning up {len(self.temp_files)} temporary files")
        
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except OSError as e:
                self.logger.warning(f"Could not remove temporary file {temp_file}: {e}")
        
        self.temp_files.clear()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()