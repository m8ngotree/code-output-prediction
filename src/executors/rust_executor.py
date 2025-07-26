"""
Simple Rust executor using rustc.
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional


class RustExecutor:
    """Executes Rust code using rustc compilation."""
    
    def __init__(self, timeout_seconds: int = 10, max_memory_mb: int = 128):
        """Initialize Rust executor."""
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        
        # Check if rustc is available
        try:
            subprocess.run(['rustc', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Rust compiler (rustc) is not installed or not in PATH")
    
    def execute(self, code: str, input_data: Optional[str] = None) -> Dict[str, Any]:
        """Execute Rust code and return results."""
        # Create temporary directory for compilation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_file = temp_path / "main.rs"
            binary_file = temp_path / "main"
            
            # Write source code
            source_file.write_text(code)
            
            try:
                # Compile the code
                compile_start = time.time()
                compile_result = subprocess.run(
                    ['rustc', str(source_file), '-o', str(binary_file)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )
                
                if compile_result.returncode != 0:
                    return {
                        "success": False,
                        "output": None,
                        "error": f"Compilation failed: {compile_result.stderr}",
                        "execution_time": 0,
                        "return_code": compile_result.returncode,
                        "language": "rust",
                        "compilation_error": True
                    }
                
                # Execute the compiled binary
                start_time = time.time()
                result = subprocess.run(
                    [str(binary_file)],
                    input=input_data,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds
                )
                
                execution_time = time.time() - start_time
                
                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr if result.returncode != 0 else None,
                    "execution_time": execution_time,
                    "return_code": result.returncode,
                    "language": "rust",
                    "compilation_error": False
                }
                
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "output": None,
                    "error": f"Execution timed out after {self.timeout_seconds} seconds",
                    "execution_time": self.timeout_seconds,
                    "return_code": -1,
                    "language": "rust"
                }
            except Exception as e:
                return {
                    "success": False,
                    "output": None,
                    "error": str(e),
                    "execution_time": 0,
                    "return_code": -1,
                    "language": "rust"
                }