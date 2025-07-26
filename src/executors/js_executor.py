"""
Simple JavaScript executor using Node.js.
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional


class JSExecutor:
    """Executes JavaScript code using Node.js."""
    
    def __init__(self, timeout_seconds: int = 10, max_memory_mb: int = 128):
        """Initialize JavaScript executor."""
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        
        # Check if Node.js is available
        try:
            subprocess.run(['node', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Node.js is not installed or not in PATH")
    
    def execute(self, code: str, input_data: Optional[str] = None) -> Dict[str, Any]:
        """Execute JavaScript code and return results."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            start_time = time.time()
            
            # Execute with Node.js
            cmd = ['node', temp_file]
            
            # Add memory limit if supported
            if self.max_memory_mb:
                cmd = ['node', f'--max-old-space-size={self.max_memory_mb}', temp_file]
            
            result = subprocess.run(
                cmd,
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
                "language": "javascript"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": None,
                "error": f"Execution timed out after {self.timeout_seconds} seconds",
                "execution_time": self.timeout_seconds,
                "return_code": -1,
                "language": "javascript"
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "execution_time": 0,
                "return_code": -1,
                "language": "javascript"
            }
        finally:
            # Clean up temporary file
            Path(temp_file).unlink(missing_ok=True)