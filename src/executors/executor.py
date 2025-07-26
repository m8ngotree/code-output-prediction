"""
Simple Python code executor with safety measures.
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional


class CodeExecutor:
    """Executes Python code safely in a subprocess."""
    
    def __init__(self, timeout_seconds: int = 10, max_memory_mb: int = 128):
        """Initialize executor with safety limits."""
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
    
    def execute(self, code: str, input_data: Optional[str] = None) -> Dict[str, Any]:
        """Execute Python code and return results."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            start_time = time.time()
            
            # Execute with timeout
            result = subprocess.run(
                ['python', temp_file],
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
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": None,
                "error": f"Execution timed out after {self.timeout_seconds} seconds",
                "execution_time": self.timeout_seconds,
                "return_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "execution_time": 0,
                "return_code": -1
            }
        finally:
            # Clean up temporary file
            Path(temp_file).unlink(missing_ok=True)