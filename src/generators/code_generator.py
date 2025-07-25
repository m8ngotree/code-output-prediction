"""
Code Generator for synthetic Python programs using OpenAI's API.

This module provides functionality to generate complex Python programs
based on application and concept seeds using GPT-4.
"""

import ast
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import openai
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..seeds.seed_manager import SeedManager
from ..utils.config import get_config, ConfigurationError


class CodeGenerationError(Exception):
    """Custom exception for code generation errors."""
    pass


class CodeGenerator:
    """
    Generates Python code using OpenAI's API based on application and concept seeds.
    
    Features:
    - Template-based prompt generation
    - Code post-processing and validation
    - Metadata tracking and persistence
    - Configurable retry logic
    - Error handling and logging
    """
    
    def __init__(self, config_path: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the CodeGenerator.
        
        Args:
            config_path: Path to configuration YAML file (deprecated - uses central config)
            api_key: OpenAI API key (if not provided, uses environment variable)
        """
        # Use centralized configuration system
        self.global_config = get_config()
        
        # For backward compatibility, load old config if provided
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._get_config_from_global()
        
        self.seed_manager = SeedManager()
        
        # Initialize OpenAI client
        api_key = api_key or self.global_config.get('api.openai.api_key') or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided either as parameter, configuration, or OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Ensure output directory exists
        output_dir = self.config.get("output", {}).get("save_directory") or self.global_config.get('generation.output.base_directory')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default to configs/code_generator.yaml relative to project root
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            config_path = project_root / "configs" / "code_generator.yaml"
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_config_from_global(self) -> Dict[str, Any]:
        """Convert global configuration to code generator format."""
        return {
            "openai": {
                "model": self.global_config.get('api.openai.model', 'gpt-3.5-turbo'),
                "max_tokens": self.global_config.get('api.openai.max_tokens', 1500),
                "temperature": self.global_config.get('api.openai.temperature', 0.5),
                "top_p": self.global_config.get('api.openai.top_p', 1.0),
                "frequency_penalty": self.global_config.get('api.openai.frequency_penalty', 0.2),
                "presence_penalty": self.global_config.get('api.openai.presence_penalty', 0.1),
                "timeout": self.global_config.get('api.openai.timeout_seconds', 30)
            },
            "retry": {
                "max_attempts": self.global_config.get('api.openai.rate_limit.max_retries', 3),
                "wait_multiplier": self.global_config.get('api.openai.rate_limit.backoff_multiplier', 2),
                "wait_min": 1,
                "wait_max": self.global_config.get('api.openai.rate_limit.max_backoff_seconds', 60)
            },
            "generation": {
                "min_functions": self.global_config.get('generation.complexity.min_functions', 2),
                "max_functions": self.global_config.get('generation.complexity.max_functions', 5),
                "include_main": self.global_config.get('generation.complexity.include_main', True),
                "include_docstrings": self.global_config.get('generation.complexity.include_docstrings', True),
                "include_type_hints": self.global_config.get('generation.complexity.include_type_hints', True)
            },
            "output": {
                "save_directory": self.global_config.get('generation.output.base_directory', 'data/generated'),
                "include_metadata": self.global_config.get('generation.output.save_metadata', True),
                "filename_format": self.global_config.get('generation.output.filename_format', 'generated_{timestamp}_{hash}.py'),
                "metadata_format": "json"
            },
            "validation": {
                "syntax_check": True,
                "import_check": True,
                "basic_execution_test": False
            },
            "prompts": {
                "system_message": """You are an expert Python programmer. Generate complete, executable Python programs that demonstrate specific programming concepts and applications.

Requirements:
- Write clean, well-documented code with docstrings
- Include multiple functions that work together
- Handle edge cases and errors appropriately
- Use the specified programming concepts effectively
- Create realistic, practical applications
- Ensure the code takes input and produces clear output
- Include a main function or execution block
- IMPORTANT: Avoid infinite loops, excessive recursion, or long-running operations
- Keep execution time under 5 seconds
- Use reasonable input sizes and iteration limits""",
                "user_template": """Create a Python program for {application} that demonstrates {concept}.

Requirements:
- Implement {min_functions}-{max_functions} functions
- Focus on {concept} as the core programming concept
- Make it a practical {application} application
- Include proper error handling and edge cases
- Add docstrings and comments for clarity
- Ensure the program is executable and produces meaningful output
- Include examples of input/output in comments
- CRITICAL: Avoid infinite loops, excessive recursion, or time-consuming operations
- Use small datasets and reasonable iteration limits (max 1000 iterations)
- Complete execution in under 5 seconds

The program should be complete, fast, and ready to run."""
            },
            "common_imports": [
                "import sys",
                "import os",
                "from typing import List, Dict, Optional, Union, Any",
                "import json",
                "import re"
            ]
        }
    
    def _create_prompt(self, application: str, concept: str) -> Tuple[str, str]:
        """
        Create system and user prompts for code generation.
        
        Args:
            application: Application seed (e.g., "web scraping")
            concept: Concept seed (e.g., "recursion")
            
        Returns:
            Tuple of (system_message, user_prompt)
        """
        system_message = self.config["prompts"]["system_message"]
        
        user_template = self.config["prompts"]["user_template"]
        user_prompt = user_template.format(
            application=application,
            concept=concept,
            min_functions=self.config["generation"]["min_functions"],
            max_functions=self.config["generation"]["max_functions"]
        )
        
        return system_message, user_prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError))
    )
    def _call_openai_api(self, system_message: str, user_prompt: str) -> str:
        """
        Call OpenAI API with retry logic.
        
        Args:
            system_message: System prompt
            user_prompt: User prompt
            
        Returns:
            Generated code as string
            
        Raises:
            CodeGenerationError: If API call fails after retries
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config["openai"]["model"],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config["openai"]["max_tokens"],
                temperature=self.config["openai"]["temperature"],
                top_p=self.config["openai"]["top_p"],
                frequency_penalty=self.config["openai"]["frequency_penalty"],
                presence_penalty=self.config["openai"]["presence_penalty"],
                timeout=self.config["openai"]["timeout"]
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.OpenAIError as e:
            raise CodeGenerationError(f"OpenAI API error: {e}")
        except Exception as e:
            raise CodeGenerationError(f"Unexpected error during API call: {e}")
    
    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract Python code from API response.
        
        Args:
            response: Raw API response
            
        Returns:
            Extracted Python code
        """
        # Look for code blocks
        code_block_pattern = r"```python\n(.*?)\n```"
        match = re.search(code_block_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # If no code block found, look for code after "python" keyword
        python_pattern = r"python\n(.*?)(?=\n\n|\Z)"
        match = re.search(python_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # If still no match, return the entire response (might be just code)
        return response.strip()
    
    def _validate_syntax(self, code: str) -> bool:
        """
        Validate Python syntax.
        
        Args:
            code: Python code to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _add_missing_imports(self, code: str) -> str:
        """
        Add common imports if they're used but not imported.
        
        Args:
            code: Python code
            
        Returns:
            Code with added imports
        """
        lines = code.split('\n')
        imports_section = []
        code_section = []
        
        # Separate existing imports from code
        in_imports = True
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')) or stripped == '':
                if in_imports:
                    imports_section.append(line)
                else:
                    code_section.append(line)
            else:
                in_imports = False
                code_section.append(line)
        
        # Check what imports might be needed
        code_text = '\n'.join(code_section)
        existing_imports = '\n'.join(imports_section)
        
        needed_imports = []
        
        # Check for common patterns and add imports if needed
        import_patterns = {
            r'\bsys\b': "import sys",
            r'\bos\b': "import os", 
            r'\bjson\b': "import json",
            r'\bre\b': "import re",
            r'\bList\b|\bDict\b|\bOptional\b|\bUnion\b|\bAny\b': "from typing import List, Dict, Optional, Union, Any"
        }
        
        for pattern, import_stmt in import_patterns.items():
            if re.search(pattern, code_text) and import_stmt not in existing_imports:
                needed_imports.append(import_stmt)
        
        # Combine imports and code
        all_imports = imports_section + needed_imports
        if all_imports and code_section:
            all_imports.append('')  # Add blank line after imports
        
        return '\n'.join(all_imports + code_section)
    
    def _ensure_executable(self, code: str) -> str:
        """
        Ensure code has proper execution structure.
        
        Args:
            code: Python code
            
        Returns:
            Code with proper main execution block
        """
        # Check if code already has main execution
        if 'if __name__ == "__main__"' in code or 'def main(' in code:
            return code
        
        # Add basic main execution if missing
        if not re.search(r'\n\s*\w+\(', code):  # No function calls at module level
            code += '\n\n# Example usage\nif __name__ == "__main__":\n    print("Program executed successfully!")'
        
        return code
    
    def _post_process_code(self, code: str) -> str:
        """
        Post-process generated code to ensure it's executable.
        
        Args:
            code: Raw generated code
            
        Returns:
            Post-processed executable code
            
        Raises:
            CodeGenerationError: If code cannot be made executable
        """
        # Extract code from response
        code = self._extract_code_from_response(code)
        
        # Add missing imports
        code = self._add_missing_imports(code)
        
        # Ensure executable structure
        if self.config["generation"]["include_main"]:
            code = self._ensure_executable(code)
        
        # Validate syntax
        if self.config["validation"]["syntax_check"] and not self._validate_syntax(code):
            raise CodeGenerationError("Generated code has syntax errors")
        
        return code
    
    def _create_metadata(self, application: str, concept: str, code: str, 
                        generation_time: float) -> Dict[str, Any]:
        """
        Create metadata for generated code.
        
        Args:
            application: Application seed used
            concept: Concept seed used
            code: Generated code
            generation_time: Time taken to generate code
            
        Returns:
            Metadata dictionary
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "seeds": {
                "application": application,
                "concept": concept
            },
            "generation_time_seconds": round(generation_time, 2),
            "code_stats": {
                "lines": len(code.split('\n')),
                "characters": len(code),
                "functions": len(re.findall(r'^def\s+\w+', code, re.MULTILINE))
            },
            "config": {
                "model": self.config["openai"]["model"],
                "temperature": self.config["openai"]["temperature"],
                "max_tokens": self.config["openai"]["max_tokens"]
            },
            "hash": hashlib.md5(code.encode()).hexdigest()[:8]
        }
    
    def _save_generated_code(self, code: str, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """
        Save generated code and metadata to files.
        
        Args:
            code: Generated Python code
            metadata: Code metadata
            
        Returns:
            Tuple of (code_filepath, metadata_filepath)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_hash = metadata["hash"]
        
        # Create filenames
        filename_template = self.config["output"]["filename_format"]
        base_filename = filename_template.format(timestamp=timestamp, hash=code_hash)
        
        code_filepath = self.output_dir / base_filename
        metadata_filepath = self.output_dir / f"{base_filename.replace('.py', '_metadata.json')}"
        
        # Save code
        with open(code_filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Save metadata
        if self.config["output"]["include_metadata"]:
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return str(code_filepath), str(metadata_filepath)
    
    def generate_code(self, application: Optional[str] = None, 
                     concept: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate Python code based on application and concept seeds.
        
        Args:
            application: Specific application seed (random if None)
            concept: Specific concept seed (random if None)
            
        Returns:
            Generation result with code, metadata, and file paths
            
        Raises:
            CodeGenerationError: If generation fails
        """
        start_time = time.time()
        
        try:
            # Get seeds
            if application is None:
                application = self.seed_manager.sample_seeds("applications", 1)[0]
            if concept is None:
                concept = self.seed_manager.sample_seeds("concepts", 1)[0]
            
            # Create prompts
            system_message, user_prompt = self._create_prompt(application, concept)
            
            # Generate code via API
            raw_response = self._call_openai_api(system_message, user_prompt)
            
            # Post-process code
            processed_code = self._post_process_code(raw_response)
            
            # Create metadata
            generation_time = time.time() - start_time
            metadata = self._create_metadata(application, concept, processed_code, generation_time)
            
            # Save files
            code_filepath, metadata_filepath = self._save_generated_code(processed_code, metadata)
            
            return {
                "success": True,
                "code": processed_code,
                "metadata": metadata,
                "files": {
                    "code": code_filepath,
                    "metadata": metadata_filepath
                },
                "seeds_used": {
                    "application": application,
                    "concept": concept
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "generation_time": time.time() - start_time
            }
    
    def generate_batch(self, count: int, applications: Optional[List[str]] = None,
                      concepts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate multiple code examples.
        
        Args:
            count: Number of code examples to generate
            applications: Specific applications to use (random if None)
            concepts: Specific concepts to use (random if None)
            
        Returns:
            List of generation results
        """
        results = []
        
        # Prepare seed lists
        if applications is None:
            applications = self.seed_manager.sample_seeds("applications", count, replace=True)
        if concepts is None:
            concepts = self.seed_manager.sample_seeds("concepts", count, replace=True)
        
        # Ensure we have enough seeds
        while len(applications) < count:
            applications.extend(self.seed_manager.sample_seeds("applications", count - len(applications), replace=True))
        while len(concepts) < count:
            concepts.extend(self.seed_manager.sample_seeds("concepts", count - len(concepts), replace=True))
        
        # Generate code examples
        for i in range(count):
            print(f"Generating code example {i+1}/{count}...")
            result = self.generate_code(applications[i], concepts[i])
            results.append(result)
            
            # Brief pause between requests to avoid rate limiting
            if i < count - 1:
                time.sleep(1)
        
        return results
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about generated code files.
        
        Returns:
            Statistics dictionary
        """
        code_files = list(self.output_dir.glob("*.py"))
        metadata_files = list(self.output_dir.glob("*_metadata.json"))
        
        total_lines = 0
        total_functions = 0
        applications_used = set()
        concepts_used = set()
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    total_lines += metadata.get("code_stats", {}).get("lines", 0)
                    total_functions += metadata.get("code_stats", {}).get("functions", 0)
                    
                    seeds = metadata.get("seeds", {})
                    if "application" in seeds:
                        applications_used.add(seeds["application"])
                    if "concept" in seeds:
                        concepts_used.add(seeds["concept"])
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        return {
            "total_files": len(code_files),
            "total_lines": total_lines,
            "total_functions": total_functions,
            "unique_applications": len(applications_used),
            "unique_concepts": len(concepts_used),
            "applications_used": sorted(list(applications_used)),
            "concepts_used": sorted(list(concepts_used))
        }