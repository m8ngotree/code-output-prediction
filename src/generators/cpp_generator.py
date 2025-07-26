"""
Simple C++ code generator using OpenAI API.
"""

import re
import time
from typing import Dict, Any, Optional

import openai
import yaml
from ..core.seed_manager import SeedManager


class CppGenerator:
    """Generates C++ code using OpenAI API."""
    
    def __init__(self, api_key: str, config_path: str = "config.yaml"):
        """Initialize C++ generator."""
        self.client = openai.OpenAI(api_key=api_key)
        self.seed_manager = SeedManager()
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    def generate_code(self) -> Dict[str, Any]:
        """Generate a single C++ code sample."""
        # Get random seeds
        application = self.seed_manager.get_random_application()
        concept = self.seed_manager.get_random_concept()
        
        # Create C++-specific prompt
        prompt = f"Create a C++ program for {application} using {concept}. " \
                f"Include {self.config['generation']['min_functions']}-{self.config['generation']['max_functions']} functions, " \
                f"use modern C++17 features, include proper headers, and have a main function. Use STL when appropriate."
        
        try:
            # Call OpenAI API
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.config['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are a C++ expert. Write modern C++17 code with proper headers, STL usage, and good practices."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config['openai']['max_tokens'],
                temperature=self.config['openai']['temperature']
            )
            generation_time = time.time() - start_time
            
            # Extract and clean code
            raw_code = response.choices[0].message.content
            code = self._extract_code(raw_code)
            
            # Validate and enhance code
            code = self._enhance_cpp_code(code)
            
            return {
                "success": True,
                "code": code,
                "language": "cpp",
                "application": application,
                "concept": concept,
                "generation_time": generation_time,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "language": "cpp",
                "application": application,
                "concept": concept
            }
    
    def _extract_code(self, text: str) -> str:
        """Extract C++ code from response text."""
        # Look for C++ code blocks
        patterns = [
            r'```cpp\n(.*?)\n```',
            r'```c\+\+\n(.*?)\n```',
            r'```\n(.*?)\n```'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no code block, assume entire text is code
        return text.strip()
    
    def _enhance_cpp_code(self, code: str) -> str:
        """Enhance C++ code with necessary additions."""
        # Add basic includes if missing
        includes_needed = []
        
        if '#include' not in code:
            # Add common includes based on code content
            if 'cout' in code or 'cin' in code or 'endl' in code:
                includes_needed.append('#include <iostream>')
            if 'vector' in code:
                includes_needed.append('#include <vector>')
            if 'string' in code:
                includes_needed.append('#include <string>')
            if 'map' in code or 'unordered_map' in code:
                includes_needed.append('#include <map>')
            if 'algorithm' in code or 'sort' in code:
                includes_needed.append('#include <algorithm>')
            
            # Add default includes if none detected
            if not includes_needed:
                includes_needed.append('#include <iostream>')
        
        # Add using namespace std if not present and needed
        needs_std = 'cout' in code or 'cin' in code or 'endl' in code or 'string' in code
        if needs_std and 'using namespace std' not in code and 'std::' not in code:
            includes_needed.append('using namespace std;')
        
        # Prepend includes
        if includes_needed:
            code = '\n'.join(includes_needed) + '\n\n' + code
        
        # Ensure main function exists
        if 'int main(' not in code:
            # If there are other functions, add a simple main
            if '(' in code and ')' in code:
                code += '\n\nint main() {\n    cout << "Program executed successfully" << endl;\n    return 0;\n}'
        
        return code