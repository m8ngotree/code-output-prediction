"""
Simple JavaScript code generator using OpenAI API.
"""

import ast
import re
import time
from typing import Dict, Any, Optional

import openai
import yaml
from ..core.seed_manager import SeedManager


class JSGenerator:
    """Generates JavaScript code using OpenAI API."""
    
    def __init__(self, api_key: str, config_path: str = "config.yaml"):
        """Initialize JavaScript generator."""
        self.client = openai.OpenAI(api_key=api_key)
        self.seed_manager = SeedManager()
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    def generate_code(self) -> Dict[str, Any]:
        """Generate a single JavaScript code sample."""
        # Get random seeds
        application = self.seed_manager.get_random_application()
        concept = self.seed_manager.get_random_concept()
        
        # Create JavaScript-specific prompt
        prompt = f"Create a JavaScript/Node.js program for {application} using {concept}. " \
                f"Include {self.config['generation']['min_functions']}-{self.config['generation']['max_functions']} functions, " \
                f"proper error handling, and use modern JavaScript (ES6+). Include a main function."
        
        try:
            # Call OpenAI API
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.config['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are a JavaScript/Node.js expert. Write clean, modern JavaScript with proper error handling."},
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
            code = self._enhance_js_code(code)
            
            return {
                "success": True,
                "code": code,
                "language": "javascript",
                "application": application,
                "concept": concept,
                "generation_time": generation_time,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "language": "javascript",
                "application": application,
                "concept": concept
            }
    
    def _extract_code(self, text: str) -> str:
        """Extract JavaScript code from response text."""
        # Look for JavaScript code blocks
        patterns = [
            r'```javascript\n(.*?)\n```',
            r'```js\n(.*?)\n```',
            r'```\n(.*?)\n```'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no code block, assume entire text is code
        return text.strip()
    
    def _enhance_js_code(self, code: str) -> str:
        """Enhance JavaScript code with necessary additions."""
        # Add main function call if missing
        if 'main()' not in code and 'function main' in code:
            code += '\n\n// Run main function\nmain();'
        
        # Add basic error handling wrapper if missing
        if 'try' not in code and 'catch' not in code:
            # Wrap main execution in try-catch
            lines = code.split('\n')
            main_call_line = None
            for i, line in enumerate(lines):
                if 'main()' in line and not line.strip().startswith('//'):
                    main_call_line = i
                    break
            
            if main_call_line is not None:
                lines[main_call_line] = f"""try {{
    {lines[main_call_line]}
}} catch (error) {{
    console.error('Error:', error.message);
    process.exit(1);
}}"""
                code = '\n'.join(lines)
        
        return code