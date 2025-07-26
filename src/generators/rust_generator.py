"""
Simple Rust code generator using OpenAI API.
"""

import re
import time
from typing import Dict, Any, Optional

import openai
import yaml
from ..core.seed_manager import SeedManager


class RustGenerator:
    """Generates Rust code using OpenAI API."""
    
    def __init__(self, api_key: str, config_path: str = "config.yaml"):
        """Initialize Rust generator."""
        self.client = openai.OpenAI(api_key=api_key)
        self.seed_manager = SeedManager()
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    def generate_code(self) -> Dict[str, Any]:
        """Generate a single Rust code sample."""
        # Get random seeds
        application = self.seed_manager.get_random_application()
        concept = self.seed_manager.get_random_concept()
        
        # Create Rust-specific prompt
        prompt = f"Create a Rust program for {application} using {concept}. " \
                f"Include {self.config['generation']['min_functions']}-{self.config['generation']['max_functions']} functions, " \
                f"proper error handling with Result types, and a main function. Use modern Rust patterns."
        
        try:
            # Call OpenAI API
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.config['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are a Rust expert. Write safe, idiomatic Rust code with proper error handling using Result types."},
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
            code = self._enhance_rust_code(code)
            
            return {
                "success": True,
                "code": code,
                "language": "rust",
                "application": application,
                "concept": concept,
                "generation_time": generation_time,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "language": "rust",
                "application": application,
                "concept": concept
            }
    
    def _extract_code(self, text: str) -> str:
        """Extract Rust code from response text."""
        # Look for Rust code blocks
        patterns = [
            r'```rust\n(.*?)\n```',
            r'```\n(.*?)\n```'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no code block, assume entire text is code
        return text.strip()
    
    def _enhance_rust_code(self, code: str) -> str:
        """Enhance Rust code with necessary additions."""
        # Ensure main function exists
        if 'fn main()' not in code:
            # If there are other functions, add a simple main
            if 'fn ' in code:
                code += '\n\nfn main() {\n    println!("Program executed successfully");\n}'
        
        # Add basic std imports if not present and needed
        needs_io = 'println!' in code or 'print!' in code
        needs_std = 'std::' in code or needs_io
        
        if needs_std and 'use std::' not in code:
            # Add common std imports at the top
            imports = []
            if 'io::stdin' in code or 'read_line' in code:
                imports.append('use std::io;')
            if 'HashMap' in code:
                imports.append('use std::collections::HashMap;')
            if 'fs::' in code:
                imports.append('use std::fs;')
            
            if imports:
                code = '\n'.join(imports) + '\n\n' + code
        
        return code