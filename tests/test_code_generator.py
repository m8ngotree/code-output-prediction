"""
Unit tests for the CodeGenerator class.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.generators.code_generator import CodeGenerator, CodeGenerationError


class TestCodeGenerator(unittest.TestCase):
    """Test cases for CodeGenerator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        # Create test configuration
        test_config = {
            "openai": {
                "model": "gpt-4",
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "timeout": 30
            },
            "retry": {
                "max_attempts": 2,
                "wait_multiplier": 1,
                "wait_min": 1,
                "wait_max": 5
            },
            "generation": {
                "min_functions": 2,
                "max_functions": 4,
                "include_main": True,
                "include_docstrings": True,
                "include_type_hints": True
            },
            "output": {
                "save_directory": str(Path(self.temp_dir) / "output"),
                "include_metadata": True,
                "filename_format": "test_{timestamp}_{hash}.py",
                "metadata_format": "json"
            },
            "validation": {
                "syntax_check": True,
                "import_check": True,
                "basic_execution_test": False
            },
            "prompts": {
                "system_message": "Test system message",
                "user_template": "Create {application} using {concept}"
            },
            "common_imports": [
                "import sys",
                "import os"
            ]
        }
        
        # Save test config
        import yaml
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_initialization(self, mock_openai, mock_seed_manager):
        """Test CodeGenerator initialization."""
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        self.assertIsNotNone(generator.config)
        self.assertIsNotNone(generator.seed_manager)
        mock_openai.assert_called_once_with(api_key="test-key")
    
    @patch('src.generators.code_generator.SeedManager')
    def test_initialization_no_api_key(self, mock_seed_manager):
        """Test initialization without API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError):
                CodeGenerator(config_path=str(self.config_path))
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_create_prompt(self, mock_openai, mock_seed_manager):
        """Test prompt creation."""
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        system_msg, user_prompt = generator._create_prompt("web scraping", "recursion")
        
        self.assertEqual(system_msg, "Test system message")
        self.assertIn("web scraping", user_prompt)
        self.assertIn("recursion", user_prompt)
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_extract_code_from_response(self, mock_openai, mock_seed_manager):
        """Test code extraction from API response."""
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        # Test code block extraction
        response_with_block = "Here's the code:\n```python\nprint('hello')\n```\nThat's it!"
        extracted = generator._extract_code_from_response(response_with_block)
        self.assertEqual(extracted, "print('hello')")
        
        # Test plain code
        plain_code = "def test():\n    return 42"
        extracted = generator._extract_code_from_response(plain_code)
        self.assertEqual(extracted, plain_code)
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_validate_syntax(self, mock_openai, mock_seed_manager):
        """Test syntax validation."""
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        # Valid syntax
        valid_code = "def test():\n    return 42"
        self.assertTrue(generator._validate_syntax(valid_code))
        
        # Invalid syntax
        invalid_code = "def test(\n    return 42"
        self.assertFalse(generator._validate_syntax(invalid_code))
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_add_missing_imports(self, mock_openai, mock_seed_manager):
        """Test adding missing imports."""
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        code = "def test():\n    sys.exit(0)\n    json.dumps({})"
        processed = generator._add_missing_imports(code)
        
        self.assertIn("import sys", processed)
        self.assertIn("import json", processed)
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_ensure_executable(self, mock_openai, mock_seed_manager):
        """Test ensuring code is executable."""
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        # Code without main
        code_without_main = "def test():\n    return 42"
        processed = generator._ensure_executable(code_without_main)
        self.assertIn('if __name__ == "__main__"', processed)
        
        # Code with main should remain unchanged
        code_with_main = "def test():\n    return 42\n\nif __name__ == '__main__':\n    test()"
        processed = generator._ensure_executable(code_with_main)
        self.assertEqual(processed, code_with_main)
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_post_process_code(self, mock_openai, mock_seed_manager):
        """Test code post-processing."""
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        raw_response = "```python\ndef test():\n    sys.exit(0)\n```"
        processed = generator._post_process_code(raw_response)
        
        self.assertIn("import sys", processed)
        self.assertIn('if __name__ == "__main__"', processed)
        self.assertIn("def test():", processed)
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_create_metadata(self, mock_openai, mock_seed_manager):
        """Test metadata creation."""
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        code = "def test():\n    return 42"
        metadata = generator._create_metadata("web scraping", "recursion", code, 1.5)
        
        self.assertIn("timestamp", metadata)
        self.assertEqual(metadata["seeds"]["application"], "web scraping")
        self.assertEqual(metadata["seeds"]["concept"], "recursion")
        self.assertEqual(metadata["generation_time_seconds"], 1.5)
        self.assertIn("code_stats", metadata)
        self.assertIn("hash", metadata)
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_save_generated_code(self, mock_openai, mock_seed_manager):
        """Test saving generated code and metadata."""
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        code = "def test():\n    return 42"
        metadata = {"hash": "abc123", "test": "data"}
        
        code_path, metadata_path = generator._save_generated_code(code, metadata)
        
        # Check files were created
        self.assertTrue(Path(code_path).exists())
        self.assertTrue(Path(metadata_path).exists())
        
        # Check content
        with open(code_path, 'r') as f:
            saved_code = f.read()
        self.assertEqual(saved_code, code)
        
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)
        self.assertEqual(saved_metadata, metadata)
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_call_openai_api_success(self, mock_openai, mock_seed_manager):
        """Test successful OpenAI API call."""
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "def test(): return 42"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        result = generator._call_openai_api("system", "user prompt")
        self.assertEqual(result, "def test(): return 42")
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_call_openai_api_error(self, mock_openai, mock_seed_manager):
        """Test OpenAI API call with error."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        with self.assertRaises(CodeGenerationError):
            generator._call_openai_api("system", "user prompt")
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    @patch('time.time')
    def test_generate_code_success(self, mock_time, mock_openai, mock_seed_manager):
        """Test successful code generation."""
        # Mock time
        mock_time.side_effect = [0, 1.5]
        
        # Mock seed manager
        mock_seed_manager_instance = Mock()
        mock_seed_manager_instance.sample_seeds.side_effect = [
            ["web scraping"], ["recursion"]
        ]
        mock_seed_manager.return_value = mock_seed_manager_instance
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "```python\ndef scrape():\n    return 'data'\n```"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        result = generator.generate_code()
        
        self.assertTrue(result["success"])
        self.assertIn("code", result)
        self.assertIn("metadata", result)
        self.assertIn("files", result)
        self.assertEqual(result["seeds_used"]["application"], "web scraping")
        self.assertEqual(result["seeds_used"]["concept"], "recursion")
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_generate_code_failure(self, mock_openai, mock_seed_manager):
        """Test code generation failure."""
        # Mock seed manager to raise error
        mock_seed_manager_instance = Mock()
        mock_seed_manager_instance.sample_seeds.side_effect = Exception("Seed error")
        mock_seed_manager.return_value = mock_seed_manager_instance
        
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        result = generator.generate_code()
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("error_type", result)
    
    @patch('src.generators.code_generator.SeedManager')
    @patch('openai.OpenAI')
    def test_generate_batch(self, mock_openai, mock_seed_manager):
        """Test batch code generation."""
        # Mock seed manager 
        mock_seed_manager_instance = Mock()
        mock_seed_manager_instance.sample_seeds.side_effect = [
            ["app1", "app2"], ["concept1", "concept2"]
        ]
        mock_seed_manager.return_value = mock_seed_manager_instance
        
        generator = CodeGenerator(
            config_path=str(self.config_path),
            api_key="test-key"
        )
        
        # Mock generate_code method
        generator.generate_code = Mock()
        generator.generate_code.side_effect = [
            {"success": True, "test": "result1"},
            {"success": True, "test": "result2"}
        ]
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            results = generator.generate_batch(2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(generator.generate_code.call_count, 2)
    
    def test_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            CodeGenerator(config_path="nonexistent.yaml", api_key="test-key")


if __name__ == "__main__":
    unittest.main()