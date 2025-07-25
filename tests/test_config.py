"""
Comprehensive tests for the configuration management system.

Tests configuration loading, validation, environment variable overrides,
and various configuration scenarios.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import (
    ConfigurationManager,
    ConfigurationFactory,
    ConfigurationError,
    EnvironmentVariableResolver,
    ConfigValidator,
    get_config,
    set_config,
    init_config,
    ConfigContext
)


class TestConfigValidator(unittest.TestCase):
    """Test configuration validation utilities."""
    
    def test_validate_positive_number(self):
        """Test positive number validation."""
        # Valid cases
        self.assertEqual(ConfigValidator.validate_positive_number(5, "test"), 5)
        self.assertEqual(ConfigValidator.validate_positive_number(3.14, "test"), 3.14)
        
        # Invalid cases
        with self.assertRaises(ConfigurationError):
            ConfigValidator.validate_positive_number(0, "test")
        with self.assertRaises(ConfigurationError):
            ConfigValidator.validate_positive_number(-1, "test")
        with self.assertRaises(ConfigurationError):
            ConfigValidator.validate_positive_number("not_a_number", "test")
    
    def test_validate_percentage(self):
        """Test percentage validation."""
        # Valid cases
        self.assertEqual(ConfigValidator.validate_percentage(0.5, "test"), 0.5)
        self.assertEqual(ConfigValidator.validate_percentage(0, "test"), 0)
        self.assertEqual(ConfigValidator.validate_percentage(1, "test"), 1)
        
        # Invalid cases
        with self.assertRaises(ConfigurationError):
            ConfigValidator.validate_percentage(-0.1, "test")
        with self.assertRaises(ConfigurationError):
            ConfigValidator.validate_percentage(1.1, "test")
    
    def test_validate_string_choice(self):
        """Test string choice validation."""
        choices = ["option1", "option2", "option3"]
        
        # Valid case
        self.assertEqual(ConfigValidator.validate_string_choice("option1", choices, "test"), "option1")
        
        # Invalid case
        with self.assertRaises(ConfigurationError):
            ConfigValidator.validate_string_choice("invalid", choices, "test")
    
    def test_validate_file_path(self):
        """Test file path validation."""
        # Valid path (string)
        result = ConfigValidator.validate_file_path("/tmp/test.txt", "test", must_exist=False)
        self.assertEqual(result, "/tmp/test.txt")
        
        # Invalid type
        with self.assertRaises(ConfigurationError):
            ConfigValidator.validate_file_path(123, "test")


class TestEnvironmentVariableResolver(unittest.TestCase):
    """Test environment variable resolution."""
    
    def setUp(self):
        """Set up test environment variables."""
        os.environ['TEST_VAR'] = 'test_value'
        os.environ['TEST_NUMBER'] = '42'
    
    def tearDown(self):
        """Clean up test environment variables."""
        os.environ.pop('TEST_VAR', None)
        os.environ.pop('TEST_NUMBER', None)
    
    def test_resolve_simple_variable(self):
        """Test simple environment variable resolution."""
        result = EnvironmentVariableResolver.resolve_string("${TEST_VAR}")
        self.assertEqual(result, "test_value")
    
    def test_resolve_with_default(self):
        """Test environment variable resolution with default value."""
        result = EnvironmentVariableResolver.resolve_string("${NONEXISTENT:default_value}")
        self.assertEqual(result, "default_value")
    
    def test_resolve_mixed_string(self):
        """Test resolution in mixed string."""
        result = EnvironmentVariableResolver.resolve_string("prefix_${TEST_VAR}_suffix")
        self.assertEqual(result, "prefix_test_value_suffix")
    
    def test_resolve_missing_variable(self):
        """Test resolution of missing environment variable."""
        with self.assertRaises(ConfigurationError):
            EnvironmentVariableResolver.resolve_string("${NONEXISTENT_VAR}")
    
    def test_resolve_nested_structure(self):
        """Test resolution in nested data structures."""
        data = {
            "key1": "${TEST_VAR}",
            "key2": {
                "nested": "${TEST_NUMBER}",
                "list": ["${TEST_VAR}", "static_value"]
            }
        }
        
        result = EnvironmentVariableResolver.resolve_value(data)
        
        expected = {
            "key1": "test_value",
            "key2": {
                "nested": "42",
                "list": ["test_value", "static_value"]
            }
        }
        
        self.assertEqual(result, expected)


class TestConfigurationManager(unittest.TestCase):
    """Test the main configuration manager."""
    
    def setUp(self):
        """Set up test configuration."""
        self.test_config = {
            "api": {
                "openai": {
                    "model": "gpt-3.5-turbo",
                    "max_tokens": 1500,
                    "temperature": 0.5
                }
            },
            "execution": {
                "timeout_seconds": 30,
                "max_memory_mb": 256
            },
            "logging": {
                "level": "INFO"
            }
        }
    
    def test_initialization_without_file(self):
        """Test initialization without configuration file."""
        manager = ConfigurationManager()
        self.assertIsInstance(manager, ConfigurationManager)
        self.assertEqual(manager.get('nonexistent', 'default'), 'default')
    
    def test_get_with_dot_notation(self):
        """Test getting values with dot notation."""
        manager = ConfigurationManager()
        manager._config = self.test_config
        
        self.assertEqual(manager.get('api.openai.model'), 'gpt-3.5-turbo')
        self.assertEqual(manager.get('api.openai.max_tokens'), 1500)
        self.assertEqual(manager.get('execution.timeout_seconds'), 30)
    
    def test_get_with_default(self):
        """Test getting values with default fallback."""
        manager = ConfigurationManager()
        manager._config = self.test_config
        
        self.assertEqual(manager.get('nonexistent.path', 'default_value'), 'default_value')
        self.assertEqual(manager.get('api.openai.nonexistent', 42), 42)
    
    def test_get_required(self):
        """Test getting required values."""
        manager = ConfigurationManager()
        manager._config = self.test_config
        
        # Should work for existing path
        result = manager.get('api.openai.model', required=True)
        self.assertEqual(result, 'gpt-3.5-turbo')
        
        # Should raise error for missing path
        with self.assertRaises(ConfigurationError):
            manager.get('nonexistent.path', required=True)
    
    def test_set_value(self):
        """Test setting configuration values."""
        manager = ConfigurationManager(strict_validation=False)
        
        manager.set('new.nested.value', 'test_value')
        self.assertEqual(manager.get('new.nested.value'), 'test_value')
    
    def test_has_method(self):
        """Test checking if configuration paths exist."""
        manager = ConfigurationManager()
        manager._config = self.test_config
        
        self.assertTrue(manager.has('api.openai.model'))
        self.assertFalse(manager.has('nonexistent.path'))
    
    def test_get_section(self):
        """Test getting entire configuration sections."""
        manager = ConfigurationManager()
        manager._config = self.test_config
        
        api_section = manager.get_section('api')
        expected = self.test_config['api']
        self.assertEqual(api_section, expected)
    
    def test_env_value_conversion(self):
        """Test environment variable value conversion."""
        manager = ConfigurationManager()
        
        # Test boolean conversion
        self.assertTrue(manager._convert_env_value('true'))
        self.assertTrue(manager._convert_env_value('True'))
        self.assertTrue(manager._convert_env_value('yes'))
        self.assertTrue(manager._convert_env_value('1'))
        
        self.assertFalse(manager._convert_env_value('false'))
        self.assertFalse(manager._convert_env_value('no'))
        self.assertFalse(manager._convert_env_value('0'))
        
        # Test numeric conversion
        self.assertEqual(manager._convert_env_value('42'), 42)
        self.assertEqual(manager._convert_env_value('3.14'), 3.14)
        self.assertEqual(manager._convert_env_value('1e-6'), 1e-6)
        
        # Test string fallback
        self.assertEqual(manager._convert_env_value('regular_string'), 'regular_string')
    
    @patch.dict(os.environ, {'COP_API_OPENAI_MODEL': 'gpt-4', 'COP_EXECUTION_TIMEOUT_SECONDS': '60'})
    def test_env_overrides(self):
        """Test environment variable overrides."""
        manager = ConfigurationManager(env_prefix="COP")
        manager._config = self.test_config.copy()
        
        # Load environment overrides
        env_overrides = manager._load_env_overrides()
        
        self.assertEqual(env_overrides['api']['openai']['model'], 'gpt-4')
        self.assertEqual(env_overrides['execution']['timeout_seconds'], 60)
    
    def test_merge_configs(self):
        """Test configuration merging."""
        manager = ConfigurationManager()
        
        base = {
            "a": {"b": 1, "c": 2},
            "d": 3
        }
        
        override = {
            "a": {"b": 10, "e": 4},
            "f": 5
        }
        
        result = manager._merge_configs(base, override)
        
        expected = {
            "a": {"b": 10, "c": 2, "e": 4},
            "d": 3,
            "f": 5
        }
        
        self.assertEqual(result, expected)


class TestConfigurationFactory(unittest.TestCase):
    """Test configuration factory methods."""
    
    def test_create_default(self):
        """Test creating default configuration."""
        config = ConfigurationFactory.create_default()
        self.assertIsInstance(config, ConfigurationManager)
        self.assertEqual(config.env_prefix, "COP")
        self.assertFalse(config.auto_reload)
        self.assertTrue(config.strict_validation)
    
    def test_create_development(self):
        """Test creating development configuration."""
        config = ConfigurationFactory.create_development()
        self.assertTrue(config.auto_reload)
        self.assertFalse(config.strict_validation)
    
    def test_create_production(self):
        """Test creating production configuration."""
        config = ConfigurationFactory.create_production()
        self.assertFalse(config.auto_reload)
        self.assertTrue(config.strict_validation)
    
    def test_create_testing(self):
        """Test creating testing configuration."""
        test_data = {"test": {"value": 42}}
        config = ConfigurationFactory.create_testing(test_data)
        
        self.assertEqual(config.env_prefix, "COP_TEST")
        self.assertFalse(config.strict_validation)
        self.assertEqual(config.get('test.value'), 42)


class TestConfigurationWithFile(unittest.TestCase):
    """Test configuration loading from files."""
    
    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "api": {
                "openai": {
                    "model": "gpt-4",
                    "temperature": 0.7
                }
            },
            "logging": {
                "level": "DEBUG"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            manager = ConfigurationManager(config_file=temp_file)
            
            self.assertEqual(manager.get('api.openai.model'), 'gpt-4')
            self.assertEqual(manager.get('api.openai.temperature'), 0.7)
            self.assertEqual(manager.get('logging.level'), 'DEBUG')
        
        finally:
            os.unlink(temp_file)
    
    def test_nonexistent_config_file(self):
        """Test handling of nonexistent configuration file."""
        manager = ConfigurationManager(config_file="/nonexistent/path/config.yaml")
        # Should not raise error, just use empty config
        self.assertEqual(manager.get('any.path', 'default'), 'default')


class TestGlobalConfiguration(unittest.TestCase):
    """Test global configuration functions."""
    
    def setUp(self):
        """Reset global configuration before each test."""
        set_config(None)
    
    def test_get_config_creates_default(self):
        """Test that get_config creates default configuration."""
        config = get_config()
        self.assertIsInstance(config, ConfigurationManager)
    
    def test_set_and_get_config(self):
        """Test setting and getting global configuration."""
        test_config = ConfigurationFactory.create_testing()
        set_config(test_config)
        
        retrieved_config = get_config()
        self.assertIs(retrieved_config, test_config)
    
    def test_init_config_development(self):
        """Test initializing configuration for development."""
        config = init_config(environment="development")
        
        self.assertTrue(config.auto_reload)
        self.assertFalse(config.strict_validation)
    
    def test_init_config_production(self):
        """Test initializing configuration for production."""
        config = init_config(environment="production")
        
        self.assertFalse(config.auto_reload)
        self.assertTrue(config.strict_validation)


class TestConfigContext(unittest.TestCase):
    """Test configuration context manager."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = ConfigurationFactory.create_testing({
            "logging": {"level": "INFO"},
            "api": {"timeout": 30}
        })
        set_config(self.config)
    
    def test_config_context_single_value(self):
        """Test context manager with single value change."""
        original_level = self.config.get('logging.level')
        
        with ConfigContext(**{'logging.level': 'DEBUG'}):
            self.assertEqual(self.config.get('logging.level'), 'DEBUG')
        
        self.assertEqual(self.config.get('logging.level'), original_level)
    
    def test_config_context_multiple_values(self):
        """Test context manager with multiple value changes."""
        original_level = self.config.get('logging.level')
        original_timeout = self.config.get('api.timeout')
        
        with ConfigContext(**{
            'logging.level': 'WARNING',
            'api.timeout': 60
        }):
            self.assertEqual(self.config.get('logging.level'), 'WARNING')
            self.assertEqual(self.config.get('api.timeout'), 60)
        
        self.assertEqual(self.config.get('logging.level'), original_level)
        self.assertEqual(self.config.get('api.timeout'), original_timeout)


class TestValidationIntegration(unittest.TestCase):
    """Test validation integration with configuration manager."""
    
    def test_validation_on_set(self):
        """Test validation when setting values."""
        manager = ConfigurationManager(strict_validation=True)
        
        # This should pass validation
        manager.set('api.openai.max_tokens', 2000)
        self.assertEqual(manager.get('api.openai.max_tokens'), 2000)
        
        # This should fail validation (negative value)
        with self.assertRaises(ConfigurationError):
            manager.set('api.openai.max_tokens', -100)
    
    def test_custom_validator_registration(self):
        """Test registering custom validators."""
        manager = ConfigurationManager(strict_validation=True)
        
        def validate_custom(value):
            if value != "expected_value":
                raise ValueError("Custom validation failed")
            return value
        
        manager.register_validator('custom.setting', validate_custom)
        
        # Should pass
        manager.set('custom.setting', 'expected_value')
        
        # Should fail
        with self.assertRaises(ConfigurationError):
            manager.set('custom.setting', 'wrong_value')


class TestEnvironmentVariableIntegration(unittest.TestCase):
    """Test environment variable integration."""
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-api-key',
        'COP_API_OPENAI_TEMPERATURE': '0.8',
        'COP_LOGGING_LEVEL': 'WARNING'
    })
    def test_env_var_resolution_in_config(self):
        """Test environment variable resolution in configuration."""
        config_data = {
            "api": {
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "temperature": 0.5
                }
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            manager = ConfigurationManager(config_file=temp_file, env_prefix="COP")
            
            # Environment variable should be resolved
            self.assertEqual(manager.get('api.openai.api_key'), 'test-api-key')
            
            # Environment override should take precedence
            self.assertEqual(manager.get('api.openai.temperature'), 0.8)
            self.assertEqual(manager.get('logging.level'), 'WARNING')
        
        finally:
            os.unlink(temp_file)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in configuration system."""
    
    def test_invalid_yaml_file(self):
        """Test handling of invalid YAML file."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_file = f.name
        
        try:
            with self.assertRaises(ConfigurationError):
                ConfigurationManager(config_file=temp_file)
        
        finally:
            os.unlink(temp_file)
    
    def test_validation_error_propagation(self):
        """Test that validation errors are properly propagated."""
        manager = ConfigurationManager(strict_validation=True)
        
        with self.assertRaises(ConfigurationError) as context:
            manager.set('api.openai.temperature', -1.0)  # Invalid temperature
        
        self.assertIn('temperature', str(context.exception).lower())


if __name__ == "__main__":
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run all tests
    unittest.main(verbosity=2)