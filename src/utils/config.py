"""
Comprehensive Configuration Management System.

This module provides centralized configuration management with:
- YAML configuration file loading
- Environment variable override support
- Configuration validation and type checking
- Nested configuration access
- Default value handling
- Configuration caching and hot-reloading
"""

import os
import re
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Callable
from datetime import datetime, timedelta
import yaml

try:
    from pydantic import BaseModel, ValidationError, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    ValidationError = Exception


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class ConfigValidator:
    """Configuration validation utilities."""
    
    @staticmethod
    def validate_positive_number(value: Union[int, float], name: str) -> Union[int, float]:
        """Validate that a number is positive."""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ConfigurationError(f"{name} must be a positive number, got {value}")
        return value
    
    @staticmethod
    def validate_non_negative_number(value: Union[int, float], name: str) -> Union[int, float]:
        """Validate that a number is non-negative."""
        if not isinstance(value, (int, float)) or value < 0:
            raise ConfigurationError(f"{name} must be non-negative, got {value}")
        return value
    
    @staticmethod
    def validate_percentage(value: Union[int, float], name: str) -> Union[int, float]:
        """Validate that a number is between 0 and 1 (percentage)."""
        if not isinstance(value, (int, float)) or not (0 <= value <= 1):
            raise ConfigurationError(f"{name} must be between 0 and 1, got {value}")
        return value
    
    @staticmethod
    def validate_string_choice(value: str, choices: List[str], name: str) -> str:
        """Validate that a string is one of the allowed choices."""
        if value not in choices:
            raise ConfigurationError(f"{name} must be one of {choices}, got '{value}'")
        return value
    
    @staticmethod
    def validate_file_path(value: str, name: str, must_exist: bool = False) -> str:
        """Validate file path."""
        if not isinstance(value, str):
            raise ConfigurationError(f"{name} must be a string path, got {type(value)}")
        
        path = Path(value)
        if must_exist and not path.exists():
            raise ConfigurationError(f"{name} path does not exist: {value}")
        
        return value
    
    @staticmethod
    def validate_directory_path(value: str, name: str, create_if_missing: bool = False) -> str:
        """Validate directory path."""
        if not isinstance(value, str):
            raise ConfigurationError(f"{name} must be a string path, got {type(value)}")
        
        path = Path(value)
        if create_if_missing:
            path.mkdir(parents=True, exist_ok=True)
        elif not path.exists():
            raise ConfigurationError(f"{name} directory does not exist: {value}")
        
        return value


class EnvironmentVariableResolver:
    """Resolves environment variables in configuration values."""
    
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    @classmethod
    def resolve_string(cls, value: str) -> str:
        """Resolve environment variables in a string value."""
        if not isinstance(value, str):
            return value
        
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = None
            
            # Handle default values: ${VAR_NAME:default_value}
            if ':' in var_name:
                var_name, default_value = var_name.split(':', 1)
            
            env_value = os.getenv(var_name, default_value)
            if env_value is None:
                raise ConfigurationError(f"Environment variable '{var_name}' not found and no default provided")
            
            return env_value
        
        return cls.ENV_VAR_PATTERN.sub(replace_env_var, value)
    
    @classmethod
    def resolve_value(cls, value: Any) -> Any:
        """Recursively resolve environment variables in configuration values."""
        if isinstance(value, str):
            return cls.resolve_string(value)
        elif isinstance(value, dict):
            return {k: cls.resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.resolve_value(item) for item in value]
        else:
            return value


class ConfigurationManager:
    """
    Comprehensive configuration management system.
    
    Features:
    - YAML configuration file loading
    - Environment variable overrides
    - Configuration validation
    - Nested value access with dot notation
    - Configuration caching and reload detection
    - Type conversion and validation
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 env_prefix: str = "COP",
                 auto_reload: bool = False,
                 strict_validation: bool = True):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to YAML configuration file
            env_prefix: Prefix for environment variable overrides
            auto_reload: Enable automatic config reloading on file changes
            strict_validation: Enable strict validation of configuration values
        """
        self.config_file = Path(config_file) if config_file else None
        self.env_prefix = env_prefix
        self.auto_reload = auto_reload
        self.strict_validation = strict_validation
        
        # Internal state
        self._config: Dict[str, Any] = {}
        self._file_mtime: Optional[float] = None
        self._env_overrides: Dict[str, Any] = {}
        self._validators: Dict[str, Callable] = {}
        self._logger = logging.getLogger(__name__)
        
        # Load initial configuration
        self.reload()
        
        # Register default validators
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register default configuration validators."""
        validators = {
            # API settings
            'api.openai.max_tokens': lambda v: ConfigValidator.validate_positive_number(v, 'api.openai.max_tokens'),
            'api.openai.temperature': lambda v: ConfigValidator.validate_non_negative_number(v, 'api.openai.temperature'),
            'api.openai.timeout_seconds': lambda v: ConfigValidator.validate_positive_number(v, 'api.openai.timeout_seconds'),
            'api.openai.rate_limit.requests_per_minute': lambda v: ConfigValidator.validate_positive_number(v, 'api.openai.rate_limit.requests_per_minute'),
            'api.openai.cost_tracking.daily_limit_usd': lambda v: ConfigValidator.validate_positive_number(v, 'api.openai.cost_tracking.daily_limit_usd'),
            
            # Generation settings
            'generation.complexity.min_functions': lambda v: ConfigValidator.validate_positive_number(v, 'generation.complexity.min_functions'),
            'generation.complexity.max_functions': lambda v: ConfigValidator.validate_positive_number(v, 'generation.complexity.max_functions'),
            'generation.complexity.max_execution_time_seconds': lambda v: ConfigValidator.validate_positive_number(v, 'generation.complexity.max_execution_time_seconds'),
            'generation.seeds.strategy': lambda v: ConfigValidator.validate_string_choice(v, ['random', 'sequential', 'weighted'], 'generation.seeds.strategy'),
            'generation.output.base_directory': lambda v: ConfigValidator.validate_directory_path(v, 'generation.output.base_directory', create_if_missing=True),
            
            # Execution settings
            'execution.timeouts.code_execution_seconds': lambda v: ConfigValidator.validate_positive_number(v, 'execution.timeouts.code_execution_seconds'),
            'execution.resources.max_memory_mb': lambda v: ConfigValidator.validate_positive_number(v, 'execution.resources.max_memory_mb'),
            'execution.parallel.max_workers': lambda v: ConfigValidator.validate_positive_number(v, 'execution.parallel.max_workers'),
            
            # Verification settings
            'verification.tolerances.numeric_absolute': lambda v: ConfigValidator.validate_positive_number(v, 'verification.tolerances.numeric_absolute'),
            'verification.tolerances.fuzzy_string_threshold': lambda v: ConfigValidator.validate_percentage(v, 'verification.tolerances.fuzzy_string_threshold'),
            
            # Logging settings
            'logging.level': lambda v: ConfigValidator.validate_string_choice(v, ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 'logging.level'),
            'logging.destinations.file.max_size_mb': lambda v: ConfigValidator.validate_positive_number(v, 'logging.destinations.file.max_size_mb'),
        }
        
        self._validators.update(validators)
    
    def reload(self) -> bool:
        """
        Reload configuration from file and environment variables.
        
        Returns:
            True if configuration was reloaded, False if no changes detected
        """
        try:
            # Check if file needs reloading
            if self.config_file and self.config_file.exists():
                current_mtime = self.config_file.stat().st_mtime
                if self._file_mtime is not None and current_mtime == self._file_mtime:
                    if not self.auto_reload:
                        return False
                
                self._file_mtime = current_mtime
                
                # Load YAML configuration
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                
                self._logger.info(f"Loaded configuration from {self.config_file}")
            else:
                file_config = {}
                if self.config_file:
                    self._logger.warning(f"Configuration file not found: {self.config_file}")
            
            # Resolve environment variables in file config
            file_config = EnvironmentVariableResolver.resolve_value(file_config)
            
            # Load environment variable overrides
            env_overrides = self._load_env_overrides()
            
            # Merge configurations (env overrides take precedence)
            self._config = self._merge_configs(file_config, env_overrides)
            
            # Validate configuration if strict mode is enabled
            if self.strict_validation:
                self._validate_config()
            
            self._logger.debug("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to reload configuration: {e}")
            raise ConfigurationError(f"Configuration reload failed: {e}")
    
    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        prefix = f"{self.env_prefix}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert environment variable name to config path
                config_path = key[len(prefix):].lower().replace('_', '.')
                
                # Try to convert value to appropriate type
                converted_value = self._convert_env_value(value)
                
                # Set nested value
                self._set_nested_value(overrides, config_path, converted_value)
        
        return overrides
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate Python type."""
        if not value:
            return None
        
        # Try boolean conversion
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Try numeric conversion
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Try JSON parsing for complex types
        if value.startswith(('{', '[', '"')):
            try:
                import json
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set a nested value in configuration using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self):
        """Validate configuration using registered validators."""
        for config_path, validator in self._validators.items():
            try:
                value = self.get(config_path)
                if value is not None:
                    validator(value)
            except Exception as e:
                raise ConfigurationError(f"Validation failed for {config_path}: {e}")
    
    def get(self, path: str, default: Any = None, required: bool = False) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            path: Configuration path (e.g., 'api.openai.model')
            default: Default value if path not found
            required: Raise error if path not found and no default
            
        Returns:
            Configuration value
            
        Raises:
            ConfigurationError: If required path not found
        """
        # Check if reload is needed
        if self.auto_reload:
            self.reload()
        
        keys = path.split('.')
        current = self._config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            if required and default is None:
                raise ConfigurationError(f"Required configuration path not found: {path}")
            return default
    
    def set(self, path: str, value: Any, validate: bool = True):
        """
        Set configuration value using dot notation.
        
        Args:
            path: Configuration path
            value: Value to set
            validate: Whether to validate the value
        """
        if validate and path in self._validators:
            try:
                self._validators[path](value)
            except Exception as e:
                raise ConfigurationError(f"Validation failed for {path}: {e}")
        
        self._set_nested_value(self._config, path, value)
    
    def has(self, path: str) -> bool:
        """Check if configuration path exists."""
        try:
            self.get(path, required=True)
            return True
        except ConfigurationError:
            return False
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.get(section, default={})
    
    def register_validator(self, path: str, validator: Callable[[Any], Any]):
        """Register a custom validator for a configuration path."""
        self._validators[path] = validator
    
    def to_dict(self) -> Dict[str, Any]:
        """Get entire configuration as dictionary."""
        return self._config.copy()
    
    def save_to_file(self, file_path: str):
        """Save current configuration to YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def get_env_info(self) -> Dict[str, Any]:
        """Get information about environment variable overrides."""
        env_vars = {}
        prefix = f"{self.env_prefix}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_path = key[len(prefix):].lower().replace('_', '.')
                env_vars[config_path] = value
        
        return {
            'prefix': self.env_prefix,
            'overrides': env_vars,
            'total_overrides': len(env_vars)
        }


class ConfigurationFactory:
    """Factory for creating pre-configured ConfigurationManager instances."""
    
    @staticmethod
    def create_default(config_file: Optional[str] = None) -> ConfigurationManager:
        """Create configuration manager with default settings."""
        if config_file is None:
            # Try to find config file in common locations
            possible_paths = [
                Path("configs/config.yaml"),
                Path("config.yaml"),
                Path("conf/config.yaml"),
                Path.cwd() / "configs" / "config.yaml"
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_file = str(path)
                    break
        
        return ConfigurationManager(
            config_file=config_file,
            env_prefix="COP",
            auto_reload=False,
            strict_validation=True
        )
    
    @staticmethod
    def create_development() -> ConfigurationManager:
        """Create configuration manager for development environment."""
        return ConfigurationManager(
            config_file="configs/config.yaml",
            env_prefix="COP",
            auto_reload=True,
            strict_validation=False
        )
    
    @staticmethod
    def create_production() -> ConfigurationManager:
        """Create configuration manager for production environment."""
        return ConfigurationManager(
            config_file="configs/config.yaml",
            env_prefix="COP",
            auto_reload=False,
            strict_validation=True
        )
    
    @staticmethod
    def create_testing(config_dict: Optional[Dict[str, Any]] = None) -> ConfigurationManager:
        """Create configuration manager for testing with in-memory config."""
        manager = ConfigurationManager(
            env_prefix="COP_TEST",
            strict_validation=False
        )
        
        if config_dict:
            manager._config = config_dict
        
        return manager


# Pydantic models for configuration validation (if available)
if PYDANTIC_AVAILABLE:
    class APIConfig(BaseModel):
        """API configuration validation model."""
        api_key: str
        model: str = "gpt-3.5-turbo"
        max_tokens: int = 1500
        temperature: float = 0.5
        timeout_seconds: int = 30
        
        @validator('max_tokens')
        def validate_max_tokens(cls, v):
            if v <= 0:
                raise ValueError('max_tokens must be positive')
            return v
        
        @validator('temperature')
        def validate_temperature(cls, v):
            if not 0 <= v <= 2:
                raise ValueError('temperature must be between 0 and 2')
            return v
    
    class ExecutionConfig(BaseModel):
        """Execution configuration validation model."""
        code_execution_seconds: int = 10
        max_memory_mb: int = 256
        max_workers: int = 4
        
        @validator('code_execution_seconds', 'max_memory_mb', 'max_workers')
        def validate_positive(cls, v):
            if v <= 0:
                raise ValueError('Value must be positive')
            return v


# Global configuration instance
_global_config: Optional[ConfigurationManager] = None


def get_config() -> ConfigurationManager:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigurationFactory.create_default()
    return _global_config


def set_config(config: ConfigurationManager):
    """Set global configuration instance."""
    global _global_config
    _global_config = config


def init_config(config_file: Optional[str] = None, 
                env_prefix: str = "COP",
                environment: str = "development") -> ConfigurationManager:
    """
    Initialize global configuration.
    
    Args:
        config_file: Path to configuration file
        env_prefix: Environment variable prefix
        environment: Environment type (development, production, testing)
        
    Returns:
        Configured ConfigurationManager instance
    """
    if environment == "development":
        config = ConfigurationFactory.create_development()
    elif environment == "production":
        config = ConfigurationFactory.create_production()
    elif environment == "testing":
        config = ConfigurationFactory.create_testing()
    else:
        config = ConfigurationFactory.create_default(config_file)
    
    set_config(config)
    return config


# Convenience functions for common configuration access
def get_api_config() -> Dict[str, Any]:
    """Get API configuration section."""
    return get_config().get_section('api')


def get_execution_config() -> Dict[str, Any]:
    """Get execution configuration section."""
    return get_config().get_section('execution')


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration section."""
    return get_config().get_section('logging')


def get_verification_config() -> Dict[str, Any]:
    """Get verification configuration section."""
    return get_config().get_section('verification')


# Context manager for temporary configuration changes
class ConfigContext:
    """Context manager for temporary configuration changes."""
    
    def __init__(self, **kwargs):
        self.changes = kwargs
        self.original_values = {}
        
    def __enter__(self):
        config = get_config()
        for path, value in self.changes.items():
            self.original_values[path] = config.get(path)
            config.set(path, value, validate=False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        config = get_config()
        for path, original_value in self.original_values.items():
            if original_value is not None:
                config.set(path, original_value, validate=False)
            # Note: Can't easily remove keys, so we leave them with None values


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    config = ConfigurationFactory.create_default()
    
    print("Configuration loaded successfully!")
    print(f"OpenAI model: {config.get('api.openai.model')}")
    print(f"Execution timeout: {config.get('execution.timeouts.code_execution_seconds')}")
    print(f"Log level: {config.get('logging.level')}")
    
    # Show environment variable info
    env_info = config.get_env_info()
    print(f"Environment overrides: {env_info}")
    
    # Test setting values
    with ConfigContext(**{'logging.level': 'DEBUG'}):
        print(f"Temporary log level: {config.get('logging.level')}")
    
    print(f"Original log level: {config.get('logging.level')}")