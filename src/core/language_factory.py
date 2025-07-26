"""
Comprehensive factory for creating language-specific generators and executors.

This module provides the LanguageFactory class which abstracts the creation
of language-specific code generators and executors, supporting multiple
programming languages with a unified interface.
"""

from typing import Tuple, Any

from ..generators.code_generator import CodeGenerator
from ..executors.executor import CodeExecutor
from ..generators.js_generator import JSGenerator
from ..executors.js_executor import JSExecutor
from ..generators.rust_generator import RustGenerator
from ..executors.rust_executor import RustExecutor
from ..generators.cpp_generator import CppGenerator
from ..executors.cpp_executor import CppExecutor


class LanguageFactory:
    """
    Factory for creating language-specific code generation and execution components.
    
    This factory provides a centralized way to instantiate the appropriate
    generator and executor pairs for different programming languages,
    maintaining consistency and extensibility across the system.
    """
    
    SUPPORTED_LANGUAGES = ["python", "javascript", "rust", "cpp"]
    
    @staticmethod
    def create_generator_and_executor(language: str, api_key: str) -> Tuple[Any, Any]:
        """
        Create appropriate generator and executor instances for the specified language.
        
        Args:
            language (str): Programming language (case-insensitive)
                          Supported: python, javascript/js, rust, cpp/c++
            api_key (str): OpenAI API key for code generation
            
        Returns:
            Tuple[Any, Any]: (generator, executor) pair for the language
            
        Raises:
            ValueError: If the specified language is not supported
        """
        language = language.lower()
        
        if language == "python":
            return CodeGenerator(api_key), CodeExecutor()
        elif language == "javascript" or language == "js":
            return JSGenerator(api_key), JSExecutor()
        elif language == "rust":
            return RustGenerator(api_key), RustExecutor()
        elif language == "cpp" or language == "c++":
            return CppGenerator(api_key), CppExecutor()
        else:
            raise ValueError(f"Unsupported language: {language}. Supported: {LanguageFactory.SUPPORTED_LANGUAGES}")
    
    @staticmethod
    def get_supported_languages() -> list:
        """
        Get a list of all supported programming languages.
        
        Returns:
            list: Copy of supported languages list to prevent external modification
        """
        return LanguageFactory.SUPPORTED_LANGUAGES.copy()