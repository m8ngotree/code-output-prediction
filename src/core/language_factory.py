"""
Comprehensive factory for creating language-specific generators and executors.

This module provides the LanguageFactory class which abstracts the creation
of language-specific code generators and executors, supporting multiple
programming languages with a unified interface.
"""

from typing import Tuple, Any

# Import generators and executors dynamically to avoid circular imports


class LanguageFactory:
    """
    Factory for creating language-specific code generation and execution components.
    
    This factory provides a centralized way to instantiate the appropriate
    generator and executor pairs for different programming languages,
    maintaining consistency and extensibility across the system.
    """
    
    SUPPORTED_LANGUAGES = ["python-library"]
    
    @staticmethod
    def create_generator_and_executor(language: str, api_key: str) -> Tuple[Any, Any]:
        """
        Create appropriate generator and executor instances for the specified language.
        
        Args:
            language (str): Programming language (case-insensitive)
                          Supported: python-library
            api_key (str): OpenAI API key for code generation
            
        Returns:
            Tuple[Any, Any]: (generator, executor) pair for the language
            
        Raises:
            ValueError: If the specified language is not supported
        """
        language = language.lower()
        
        if language == "python-library":
            from ..generators.library_task_factory import LibraryTaskFactory
            from ..executors.library_executor import LibraryCodeExecutor
            return LibraryTaskFactory(api_key), LibraryCodeExecutor()
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