"""
Comprehensive seed manager for diverse code generation scenarios.

This module provides the SeedManager class which manages application domains
and programming concepts used to generate varied and challenging code samples.
It supports fallback mechanisms and extensible seed data sources.
"""

import json
import random
from pathlib import Path
from typing import List


class SeedManager:
    """
    Manages seed data for diversified code generation.
    
    This class loads and provides random access to application domains and
    programming concepts, enabling the generation of varied code samples
    across different problem domains and complexity levels.
    
    Attributes:
        seeds_dir (Path): Directory containing seed data files
        applications (List[str]): Available application domains
        concepts (List[str]): Available programming concepts
    """
    
    def __init__(self, seeds_dir: str = "data/seeds"):
        """
        Initialize the seed manager with data from specified directory.
        
        Args:
            seeds_dir (str): Path to directory containing seed JSON files
                           (default: "data/seeds")
        """
        self.seeds_dir = Path(seeds_dir)
        self.applications = self._load_seeds("applications.json")
        self.concepts = self._load_seeds("concepts.json")
    
    def _load_seeds(self, filename: str) -> List[str]:
        """
        Load seed data from JSON file with error handling.
        
        Args:
            filename (str): Name of JSON file to load (e.g., "applications.json")
            
        Returns:
            List[str]: List of seed values, empty list if file doesn't exist
            
        Raises:
            json.JSONDecodeError: If file contains invalid JSON
        """
        file_path = self.seeds_dir / filename
        if not file_path.exists():
            return []
        
        with open(file_path) as f:
            data = json.load(f)
            seed_type = filename.replace('.json', '')
            return data.get(seed_type, [])
    
    def get_random_application(self) -> str:
        """
        Get a random application domain for code generation.
        
        Returns:
            str: Random application domain, defaults to "general" if none available
        """
        return random.choice(self.applications) if self.applications else "general"
    
    def get_random_concept(self) -> str:
        """
        Get a random programming concept for code generation.
        
        Returns:
            str: Random programming concept, defaults to "functions" if none available
        """
        return random.choice(self.concepts) if self.concepts else "functions"