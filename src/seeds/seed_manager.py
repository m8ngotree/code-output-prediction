"""
Seed Management System for Code Output Prediction

This module provides functionality to manage seed data for generating
synthetic code examples based on different applications and concepts.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union


class SeedManager:
    """
    Manages seed data for code generation, supporting applications and concepts.
    
    The SeedManager handles loading, sampling, and managing seed data from JSON files
    stored in the data/seeds/ directory.
    """
    
    def __init__(self, seeds_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the SeedManager.
        
        Args:
            seeds_dir: Path to the seeds directory. Defaults to 'data/seeds/'
        """
        if seeds_dir is None:
            # Default to data/seeds/ relative to project root
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            self.seeds_dir = project_root / "data" / "seeds"
        else:
            self.seeds_dir = Path(seeds_dir)
        
        self.seeds_data: Dict[str, List[str]] = {}
        self._load_all_seeds()
    
    def _load_all_seeds(self) -> None:
        """Load all seed data from JSON files in the seeds directory."""
        if not self.seeds_dir.exists():
            raise FileNotFoundError(f"Seeds directory not found: {self.seeds_dir}")
        
        # Load applications and concepts
        for seed_type in ["applications", "concepts"]:
            self._load_seeds_from_file(seed_type)
    
    def _load_seeds_from_file(self, seed_type: str) -> None:
        """
        Load seeds from a specific JSON file.
        
        Args:
            seed_type: Type of seeds to load ('applications' or 'concepts')
        """
        file_path = self.seeds_dir / f"{seed_type}.json"
        
        if not file_path.exists():
            print(f"Warning: {file_path} not found. Creating empty seed list.")
            self.seeds_data[seed_type] = []
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.seeds_data[seed_type] = data.get(seed_type, [])
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
    
    def get_all_seeds(self, seed_type: str) -> List[str]:
        """
        Get all seeds of a specific type.
        
        Args:
            seed_type: Type of seeds to retrieve ('applications' or 'concepts')
            
        Returns:
            List of all seeds of the specified type
            
        Raises:
            ValueError: If seed_type is not supported
        """
        if seed_type not in self.seeds_data:
            raise ValueError(f"Unsupported seed type: {seed_type}. "
                           f"Available types: {list(self.seeds_data.keys())}")
        
        return self.seeds_data[seed_type].copy()
    
    def sample_seeds(self, seed_type: str, n: int, replace: bool = False) -> List[str]:
        """
        Randomly sample N seeds from a specific category.
        
        Args:
            seed_type: Type of seeds to sample from ('applications' or 'concepts')
            n: Number of seeds to sample
            replace: Whether to allow sampling with replacement
            
        Returns:
            List of randomly sampled seeds
            
        Raises:
            ValueError: If seed_type is not supported or n is invalid
        """
        if seed_type not in self.seeds_data:
            raise ValueError(f"Unsupported seed type: {seed_type}. "
                           f"Available types: {list(self.seeds_data.keys())}")
        
        available_seeds = self.seeds_data[seed_type]
        
        if n <= 0:
            raise ValueError("Number of samples must be positive")
        
        if not replace and n > len(available_seeds):
            raise ValueError(f"Cannot sample {n} seeds without replacement from "
                           f"{len(available_seeds)} available seeds")
        
        if replace:
            return random.choices(available_seeds, k=n)
        else:
            return random.sample(available_seeds, k=n)
    
    def add_seed(self, seed_type: str, seed: str) -> None:
        """
        Add a new seed to a specific category.
        
        Args:
            seed_type: Type of seed to add to ('applications' or 'concepts')
            seed: The seed string to add
            
        Raises:
            ValueError: If seed_type is not supported or seed already exists
        """
        if seed_type not in self.seeds_data:
            raise ValueError(f"Unsupported seed type: {seed_type}. "
                           f"Available types: {list(self.seeds_data.keys())}")
        
        if seed in self.seeds_data[seed_type]:
            raise ValueError(f"Seed '{seed}' already exists in {seed_type}")
        
        self.seeds_data[seed_type].append(seed)
    
    def add_seeds(self, seed_type: str, seeds: List[str]) -> None:
        """
        Add multiple new seeds to a specific category.
        
        Args:
            seed_type: Type of seeds to add to ('applications' or 'concepts')
            seeds: List of seed strings to add
            
        Raises:
            ValueError: If seed_type is not supported or any seed already exists
        """
        if seed_type not in self.seeds_data:
            raise ValueError(f"Unsupported seed type: {seed_type}. "
                           f"Available types: {list(self.seeds_data.keys())}")
        
        # Check for duplicates before adding any
        existing_seeds = set(self.seeds_data[seed_type])
        for seed in seeds:
            if seed in existing_seeds:
                raise ValueError(f"Seed '{seed}' already exists in {seed_type}")
        
        # Add all seeds
        self.seeds_data[seed_type].extend(seeds)
    
    def save_seeds(self, seed_type: Optional[str] = None) -> None:
        """
        Save updated seeds back to JSON files.
        
        Args:
            seed_type: Specific seed type to save. If None, saves all types.
        """
        seed_types_to_save = [seed_type] if seed_type else list(self.seeds_data.keys())
        
        for stype in seed_types_to_save:
            if stype not in self.seeds_data:
                raise ValueError(f"Unsupported seed type: {stype}")
            
            file_path = self.seeds_dir / f"{stype}.json"
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save data
            data = {stype: sorted(self.seeds_data[stype])}  # Sort for consistency
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_seed_count(self, seed_type: str) -> int:
        """
        Get the number of seeds in a specific category.
        
        Args:
            seed_type: Type of seeds to count
            
        Returns:
            Number of seeds in the category
        """
        if seed_type not in self.seeds_data:
            raise ValueError(f"Unsupported seed type: {seed_type}")
        
        return len(self.seeds_data[seed_type])
    
    def get_available_types(self) -> List[str]:
        """
        Get list of available seed types.
        
        Returns:
            List of available seed types
        """
        return list(self.seeds_data.keys())
    
    def remove_seed(self, seed_type: str, seed: str) -> bool:
        """
        Remove a seed from a specific category.
        
        Args:
            seed_type: Type of seed to remove from
            seed: The seed string to remove
            
        Returns:
            True if seed was removed, False if it didn't exist
            
        Raises:
            ValueError: If seed_type is not supported
        """
        if seed_type not in self.seeds_data:
            raise ValueError(f"Unsupported seed type: {seed_type}")
        
        try:
            self.seeds_data[seed_type].remove(seed)
            return True
        except ValueError:
            return False
    
    def clear_seeds(self, seed_type: str) -> None:
        """
        Clear all seeds from a specific category.
        
        Args:
            seed_type: Type of seeds to clear
            
        Raises:
            ValueError: If seed_type is not supported
        """
        if seed_type not in self.seeds_data:
            raise ValueError(f"Unsupported seed type: {seed_type}")
        
        self.seeds_data[seed_type].clear()
    
    def __repr__(self) -> str:
        """String representation of the SeedManager."""
        counts = {stype: len(seeds) for stype, seeds in self.seeds_data.items()}
        return f"SeedManager(seeds_dir='{self.seeds_dir}', counts={counts})"