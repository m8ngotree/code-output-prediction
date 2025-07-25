"""
Unit tests for the SeedManager class.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.seeds.seed_manager import SeedManager


class TestSeedManager(unittest.TestCase):
    """Test cases for SeedManager functionality."""
    
    def setUp(self):
        """Set up test fixtures with temporary directory and sample data."""
        self.temp_dir = tempfile.mkdtemp()
        self.seeds_dir = Path(self.temp_dir) / "seeds"
        self.seeds_dir.mkdir()
        
        # Create sample seed files
        self.sample_applications = ["web scraping", "data analysis", "game development"]
        self.sample_concepts = ["recursion", "dynamic programming", "graph algorithms"]
        
        # Write sample data to JSON files
        with open(self.seeds_dir / "applications.json", 'w') as f:
            json.dump({"applications": self.sample_applications}, f)
        
        with open(self.seeds_dir / "concepts.json", 'w') as f:
            json.dump({"concepts": self.sample_concepts}, f)
        
        self.manager = SeedManager(seeds_dir=self.seeds_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test SeedManager initialization."""
        self.assertEqual(self.manager.seeds_dir, self.seeds_dir)
        self.assertIn("applications", self.manager.seeds_data)
        self.assertIn("concepts", self.manager.seeds_data)
    
    def test_load_seeds_from_file(self):
        """Test loading seeds from JSON files."""
        self.assertEqual(
            self.manager.seeds_data["applications"], 
            self.sample_applications
        )
        self.assertEqual(
            self.manager.seeds_data["concepts"], 
            self.sample_concepts
        )
    
    def test_get_all_seeds(self):
        """Test getting all seeds of specific types."""
        apps = self.manager.get_all_seeds("applications")
        concepts = self.manager.get_all_seeds("concepts")
        
        self.assertEqual(apps, self.sample_applications)
        self.assertEqual(concepts, self.sample_concepts)
        
        # Test that returned list is a copy
        apps.append("new app")
        self.assertNotEqual(
            len(self.manager.seeds_data["applications"]), 
            len(apps)
        )
    
    def test_get_all_seeds_invalid_type(self):
        """Test error handling for invalid seed types."""
        with self.assertRaises(ValueError):
            self.manager.get_all_seeds("invalid_type")
    
    def test_sample_seeds_without_replacement(self):
        """Test sampling seeds without replacement."""
        sampled = self.manager.sample_seeds("applications", 2, replace=False)
        
        self.assertEqual(len(sampled), 2)
        self.assertEqual(len(set(sampled)), 2)  # No duplicates
        
        for seed in sampled:
            self.assertIn(seed, self.sample_applications)
    
    def test_sample_seeds_with_replacement(self):
        """Test sampling seeds with replacement."""
        sampled = self.manager.sample_seeds("applications", 5, replace=True)
        
        self.assertEqual(len(sampled), 5)
        for seed in sampled:
            self.assertIn(seed, self.sample_applications)
    
    def test_sample_seeds_too_many_without_replacement(self):
        """Test error when sampling too many seeds without replacement."""
        with self.assertRaises(ValueError):
            self.manager.sample_seeds("applications", 10, replace=False)
    
    def test_sample_seeds_invalid_n(self):
        """Test error handling for invalid sample sizes."""
        with self.assertRaises(ValueError):
            self.manager.sample_seeds("applications", 0)
        
        with self.assertRaises(ValueError):
            self.manager.sample_seeds("applications", -1)
    
    def test_sample_seeds_invalid_type(self):
        """Test error handling for invalid seed types in sampling."""
        with self.assertRaises(ValueError):
            self.manager.sample_seeds("invalid_type", 1)
    
    def test_add_seed(self):
        """Test adding a single seed."""
        new_seed = "blockchain development"
        initial_count = len(self.manager.seeds_data["applications"])
        
        self.manager.add_seed("applications", new_seed)
        
        self.assertEqual(
            len(self.manager.seeds_data["applications"]), 
            initial_count + 1
        )
        self.assertIn(new_seed, self.manager.seeds_data["applications"])
    
    def test_add_duplicate_seed(self):
        """Test error handling when adding duplicate seeds."""
        with self.assertRaises(ValueError):
            self.manager.add_seed("applications", "web scraping")
    
    def test_add_seed_invalid_type(self):
        """Test error handling for invalid seed types when adding."""
        with self.assertRaises(ValueError):
            self.manager.add_seed("invalid_type", "new seed")
    
    def test_add_seeds_multiple(self):
        """Test adding multiple seeds at once."""
        new_seeds = ["blockchain development", "mobile app development"]
        initial_count = len(self.manager.seeds_data["applications"])
        
        self.manager.add_seeds("applications", new_seeds)
        
        self.assertEqual(
            len(self.manager.seeds_data["applications"]), 
            initial_count + 2
        )
        for seed in new_seeds:
            self.assertIn(seed, self.manager.seeds_data["applications"])
    
    def test_add_seeds_with_duplicate(self):
        """Test error handling when adding multiple seeds with duplicates."""
        new_seeds = ["blockchain development", "web scraping"]  # web scraping exists
        
        with self.assertRaises(ValueError):
            self.manager.add_seeds("applications", new_seeds)
        
        # Verify no seeds were added
        self.assertNotIn("blockchain development", self.manager.seeds_data["applications"])
    
    def test_save_seeds(self):
        """Test saving seeds to files."""
        # Add a new seed
        self.manager.add_seed("applications", "blockchain development")
        
        # Save to file
        self.manager.save_seeds("applications")
        
        # Verify file was updated
        with open(self.seeds_dir / "applications.json", 'r') as f:
            data = json.load(f)
        
        self.assertIn("blockchain development", data["applications"])
    
    def test_save_all_seeds(self):
        """Test saving all seed types."""
        # Add seeds to both types
        self.manager.add_seed("applications", "blockchain development")
        self.manager.add_seed("concepts", "machine learning algorithms")
        
        # Save all
        self.manager.save_seeds()
        
        # Verify both files were updated
        with open(self.seeds_dir / "applications.json", 'r') as f:
            apps_data = json.load(f)
        
        with open(self.seeds_dir / "concepts.json", 'r') as f:
            concepts_data = json.load(f)
        
        self.assertIn("blockchain development", apps_data["applications"])
        self.assertIn("machine learning algorithms", concepts_data["concepts"])
    
    def test_get_seed_count(self):
        """Test getting seed counts."""
        self.assertEqual(
            self.manager.get_seed_count("applications"), 
            len(self.sample_applications)
        )
        self.assertEqual(
            self.manager.get_seed_count("concepts"), 
            len(self.sample_concepts)
        )
    
    def test_get_seed_count_invalid_type(self):
        """Test error handling for invalid seed types in count."""
        with self.assertRaises(ValueError):
            self.manager.get_seed_count("invalid_type")
    
    def test_get_available_types(self):
        """Test getting available seed types."""
        types = self.manager.get_available_types()
        self.assertIn("applications", types)
        self.assertIn("concepts", types)
        self.assertEqual(len(types), 2)
    
    def test_remove_seed(self):
        """Test removing seeds."""
        # Remove existing seed
        result = self.manager.remove_seed("applications", "web scraping")
        self.assertTrue(result)
        self.assertNotIn("web scraping", self.manager.seeds_data["applications"])
        
        # Try to remove non-existent seed
        result = self.manager.remove_seed("applications", "non-existent")
        self.assertFalse(result)
    
    def test_remove_seed_invalid_type(self):
        """Test error handling for invalid seed types in removal."""
        with self.assertRaises(ValueError):
            self.manager.remove_seed("invalid_type", "seed")
    
    def test_clear_seeds(self):
        """Test clearing all seeds of a type."""
        self.manager.clear_seeds("applications")
        self.assertEqual(len(self.manager.seeds_data["applications"]), 0)
    
    def test_clear_seeds_invalid_type(self):
        """Test error handling for invalid seed types in clearing."""
        with self.assertRaises(ValueError):
            self.manager.clear_seeds("invalid_type")
    
    def test_repr(self):
        """Test string representation of SeedManager."""
        repr_str = repr(self.manager)
        self.assertIn("SeedManager", repr_str)
        self.assertIn(str(self.seeds_dir), repr_str)
        self.assertIn("applications", repr_str)
        self.assertIn("concepts", repr_str)
    
    def test_missing_seed_file(self):
        """Test handling of missing seed files."""
        # Create manager with empty directory
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        
        with patch('builtins.print') as mock_print:
            manager = SeedManager(seeds_dir=empty_dir)
            
            # Should create empty lists for missing files
            self.assertEqual(manager.seeds_data.get("applications", []), [])
            self.assertEqual(manager.seeds_data.get("concepts", []), [])
            
            # Should print warnings
            self.assertTrue(mock_print.called)
    
    def test_invalid_json_file(self):
        """Test handling of invalid JSON files."""
        # Create invalid JSON file
        invalid_file = self.seeds_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        with self.assertRaises(ValueError):
            # Try to load from invalid file
            manager = SeedManager(seeds_dir=self.seeds_dir)
            manager._load_seeds_from_file("invalid")
    
    def test_default_seeds_directory(self):
        """Test default seeds directory resolution."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch.object(SeedManager, '_load_all_seeds'):
            manager = SeedManager()
            
            # Should resolve to project_root/data/seeds
            self.assertTrue(str(manager.seeds_dir).endswith("data/seeds"))


if __name__ == "__main__":
    unittest.main()