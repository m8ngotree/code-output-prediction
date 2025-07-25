"""
Comprehensive unit tests for the output verification system.

Tests all comparison types, normalization functions, scoring mechanisms,
and edge cases to ensure robust verification of code execution outputs.
"""

import json
import math
import pytest
import unittest
from typing import Any, Dict, List

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verifiers.output_verifier import (
    OutputVerifier,
    VerificationConfig,
    VerificationResult,
    ComparisonType,
    OutputNormalizer,
    OutputComparator
)


class TestVerificationConfig(unittest.TestCase):
    """Test verification configuration settings."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = VerificationConfig()
        
        # String settings
        self.assertFalse(config.case_sensitive)
        self.assertTrue(config.ignore_whitespace)
        self.assertTrue(config.normalize_unicode)
        self.assertTrue(config.strip_strings)
        
        # Numeric settings
        self.assertEqual(config.numeric_tolerance, 1e-6)
        self.assertEqual(config.relative_tolerance, 1e-9)
        self.assertTrue(config.allow_infinity)
        self.assertFalse(config.allow_nan)
        
        # Fuzzy matching
        self.assertEqual(config.fuzzy_threshold, 0.8)
        self.assertTrue(config.enable_partial_credit)
        self.assertEqual(config.min_partial_score, 0.1)


class TestOutputNormalizer(unittest.TestCase):
    """Test output normalization functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = VerificationConfig()
        self.normalizer = OutputNormalizer(self.config)
    
    def test_normalize_string_basic(self):
        """Test basic string normalization."""
        # Case normalization
        result = self.normalizer.normalize_string("Hello World")
        self.assertEqual(result, "hello world")
        
        # Whitespace normalization
        result = self.normalizer.normalize_string("  hello    world  ")
        self.assertEqual(result, "hello world")
        
        # Multiple whitespace types
        result = self.normalizer.normalize_string("hello\t\n  world")
        self.assertEqual(result, "hello world")
    
    def test_normalize_string_case_sensitive(self):
        """Test string normalization with case sensitivity."""
        config = VerificationConfig()
        config.case_sensitive = True
        normalizer = OutputNormalizer(config)
        
        result = normalizer.normalize_string("Hello World")
        self.assertEqual(result, "Hello World")
    
    def test_normalize_string_preserve_whitespace(self):
        """Test string normalization preserving whitespace."""
        config = VerificationConfig()
        config.ignore_whitespace = False
        config.strip_strings = False
        normalizer = OutputNormalizer(config)
        
        result = normalizer.normalize_string("  hello    world  ")
        self.assertEqual(result, "  hello    world  ")
    
    def test_normalize_number_integers(self):
        """Test number normalization with integers."""
        self.assertEqual(self.normalizer.normalize_number(42), 42.0)
        self.assertEqual(self.normalizer.normalize_number("42"), 42.0)
        self.assertEqual(self.normalizer.normalize_number(" 42 "), 42.0)
    
    def test_normalize_number_floats(self):
        """Test number normalization with floats."""
        self.assertEqual(self.normalizer.normalize_number(3.14), 3.14)
        self.assertEqual(self.normalizer.normalize_number("3.14"), 3.14)
        self.assertEqual(self.normalizer.normalize_number("3.14e2"), 314.0)
    
    def test_normalize_number_special_values(self):
        """Test number normalization with special values."""
        # Infinity
        self.assertTrue(math.isinf(self.normalizer.normalize_number("inf")))
        self.assertTrue(math.isinf(self.normalizer.normalize_number("infinity")))
        self.assertTrue(math.isinf(self.normalizer.normalize_number("+inf")))
        
        # Negative infinity
        result = self.normalizer.normalize_number("-inf")
        self.assertTrue(math.isinf(result) and result < 0)
        
        # NaN
        self.assertTrue(math.isnan(self.normalizer.normalize_number("nan")))
        self.assertTrue(math.isnan(self.normalizer.normalize_number("none")))
        self.assertTrue(math.isnan(self.normalizer.normalize_number("null")))
    
    def test_normalize_number_invalid(self):
        """Test number normalization with invalid input."""
        with self.assertRaises(ValueError):
            self.normalizer.normalize_number("not a number")
        
        with self.assertRaises(ValueError):
            self.normalizer.normalize_number([1, 2, 3])
    
    def test_normalize_list_from_string(self):
        """Test list normalization from string representations."""
        # JSON-like format
        result = self.normalizer.normalize_list("[1, 2, 3]")
        self.assertEqual(result, [1, 2, 3])
        
        # Python list format
        result = self.normalizer.normalize_list("['a', 'b', 'c']")
        self.assertEqual(result, ['a', 'b', 'c'])
        
        # Comma-separated values
        result = self.normalizer.normalize_list("a, b, c")
        self.assertEqual(result, ['a', 'b', 'c'])
        
        # Different delimiters
        result = self.normalizer.normalize_list("a;b;c")
        self.assertEqual(result, ['a', 'b', 'c'])
        
        result = self.normalizer.normalize_list("a|b|c")
        self.assertEqual(result, ['a', 'b', 'c'])
    
    def test_normalize_list_from_list(self):
        """Test list normalization from actual lists."""
        input_list = [1, 2, 3]
        result = self.normalizer.normalize_list(input_list)
        self.assertEqual(result, input_list)
    
    def test_normalize_list_from_iterable(self):
        """Test list normalization from iterables."""
        result = self.normalizer.normalize_list(range(3))
        self.assertEqual(result, [0, 1, 2])
        
        result = self.normalizer.normalize_list((1, 2, 3))
        self.assertEqual(result, [1, 2, 3])
    
    def test_normalize_json_from_string(self):
        """Test JSON normalization from string."""
        json_str = '{"name": "test", "value": 42}'
        result = self.normalizer.normalize_json(json_str)
        expected = {"name": "test", "value": 42}
        self.assertEqual(result, expected)
    
    def test_normalize_json_from_dict(self):
        """Test JSON normalization from dictionary."""
        input_dict = {"name": "test", "value": 42}
        result = self.normalizer.normalize_json(input_dict)
        self.assertEqual(result, input_dict)
    
    def test_normalize_json_invalid(self):
        """Test JSON normalization with invalid input."""
        with self.assertRaises(ValueError):
            self.normalizer.normalize_json("invalid json")
        
        with self.assertRaises(ValueError):
            self.normalizer.normalize_json([1, 2, 3])


class TestOutputComparator(unittest.TestCase):
    """Test output comparison functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = VerificationConfig()
        self.comparator = OutputComparator(self.config)
    
    def test_compare_exact_string_match(self):
        """Test exact string comparison with matches."""
        result = self.comparator.compare_exact_string("hello", "hello")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.comparison_type, ComparisonType.EXACT_STRING)
    
    def test_compare_exact_string_mismatch(self):
        """Test exact string comparison with mismatches."""
        result = self.comparator.compare_exact_string("hello", "world")
        self.assertFalse(result.is_correct)
        self.assertEqual(result.score, 0.0)
        self.assertIn("string_mismatch", result.mismatch_details["type"])
    
    def test_compare_exact_string_normalization(self):
        """Test exact string comparison with normalization."""
        # Case insensitive
        result = self.comparator.compare_exact_string("Hello", "hello")
        self.assertTrue(result.is_correct)
        
        # Whitespace normalization
        result = self.comparator.compare_exact_string("  hello  world  ", "hello world")
        self.assertTrue(result.is_correct)
    
    def test_compare_fuzzy_string_high_similarity(self):
        """Test fuzzy string comparison with high similarity."""
        result = self.comparator.compare_fuzzy_string("hello world", "hello word")
        self.assertTrue(result.is_correct)  # Should pass threshold
        self.assertGreater(result.score, 0.8)
    
    def test_compare_fuzzy_string_low_similarity(self):
        """Test fuzzy string comparison with low similarity."""
        result = self.comparator.compare_fuzzy_string("hello", "xyz")
        self.assertFalse(result.is_correct)
        self.assertLess(result.score, 0.8)
    
    def test_compare_fuzzy_string_partial_credit(self):
        """Test fuzzy string comparison partial credit."""
        result = self.comparator.compare_fuzzy_string("hello world", "hello")
        self.assertGreater(result.score, 0.0)
        self.assertIn("similarity_ratio", result.partial_credit_details)
    
    def test_compare_numeric_exact_match(self):
        """Test numeric comparison with exact matches."""
        result = self.comparator.compare_numeric(42, 42)
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)
        
        result = self.comparator.compare_numeric(3.14, 3.14)
        self.assertTrue(result.is_correct)
    
    def test_compare_numeric_within_tolerance(self):
        """Test numeric comparison within tolerance."""
        result = self.comparator.compare_numeric(1.0, 1.0000001)  # Within absolute tolerance
        self.assertTrue(result.is_correct)
        
        result = self.comparator.compare_numeric(1000000.0, 1000000.1)  # Within relative tolerance
        self.assertTrue(result.is_correct)
    
    def test_compare_numeric_outside_tolerance(self):
        """Test numeric comparison outside tolerance."""
        result = self.comparator.compare_numeric(1.0, 2.0)
        self.assertFalse(result.is_correct)
        self.assertEqual(result.score, 0.0)
    
    def test_compare_numeric_special_values(self):
        """Test numeric comparison with special values."""
        # Infinity
        result = self.comparator.compare_numeric(float('inf'), float('inf'))
        self.assertTrue(result.is_correct)
        
        # NaN (should not be equal by default)
        config = VerificationConfig()
        config.allow_nan = True
        comparator = OutputComparator(config)
        result = comparator.compare_numeric(float('nan'), float('nan'))
        self.assertTrue(result.is_correct)
    
    def test_compare_list_ordered_match(self):
        """Test ordered list comparison with matches."""
        result = self.comparator.compare_list([1, 2, 3], [1, 2, 3], ordered=True)
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)
    
    def test_compare_list_ordered_mismatch(self):
        """Test ordered list comparison with mismatches."""
        result = self.comparator.compare_list([1, 2, 3], [3, 2, 1], ordered=True)
        self.assertFalse(result.is_correct)
    
    def test_compare_list_unordered_match(self):
        """Test unordered list comparison with matches."""
        result = self.comparator.compare_list([1, 2, 3], [3, 1, 2], ordered=False)
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)
    
    def test_compare_list_partial_credit(self):
        """Test list comparison with partial credit."""
        config = VerificationConfig()
        config.partial_list_credit = True
        comparator = OutputComparator(config)
        
        result = comparator.compare_list([1, 2, 3], [1, 2, 4], ordered=True)
        self.assertGreater(result.score, 0.0)
        self.assertLess(result.score, 1.0)
    
    def test_compare_json_exact_match(self):
        """Test JSON comparison with exact matches."""
        json1 = {"name": "test", "value": 42}
        json2 = {"name": "test", "value": 42}
        result = self.comparator.compare_json(json1, json2)
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)
    
    def test_compare_json_key_order(self):
        """Test JSON comparison with different key orders."""
        json1 = {"a": 1, "b": 2}
        json2 = {"b": 2, "a": 1}
        result = self.comparator.compare_json(json1, json2)
        self.assertTrue(result.is_correct)  # Order shouldn't matter by default
    
    def test_compare_json_missing_keys(self):
        """Test JSON comparison with missing keys."""
        json1 = {"a": 1, "b": 2}
        json2 = {"a": 1}
        result = self.comparator.compare_json(json1, json2)
        self.assertFalse(result.is_correct)
        self.assertIn("Extra keys", result.mismatch_details["differences"][0])
    
    def test_compare_json_nested_structures(self):
        """Test JSON comparison with nested structures."""
        json1 = {"user": {"name": "test", "age": 25}, "active": True}
        json2 = {"user": {"name": "test", "age": 25}, "active": True}
        result = self.comparator.compare_json(json1, json2)
        self.assertTrue(result.is_correct)
    
    def test_compare_regex_match(self):
        """Test regex comparison with matches."""
        result = self.comparator.compare_regex("hello123", r"hello\d+")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)
        self.assertIn("matched_text", result.partial_credit_details)
    
    def test_compare_regex_no_match(self):
        """Test regex comparison with no matches."""
        result = self.comparator.compare_regex("hello", r"\d+")
        self.assertFalse(result.is_correct)
        self.assertEqual(result.score, 0.0)
        self.assertIn("regex_no_match", result.mismatch_details["type"])
    
    def test_compare_regex_invalid_pattern(self):
        """Test regex comparison with invalid pattern."""
        result = self.comparator.compare_regex("test", r"[invalid")
        self.assertFalse(result.is_correct)
        self.assertIn("regex_compile_error", result.mismatch_details["type"])


class TestOutputVerifier(unittest.TestCase):
    """Test main output verifier functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.verifier = OutputVerifier()
    
    def test_auto_detect_numeric(self):
        """Test automatic detection of numeric comparison."""
        result = self.verifier.verify_output(42, 42)
        self.assertEqual(result.comparison_type, ComparisonType.NUMERIC)
        self.assertTrue(result.is_correct)
    
    def test_auto_detect_list(self):
        """Test automatic detection of list comparison."""
        result = self.verifier.verify_output([1, 2, 3], [1, 2, 3])
        self.assertEqual(result.comparison_type, ComparisonType.LIST_ORDERED)
        self.assertTrue(result.is_correct)
    
    def test_auto_detect_json(self):
        """Test automatic detection of JSON comparison."""
        result = self.verifier.verify_output({"key": "value"}, {"key": "value"})
        self.assertEqual(result.comparison_type, ComparisonType.JSON_OBJECT)
        self.assertTrue(result.is_correct)
    
    def test_auto_detect_string(self):
        """Test automatic detection of string comparison."""
        result = self.verifier.verify_output("hello", "hello")
        self.assertEqual(result.comparison_type, ComparisonType.FUZZY_STRING)
        self.assertTrue(result.is_correct)
    
    def test_explicit_comparison_type(self):
        """Test using explicit comparison type."""
        result = self.verifier.verify_output("hello", "hello", ComparisonType.EXACT_STRING)
        self.assertEqual(result.comparison_type, ComparisonType.EXACT_STRING)
        self.assertTrue(result.is_correct)
    
    def test_verify_batch_basic(self):
        """Test batch verification with basic cases."""
        predictions = [
            {"predicted": 42, "actual": 42},
            {"predicted": "hello", "actual": "hello"},
            {"predicted": [1, 2], "actual": [1, 2]},
            {"predicted": {"key": "value"}, "actual": {"key": "value"}}
        ]
        
        results = self.verifier.verify_batch(predictions)
        
        self.assertEqual(results["statistics"]["total_predictions"], 4)
        self.assertEqual(results["statistics"]["correct_predictions"], 4)
        self.assertEqual(results["statistics"]["accuracy"], 1.0)
        self.assertEqual(len(results["results"]), 4)
    
    def test_verify_batch_mixed_results(self):
        """Test batch verification with mixed success/failure."""
        predictions = [
            {"predicted": 42, "actual": 42},           # Correct
            {"predicted": "hello", "actual": "world"}, # Incorrect
            {"predicted": [1, 2], "actual": [1, 2]},   # Correct
            {"predicted": {"a": 1}, "actual": {"b": 2}} # Incorrect
        ]
        
        results = self.verifier.verify_batch(predictions)
        
        self.assertEqual(results["statistics"]["total_predictions"], 4)
        self.assertEqual(results["statistics"]["correct_predictions"], 2)
        self.assertEqual(results["statistics"]["accuracy"], 0.5)
    
    def test_verify_batch_with_explicit_types(self):
        """Test batch verification with explicit comparison types."""
        predictions = [
            {"predicted": "42", "actual": "42", "type": "exact_string"},
            {"predicted": "42", "actual": 42, "type": "numeric"}
        ]
        
        results = self.verifier.verify_batch(predictions)
        
        # Both should be correct but use different comparison types
        self.assertEqual(results["statistics"]["correct_predictions"], 2)
        self.assertIn("exact_string", results["statistics"]["comparison_type_counts"])
        self.assertIn("numeric", results["statistics"]["comparison_type_counts"])
    
    def test_verify_batch_error_handling(self):
        """Test batch verification error handling."""
        predictions = [
            {"predicted": 42, "actual": 42},           # Valid
            {"missing_key": "value"},                  # Invalid - missing keys
            {"predicted": "test", "actual": "test"}    # Valid
        ]
        
        results = self.verifier.verify_batch(predictions)
        
        self.assertEqual(results["statistics"]["total_predictions"], 3)
        self.assertEqual(results["statistics"]["error_count"], 1)
        self.assertEqual(results["statistics"]["correct_predictions"], 2)
    
    def test_generate_verification_report(self):
        """Test verification report generation."""
        predictions = [
            {"predicted": 42, "actual": 42},
            {"predicted": "hello", "actual": "world"},
            {"predicted": [1, 2], "actual": [1, 2, 3]}
        ]
        
        batch_results = self.verifier.verify_batch(predictions)
        report = self.verifier.generate_verification_report(batch_results)
        
        self.assertIn("summary", report)
        self.assertIn("comparison_type_analysis", report)
        self.assertIn("failure_patterns", report)
        self.assertIn("configuration", report)
        
        # Check summary statistics
        self.assertEqual(report["summary"]["total_predictions"], 3)
        self.assertLess(report["summary"]["accuracy"], 1.0)  # Some failures expected


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.verifier = OutputVerifier()
    
    def test_empty_values(self):
        """Test comparison with empty values."""
        # Empty strings
        result = self.verifier.verify_output("", "")
        self.assertTrue(result.is_correct)
        
        # Empty lists
        result = self.verifier.verify_output([], [])
        self.assertTrue(result.is_correct)
        
        # Empty dictionaries
        result = self.verifier.verify_output({}, {})
        self.assertTrue(result.is_correct)
    
    def test_none_values(self):
        """Test comparison with None values."""
        result = self.verifier.verify_output(None, None)
        self.assertTrue(result.is_correct)
        
        result = self.verifier.verify_output(None, "not none")
        self.assertFalse(result.is_correct)
    
    def test_mixed_types(self):
        """Test comparison with mixed data types."""
        # String vs Number
        result = self.verifier.verify_output("42", 42, ComparisonType.NUMERIC)
        self.assertTrue(result.is_correct)
        
        # List vs String representation
        result = self.verifier.verify_output("[1, 2, 3]", [1, 2, 3], ComparisonType.LIST_ORDERED)
        self.assertTrue(result.is_correct)
    
    def test_large_data_structures(self):
        """Test comparison with large data structures."""
        large_list = list(range(1000))
        result = self.verifier.verify_output(large_list, large_list)
        self.assertTrue(result.is_correct)
        
        large_dict = {f"key_{i}": f"value_{i}" for i in range(100)}
        result = self.verifier.verify_output(large_dict, large_dict)
        self.assertTrue(result.is_correct)
    
    def test_unicode_handling(self):
        """Test Unicode string handling."""
        unicode_str1 = "héllo wörld"
        unicode_str2 = "héllo wörld"
        result = self.verifier.verify_output(unicode_str1, unicode_str2)
        self.assertTrue(result.is_correct)
        
        # Test normalization
        unicode_str3 = "café"  # Different Unicode representations
        unicode_str4 = "café"
        result = self.verifier.verify_output(unicode_str3, unicode_str4)
        self.assertTrue(result.is_correct)
    
    def test_floating_point_precision(self):
        """Test floating point precision issues."""
        # Values that are close but not exactly equal due to floating point
        result = self.verifier.verify_output(0.1 + 0.2, 0.3)
        self.assertTrue(result.is_correct)  # Should pass with tolerance
        
        # Very small numbers
        result = self.verifier.verify_output(1e-10, 1.1e-10)
        self.assertTrue(result.is_correct)  # Within relative tolerance
    
    def test_deeply_nested_structures(self):
        """Test deeply nested data structures."""
        nested_dict = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": [1, 2, {"nested_list": [3, 4, 5]}]
                    }
                }
            }
        }
        
        result = self.verifier.verify_output(nested_dict, nested_dict)
        self.assertTrue(result.is_correct)
    
    def test_special_float_values(self):
        """Test special floating point values."""
        # Test infinity
        result = self.verifier.verify_output(float('inf'), float('inf'))
        self.assertTrue(result.is_correct)
        
        # Test negative infinity
        result = self.verifier.verify_output(float('-inf'), float('-inf'))
        self.assertTrue(result.is_correct)
        
        # Test different infinities
        result = self.verifier.verify_output(float('inf'), float('-inf'))
        self.assertFalse(result.is_correct)
    
    def test_unsupported_comparison_type(self):
        """Test unsupported comparison type handling."""
        with self.assertRaises(ValueError):
            # This should raise an error for invalid comparison type
            self.verifier.verify_output("test", "test", "invalid_type")


class TestConfigurationOptions(unittest.TestCase):
    """Test different configuration options."""
    
    def test_case_sensitive_configuration(self):
        """Test case-sensitive string comparison configuration."""
        config = VerificationConfig()
        config.case_sensitive = True
        verifier = OutputVerifier(config)
        
        result = verifier.verify_output("Hello", "hello")
        self.assertFalse(result.is_correct)
    
    def test_strict_whitespace_configuration(self):
        """Test strict whitespace handling configuration."""
        config = VerificationConfig()
        config.ignore_whitespace = False
        config.strip_strings = False
        verifier = OutputVerifier(config)
        
        result = verifier.verify_output("hello world", "hello  world")
        self.assertFalse(result.is_correct)
    
    def test_numeric_tolerance_configuration(self):
        """Test numeric tolerance configuration."""
        config = VerificationConfig()
        config.numeric_tolerance = 1e-3  # Larger tolerance
        verifier = OutputVerifier(config)
        
        result = verifier.verify_output(1.0, 1.0005)  # Should pass with larger tolerance
        self.assertTrue(result.is_correct)
    
    def test_fuzzy_threshold_configuration(self):
        """Test fuzzy matching threshold configuration."""
        config = VerificationConfig()
        config.fuzzy_threshold = 0.9  # Higher threshold
        verifier = OutputVerifier(config)
        
        result = verifier.verify_output("hello world", "hello word")
        # Might fail with higher threshold
        self.assertLessEqual(result.score, 1.0)
    
    def test_partial_credit_disabled(self):
        """Test with partial credit disabled."""
        config = VerificationConfig()
        config.enable_partial_credit = False
        verifier = OutputVerifier(config)
        
        result = verifier.verify_output("hello world", "hello", ComparisonType.FUZZY_STRING)
        # Should be binary: either 0 or 1, no partial credit
        self.assertIn(result.score, [0.0, 1.0])


if __name__ == "__main__":
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run all tests
    unittest.main(verbosity=2)