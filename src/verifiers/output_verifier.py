"""
Output Verification System for Code Execution Results.

This module provides comprehensive functionality to compare predicted outputs
with actual execution results, supporting various data types and comparison methods.
"""

import json
import math
import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import logging


class ComparisonType(Enum):
    """Types of output comparison methods."""
    EXACT_STRING = "exact_string"
    NUMERIC = "numeric"
    LIST_ORDERED = "list_ordered"
    LIST_UNORDERED = "list_unordered"
    JSON_OBJECT = "json_object"
    REGEX_PATTERN = "regex_pattern"
    FUZZY_STRING = "fuzzy_string"
    AUTO_DETECT = "auto_detect"


class VerificationResult:
    """Represents the result of a single verification comparison."""
    
    def __init__(self):
        self.is_correct: bool = False
        self.score: float = 0.0
        self.confidence: float = 0.0
        self.comparison_type: ComparisonType = ComparisonType.EXACT_STRING
        self.predicted_value: Any = None
        self.actual_value: Any = None
        self.normalized_predicted: Any = None
        self.normalized_actual: Any = None
        self.mismatch_details: Dict[str, Any] = {}
        self.partial_credit_details: Dict[str, Any] = {}
        self.timestamp: str = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "is_correct": self.is_correct,
            "score": self.score,
            "confidence": self.confidence,
            "comparison_type": self.comparison_type.value,
            "predicted_value": self.predicted_value,
            "actual_value": self.actual_value,
            "normalized_predicted": self.normalized_predicted,
            "normalized_actual": self.normalized_actual,
            "mismatch_details": self.mismatch_details,
            "partial_credit_details": self.partial_credit_details,
            "timestamp": self.timestamp
        }


class VerificationConfig:
    """Configuration for output verification settings."""
    
    def __init__(self, global_config=None):
        # Import here to avoid circular imports
        try:
            from ..utils.config import get_config
            config = global_config or get_config()
        except ImportError:
            config = None
        
        # String comparison settings
        self.case_sensitive: bool = config.get('verification.formatting.case_sensitive', False) if config else False
        self.ignore_whitespace: bool = config.get('verification.formatting.ignore_whitespace', True) if config else True
        self.normalize_unicode: bool = config.get('verification.formatting.normalize_unicode', True) if config else True
        self.strip_strings: bool = config.get('verification.formatting.strip_strings', True) if config else True
        
        # Numeric comparison settings
        self.numeric_tolerance: float = config.get('verification.tolerances.numeric_absolute', 1e-6) if config else 1e-6
        self.relative_tolerance: float = config.get('verification.tolerances.numeric_relative', 1e-9) if config else 1e-9
        self.allow_infinity: bool = True
        self.allow_nan: bool = False
        
        # List comparison settings
        self.list_element_tolerance: float = config.get('verification.tolerances.numeric_absolute', 1e-6) if config else 1e-6
        self.partial_list_credit: bool = config.get('verification.scoring.enable_partial_credit', True) if config else True
        self.list_order_matters: bool = True
        
        # JSON comparison settings
        self.json_key_order_matters: bool = False
        self.json_partial_credit: bool = config.get('verification.scoring.enable_partial_credit', True) if config else True
        self.json_type_strict: bool = False
        
        # Fuzzy matching settings
        self.fuzzy_threshold: float = config.get('verification.tolerances.fuzzy_string_threshold', 0.8) if config else 0.8
        self.enable_partial_credit: bool = config.get('verification.scoring.enable_partial_credit', True) if config else True
        self.min_partial_score: float = config.get('verification.scoring.min_partial_score', 0.1) if config else 0.1
        
        # Regex settings
        self.regex_flags: int = re.IGNORECASE | re.MULTILINE
        self.regex_timeout: float = 1.0


class OutputNormalizer:
    """Handles normalization of different output types."""
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def normalize_string(self, value: str) -> str:
        """Normalize string value according to configuration."""
        if not isinstance(value, str):
            value = str(value)
        
        # Unicode normalization
        if self.config.normalize_unicode:
            value = unicodedata.normalize('NFKC', value)
        
        # Case normalization
        if not self.config.case_sensitive:
            value = value.lower()
        
        # Whitespace handling
        if self.config.ignore_whitespace:
            # Normalize whitespace but preserve structure
            value = re.sub(r'\s+', ' ', value)
        
        # Strip whitespace
        if self.config.strip_strings:
            value = value.strip()
        
        return value
    
    def normalize_number(self, value: Union[int, float, str]) -> float:
        """Normalize numeric value."""
        if isinstance(value, str):
            # Try to parse string as number
            value = value.strip()
            
            # Handle special cases
            if value.lower() in ['inf', 'infinity', '+inf', '+infinity']:
                return float('inf')
            elif value.lower() in ['-inf', '-infinity']:
                return float('-inf')
            elif value.lower() in ['nan', 'none', 'null']:
                return float('nan')
            
            # Try to parse as number
            try:
                if '.' in value or 'e' in value.lower():
                    return float(value)
                else:
                    return float(int(value))
            except ValueError:
                raise ValueError(f"Cannot convert '{value}' to number")
        
        elif isinstance(value, (int, float)):
            return float(value)
        
        else:
            raise ValueError(f"Cannot convert {type(value)} to number")
    
    def normalize_list(self, value: Union[List, str]) -> List[Any]:
        """Normalize list value."""
        if isinstance(value, str):
            # Try to parse string as list
            value = value.strip()
            
            # Handle different list formats
            if value.startswith('[') and value.endswith(']'):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # Try to parse as Python list
                    try:
                        return eval(value)
                    except:
                        # Split by comma as fallback
                        return [item.strip() for item in value[1:-1].split(',') if item.strip()]
            else:
                # Split by common delimiters
                for delimiter in [',', ';', '|', '\n', '\t']:
                    if delimiter in value:
                        return [item.strip() for item in value.split(delimiter) if item.strip()]
                
                # Single item
                return [value]
        
        elif isinstance(value, list):
            return value
        
        elif hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
            return list(value)
        
        else:
            return [value]
    
    def normalize_json(self, value: Union[Dict, str]) -> Dict[str, Any]:
        """Normalize JSON object value."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Try to parse as Python dict
                try:
                    return eval(value)
                except:
                    raise ValueError(f"Cannot parse '{value}' as JSON")
        
        elif isinstance(value, dict):
            return value
        
        else:
            raise ValueError(f"Cannot convert {type(value)} to JSON object")


class OutputComparator:
    """Handles different types of output comparisons."""
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        self.normalizer = OutputNormalizer(config)
        self.logger = logging.getLogger(__name__)
    
    def compare_exact_string(self, predicted: str, actual: str) -> VerificationResult:
        """Compare strings exactly."""
        result = VerificationResult()
        result.comparison_type = ComparisonType.EXACT_STRING
        result.predicted_value = predicted
        result.actual_value = actual
        
        # Normalize strings
        norm_predicted = self.normalizer.normalize_string(predicted)
        norm_actual = self.normalizer.normalize_string(actual)
        
        result.normalized_predicted = norm_predicted
        result.normalized_actual = norm_actual
        
        # Compare
        result.is_correct = norm_predicted == norm_actual
        result.score = 1.0 if result.is_correct else 0.0
        result.confidence = 1.0
        
        if not result.is_correct:
            result.mismatch_details = {
                "type": "string_mismatch",
                "predicted_length": len(norm_predicted),
                "actual_length": len(norm_actual),
                "common_prefix": self._find_common_prefix(norm_predicted, norm_actual),
                "common_suffix": self._find_common_suffix(norm_predicted, norm_actual)
            }
        
        return result
    
    def compare_fuzzy_string(self, predicted: str, actual: str) -> VerificationResult:
        """Compare strings with fuzzy matching."""
        result = VerificationResult()
        result.comparison_type = ComparisonType.FUZZY_STRING
        result.predicted_value = predicted
        result.actual_value = actual
        
        # Normalize strings
        norm_predicted = self.normalizer.normalize_string(predicted)
        norm_actual = self.normalizer.normalize_string(actual)
        
        result.normalized_predicted = norm_predicted
        result.normalized_actual = norm_actual
        
        # Calculate similarity
        similarity = SequenceMatcher(None, norm_predicted, norm_actual).ratio()
        
        result.is_correct = similarity >= self.config.fuzzy_threshold
        result.score = similarity
        result.confidence = min(1.0, similarity * 1.2)  # Boost confidence for high similarity
        
        if self.config.enable_partial_credit and similarity >= self.config.min_partial_score:
            result.partial_credit_details = {
                "similarity_ratio": similarity,
                "threshold_met": similarity >= self.config.fuzzy_threshold,
                "edit_distance": self._calculate_edit_distance(norm_predicted, norm_actual)
            }
        
        if not result.is_correct:
            result.mismatch_details = {
                "type": "fuzzy_string_mismatch",
                "similarity": similarity,
                "threshold": self.config.fuzzy_threshold,
                "edit_distance": self._calculate_edit_distance(norm_predicted, norm_actual)
            }
        
        return result
    
    def compare_numeric(self, predicted: Union[int, float, str], 
                       actual: Union[int, float, str]) -> VerificationResult:
        """Compare numeric values with tolerance."""
        result = VerificationResult()
        result.comparison_type = ComparisonType.NUMERIC
        result.predicted_value = predicted
        result.actual_value = actual
        
        try:
            # Normalize numbers
            norm_predicted = self.normalizer.normalize_number(predicted)
            norm_actual = self.normalizer.normalize_number(actual)
            
            result.normalized_predicted = norm_predicted
            result.normalized_actual = norm_actual
            
            # Handle special cases
            if math.isnan(norm_predicted) and math.isnan(norm_actual):
                result.is_correct = self.config.allow_nan
                result.score = 1.0 if result.is_correct else 0.0
                result.confidence = 1.0
                return result
            
            if math.isinf(norm_predicted) or math.isinf(norm_actual):
                result.is_correct = (norm_predicted == norm_actual) and self.config.allow_infinity
                result.score = 1.0 if result.is_correct else 0.0
                result.confidence = 1.0
                return result
            
            # Calculate difference
            abs_diff = abs(norm_predicted - norm_actual)
            rel_diff = abs_diff / max(abs(norm_actual), 1e-10) if norm_actual != 0 else abs_diff
            
            # Check tolerance
            within_abs_tolerance = abs_diff <= self.config.numeric_tolerance
            within_rel_tolerance = rel_diff <= self.config.relative_tolerance
            
            result.is_correct = within_abs_tolerance or within_rel_tolerance
            
            # Calculate score with partial credit
            if result.is_correct:
                result.score = 1.0
            elif self.config.enable_partial_credit:
                # Give partial credit based on how close the values are
                score_abs = max(0, 1 - abs_diff / (self.config.numeric_tolerance * 10))
                score_rel = max(0, 1 - rel_diff / (self.config.relative_tolerance * 10))
                result.score = max(score_abs, score_rel, self.config.min_partial_score)
            else:
                result.score = 0.0
            
            result.confidence = 1.0
            
            if not result.is_correct:
                result.mismatch_details = {
                    "type": "numeric_mismatch",
                    "absolute_difference": abs_diff,
                    "relative_difference": rel_diff,
                    "absolute_tolerance": self.config.numeric_tolerance,
                    "relative_tolerance": self.config.relative_tolerance
                }
            
            if result.score > 0 and result.score < 1:
                result.partial_credit_details = {
                    "absolute_difference": abs_diff,
                    "relative_difference": rel_diff,
                    "partial_score_basis": "proximity_to_tolerance"
                }
        
        except ValueError as e:
            result.is_correct = False
            result.score = 0.0
            result.confidence = 0.0
            result.mismatch_details = {
                "type": "numeric_parse_error",
                "error": str(e)
            }
        
        return result
    
    def compare_list(self, predicted: Union[List, str], actual: Union[List, str], 
                    ordered: bool = True) -> VerificationResult:
        """Compare list values."""
        result = VerificationResult()
        result.comparison_type = ComparisonType.LIST_ORDERED if ordered else ComparisonType.LIST_UNORDERED
        result.predicted_value = predicted
        result.actual_value = actual
        
        try:
            # Normalize lists
            norm_predicted = self.normalizer.normalize_list(predicted)
            norm_actual = self.normalizer.normalize_list(actual)
            
            result.normalized_predicted = norm_predicted
            result.normalized_actual = norm_actual
            
            if ordered:
                # Ordered comparison
                result.is_correct = len(norm_predicted) == len(norm_actual)
                
                if result.is_correct:
                    # Compare elements in order
                    matching_elements = 0
                    element_scores = []
                    
                    for i, (pred_elem, actual_elem) in enumerate(zip(norm_predicted, norm_actual)):
                        elem_result = self._compare_list_element(pred_elem, actual_elem)
                        element_scores.append(elem_result.score)
                        if elem_result.is_correct:
                            matching_elements += 1
                        else:
                            result.is_correct = False
                    
                    if self.config.partial_list_credit and len(norm_actual) > 0:
                        result.score = sum(element_scores) / len(element_scores)
                    else:
                        result.score = 1.0 if result.is_correct else 0.0
                    
                    result.partial_credit_details = {
                        "matching_elements": matching_elements,
                        "total_elements": len(norm_actual),
                        "element_scores": element_scores
                    }
                
                else:
                    result.score = 0.0
            
            else:
                # Unordered comparison
                pred_set = set(str(x) for x in norm_predicted)
                actual_set = set(str(x) for x in norm_actual)
                
                result.is_correct = pred_set == actual_set
                
                if self.config.partial_list_credit:
                    intersection = pred_set.intersection(actual_set)
                    union = pred_set.union(actual_set)
                    result.score = len(intersection) / len(union) if union else 1.0
                else:
                    result.score = 1.0 if result.is_correct else 0.0
                
                result.partial_credit_details = {
                    "predicted_unique": len(pred_set),
                    "actual_unique": len(actual_set),
                    "intersection": len(pred_set.intersection(actual_set)),
                    "jaccard_similarity": result.score
                }
            
            result.confidence = 1.0
            
            if not result.is_correct:
                result.mismatch_details = {
                    "type": "list_mismatch",
                    "predicted_length": len(norm_predicted),
                    "actual_length": len(norm_actual),
                    "ordered_comparison": ordered
                }
        
        except ValueError as e:
            result.is_correct = False
            result.score = 0.0
            result.confidence = 0.0
            result.mismatch_details = {
                "type": "list_parse_error",
                "error": str(e)
            }
        
        return result
    
    def compare_json(self, predicted: Union[Dict, str], actual: Union[Dict, str]) -> VerificationResult:
        """Compare JSON objects."""
        result = VerificationResult()
        result.comparison_type = ComparisonType.JSON_OBJECT
        result.predicted_value = predicted
        result.actual_value = actual
        
        try:
            # Normalize JSON objects
            norm_predicted = self.normalizer.normalize_json(predicted)
            norm_actual = self.normalizer.normalize_json(actual)
            
            result.normalized_predicted = norm_predicted
            result.normalized_actual = norm_actual
            
            # Compare JSON objects
            comparison_result = self._compare_json_recursive(norm_predicted, norm_actual)
            
            result.is_correct = comparison_result["exact_match"]
            result.score = comparison_result["similarity_score"]
            result.confidence = 1.0
            
            result.partial_credit_details = comparison_result["details"]
            
            if not result.is_correct:
                result.mismatch_details = {
                    "type": "json_mismatch",
                    "predicted_keys": list(norm_predicted.keys()) if isinstance(norm_predicted, dict) else None,
                    "actual_keys": list(norm_actual.keys()) if isinstance(norm_actual, dict) else None,
                    "differences": comparison_result["differences"]
                }
        
        except ValueError as e:
            result.is_correct = False
            result.score = 0.0
            result.confidence = 0.0
            result.mismatch_details = {
                "type": "json_parse_error",
                "error": str(e)
            }
        
        return result
    
    def compare_regex(self, predicted: str, pattern: str) -> VerificationResult:
        """Compare string against regex pattern."""
        result = VerificationResult()
        result.comparison_type = ComparisonType.REGEX_PATTERN
        result.predicted_value = predicted
        result.actual_value = pattern
        
        try:
            # Normalize string
            norm_predicted = self.normalizer.normalize_string(predicted)
            result.normalized_predicted = norm_predicted
            result.normalized_actual = pattern
            
            # Compile and match regex
            regex = re.compile(pattern, self.config.regex_flags)
            match = regex.search(norm_predicted)
            
            result.is_correct = match is not None
            result.score = 1.0 if result.is_correct else 0.0
            result.confidence = 1.0
            
            if match:
                result.partial_credit_details = {
                    "match_start": match.start(),
                    "match_end": match.end(),
                    "matched_text": match.group(0),
                    "groups": match.groups()
                }
            else:
                result.mismatch_details = {
                    "type": "regex_no_match",
                    "pattern": pattern,
                    "flags": self.config.regex_flags
                }
        
        except re.error as e:
            result.is_correct = False
            result.score = 0.0
            result.confidence = 0.0
            result.mismatch_details = {
                "type": "regex_compile_error",
                "error": str(e)
            }
        
        return result
    
    def _compare_list_element(self, predicted: Any, actual: Any) -> VerificationResult:
        """Compare individual list elements."""
        # Try numeric comparison first
        try:
            return self.compare_numeric(predicted, actual)
        except ValueError:
            pass
        
        # Fall back to string comparison
        return self.compare_exact_string(str(predicted), str(actual))
    
    def _compare_json_recursive(self, predicted: Any, actual: Any) -> Dict[str, Any]:
        """Recursively compare JSON structures."""
        if type(predicted) != type(actual):
            return {
                "exact_match": False,
                "similarity_score": 0.0,
                "differences": [f"Type mismatch: {type(predicted)} vs {type(actual)}"],
                "details": {"type_mismatch": True}
            }
        
        if isinstance(predicted, dict):
            return self._compare_json_dict(predicted, actual)
        elif isinstance(predicted, list):
            return self._compare_json_list(predicted, actual)
        else:
            # Compare primitive values
            if predicted == actual:
                return {
                    "exact_match": True,
                    "similarity_score": 1.0,
                    "differences": [],
                    "details": {}
                }
            else:
                return {
                    "exact_match": False,
                    "similarity_score": 0.0,
                    "differences": [f"Value mismatch: {predicted} vs {actual}"],
                    "details": {"value_mismatch": True}
                }
    
    def _compare_json_dict(self, predicted: Dict, actual: Dict) -> Dict[str, Any]:
        """Compare JSON dictionaries."""
        pred_keys = set(predicted.keys())
        actual_keys = set(actual.keys())
        
        common_keys = pred_keys.intersection(actual_keys)
        missing_keys = actual_keys - pred_keys
        extra_keys = pred_keys - actual_keys
        
        exact_match = True
        total_score = 0.0
        differences = []
        
        # Compare common keys
        for key in common_keys:
            key_result = self._compare_json_recursive(predicted[key], actual[key])
            if not key_result["exact_match"]:
                exact_match = False
                differences.extend([f"Key '{key}': {diff}" for diff in key_result["differences"]])
            total_score += key_result["similarity_score"]
        
        # Handle missing/extra keys
        if missing_keys:
            exact_match = False
            differences.append(f"Missing keys: {list(missing_keys)}")
        
        if extra_keys:
            exact_match = False
            differences.append(f"Extra keys: {list(extra_keys)}")
        
        # Calculate overall similarity
        total_keys = len(pred_keys.union(actual_keys))
        if total_keys > 0:
            similarity_score = (total_score + len(common_keys) - len(missing_keys) - len(extra_keys)) / total_keys
            similarity_score = max(0.0, min(1.0, similarity_score))
        else:
            similarity_score = 1.0
        
        return {
            "exact_match": exact_match,
            "similarity_score": similarity_score,
            "differences": differences,
            "details": {
                "common_keys": len(common_keys),
                "missing_keys": len(missing_keys),
                "extra_keys": len(extra_keys),
                "total_keys": total_keys
            }
        }
    
    def _compare_json_list(self, predicted: List, actual: List) -> Dict[str, Any]:
        """Compare JSON lists."""
        if len(predicted) != len(actual):
            similarity_score = 0.0 if not self.config.json_partial_credit else min(len(predicted), len(actual)) / max(len(predicted), len(actual))
            return {
                "exact_match": False,
                "similarity_score": similarity_score,
                "differences": [f"Length mismatch: {len(predicted)} vs {len(actual)}"],
                "details": {"length_mismatch": True}
            }
        
        exact_match = True
        total_score = 0.0
        differences = []
        
        for i, (pred_item, actual_item) in enumerate(zip(predicted, actual)):
            item_result = self._compare_json_recursive(pred_item, actual_item)
            if not item_result["exact_match"]:
                exact_match = False
                differences.extend([f"Index {i}: {diff}" for diff in item_result["differences"]])
            total_score += item_result["similarity_score"]
        
        similarity_score = total_score / len(actual) if actual else 1.0
        
        return {
            "exact_match": exact_match,
            "similarity_score": similarity_score,
            "differences": differences,
            "details": {"list_length": len(actual)}
        }
    
    def _find_common_prefix(self, str1: str, str2: str) -> str:
        """Find common prefix of two strings."""
        min_len = min(len(str1), len(str2))
        for i in range(min_len):
            if str1[i] != str2[i]:
                return str1[:i]
        return str1[:min_len]
    
    def _find_common_suffix(self, str1: str, str2: str) -> str:
        """Find common suffix of two strings."""
        min_len = min(len(str1), len(str2))
        for i in range(1, min_len + 1):
            if str1[-i] != str2[-i]:
                return str1[len(str1) - i + 1:]
        return str1[len(str1) - min_len:]
    
    def _calculate_edit_distance(self, str1: str, str2: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(str1) < len(str2):
            return self._calculate_edit_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class OutputVerifier:
    """
    Main verification system for comparing predicted and actual outputs.
    
    Supports multiple comparison types with smart normalization and scoring.
    """
    
    def __init__(self, config: Optional[VerificationConfig] = None):
        """
        Initialize the output verifier.
        
        Args:
            config: Verification configuration. Uses defaults if None.
        """
        self.config = config or VerificationConfig()
        self.comparator = OutputComparator(self.config)
        self.logger = logging.getLogger(__name__)
    
    def verify_output(self, predicted: Any, actual: Any, 
                     comparison_type: ComparisonType = ComparisonType.AUTO_DETECT) -> VerificationResult:
        """
        Verify a single output prediction against actual result.
        
        Args:
            predicted: Predicted output value
            actual: Actual output value
            comparison_type: Type of comparison to perform
            
        Returns:
            VerificationResult with detailed comparison information
        """
        if comparison_type == ComparisonType.AUTO_DETECT:
            comparison_type = self._detect_comparison_type(predicted, actual)
        
        self.logger.debug(f"Verifying with {comparison_type.value}: {predicted} vs {actual}")
        
        if comparison_type == ComparisonType.EXACT_STRING:
            return self.comparator.compare_exact_string(str(predicted), str(actual))
        elif comparison_type == ComparisonType.FUZZY_STRING:
            return self.comparator.compare_fuzzy_string(str(predicted), str(actual))
        elif comparison_type == ComparisonType.NUMERIC:
            return self.comparator.compare_numeric(predicted, actual)
        elif comparison_type == ComparisonType.LIST_ORDERED:
            return self.comparator.compare_list(predicted, actual, ordered=True)
        elif comparison_type == ComparisonType.LIST_UNORDERED:
            return self.comparator.compare_list(predicted, actual, ordered=False)
        elif comparison_type == ComparisonType.JSON_OBJECT:
            return self.comparator.compare_json(predicted, actual)
        elif comparison_type == ComparisonType.REGEX_PATTERN:
            return self.comparator.compare_regex(str(predicted), str(actual))
        else:
            raise ValueError(f"Unsupported comparison type: {comparison_type}")
    
    def verify_batch(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify a batch of predictions efficiently.
        
        Args:
            predictions: List of prediction dictionaries with 'predicted', 'actual', and optional 'type' keys
            
        Returns:
            Dictionary with batch verification results and statistics
        """
        results = []
        stats = {
            "total_predictions": len(predictions),
            "correct_predictions": 0,
            "average_score": 0.0,
            "comparison_type_counts": {},
            "error_count": 0
        }
        
        for i, prediction in enumerate(predictions):
            try:
                predicted = prediction["predicted"]
                actual = prediction["actual"]
                comp_type = ComparisonType(prediction.get("type", "auto_detect"))
                
                result = self.verify_output(predicted, actual, comp_type)
                results.append({
                    "index": i,
                    "result": result.to_dict()
                })
                
                # Update statistics
                if result.is_correct:
                    stats["correct_predictions"] += 1
                
                stats["average_score"] += result.score
                
                comp_type_str = result.comparison_type.value
                stats["comparison_type_counts"][comp_type_str] = stats["comparison_type_counts"].get(comp_type_str, 0) + 1
                
            except Exception as e:
                self.logger.error(f"Error verifying prediction {i}: {e}")
                stats["error_count"] += 1
                results.append({
                    "index": i,
                    "error": str(e)
                })
        
        # Finalize statistics
        if stats["total_predictions"] > 0:
            stats["accuracy"] = stats["correct_predictions"] / stats["total_predictions"]
            stats["average_score"] /= stats["total_predictions"]
        else:
            stats["accuracy"] = 0.0
        
        return {
            "results": results,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_verification_report(self, batch_results: Dict[str, Any], 
                                   output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive verification report.
        
        Args:
            batch_results: Results from verify_batch
            output_file: Optional file path to save report
            
        Returns:
            Detailed verification report
        """
        stats = batch_results["statistics"]
        results = batch_results["results"]
        
        # Analyze results by comparison type
        type_analysis = {}
        for comp_type, count in stats["comparison_type_counts"].items():
            type_results = [r for r in results if r.get("result", {}).get("comparison_type") == comp_type]
            if type_results:
                type_scores = [r["result"]["score"] for r in type_results if "result" in r]
                type_analysis[comp_type] = {
                    "count": count,
                    "accuracy": sum(1 for r in type_results if r.get("result", {}).get("is_correct", False)) / count,
                    "average_score": sum(type_scores) / len(type_scores) if type_scores else 0.0,
                    "min_score": min(type_scores) if type_scores else 0.0,
                    "max_score": max(type_scores) if type_scores else 0.0
                }
        
        # Identify common failure patterns
        failures = [r for r in results if r.get("result", {}).get("score", 0) < 1.0]
        failure_patterns = self._analyze_failure_patterns(failures)
        
        report = {
            "summary": {
                "total_predictions": stats["total_predictions"],
                "accuracy": stats["accuracy"],
                "average_score": stats["average_score"],
                "error_rate": stats["error_count"] / stats["total_predictions"] if stats["total_predictions"] > 0 else 0
            },
            "comparison_type_analysis": type_analysis,
            "failure_patterns": failure_patterns,
            "detailed_results": results,
            "configuration": {
                "case_sensitive": self.config.case_sensitive,
                "numeric_tolerance": self.config.numeric_tolerance,
                "fuzzy_threshold": self.config.fuzzy_threshold,
                "partial_credit_enabled": self.config.enable_partial_credit
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Verification report saved to {output_file}")
        
        return report
    
    def _detect_comparison_type(self, predicted: Any, actual: Any) -> ComparisonType:
        """Automatically detect the appropriate comparison type."""
        # Try to detect numeric values
        if self._looks_like_number(predicted) and self._looks_like_number(actual):
            return ComparisonType.NUMERIC
        
        # Try to detect lists
        if self._looks_like_list(predicted) or self._looks_like_list(actual):
            return ComparisonType.LIST_ORDERED
        
        # Try to detect JSON objects
        if self._looks_like_json(predicted) or self._looks_like_json(actual):
            return ComparisonType.JSON_OBJECT
        
        # Default to fuzzy string matching for better partial credit
        return ComparisonType.FUZZY_STRING
    
    def _looks_like_number(self, value: Any) -> bool:
        """Check if value looks like a number."""
        if isinstance(value, (int, float)):
            return True
        
        if isinstance(value, str):
            value = value.strip()
            try:
                float(value)
                return True
            except ValueError:
                return value.lower() in ['inf', 'infinity', '-inf', '-infinity', 'nan']
        
        return False
    
    def _looks_like_list(self, value: Any) -> bool:
        """Check if value looks like a list."""
        if isinstance(value, list):
            return True
        
        if isinstance(value, str):
            value = value.strip()
            return (value.startswith('[') and value.endswith(']')) or \
                   (',' in value) or ('|' in value) or (';' in value)
        
        return hasattr(value, '__iter__') and not isinstance(value, (str, dict))
    
    def _looks_like_json(self, value: Any) -> bool:
        """Check if value looks like a JSON object."""
        if isinstance(value, dict):
            return True
        
        if isinstance(value, str):
            value = value.strip()
            if value.startswith('{') and value.endswith('}'):
                try:
                    json.loads(value)
                    return True
                except json.JSONDecodeError:
                    pass
        
        return False
    
    def _analyze_failure_patterns(self, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze common patterns in verification failures."""
        patterns = {
            "common_mismatch_types": {},
            "frequent_errors": {},
            "partial_credit_distribution": {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        }
        
        for failure in failures:
            if "result" not in failure:
                continue
            
            result = failure["result"]
            
            # Analyze mismatch types
            if "mismatch_details" in result and result["mismatch_details"]:
                mismatch_type = result["mismatch_details"].get("type", "unknown")
                patterns["common_mismatch_types"][mismatch_type] = patterns["common_mismatch_types"].get(mismatch_type, 0) + 1
            
            # Analyze partial credit distribution
            score = result.get("score", 0.0)
            if score < 0.2:
                patterns["partial_credit_distribution"]["0.0-0.2"] += 1
            elif score < 0.4:
                patterns["partial_credit_distribution"]["0.2-0.4"] += 1
            elif score < 0.6:
                patterns["partial_credit_distribution"]["0.4-0.6"] += 1
            elif score < 0.8:
                patterns["partial_credit_distribution"]["0.6-0.8"] += 1
            else:
                patterns["partial_credit_distribution"]["0.8-1.0"] += 1
        
        return patterns