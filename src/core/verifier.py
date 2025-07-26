"""
Simple output verification for generated code.
"""

import re
from typing import Dict, Any, Optional


class OutputVerifier:
    """Verifies that code execution produces expected results."""
    
    def verify(self, expected: str, actual: str, verification_type: str = "exact") -> Dict[str, Any]:
        """Verify actual output matches expected output."""
        if verification_type == "exact":
            return self._exact_match(expected, actual)
        elif verification_type == "contains":
            return self._contains_match(expected, actual)
        elif verification_type == "numeric":
            return self._numeric_match(expected, actual)
        else:
            return {"is_correct": False, "score": 0.0, "error": f"Unknown verification type: {verification_type}"}
    
    def _exact_match(self, expected: str, actual: str) -> Dict[str, Any]:
        """Check for exact string match."""
        expected_clean = expected.strip()
        actual_clean = actual.strip()
        is_correct = expected_clean == actual_clean
        
        return {
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "verification_type": "exact",
            "expected": expected_clean,
            "actual": actual_clean
        }
    
    def _contains_match(self, expected: str, actual: str) -> Dict[str, Any]:
        """Check if actual output contains expected string."""
        is_correct = expected.strip() in actual.strip()
        
        return {
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "verification_type": "contains",
            "expected": expected.strip(),
            "actual": actual.strip()
        }
    
    def _numeric_match(self, expected: str, actual: str) -> Dict[str, Any]:
        """Check for numeric match with tolerance."""
        try:
            expected_num = float(expected.strip())
            actual_num = float(actual.strip())
            
            # Allow small tolerance for floating point comparison
            tolerance = 1e-6
            is_correct = abs(expected_num - actual_num) < tolerance
            
            return {
                "is_correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "verification_type": "numeric",
                "expected": expected_num,
                "actual": actual_num,
                "tolerance": tolerance
            }
        except ValueError:
            return {
                "is_correct": False,
                "score": 0.0,
                "verification_type": "numeric",
                "error": "Could not parse numeric values"
            }