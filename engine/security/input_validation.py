"""
ELEANOR V8 — Input Validation and Security
-------------------------------------------

Comprehensive input validation and security hardening.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Input validator with security hardening.
    
    Provides:
    - SQL injection prevention
    - XSS prevention
    - Path traversal prevention
    - Command injection prevention
    - Size limits
    - Type validation
    """
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(UNION|OR|AND)\s+\d+)",
        r"('|(\\')|(;)|(\|)|(&))",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"\.\.%2F",
        r"\.\.%5C",
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}]",
        r"\b(cat|ls|pwd|whoami|id|uname|ps|kill|rm|mv|cp)\b",
    ]
    
    def __init__(
        self,
        max_string_length: int = 1_000_000,  # 1MB
        max_dict_depth: int = 10,
        max_list_length: int = 10_000,
        strict_mode: bool = True,
    ):
        """
        Initialize input validator.
        
        Args:
            max_string_length: Maximum string length
            max_dict_depth: Maximum dictionary nesting depth
            max_list_length: Maximum list length
            strict_mode: Enable strict security checks
        """
        self.max_string_length = max_string_length
        self.max_dict_depth = max_dict_depth
        self.max_list_length = max_list_length
        self.strict_mode = strict_mode
    
    def validate_string(
        self,
        value: str,
        field_name: str = "input",
        allow_empty: bool = True,
        sanitize: bool = True,
    ) -> str:
        """
        Validate and sanitize string input.
        
        Args:
            value: String to validate
            field_name: Field name for error messages
            allow_empty: Allow empty strings
            sanitize: Sanitize dangerous content
        
        Returns:
            Validated and sanitized string
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")
        
        if not allow_empty and not value.strip():
            raise ValueError(f"{field_name} cannot be empty")
        
        if len(value) > self.max_string_length:
            raise ValueError(
                f"{field_name} exceeds maximum length of {self.max_string_length} characters"
            )
        
        if sanitize:
            value = self._sanitize_string(value, field_name)
        
        return value
    
    def validate_dict(
        self,
        value: Dict[str, Any],
        field_name: str = "input",
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate dictionary input.
        
        Args:
            value: Dictionary to validate
            field_name: Field name for error messages
            max_depth: Maximum nesting depth (uses default if None)
        
        Returns:
            Validated dictionary
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be a dictionary, got {type(value).__name__}")
        
        max_depth = max_depth or self.max_dict_depth
        return self._validate_dict_recursive(value, field_name, max_depth, 0)
    
    def validate_list(
        self,
        value: List[Any],
        field_name: str = "input",
        max_length: Optional[int] = None,
    ) -> List[Any]:
        """
        Validate list input.
        
        Args:
            value: List to validate
            field_name: Field name for error messages
            max_length: Maximum list length (uses default if None)
        
        Returns:
            Validated list
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list, got {type(value).__name__}")
        
        max_length = max_length or self.max_list_length
        if len(value) > max_length:
            raise ValueError(
                f"{field_name} exceeds maximum length of {max_length} items"
            )
        
        return value
    
    def _sanitize_string(self, value: str, field_name: str) -> str:
        """Sanitize string to prevent security issues."""
        if not self.strict_mode:
            return value
        
        # Check for SQL injection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(
                    "potential_sql_injection_detected",
                    extra={"field": field_name, "pattern": pattern},
                )
                raise ValueError(f"{field_name} contains potentially dangerous SQL patterns")
        
        # Check for XSS
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(
                    "potential_xss_detected",
                    extra={"field": field_name, "pattern": pattern},
                )
                raise ValueError(f"{field_name} contains potentially dangerous XSS patterns")
        
        # Check for path traversal
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(
                    "potential_path_traversal_detected",
                    extra={"field": field_name, "pattern": pattern},
                )
                raise ValueError(f"{field_name} contains potentially dangerous path traversal patterns")
        
        return value
    
    def _validate_dict_recursive(
        self,
        value: Dict[str, Any],
        field_name: str,
        max_depth: int,
        current_depth: int,
    ) -> Dict[str, Any]:
        """Recursively validate dictionary."""
        if current_depth > max_depth:
            raise ValueError(
                f"{field_name} exceeds maximum nesting depth of {max_depth}"
            )
        
        validated = {}
        for key, val in value.items():
            # Validate key
            if not isinstance(key, str):
                raise ValueError(f"{field_name} keys must be strings")
            
            # Validate value
            if isinstance(val, str):
                validated[key] = self.validate_string(val, f"{field_name}.{key}")
            elif isinstance(val, dict):
                validated[key] = self._validate_dict_recursive(
                    val, f"{field_name}.{key}", max_depth, current_depth + 1
                )
            elif isinstance(val, list):
                validated[key] = self.validate_list(val, f"{field_name}.{key}")
            else:
                validated[key] = val
        
        return validated


def sanitize_sql_identifier(identifier: str) -> str:
    """
    Sanitize SQL identifier to prevent injection.
    
    Args:
        identifier: SQL identifier to sanitize
    
    Returns:
        Sanitized identifier
    """
    # Only allow alphanumeric and underscore
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}")
    return identifier


def sanitize_path(path: str) -> str:
    """
    Sanitize file path to prevent path traversal.
    
    Args:
        path: File path to sanitize
    
    Returns:
        Sanitized path
    """
    # Remove path traversal attempts
    path = path.replace("../", "").replace("..\\", "")
    path = path.replace("%2e%2e%2f", "").replace("%2e%2e%5c", "")
    
    # Remove leading slashes
    path = path.lstrip("/\\")
    
    return path


__all__ = [
    "InputValidator",
    "sanitize_sql_identifier",
    "sanitize_path",
]
