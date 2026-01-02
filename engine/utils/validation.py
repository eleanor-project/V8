"""Input validation utilities for ELEANOR V8 Engine.

Provides security-focused input validation to prevent:
- Prompt injection attacks
- Resource exhaustion
- Malicious context payloads
"""

import re
from typing import Any, Dict, Optional

from engine.exceptions import InputValidationError


class InputValidator:
    """Validates and sanitizes user inputs to the engine."""

    # Configuration
    MAX_TEXT_LENGTH = 100_000  # 100K characters
    MAX_CONTEXT_KEYS = 50
    MAX_CONTEXT_VALUE_LENGTH = 10_000
    MAX_CONTEXT_DEPTH = 5
    
    # Patterns for suspicious content
    PROMPT_INJECTION_PATTERNS = [
        r'ignore\s+previous\s+instructions',
        r'disregard\s+all\s+rules',
        r'system\s*:\s*you\s+are',
        r'<\s*script\s*>',
        r'javascript\s*:',
    ]
    
    @classmethod
    def validate_text_input(cls, text: str, field_name: str = "text") -> str:
        """Validate and sanitize text input.
        
        Args:
            text: Input text to validate
            field_name: Name of the field for error messages
            
        Returns:
            Validated text
            
        Raises:
            InputValidationError: If validation fails
        """
        if not isinstance(text, str):
            raise InputValidationError(
                f"{field_name} must be a string, got {type(text).__name__}"
            )
        
        if not text.strip():
            raise InputValidationError(f"{field_name} cannot be empty")
        
        if len(text) > cls.MAX_TEXT_LENGTH:
            raise InputValidationError(
                f"{field_name} exceeds maximum length of {cls.MAX_TEXT_LENGTH} characters",
                details={"length": len(text), "max_length": cls.MAX_TEXT_LENGTH}
            )
        
        # Check for prompt injection patterns
        text_lower = text.lower()
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                raise InputValidationError(
                    f"{field_name} contains suspicious patterns that may indicate prompt injection",
                    details={"pattern": pattern}
                )
        
        return text
    
    @classmethod
    def validate_context(
        cls,
        context: Optional[Dict[str, Any]],
        max_depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """Validate context dictionary.
        
        Args:
            context: Context dictionary to validate
            max_depth: Maximum nesting depth (default: MAX_CONTEXT_DEPTH)
            
        Returns:
            Validated context
            
        Raises:
            InputValidationError: If validation fails
        """
        if context is None:
            return {}
        
        if not isinstance(context, dict):
            raise InputValidationError(
                f"context must be a dictionary, got {type(context).__name__}"
            )
        
        if len(context) > cls.MAX_CONTEXT_KEYS:
            raise InputValidationError(
                f"context exceeds maximum of {cls.MAX_CONTEXT_KEYS} keys",
                details={"keys": len(context), "max_keys": cls.MAX_CONTEXT_KEYS}
            )
        
        max_depth = max_depth or cls.MAX_CONTEXT_DEPTH
        cls._validate_dict_depth(context, current_depth=0, max_depth=max_depth)
        cls._validate_context_values(context)
        
        return context
    
    @classmethod
    def _validate_dict_depth(
        cls,
        obj: Any,
        current_depth: int,
        max_depth: int
    ) -> None:
        """Recursively validate dictionary nesting depth."""
        if current_depth > max_depth:
            raise InputValidationError(
                f"context nesting exceeds maximum depth of {max_depth}",
                details={"max_depth": max_depth}
            )
        
        if isinstance(obj, dict):
            for value in obj.values():
                cls._validate_dict_depth(value, current_depth + 1, max_depth)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                cls._validate_dict_depth(item, current_depth + 1, max_depth)
    
    @classmethod
    def _validate_context_values(cls, context: Dict[str, Any]) -> None:
        """Validate context dictionary values."""
        for key, value in context.items():
            # Check key is string
            if not isinstance(key, str):
                raise InputValidationError(
                    f"context keys must be strings, got {type(key).__name__} for key {key}"
                )
            
            # Check string values don't exceed length limits
            if isinstance(value, str) and len(value) > cls.MAX_CONTEXT_VALUE_LENGTH:
                raise InputValidationError(
                    f"context value for '{key}' exceeds maximum length",
                    details={
                        "key": key,
                        "length": len(value),
                        "max_length": cls.MAX_CONTEXT_VALUE_LENGTH
                    }
                )
    
    @classmethod
    def sanitize_for_logging(
        cls,
        text: str,
        max_length: int = 500,
        mask_patterns: Optional[list] = None
    ) -> str:
        """Sanitize text for safe logging.
        
        Args:
            text: Text to sanitize
            max_length: Maximum length to include in logs
            mask_patterns: Regex patterns to mask (e.g., API keys)
            
        Returns:
            Sanitized text safe for logging
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "... [truncated]"
        
        # Mask sensitive patterns
        if mask_patterns:
            for pattern in mask_patterns:
                text = re.sub(pattern, "[REDACTED]", text)
        
        # Default masking for common secrets
        text = re.sub(r'api[_-]?key[\s:=]+[\w-]+', 'api_key=[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'token[\s:=]+[\w.-]+', 'token=[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'password[\s:=]+\S+', 'password=[REDACTED]', text, flags=re.IGNORECASE)
        
        return text


def validate_trace_id(trace_id: Optional[str]) -> str:
    """Validate or generate a trace ID.
    
    Args:
        trace_id: Optional trace ID to validate
        
    Returns:
        Valid trace ID
        
    Raises:
        InputValidationError: If trace_id format is invalid
    """
    import uuid
    
    if trace_id is None:
        return str(uuid.uuid4())
    
    if not isinstance(trace_id, str):
        raise InputValidationError(f"trace_id must be a string, got {type(trace_id).__name__}")
    
    # Validate UUID format if it looks like a UUID
    uuid_like = trace_id.count("-") >= 4 or (len(trace_id) == 36 and "-" in trace_id)
    if uuid_like:
        try:
            uuid.UUID(trace_id)
        except ValueError:
            raise InputValidationError(f"trace_id is not a valid UUID: {trace_id}")
    
    # Otherwise just check it's a reasonable identifier
    if not re.match(r'^[a-zA-Z0-9_-]{1,128}$', trace_id):
        raise InputValidationError(
            "trace_id must be alphanumeric with hyphens/underscores, max 128 chars"
        )
    
    return trace_id
