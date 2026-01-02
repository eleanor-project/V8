"""
ELEANOR V8 — Input Validation for Constitutional Safety

Validation ensures:
1. Constitutional evaluations are not compromised by malicious inputs
2. Resource exhaustion is prevented
3. Prompt injection attacks are detected
4. Validation failures are clear constitutional signals

Principle: Input validation protects the integrity of constitutional
evaluation, not just system security.
"""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
import json

from engine.exceptions import InputValidationError


# ============================================================================
# CONFIGURATION
# ============================================================================

class ValidationConfig(BaseModel):
    """Configuration for input validation."""
    
    max_text_length: int = Field(default=100_000, gt=0, description="Maximum input text length")
    max_context_size: int = Field(default=1_000_000, gt=0, description="Maximum context JSON size in bytes")
    max_context_depth: int = Field(default=10, gt=0, description="Maximum nested dict depth")
    
    enable_injection_detection: bool = Field(default=True, description="Enable prompt injection detection")
    enable_malicious_pattern_detection: bool = Field(default=True, description="Detect malicious patterns")
    
    # Patterns that may indicate prompt injection
    injection_patterns: List[str] = Field(default_factory=lambda: [
        r"ignore (previous|above|all) (instructions?|prompts?|rules?)",
        r"disregard (previous|above|all)",
        r"override (system|safety|constitutional)",
        r"you (are|must) now",
        r"jailbreak",
        r"DAN mode",
    ])


# ============================================================================
# VALIDATED INPUT TYPE
# ============================================================================

class ValidatedInput(BaseModel):
    """
    Input that has passed validation checks.
    
    Presence of this type guarantees input has been sanitized and
    is safe for constitutional evaluation.
    """
    
    text: str = Field(..., description="Validated input text")
    context: Dict[str, Any] = Field(default_factory=dict, description="Validated context")
    trace_id: str = Field(..., description="Audit trail identifier")
    
    # Validation metadata
    validation_warnings: List[str] = Field(
        default_factory=list,
        description="Non-blocking warnings from validation"
    )
    sanitization_applied: bool = Field(
        default=False,
        description="Whether input was modified during sanitization"
    )
    
    class Config:
        frozen = True  # Validated inputs are immutable


# ============================================================================
# VALIDATOR
# ============================================================================

class InputValidator:
    """
    Validates inputs before constitutional evaluation.
    
    Validation failures are constitutional signals - they indicate
    that the system cannot evaluate the input with integrity.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        
        # Compile injection patterns
        self.injection_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.injection_patterns
        ]
    
    def validate(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> ValidatedInput:
        """
        Validate input text and context.
        
        Raises:
            InputValidationError: If input fails validation
        
        Returns:
            ValidatedInput: Immutable validated input
        """
        import uuid
        
        context = context or {}
        trace_id = trace_id or str(uuid.uuid4())
        warnings: List[str] = []
        sanitization_applied = False
        
        # Validate text
        text, text_warnings, text_sanitized = self._validate_text(text)
        warnings.extend(text_warnings)
        sanitization_applied = sanitization_applied or text_sanitized
        
        # Validate context
        context, ctx_warnings, ctx_sanitized = self._validate_context(context)
        warnings.extend(ctx_warnings)
        sanitization_applied = sanitization_applied or ctx_sanitized
        
        return ValidatedInput(
            text=text,
            context=context,
            trace_id=trace_id,
            validation_warnings=warnings,
            sanitization_applied=sanitization_applied,
        )
    
    def _validate_text(self, text: str) -> tuple[str, List[str], bool]:
        """
        Validate and sanitize text input.
        
        Returns:
            (sanitized_text, warnings, was_sanitized)
        """
        warnings: List[str] = []
        sanitized = False
        
        # Type check
        if not isinstance(text, str):
            raise InputValidationError(
                "Input text must be a string",
                validation_type="type_error",
                field="text",
                context={"received_type": type(text).__name__},
            )
        
        # Empty check
        if not text or not text.strip():
            raise InputValidationError(
                "Input text cannot be empty",
                validation_type="empty_input",
                field="text",
            )
        
        # Length check
        if len(text) > self.config.max_text_length:
            raise InputValidationError(
                f"Input text exceeds maximum length ({self.config.max_text_length} characters)",
                validation_type="size_limit",
                field="text",
                context={
                    "length": len(text),
                    "max_length": self.config.max_text_length,
                },
            )
        
        # Injection detection
        if self.config.enable_injection_detection:
            for pattern in self.injection_regexes:
                if pattern.search(text):
                    warnings.append(
                        f"Potential prompt injection detected: pattern '{pattern.pattern}'"
                    )
        
        # Malicious pattern detection
        if self.config.enable_malicious_pattern_detection:
            # Check for excessive repetition (potential DoS)
            if self._has_excessive_repetition(text):
                warnings.append("Excessive character repetition detected")
            
            # Check for null bytes
            if '\x00' in text:
                text = text.replace('\x00', '')
                sanitized = True
                warnings.append("Null bytes removed from input")
        
        return text, warnings, sanitized
    
    def _validate_context(self, context: Dict[str, Any]) -> tuple[Dict[str, Any], List[str], bool]:
        """
        Validate and sanitize context dictionary.
        
        Returns:
            (sanitized_context, warnings, was_sanitized)
        """
        warnings: List[str] = []
        sanitized = False
        
        # Size check
        try:
            serialized = json.dumps(context, default=str)
            size = len(serialized.encode('utf-8'))
            
            if size > self.config.max_context_size:
                raise InputValidationError(
                    f"Context payload exceeds size limit ({self.config.max_context_size} bytes)",
                    validation_type="size_limit",
                    field="context",
                    context={
                        "size": size,
                        "max_size": self.config.max_context_size,
                    },
                )
        except (TypeError, ValueError) as e:
            raise InputValidationError(
                "Context must be JSON-serializable",
                validation_type="serialization_error",
                field="context",
                context={"error": str(e)},
            )
        
        # Depth check (prevent deeply nested malicious payloads)
        max_depth = self._get_dict_depth(context)
        if max_depth > self.config.max_context_depth:
            raise InputValidationError(
                f"Context nesting exceeds maximum depth ({self.config.max_context_depth})",
                validation_type="depth_limit",
                field="context",
                context={
                    "depth": max_depth,
                    "max_depth": self.config.max_context_depth,
                },
            )
        
        # Sanitize reserved keys that might interfere with governance
        reserved_keys = {
            "_escalation_override",
            "_skip_human_review",
            "_bypass_governance",
            "_suppress_dissent",
        }
        
        context_copy = context.copy()
        for key in reserved_keys:
            if key in context_copy:
                del context_copy[key]
                sanitized = True
                warnings.append(f"Removed reserved key: {key}")
        
        return context_copy, warnings, sanitized
    
    @staticmethod
    def _has_excessive_repetition(text: str, threshold: float = 0.3) -> bool:
        """
        Check for excessive character repetition (potential DoS).
        
        Returns True if more than threshold of characters are repetitions.
        """
        if len(text) < 10:
            return False
        
        # Count longest consecutive character runs
        max_run = 1
        current_run = 1
        
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        return max_run / len(text) > threshold
    
    @staticmethod
    def _get_dict_depth(d: Any, current_depth: int = 0) -> int:
        """
        Calculate maximum nesting depth of dictionary.
        """
        if not isinstance(d, dict):
            return current_depth
        
        if not d:
            return current_depth + 1
        
        return max(
            InputValidator._get_dict_depth(v, current_depth + 1)
            for v in d.values()
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_input(
    text: str,
    context: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    config: Optional[ValidationConfig] = None,
) -> ValidatedInput:
    """
    Convenience function for input validation.
    
    Usage:
        validated = validate_input(
            text="User input here",
            context={"domain": "healthcare"},
            trace_id="trace-123"
        )
    
    Raises:
        InputValidationError: If validation fails
    """
    validator = InputValidator(config=config)
    return validator.validate(text, context, trace_id)


def validate_input_safe(
    text: str,
    context: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    config: Optional[ValidationConfig] = None,
) -> tuple[Optional[ValidatedInput], Optional[InputValidationError]]:
    """
    Safe input validation that returns error instead of raising.
    
    Returns:
        (validated_input, error) - one will be None
    
    Usage:
        validated, error = validate_input_safe(text="...")
        if error:
            handle_validation_error(error)
        else:
            process(validated)
    """
    try:
        validated = validate_input(text, context, trace_id, config)
        return validated, None
    except InputValidationError as e:
        return None, e


__all__ = [
    "ValidationConfig",
    "ValidatedInput",
    "InputValidator",
    "validate_input",
    "validate_input_safe",
]
