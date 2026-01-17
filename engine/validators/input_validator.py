"""Composed validator for engine inputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from engine.exceptions import InputValidationError
from engine.utils.validation import validate_trace_id

from .config import ValidationConfig
from .context_validator import ContextValidator
from .text_validator import TextValidator
from .types import ValidatedInput


class InputValidator:
    """
    Validates inputs before constitutional evaluation.

    Validation failures are constitutional signals - they indicate
    that the system cannot evaluate the input with integrity.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.text_validator = TextValidator(self.config)
        self.context_validator = ContextValidator(self.config)

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
        context = context or {}
        trace_id = validate_trace_id(trace_id)
        warnings: List[str] = []
        sanitization_applied = False

        text, text_warnings, text_sanitized = self.text_validator.validate(text)
        warnings.extend(text_warnings)
        sanitization_applied = sanitization_applied or text_sanitized

        context, ctx_warnings, ctx_sanitized = self.context_validator.validate(context)
        warnings.extend(ctx_warnings)
        sanitization_applied = sanitization_applied or ctx_sanitized

        return ValidatedInput(
            text=text,
            context=context,
            trace_id=trace_id,
            validation_warnings=warnings,
            sanitization_applied=sanitization_applied,
        )


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
    except InputValidationError as exc:
        return None, exc


__all__ = ["InputValidator", "validate_input", "validate_input_safe"]
