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

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, ConfigDict

from engine.exceptions import InputValidationError
from engine.utils.validation import validate_trace_id


# ============================================================================
# CONFIGURATION
# ============================================================================


class ValidationConfig(BaseModel):
    """Configuration for input validation."""

    max_text_length: int = Field(default=100_000, gt=0, description="Maximum input text length")
    max_context_size: int = Field(
        default=1_000_000, gt=0, description="Maximum context JSON size in bytes"
    )
    max_context_depth: int = Field(default=5, gt=0, description="Maximum nested dict depth")
    max_context_keys: int = Field(default=100, gt=0, description="Maximum context keys allowed")
    max_context_string_length: int = Field(
        default=50_000,
        gt=0,
        description="Maximum length for any single context string",
    )

    enable_injection_detection: bool = Field(
        default=True, description="Enable prompt injection detection"
    )
    enable_malicious_pattern_detection: bool = Field(
        default=True, description="Detect malicious patterns"
    )
    reject_on_injection: bool = Field(
        default=True, description="Reject inputs on injection detection"
    )

    # Patterns that may indicate prompt injection
    injection_patterns: List[str] = Field(
        default_factory=lambda: [
            r"ignore (previous|above|all) (instructions?|prompts?|rules?)",
            r"disregard (previous|above|all)",
            r"override (system|safety|constitutional)",
            r"disregard all",
            r"system\s*:\s*",
            r"<\|endoftext\|>",
            r"you (are|must) now",
            r"jailbreak",
            r"DAN mode",
            r"<\s*script\s*>",
            r"javascript\s*:",
        ]
    )

    override_keys: Set[str] = Field(
        default_factory=lambda: {
            "skip_router",
            "model_output",
            "model_metadata",
            "input_text_override",
            "force_model_output",
            "domain",
            "detectors",
        }
    )


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
        default_factory=list, description="Non-blocking warnings from validation"
    )
    sanitization_applied: bool = Field(
        default=False, description="Whether input was modified during sanitization"
    )

    model_config = ConfigDict(frozen=True)


# ============================================================================
# VALIDATOR
# ============================================================================


class InputValidator:
    """
    Validates inputs before constitutional evaluation.

    Validation failures are constitutional signals - they indicate
    that the system cannot evaluate the input with integrity.
    """

    _CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

        # Compile injection patterns
        self.injection_regexes = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.config.injection_patterns
        ]
        self.override_keys = set(self.config.override_keys)

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

        sanitized_text, text_sanitized = self._sanitize_string(text)
        if text_sanitized:
            sanitized = True
            warnings.append("Input text normalized and control characters removed")

        # Length check (after sanitization)
        if len(sanitized_text) > self.config.max_text_length:
            raise InputValidationError(
                f"Input text exceeds maximum length ({self.config.max_text_length} characters)",
                validation_type="size_limit",
                field="text",
                context={
                    "length": len(sanitized_text),
                    "max_length": self.config.max_text_length,
                },
            )

        # Injection detection
        if self.config.enable_injection_detection:
            for pattern in self.injection_regexes:
                if pattern.search(sanitized_text):
                    if self.config.reject_on_injection:
                        raise InputValidationError(
                            "Potentially malicious input detected",
                            validation_type="prompt_injection",
                            field="text",
                            context={"pattern": pattern.pattern},
                        )
                    warnings.append(
                        f"Potential prompt injection detected: pattern '{pattern.pattern}'"
                    )

        # Malicious pattern detection
        if self.config.enable_malicious_pattern_detection:
            # Check for excessive repetition (potential DoS)
            if self._has_excessive_repetition(sanitized_text):
                warnings.append("Excessive character repetition detected")

        return sanitized_text, warnings, sanitized

    def _sanitize_string(self, value: str) -> Tuple[str, bool]:
        normalized = unicodedata.normalize("NFKC", value)
        sanitized = self._CONTROL_CHAR_PATTERN.sub("", normalized)
        return sanitized, sanitized != value

    def _validate_context(self, context: Dict[str, Any]) -> tuple[Dict[str, Any], List[str], bool]:
        """
        Validate and sanitize context dictionary.

        Returns:
            (sanitized_context, warnings, was_sanitized)
        """
        warnings: List[str] = []
        sanitized = False

        if not isinstance(context, dict):
            raise InputValidationError(
                "Context must be a dictionary",
                validation_type="type_error",
                field="context",
                context={"received_type": type(context).__name__},
            )

        total_keys = self._count_keys(context)
        if total_keys > self.config.max_context_keys:
            raise InputValidationError(
                f"Context exceeds maximum key count ({self.config.max_context_keys})",
                validation_type="size_limit",
                field="context",
                context={
                    "keys": total_keys,
                    "max_keys": self.config.max_context_keys,
                },
            )

        context_copy, context_sanitized = self._sanitize_context_value(context, current_depth=0)
        if context_sanitized:
            sanitized = True
            warnings.append("Context values normalized and control characters removed")

        # Sanitize reserved keys that might interfere with governance
        reserved_keys = {
            "_escalation_override",
            "_skip_human_review",
            "_bypass_governance",
            "_suppress_dissent",
        }

        for key in reserved_keys:
            if key in context_copy:
                del context_copy[key]
                sanitized = True
                warnings.append(f"Removed reserved key: {key}")

        self._validate_override_keys(context_copy, warnings)

        # Size check + JSON serializability
        try:
            serialized = json.dumps(context_copy, ensure_ascii=True)
            size = len(serialized.encode("utf-8"))
        except (TypeError, ValueError) as e:
            raise InputValidationError(
                "Context must be JSON-serializable",
                validation_type="serialization_error",
                field="context",
                context={"error": str(e)},
            )

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

        # Depth check (prevent deeply nested malicious payloads)
        max_depth = self._get_dict_depth(context_copy)
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

        return context_copy, warnings, sanitized

    def _sanitize_context_value(self, value: Any, current_depth: int) -> Tuple[Any, bool]:
        if current_depth > self.config.max_context_depth:
            raise InputValidationError(
                f"Context nesting exceeds maximum depth ({self.config.max_context_depth})",
                validation_type="depth_limit",
                field="context",
                context={"max_depth": self.config.max_context_depth},
            )

        if isinstance(value, dict):
            sanitized_dict: Dict[str, Any] = {}
            changed = False
            for key, item in value.items():
                if not isinstance(key, str):
                    raise InputValidationError(
                        "Context keys must be strings",
                        validation_type="type_error",
                        field="context",
                        context={"received_type": type(key).__name__},
                    )
                sanitized_item, item_changed = self._sanitize_context_value(
                    item,
                    current_depth=current_depth + 1,
                )
                sanitized_dict[key] = sanitized_item
                changed = changed or item_changed
            return sanitized_dict, changed

        if isinstance(value, list):
            sanitized_list: List[Any] = []
            changed = False
            for item in value:
                sanitized_item, item_changed = self._sanitize_context_value(
                    item,
                    current_depth=current_depth + 1,
                )
                sanitized_list.append(sanitized_item)
                changed = changed or item_changed
            return sanitized_list, changed

        if isinstance(value, tuple):
            sanitized_tuple: List[Any] = []
            changed = True
            for item in value:
                sanitized_item, item_changed = self._sanitize_context_value(
                    item,
                    current_depth=current_depth + 1,
                )
                sanitized_tuple.append(sanitized_item)
                changed = changed or item_changed
            return sanitized_tuple, changed

        if isinstance(value, str):
            sanitized_value, value_changed = self._sanitize_string(value)
            if len(sanitized_value) > self.config.max_context_string_length:
                raise InputValidationError(
                    "Context string value exceeds maximum length",
                    validation_type="size_limit",
                    field="context",
                    context={
                        "length": len(sanitized_value),
                        "max_length": self.config.max_context_string_length,
                    },
                )
            return sanitized_value, value_changed

        if isinstance(value, (int, float, bool)) or value is None:
            return value, False

        raise InputValidationError(
            "Context values must be JSON-serializable primitives",
            validation_type="serialization_error",
            field="context",
            context={"received_type": type(value).__name__},
        )

    def _validate_override_keys(self, context: Dict[str, Any], warnings: List[str]) -> None:
        unknown_override_keys = {
            key
            for key in context
            if (key.startswith("_") or "override" in key or key.startswith("force_"))
            and key not in self.override_keys
        }
        if unknown_override_keys:
            raise InputValidationError(
                "Unknown context override keys detected",
                validation_type="override_key",
                field="context",
                context={"keys": sorted(unknown_override_keys)},
            )

        if "skip_router" in context:
            if not isinstance(context["skip_router"], bool):
                raise InputValidationError(
                    "skip_router must be a boolean",
                    validation_type="type_error",
                    field="context.skip_router",
                )
            if context["skip_router"] and "model_output" not in context:
                raise InputValidationError(
                    "model_output required when skip_router=True",
                    validation_type="override_missing",
                    field="context.model_output",
                )

        if "model_output" in context:
            model_output = context["model_output"]
            if isinstance(model_output, str):
                if len(model_output) > self.config.max_text_length:
                    raise InputValidationError(
                        "model_output exceeds maximum length",
                        validation_type="size_limit",
                        field="context.model_output",
                        context={
                            "length": len(model_output),
                            "max_length": self.config.max_text_length,
                        },
                    )
            elif model_output is None or isinstance(model_output, (dict, list, int, float, bool)):
                pass
            else:
                raise InputValidationError(
                    "model_output must be JSON-serializable",
                    validation_type="serialization_error",
                    field="context.model_output",
                    context={"received_type": type(model_output).__name__},
                )

        if "model_metadata" in context and not isinstance(context["model_metadata"], dict):
            raise InputValidationError(
                "model_metadata must be a dictionary",
                validation_type="type_error",
                field="context.model_metadata",
                context={"received_type": type(context["model_metadata"]).__name__},
            )

        if "input_text_override" in context:
            override_text = context["input_text_override"]
            if not isinstance(override_text, str):
                raise InputValidationError(
                    "input_text_override must be a string",
                    validation_type="type_error",
                    field="context.input_text_override",
                    context={"received_type": type(override_text).__name__},
                )
            if len(override_text) > self.config.max_text_length:
                raise InputValidationError(
                    "input_text_override exceeds maximum length",
                    validation_type="size_limit",
                    field="context.input_text_override",
                    context={
                        "length": len(override_text),
                        "max_length": self.config.max_text_length,
                    },
                )

        if "force_model_output" in context and not isinstance(context["force_model_output"], bool):
            raise InputValidationError(
                "force_model_output must be a boolean",
                validation_type="type_error",
                field="context.force_model_output",
                context={"received_type": type(context["force_model_output"]).__name__},
            )

        if "model_output" in context and not context.get("skip_router", False):
            warnings.append("model_output provided without skip_router=True; value will be ignored")

    def _count_keys(self, obj: Any) -> int:
        if isinstance(obj, dict):
            return sum(self._count_keys(value) for value in obj.values()) + len(obj)
        if isinstance(obj, list):
            return sum(self._count_keys(value) for value in obj)
        return 0

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
            if text[i] == text[i - 1]:
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

        return max(InputValidator._get_dict_depth(v, current_depth + 1) for v in d.values())


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
