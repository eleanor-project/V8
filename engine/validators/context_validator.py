"""Context validation and sanitization logic."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Tuple

from engine.exceptions import ValidationError as InputValidationError

from .config import ValidationConfig


class ContextValidator:
    """Validate and sanitize context dictionaries."""

    _CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.override_keys = set(self.config.override_keys)

    def validate(self, context: Dict[str, Any]) -> tuple[Dict[str, Any], List[str], bool]:
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

        total_keys = self._count_keys(context, visited=set())
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

        for key in list(context_copy.keys()):
            if key.startswith("_") and "override" not in key and key not in self.override_keys:
                del context_copy[key]
                sanitized = True
                warnings.append(f"Removed reserved key: {key}")

        self._validate_override_keys(context_copy, warnings)

        try:
            serialized = json.dumps(context_copy, ensure_ascii=True)
            size = len(serialized.encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                "Context must be JSON-serializable",
                validation_type="serialization_error",
                field="context",
                context={"error": str(exc)},
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

    def _sanitize_string(self, value: str) -> Tuple[str, bool]:
        normalized = unicodedata.normalize("NFKC", value)
        sanitized = self._CONTROL_CHAR_PATTERN.sub("", normalized)
        return sanitized, sanitized != value

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

    def _count_keys(self, obj: Any, visited: set[int]) -> int:
        if isinstance(obj, (dict, list)):
            obj_id = id(obj)
            if obj_id in visited:
                raise InputValidationError(
                    "Circular reference detected in context",
                    validation_type="circular_reference",
                    field="context",
                )
            visited.add(obj_id)

        if isinstance(obj, dict):
            return sum(self._count_keys(value, visited) for value in obj.values()) + len(obj)
        if isinstance(obj, list):
            return sum(self._count_keys(value, visited) for value in obj)
        return 0

    @staticmethod
    def _get_dict_depth(d: Any, current_depth: int = 0) -> int:
        """
        Calculate maximum nesting depth of dictionary.
        """
        if not isinstance(d, dict):
            return current_depth

        if not d:
            return current_depth + 1

        return max(ContextValidator._get_dict_depth(v, current_depth + 1) for v in d.values())


__all__ = ["ContextValidator"]
