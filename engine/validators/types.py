"""Validated input type definitions."""

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class ValidatedInput(BaseModel):
    """
    Input that has passed validation checks.

    Presence of this type guarantees input has been sanitized and
    is safe for constitutional evaluation.
    """

    text: str = Field(..., description="Validated input text")
    context: Dict[str, Any] = Field(default_factory=dict, description="Validated context")
    trace_id: str = Field(..., description="Audit trail identifier")

    validation_warnings: List[str] = Field(
        default_factory=list, description="Non-blocking warnings from validation"
    )
    sanitization_applied: bool = Field(
        default=False, description="Whether input was modified during sanitization"
    )

    model_config = ConfigDict(frozen=True)

    def __iter__(self):
        # Provide tuple-unpacking support (text, context) for legacy callers.
        return iter((self.text, self.context))


__all__ = ["ValidatedInput"]
