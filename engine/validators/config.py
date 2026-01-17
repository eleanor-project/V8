"""Validation configuration for input and context checks."""

from typing import List, Set

from pydantic import BaseModel, Field


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

    injection_patterns: List[str] = Field(
        default_factory=lambda: [
            r"ignore\s+(?:previous|above|all)\s+.*?(?:instructions?|prompts?|rules?)",
            r"disregard (previous|above|all)",
            r"disregard constitution",
            r"override (system|safety|constitutional)",
            r"disregard all",
            r"system\s*:\s*",
            r"<\|im_start\|>",
            r"<\|endoftext\|>",
            r"\bhuman\s*:",
            r"new instructions",
            r"reset your guidelines",
            r"follow these rules",
            r"you (are|must) now",
            r"jailbreak",
            r"DAN mode",
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


__all__ = ["ValidationConfig"]
