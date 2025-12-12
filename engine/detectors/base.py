"""
ELEANOR V8 — Base Detector
---------------------------

Abstract base class for all detectors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from .signals import DetectorSignal


class Detector(ABC):
    """
    Abstract base class for all detectors.

    All detectors must implement the detect() method which analyzes text
    and returns a DetectorSignal with severity, violations, and evidence.
    """

    @abstractmethod
    async def detect(self, text: str, context: Dict[str, Any]) -> DetectorSignal:
        """
        Detect violations in the provided text.

        Args:
            text: Text to analyze (typically model output)
            context: Additional context (input, domain, etc.)

        Returns:
            DetectorSignal with severity, violations, and evidence
        """
        ...
