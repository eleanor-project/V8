from ..base import Detector
from ..utils import simple_pattern_detector

class CascadingFailureDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(chain reaction|domino effect|cascade|ripple effect)\b",
            r"\b(systemic|widespread|large-?scale)\s+(failure|collapse|damage)\b",
            r"\b(one failure leads to|could trigger further|propagate failure)\b",
        ]
        keywords = ["cascading failure", "interdependent failure", "system-wide outage"]
        high_keywords = ["domino effect", "systemic collapse"]
        return simple_pattern_detector(
            name="cascading_failure",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
