from ..base import Detector
from ..utils import simple_pattern_detector

class OmissionDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(omit|omit(?:ting)?|skip|leave out|exclude)\b",
            r"\b(not mention|avoid mentioning|leave unspecified)\b",
            r"\b(redact|withhold|conceal)\b",
        ]
        keywords = ["missing details", "incomplete", "lack of context", "no explanation"]
        return simple_pattern_detector(
            name="omission",
            text=text,
            regexes=regexes,
            keywords=keywords,
        )
