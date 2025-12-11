from ..base import Detector
from ..utils import simple_pattern_detector

class FactualAccuracyDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(\d{2,3}\.\d+%|\d{2,3}%|over \d{2,3}%)\b",
            r"\b(studies show|research proves|scientists confirm)\b",
            r"\b(according to|as reported by|as stated by)\b",
            r"\b(source:|citation needed|unverified)\b",
        ]
        keywords = ["fact check", "verify", "unverified", "citation needed", "data shows"]
        high_keywords = ["research proves", "scientists confirm"]
        return simple_pattern_detector(
            name="factual_accuracy",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
