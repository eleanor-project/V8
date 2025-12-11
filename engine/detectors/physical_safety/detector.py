from ..base import Detector
from ..utils import simple_pattern_detector

class PhysicalSafetyDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(make|build|create)\s+(a|an)?\s*(bomb|explosive|weapon|molotov)\b",
            r"\b(bypass|disable|break)\s+(safety|alarm|lock|guard)\b",
            r"\b(harm|injure|kill)\s+(someone|people|them)\b",
            r"\b(poison|toxic|hazardous)\s+(substance|chemical)\b",
        ]
        keywords = ["unsafe", "dangerous", "handle chemicals", "risk of fire", "explosive"]
        high_keywords = ["kill", "bomb", "weapon", "molotov", "explosive"]
        return simple_pattern_detector(
            name="physical_safety",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
