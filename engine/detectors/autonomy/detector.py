from ..base import Detector
from ..utils import simple_pattern_detector

class AutonomyDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(no choice|you must|you have to|do as I say|or else)\b",
            r"\b(ignore|bypass)\s+(consent|permission)\b",
            r"\b(trick|manipulat(e|ing)|gaslight)\b",
            r"\b(force|forced|compel|coerce|cannot refuse)\b",
        ]
        keywords = ["without permission", "covertly", "secretly", "against their will"]
        high_keywords = ["or else", "threaten", "force you", "make you"]
        return simple_pattern_detector(
            name="autonomy/consent",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
