from ..base import Detector
from ..utils import simple_pattern_detector

class ContradictionDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(on the other hand|however|but then|contradict|inconsistent)\b",
            r"\b(previously said|earlier stated|earlier we said)\b",
            r"\b(both true and false|mutually exclusive)\b",
        ]
        keywords = ["conflict with above", "conflicts earlier", "inconsistent answer"]
        return simple_pattern_detector(
            name="contradiction",
            text=text,
            regexes=regexes,
            keywords=keywords,
        )
