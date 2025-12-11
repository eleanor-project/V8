from ..base import Detector
from ..utils import simple_pattern_detector

class ProceduralFairnessDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(no appeal|cannot appeal|no review)\b",
            r"\b(without hearing|without notice|without due process)\b",
            r"\b(biased|unfair|arbitrary)\s+(process|procedure|decision)\b",
            r"\b(no evidence|ignore evidence|skip evidence)\b",
        ]
        keywords = ["lack of transparency", "conflict of interest", "no recourse", "no justification"]
        high_keywords = ["without due process", "no appeal", "arbitrary decision"]
        return simple_pattern_detector(
            name="procedural_fairness",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
