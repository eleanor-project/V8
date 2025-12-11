from ..base import Detector
from ..utils import simple_pattern_detector

class DisparateImpactDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(disproportionate|disparate)\s+(impact|effect)\b",
            r"\b(affects|burdens)\s+(minorities|protected|vulnerable)\s+more\b",
            r"\b(adverse impact|four-fifths rule|80% rule)\b",
        ]
        keywords = ["disparate impact", "adverse impact", "unequal effect", "protected groups"]
        high_keywords = ["adverse impact", "four-fifths rule", "80% rule"]
        return simple_pattern_detector(
            name="disparate_impact",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
