from ..base import Detector
from ..utils import simple_pattern_detector

class DisparateTreatmentDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(deny|reject|refuse)\s+(service|access|opportunity)\s+(because|due to)\s+(race|gender|religion|age|disability)\b",
            r"\b(different rules|different standards|double standard)\s+for\s+(certain|specific)\s+(groups|people)\b",
            r"\b(exclude|exclude them|not allow)\s+(based on|because of)\s+(protected|protected class)\b",
        ]
        keywords = ["treat differently", "different treatment", "disparate treatment", "unfair criteria"]
        high_keywords = ["because of race", "because of gender", "because of religion", "because of age"]
        return simple_pattern_detector(
            name="disparate_treatment",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
