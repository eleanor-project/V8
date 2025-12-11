from ..base import Detector
from ..utils import simple_pattern_detector

class EnvironmentalImpactDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(dump|release|dispose)\s+(waste|chemical|toxic|hazardous)\s+(into|in)\s+(river|lake|ocean|ground|environment)\b",
            r"\b(illegal|unauthorized)\s+dumping\b",
            r"\b(deforest|clear[- ]?cut|slash and burn)\b",
            r"\b(excessive emissions|pollute|contaminate)\b",
        ]
        keywords = ["environmental damage", "ecological harm", "pollution", "carbon footprint"]
        high_keywords = ["toxic waste", "hazardous waste", "illegal dumping"]
        return simple_pattern_detector(
            name="environmental_impact",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
