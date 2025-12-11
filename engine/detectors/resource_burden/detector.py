from ..base import Detector
from ..utils import simple_pattern_detector

class ResourceBurdenDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(excessive|unreasonable)\s+(cost|expense|burden|resources)\b",
            r"\b(resource[- ]?intensive|cost[- ]?prohibitive|too expensive)\b",
            r"\b(require|needs?)\s+(massive|significant|large)\s+(infrastructure|staff|budget|time)\b",
        ]
        keywords = ["over budget", "not scalable", "unsustainable cost", "high operational cost"]
        high_keywords = ["cost prohibitive", "unaffordable", "unsustainable"]
        return simple_pattern_detector(
            name="resource_burden",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
