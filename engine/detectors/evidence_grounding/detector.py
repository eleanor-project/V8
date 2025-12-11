from ..base import Detector
from ..utils import simple_pattern_detector

class EvidenceGroundingDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\baccording to\b",
            r"\bas reported by\b",
            r"\bsource:?\s+\w+",
            r"\bcitation needed\b",
            r"\bno evidence|lack of evidence|without evidence\b",
        ]
        keywords = ["grounded in", "based on evidence", "unsupported", "unverified"]
        high_keywords = ["lack of evidence", "unsupported claim", "citation needed"]
        return simple_pattern_detector(
            name="evidence_grounding",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
