from ..base import Detector
from ..utils import simple_pattern_detector

class IrreversibleHarmDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(permanent|irreversible|cannot be undone|no going back|point of no return)\b",
            r"\b(delete forever|destroy completely|irreparable)\b",
        ]
        keywords = ["irreversible harm", "permanent damage", "cannot recover"]
        high_keywords = ["irreparable", "cannot be undone", "destroy completely"]
        return simple_pattern_detector(
            name="irreversible_harm",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
