from ..base import Detector
from ..utils import simple_pattern_detector

class FeasibilityDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(impossible|not feasible|cannot be done|unrealistic|impractical)\b",
            r"\b(without resources|without budget|no time|insufficient)\b",
            r"\b(require[s]? technology that doesn't exist)\b",
        ]
        keywords = ["overly complex", "not scalable", "unrealistic timeline", "no resources"]
        return simple_pattern_detector(
            name="feasibility",
            text=text,
            regexes=regexes,
            keywords=keywords,
        )
