from ..base import Detector
from ..utils import simple_pattern_detector

class CascadingPragmaticFailureDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(if this fails then everything fails|one failure will break the whole system)\b",
            r"\b(no rollback|no contingency|no fallback)\b",
            r"\b(chained dependencies|tight coupling|fragile integration)\b",
        ]
        keywords = ["cascading pragmatic failure", "no contingency", "no rollback", "no failover"]
        high_keywords = ["everything fails", "break the whole system"]
        return simple_pattern_detector(
            name="cascading_pragmatic_failure",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
