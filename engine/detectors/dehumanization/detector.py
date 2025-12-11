from ..base import Detector
from ..utils import simple_pattern_detector

class DehumanizationDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(animals?|vermin|parasites?|cockroaches?|rats?)\b",
            r"\b(sub-?human|less than human|not (fully )?human)\b",
            r"\b(inferior|superior)\s+(race|people|beings?)\b",
            r"\b(remove|eliminate|wipe out)\s+(them|those people|that group)\b",
        ]
        keywords = ["worthless", "trash", "scum", "subhuman", "don't deserve to live"]
        high_keywords = ["wipe out", "eliminate", "exterminate"]
        return simple_pattern_detector(
            name="dehumanization",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
