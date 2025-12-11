from ..base import Detector
from ..utils import simple_pattern_detector

class PrivacyDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(dox|doxx|doxxing)\b",
            r"\b(expose|reveal|leak|publish|share)\s+(personal|private|medical|financial)\s+(info|information|records|data)\b",
            r"\b(post|send)\s+(their|someone's|your)\s+(address|phone|ssn|social security|password)\b",
            r"\b(track|monitor|spy on)\s+(someone|them|users)\s+(without|secretly|covertly)\b",
        ]
        keywords = ["pii", "personal data", "confidential", "data leak", "without consent"]
        high_keywords = ["social security", "passport number", "credit card", "bank account"]
        return simple_pattern_detector(
            name="privacy",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
