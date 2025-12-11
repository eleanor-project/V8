from ..base import Detector
from ..utils import simple_pattern_detector

class PsychologicalHarmDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(kill yourself|suicide|self-harm|self harm|selfharm)\b",
            r"\b(you're worthless|no one cares|should die|better off dead)\b",
            r"\b(shame you|humiliate you|degrade you|dehumanize)\b",
            r"\b(trauma|triggering|panic attack)\b",
        ]
        keywords = [
            "bullying", "harassment", "intimidation", "emotional abuse",
            "psychological abuse", "threaten mental", "gaslight",
        ]
        high_keywords = ["kill yourself", "suicide", "self-harm", "better off dead"]
        return simple_pattern_detector(
            name="psychological_harm",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
