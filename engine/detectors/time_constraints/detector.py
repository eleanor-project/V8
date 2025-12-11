from ..base import Detector
from ..signals import DetectorSignal

class TimeConstraintsDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        lowered = text.lower()
        urgency_terms = [
            "immediately", "asap", "right now", "urgent", "deadline",
            "within hours", "no time", "time critical", "rush", "emergency",
        ]
        matches = [t for t in urgency_terms if t in lowered]
        violation = bool(matches)
        return DetectorSignal(
            violation=violation,
            severity="S1" if violation else "S0",
            description="Time-critical or urgent request detected" if violation else "No time constraint indicators",
            confidence=0.55 if violation else 0.0,
            metadata={"terms": matches}
        )
