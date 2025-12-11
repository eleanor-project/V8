from ..base import Detector
from ..signals import DetectorSignal

class TimeConstraintsDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        # TODO: Implement real detection logic
        return DetectorSignal(
            violation=False,
            severity="S0",
            description="Stub detector for time constraints",
            confidence=0.0,
            metadata={}
        )
