from ..base import Detector
from ..signals import DetectorSignal

class OmissionDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        # TODO: Implement real detection logic
        return DetectorSignal(
            violation=False,
            severity="S0",
            description="Stub detector for omission",
            confidence=0.0,
            metadata={}
        )
