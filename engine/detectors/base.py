from abc import ABC, abstractmethod
from .signals import DetectorSignal

class Detector(ABC):
    @abstractmethod
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        ...
