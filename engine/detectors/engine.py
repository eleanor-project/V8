import asyncio
from typing import Dict, Any
from .signals import DetectorSignal

class DetectorEngineV8:
    def __init__(self, detectors: Dict[str, Any], timeout: float = 5.0):
        self.detectors = detectors
        self.timeout = timeout

    async def run(self, text: str, context: dict):
        tasks = {
            name: asyncio.create_task(det.detect(text, context))
            for name, det in self.detectors.items()
        }
        results = {}
        for name, task in tasks.items():
            try:
                sig: DetectorSignal = await asyncio.wait_for(task, self.timeout)
            except Exception as e:
                sig = DetectorSignal(violation=False, description=str(e))
            results[name] = sig
        return results
