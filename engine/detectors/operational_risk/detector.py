from ..base import Detector
from ..utils import simple_pattern_detector

class OperationalRiskDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(single point of failure|no redundancy|no backup)\b",
            r"\b(outage|downtime|service interruption)\b",
            r"\b(unmonitored|no monitoring|lack of monitoring)\b",
            r"\b(no incident response|no rollback|no recovery)\b",
        ]
        keywords = ["operational risk", "resilience gap", "no failover", "weak controls"]
        high_keywords = ["single point of failure", "no backup", "no rollback"]
        return simple_pattern_detector(
            name="operational_risk",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
