from ..base import Detector
from ..utils import simple_pattern_detector

class EmbeddingBiasDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        # Heuristic: look for value-laden stereotypes suggestive of biased associations
        regexes = [
            r"\b(typical|all|most)\s+(men|women|group|people)\s+(are|do|have|think)\b",
            r"\b(stereotype|bias|prejudiced)\b",
            r"\b(inherently|naturally)\s+(violent|lazy|criminal|inferior|superior)\b",
        ]
        keywords = ["embedding bias", "representation bias", "stereotypical association"]
        high_keywords = ["inherently inferior", "naturally violent", "genetically inferior"]
        return simple_pattern_detector(
            name="embedding_bias",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
