from ..base import Detector
from ..utils import simple_pattern_detector

class StructuralDisadvantageDetector(Detector):
    async def detect(self, text: str, context: dict) -> DetectorSignal:
        regexes = [
            r"\b(systemic|structural|institutional)\s+(bias|racism|discrimination|inequality)\b",
            r"\b(marginalized|underrepresented|disadvantaged)\s+(groups?|communities)\b",
            r"\b(unequal access|barriers to|exclusion from)\b",
        ]
        keywords = ["historical injustice", "structural inequity", "systemic exclusion", "legacy bias"]
        high_keywords = ["structural racism", "institutional racism", "systemic discrimination"]
        return simple_pattern_detector(
            name="structural_disadvantage",
            text=text,
            regexes=regexes,
            keywords=keywords,
            high_keywords=high_keywords,
        )
