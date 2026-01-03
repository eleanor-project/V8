"""
ELEANOR V8 — Precedent Drift Detection
-----------------------------------------

Drift = Long-term deviation of the system's decision patterns compared to
institutional expectations.

Used for:
    - Calibration
    - Governance oversight
    - Detecting critic misbehavior
    - Ensuring constitutional stability over time
"""

from typing import List, Dict, Any
import statistics


class PrecedentDriftV8:
    def compute_drift(self, past_alignment_scores: List[float]) -> Dict[str, Any]:
        """
        past_alignment_scores: list of (0–1) alignment scores from prior decisions.

        Returns:
            drift_signal (0 = stable, 1 = major drift)
        """
        if not past_alignment_scores:
            return {"drift_score": 0.0, "signal": "stable"}

        mean = statistics.mean(past_alignment_scores)
        stdev = statistics.pstdev(past_alignment_scores)

        # High variance or mean far from 1.0 indicates drift
        drift_score = min(1.0, (abs(1 - mean) + stdev) / 2)

        signal = (
            "stable" if drift_score < 0.2 else "monitor" if drift_score < 0.5 else "drift_warning"
        )

        return {"drift_score": float(drift_score), "signal": signal}
