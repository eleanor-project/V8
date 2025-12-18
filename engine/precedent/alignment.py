"""
ELEANOR V8 — Precedent Alignment & Drift Detection Engine
---------------------------------------------------------

This module computes:

• Semantic alignment between current critic findings and retrieved precedents
• Constitutional conflict scoring
• Precedent support strength
• Jurisprudential drift detection (is the system departing from historical rulings?)
• Cluster consensus vs conflict clustering
• Novelty indicators (no similar past cases)
• Stability metadata for the Aggregator and Evidence Recorder

Inputs:
    critics: { critic_name: { severity, violations, justification } }
    precedent_cases: [
        {
            "text": <precedent text>,
            "metadata": {
                "decision": "allow|deny|constrained_allow|escalate",
                "critic_severity": {...},
                "timestamp": ...
            },
            "embedding": [...]
        }
    ]
    query_embedding: vector

Outputs:
    {
        "alignment_score": float (-1 to 1),
        "support_strength": float (0 to 1),
        "conflict_level": float (0 to 1),
        "drift_score": float (0 to 1),
        "clusters": [...],
        "is_novel": bool,
        "analysis": "text summary"
    }
"""

from typing import List, Dict, Any
import math
import statistics


# ============================================================
# Utility: cosine similarity
# ============================================================

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ============================================================
# Precedent Alignment Engine
# ============================================================

class PrecedentAlignmentEngineV8:

    def __init__(self):
        pass

    # ----------------------------------------------------------
    # MAIN ENTRYPOINT
    # ----------------------------------------------------------
    def analyze(
        self,
        critics: Dict[str, Dict[str, Any]],
        precedent_cases: List[Dict[str, Any]],
        query_embedding: List[float],
    ) -> Dict[str, Any]:
        """
        Returns full structured precedent analysis.
        """

        if not precedent_cases:
            return self._novel_case()

        # Normalize missing fields to avoid crashes on partial precedent entries
        normalized_cases = []
        for case in precedent_cases:
            normalized_cases.append({
                **case,
                "embedding": case.get("embedding", []),
                "metadata": case.get("metadata", {}),
            })
        precedent_cases = normalized_cases

        # Step 1: similarity scores
        similarities = [cosine(query_embedding, p["embedding"]) for p in precedent_cases]

        # Step 2: compute alignment with outcomes
        alignment_score = self._compute_alignment(similarities, precedent_cases)

        # Step 3: conflict detection
        conflict_level = self._compute_conflict(precedent_cases, similarities)

        # Step 4: support strength (consensus weight)
        support_strength = self._compute_support(alignment_score, conflict_level)

        # Step 5: jurisprudential drift detection
        drift_score = self._compute_drift(critics, precedent_cases, similarities)

        # Step 6: clustering metadata
        clusters = self._cluster_cases(precedent_cases, similarities)

        # Build final package
        return {
            "alignment_score": alignment_score,
            "support_strength": support_strength,
            "conflict_level": conflict_level,
            "drift_score": drift_score,
            "clusters": clusters,
            "is_novel": False,
            "analysis": self._summary(alignment_score, support_strength, conflict_level, drift_score)
        }

    # ----------------------------------------------------------
    # NOVEL CASE HANDLER
    # ----------------------------------------------------------
    def _novel_case(self) -> Dict[str, Any]:
        return {
            "alignment_score": 0.0,
            "support_strength": 0.0,
            "conflict_level": 0.0,
            "drift_score": 0.0,
            "clusters": [],
            "is_novel": True,
            "analysis": "No relevant precedent found; treat as novel case."
        }

    # ----------------------------------------------------------
    # STEP 1: Precedent alignment computation
    # ----------------------------------------------------------
    def _compute_alignment(self, similarities: List[float], cases: List[Dict[str, Any]]) -> float:
        """
        alignment_score = weighted agreement between similar cases
        Returns a value in [-1, 1]:

        -1.0 → strongly contradicts precedent
         0.0 → no consistent pattern
         1.0 → strongly supported by precedent
        """

        weighted = 0.0
        total_sim = sum(similarities) or 1.0

        for sim, case in zip(similarities, cases):
            decision = case["metadata"].get("decision", "allow")

            # Map decisions to directional support:
            #   allow → +1
            #   constrained_allow → +0.5
            #   escalate → 0
            #   deny → -1
            if decision == "allow":
                score = 1.0
            elif decision == "constrained_allow":
                score = 0.5
            elif decision == "escalate":
                score = 0.0
            elif decision == "deny":
                score = -1.0
            else:
                score = 0.0

            weighted += sim * score

        return max(-1.0, min(1.0, weighted / total_sim))

    # ----------------------------------------------------------
    # STEP 2: Conflict detection
    # ----------------------------------------------------------
    def _compute_conflict(self, cases, similarities):
        """
        Measures contradictions among similar cases.
        High conflict → many similarly relevant precedents disagree.

        conflict_level = variance of decision values weighted by similarity
        """

        decision_map = {
            "allow": 1.0,
            "constrained_allow": 0.5,
            "escalate": 0.0,
            "deny": -1.0
        }

        weighted_scores = []
        for sim, case in zip(similarities, cases):
            decision = case["metadata"].get("decision", "allow")
            weighted_scores.append(decision_map.get(decision, 0.0) * sim)

        if len(weighted_scores) <= 1:
            return 0.0

        var = statistics.pvariance(weighted_scores)
        return max(0.0, min(1.0, var))

    # ----------------------------------------------------------
    # STEP 3: Support strength
    # ----------------------------------------------------------
    def _compute_support(self, alignment, conflict):
        """
        Measures how reliable and consistent the precedent support is.

        High support: strong alignment + low conflict.
        """

        support = alignment * (1 - conflict)
        # Normalize to 0–1
        return (support + 1) / 2

    # ----------------------------------------------------------
    # STEP 4: Drift detection
    # ----------------------------------------------------------
    def _compute_drift(self, critics, cases, similarities):
        """
        Measures how far the current case deviates from the
        "center of mass" of historical precedent.

        Drift increases when:
         • critic severities differ from past cases of similar shape
         • precedent cluster is internally stable but disagrees with critics
        """

        if not cases:
            return 0.0

        # Compute precedent aggregate severity
        precedent_severities = []

        for case in cases:
            sev_map = case["metadata"].get("critic_severity", {})
            precedent_severities.append(
                statistics.mean(sev_map.values()) if sev_map else 0.0
            )

        avg_precedent = statistics.mean(precedent_severities)
        avg_current = statistics.mean(c["severity"] for c in critics.values())

        # Drift = difference normalized to [0,1]
        diff = abs(avg_current - avg_precedent) / 3.0
        return max(0.0, min(1.0, diff))

    # ----------------------------------------------------------
    # STEP 5: Precedent clustering metadata
    # ----------------------------------------------------------
    def _cluster_cases(self, cases, similarities):
        """
        Lightweight cluster labeling based on decision distribution.
        """

        clusters = {
            "supportive": [],
            "neutral": [],
            "contradictory": []
        }

        for sim, case in zip(similarities, cases):
            decision = case["metadata"].get("decision", "allow")
            if decision in ("allow", "constrained_allow"):
                clusters["supportive"].append(case)
            elif decision == "escalate":
                clusters["neutral"].append(case)
            else:
                clusters["contradictory"].append(case)

        return clusters

    # ----------------------------------------------------------
    # STEP 6: Natural language summary
    # ----------------------------------------------------------
    def _summary(self, alignment, support, conflict, drift):
        """
        Returns a human-readable jurisprudential diagnostic.
        """

        return (
            f"Precedent alignment: {alignment:.2f}. "
            f"Support strength: {support:.2f}. "
            f"Conflict level: {conflict:.2f}. "
            f"Drift score: {drift:.2f}. "
        )
