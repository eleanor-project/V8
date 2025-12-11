# file: orchestrator/decision.py

def aggregate_decision(critic_outputs: dict) -> dict:
    """
    Applies Eleanor's lexicographic constitutional hierarchy
    AND uncertainty-aware reasoning.
    """
    
    rights = critic_outputs["rights"]
    fairness = critic_outputs["fairness"]
    risk = critic_outputs["risk"]
    truth = critic_outputs["truth"]
    pragmatics = critic_outputs["pragmatics"]
    
    # 1. HARD CONSTRAINTS — Rights Violations
    if rights["Constitutional Principle"] != "None" and rights["Confidence"] > 0.50:
        return {
            "Final Decision": "Rights-based constraint triggered.",
            "Grounding Critic": "Rights",
            "Details": rights
        }
    
    # 2. SECONDARY — Unacceptable Risk
    if risk["Constitutional Principle"] != "None" and risk["Confidence"] > 0.60:
        return {
            "Final Decision": "High-risk action prohibited.",
            "Grounding Critic": "Risk",
            "Details": risk
        }
    
    # 3. FAIRNESS — Distributional Harms
    if fairness["Constitutional Principle"] != "None" and fairness["Confidence"] > 0.60:
        return {
            "Final Decision": "Fairness constraint triggered.",
            "Grounding Critic": "Fairness",
            "Details": fairness
        }
    
    # 4. TRUTH — Disinformation or deception
    if truth["Constitutional Principle"] != "None" and truth["Confidence"] > 0.70:
        return {
            "Final Decision": "Truthfulness constraint triggered.",
            "Grounding Critic": "Truth",
            "Details": truth
        }
    
    # 5. PRAGMATICS — Feasibility concerns
    if pragmatics["Confidence"] < 0.40:
        return {
            "Final Decision": "Insufficient feasibility certainty.",
            "Grounding Critic": "Pragmatics",
            "Details": pragmatics
        }
    
    # If no constraints triggered → permissible with mitigation
    return {
        "Final Decision": "No constitutional barriers detected.",
        "Mitigations": {
            "rights": rights["Mitigation"],
            "fairness": fairness["Mitigation"],
            "risk": risk["Mitigation"],
            "truth": truth["Mitigation"],
            "pragmatics": pragmatics["Mitigation"],
        },
        "critic_outputs": critic_outputs
    }
