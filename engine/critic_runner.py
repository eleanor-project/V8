from llm.factory import get_llm
from critics.truth import TruthCritic
from critics.fairness import FairnessCritic
from critics.risk import RiskCritic
from critics.pragmatics import PragmaticsCritic


def run_critics(prompt: str):
    """
    Execute all constitutional critics independently.
    """
    llm = get_llm()

    critics = [
        TruthCritic(llm),
        FairnessCritic(llm),
        RiskCritic(llm),
        PragmaticsCritic(llm),
    ]

    results = []
    for critic in critics:
        results.append(critic.evaluate(prompt))

    return results
