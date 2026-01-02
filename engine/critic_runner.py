from llm.factory import get_llm
from critics.truth import truth_critic
from critics.fairness import fairness_critic
from critics.risk import risk_critic
from critics.pragmatics import pragmatics_critic


def run_critics(prompt: str):
    """
    Execute all constitutional critics independently.
    """
    llm = get_llm()

    return [
        truth_critic(prompt, llm),
        fairness_critic(prompt, llm),
        risk_critic(prompt, llm),
        pragmatics_critic(prompt, llm),
    ]
