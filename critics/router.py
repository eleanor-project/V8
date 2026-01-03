from typing import List, Callable
from concurrent.futures import ThreadPoolExecutor

from .base import CriticOutput
from .llm import LLMClient
from .truth import truth_critic
from .fairness import fairness_critic
from .risk import risk_critic
from .pragmatics import pragmatics_critic
from .autonomy import autonomy_critic
from .dignity import dignity_critic


CriticFn = Callable[[str, LLMClient], CriticOutput]

CRITICS: List[CriticFn] = [
    truth_critic,
    fairness_critic,
    risk_critic,
    pragmatics_critic,
    autonomy_critic,
    dignity_critic,
]


def run_critics(prompt: str, llm: LLMClient) -> List[CriticOutput]:
    with ThreadPoolExecutor(max_workers=len(CRITICS)) as pool:
        futures = [pool.submit(fn, prompt, llm) for fn in CRITICS]
        return [f.result() for f in futures]
