# orchestrator/engine.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .critics import (
    call_rights_critic,
    call_fairness_critic,
    call_risk_critic,
    call_truth_critic,
    call_pragmatics_critic
)
from precedent import PrecedentStorage, PrecedentSearch

# Initialize precedent system
precedent_storage = PrecedentStorage()
precedent_search = PrecedentSearch(precedent_storage)

def run_all_critics(input_text: str) -> dict:
    """
    Run




