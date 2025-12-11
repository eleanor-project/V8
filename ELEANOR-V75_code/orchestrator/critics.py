# orchestrator/critics.py
import re
import requests
from typing import Dict
from .config import settings

def _parse_critic_output(text: str) -> Dict[str, str]:
    """
    Parse critic structured text into a dict.
    Expected format:
    <Critic Name> Assessment:
    - Claim: ...
    - Evidence: ...
    - Constitutional Principle: ...
    - Confidence: 0.95
    - Mitigation: ...
    """
    def extract(label: str, default: str = "") -> str:
        pattern = rf"- {label}:\s*(.*)"
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else default
    
    claim = extract("Claim")
    evidence = extract("Evidence")
    principle = extract("Constitutional Principle")
    confidence_raw = extract("Confidence", "0.0")
    mitigation = extract("Mitigation")
    
    try:
        confidence = float(re.findall(r"[\d.]+", confidence_raw)[0])
    except Exception:
        confidence = 0.0
    
    return {
        "claim": claim,
        "evidence": evidence,
        "constitutional_principle": principle,
        "confidence": confidence,
        "mitigation": mitigation,
    }

def call_critic(model_name: str, input_text: str, critic_type: str) -> Dict[str, str]:
    """
    Call a specific Eleanor critic model via Ollama's /api/chat endpoint.
    """
    url = f"{settings.OLLAMA_HOST}/api/chat"
    
    system_prompts = {
        "rights": ("You are the Eleanor Rights Critic. "
                  "Evaluate for rights, dignity, non-discrimination, autonomy, and privacy. "
                  "Respond ONLY in the following format:\n\n"
                  "Rights Critic Assessment:\n"
                  "- Claim: ...\n"
                  "- Evidence: ...\n"
                  "- Constitutional Principle: ...\n"
                  "- Confidence: <0.0–1.0>\n"
                  "- Mitigation: ...\n"),
        "fairness": ("You are the Eleanor Fairness Critic. "
                    "Evaluate for distributional fairness, subgroup impacts, and equity. "
                    "Respond ONLY in the following format:\n\n"
                    "Fairness Critic Assessment:\n"
                    "- Claim: ...\n"
                    "- Evidence: ...\n"
                    "- Constitutional Principle: ...\n"
                    "- Confidence: <0.0–1.0>\n"
                    "- Mitigation: ...\n"),
        "risk": ("You are the Eleanor Risk Critic. "
                "Evaluate for harm likelihood, severity, reversibility, and precautionary principles. "
                "Respond ONLY in the following format:\n\n"
                "Risk Critic Assessment:\n"
                "- Claim: ...\n"
                "- Evidence: ...\n"
                "- Constitutional Principle: ...\n"
                "- Confidence: <0.0–1.0>\n"
                "- Mitigation: ...\n"),
        "truth": ("You are the Eleanor Truth Critic. "
                 "Evaluate for factual accuracy, deception risk, omission, and misleading content. "
                 "Respond ONLY in the following format:\n\n"
                 "Truth Critic Assessment:\n"
                 "- Claim: ...\n"
                 "- Evidence: ...\n"
                 "- Constitutional Principle: ...\n"
                 "- Confidence: <0.0–1.0>\n"
                 "- Mitigation: ...\n"),
        "pragmatics": ("You are the Eleanor Pragmatics Critic. "
                      "Evaluate for feasibility, cost, proportionality, and operational constraints. "
                      "Respond ONLY in the following format:\n\n"
                      "Pragmatics Critic Assessment:\n"
                      "- Claim: ...\n"
                      "- Evidence: ...\n"
                      "- Constitutional Principle: ...\n"
                      "- Confidence: <0.0–1.0>\n"
                      "- Mitigation: ...\n"),
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompts[critic_type]},
            {"role": "user", "content": input_text},
        ],
        "stream": False,
    }
    
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Empty response from {critic_type.title()} Critic model")
    
    return _parse_critic_output(content)

def call_rights_critic(input_text: str) -> Dict[str, str]:
    return call_critic(settings.RIGHTS_MODEL_NAME, input_text, "rights")

def call_fairness_critic(input_text: str) -> Dict[str, str]:
    return call_critic(settings.FAIRNESS_MODEL_NAME, input_text, "fairness")

def call_risk_critic(input_text: str) -> Dict[str, str]:
    return call_critic(settings.RISK_MODEL_NAME, input_text, "risk")

def call_truth_critic(input_text: str) -> Dict[str, str]:
    return call_critic(settings.TRUTH_MODEL_NAME, input_text, "truth")

def call_pragmatics_critic(input_text: str) -> Dict[str, str]:
    return call_critic(settings.PRAGMATICS_MODEL_NAME, input_text, "pragmatics")
