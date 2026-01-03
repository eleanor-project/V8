"""
ELEANOR Audit Logging.

Every deliberation leaves a trail.
This module records:
- User prompts
- Critic outputs
- Aggregation results
- Timestamps
- Backend used

This enables:
- Retrospective analysis
- Pattern detection
- Accountability
- Governance transparency
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict
from critics.schema import CriticOutput


# Determine log file path
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "eleanor_audit.jsonl"


def ensure_log_dir():
    """Ensure the logs directory exists."""
    LOG_DIR.mkdir(exist_ok=True)


def log_deliberation(
    prompt: str, critic_outputs: List[CriticOutput], aggregation_result: Dict, backend: str = "mock"
):
    """
    Log a complete deliberation to audit trail.

    Args:
        prompt: The user's input prompt
        critic_outputs: List of critic reasoning outputs
        aggregation_result: The final aggregated result
        backend: Which LLM backend was used
    """
    ensure_log_dir()

    # Build audit record
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "backend": backend,
        "prompt": prompt,
        "critics": [
            {
                "critic": c.critic,
                "concern": c.concern,
                "severity": c.severity,
                "principle": c.principle,
                "uncertainty": c.uncertainty,
                "rationale": c.rationale,
                "precedent": c.precedent,
            }
            for c in critic_outputs
        ],
        "deliberation": aggregation_result.get("deliberation", []),
        "final_answer": aggregation_result.get("final_answer", ""),
    }

    # Append to JSONL file
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def get_recent_deliberations(limit: int = 10) -> List[Dict]:
    """
    Retrieve the most recent deliberations from the audit log.

    Args:
        limit: Maximum number of records to return

    Returns:
        List of audit records, most recent first
    """
    if not LOG_FILE.exists():
        return []

    records = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Return most recent first
    return records[-limit:][::-1]
