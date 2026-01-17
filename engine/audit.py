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
from typing import Any, List, Dict
from engine.schemas.pipeline_types import CriticResult


# Determine log file path
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "eleanor_audit.jsonl"


def ensure_log_dir():
    """Ensure the logs directory exists."""
    LOG_DIR.mkdir(exist_ok=True)


def _get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _format_precedent(item: Any) -> Any:
    precedent = _get_value(item, "precedent")
    if precedent:
        return precedent
    refs = _get_value(item, "precedent_refs")
    if isinstance(refs, list):
        return ", ".join(str(r) for r in refs) if refs else None
    return refs


def log_deliberation(
    prompt: str, critic_outputs: List[CriticResult], aggregation_result: Dict, backend: str = "mock"
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
                "critic": _get_value(c, "critic", _get_value(c, "critic_id", "unknown")),
                "concern": _get_value(c, "concern", _get_value(c, "justification", "")),
                "severity": _get_value(c, "severity", _get_value(c, "score", 0.0)),
                "principle": _get_value(c, "principle"),
                "uncertainty": _get_value(c, "uncertainty"),
                "rationale": _get_value(c, "rationale", _get_value(c, "justification", "")),
                "precedent": _format_precedent(c),
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
