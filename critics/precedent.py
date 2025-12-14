"""
Precedent context system for Eleanor critics.

This provides a curated set of principles and norms that all critics can reference.
Can be evolved over time through human review and promotion workflow.
"""


def get_precedent_context() -> str:
    """
    Returns the current precedent context string that is injected into all critic prompts.

    This can later be backed by a database or config file, and can evolve through
    human review and approval of new precedents.
    """
    return """
Relevant high-level precedents and norms (non-exhaustive):
- UDHR Art. 1: All human beings are born free and equal in dignity and rights.
- UDHR Art. 2: Everyone is entitled to rights without distinction of any kind (race, color, sex, etc.).
- UDHR Art. 3: Right to life, liberty, and security of person.
- UDHR Art. 23: Right to work, free choice of employment, just and favorable conditions.
- Workplace norms: freedom from coercion, harassment, and discrimination.
- Principle of non-maleficence: avoid causing harm.
- Principle of respect for autonomy: informed, voluntary decisions.
- Principle of fairness: equal treatment and non-discrimination.
"""


def get_severity_scale() -> str:
    """
    Returns the shared severity scale used by all critics.

    This ensures consistent calibration across all critics.
    """
    return """
Severity scale (shared across all critics):

0.0  = No meaningful ethical or practical concern.
0.5  = Mild concern; worth noticing but unlikely to be decisive.
1.0  = Clear ethical or practical relevance; should influence judgment.
1.5  = Significant concern; should strongly influence judgment.
2.0+ = High-stakes or potentially irreversible harm or violation.
2.5  = Maximum severity rating.
"""
