"""
Tests for Precedent Conflict and Drift Detection
------------------------------------------------

Tests PrecedentConflictV8 and PrecedentDriftV8 modules.
"""

from engine.precedent.conflict import PrecedentConflictV8
from engine.precedent.drift import PrecedentDriftV8


# ============================================================
# Precedent Conflict Tests
# ============================================================

def test_conflict_no_conflict_when_no_precedent():
    """Test that no conflict is detected when there's no precedent."""
    detector = PrecedentConflictV8()

    result = detector.detect(
        precedent_case=None,
        deliberation_state={"values_violated": ["privacy", "autonomy"]}
    )

    assert result["conflict_detected"] is False
    assert result["reasons"] == []


def test_conflict_no_conflict_when_identical():
    """Test that no conflict is detected when precedent matches current state."""
    detector = PrecedentConflictV8()

    precedent = {
        "violated_values": ["privacy"],
        "priority_order": ["dignity", "autonomy", "privacy"]
    }

    deliberation = {
        "values_violated": ["privacy"],
        "constitutional_priority_order": ["dignity", "autonomy", "privacy"]
    }

    result = detector.detect(precedent, deliberation)

    assert result["conflict_detected"] is False
    assert result["reasons"] == []


def test_conflict_newly_violated_values():
    """Test detection of newly violated values."""
    detector = PrecedentConflictV8()

    precedent = {
        "violated_values": ["privacy"],
        "priority_order": ["dignity", "autonomy"]
    }

    deliberation = {
        "values_violated": ["privacy", "autonomy"],  # autonomy is newly violated
        "constitutional_priority_order": ["dignity", "autonomy"]
    }

    result = detector.detect(precedent, deliberation)

    assert result["conflict_detected"] is True
    assert len(result["reasons"]) == 1
    assert "autonomy" in result["reasons"][0]
    assert "Newly violated values" in result["reasons"][0]


def test_conflict_reversed_violations():
    """Test detection of reversed violations."""
    detector = PrecedentConflictV8()

    precedent = {
        "violated_values": ["privacy", "autonomy"],
        "priority_order": ["dignity"]
    }

    deliberation = {
        "values_violated": ["privacy"],  # autonomy no longer violated
        "constitutional_priority_order": ["dignity"]
    }

    result = detector.detect(precedent, deliberation)

    assert result["conflict_detected"] is True
    assert len(result["reasons"]) == 1
    assert "autonomy" in result["reasons"][0]
    assert "Reversal" in result["reasons"][0]


def test_conflict_priority_mismatch():
    """Test detection of priority hierarchy mismatch."""
    detector = PrecedentConflictV8()

    precedent = {
        "violated_values": [],
        "priority_order": ["dignity", "autonomy", "privacy"]
    }

    deliberation = {
        "values_violated": [],
        "constitutional_priority_order": ["autonomy", "dignity", "privacy"]  # Different order
    }

    result = detector.detect(precedent, deliberation)

    assert result["conflict_detected"] is True
    assert len(result["reasons"]) == 1
    assert "Priority hierarchy mismatch" in result["reasons"][0]


def test_conflict_multiple_conflicts():
    """Test detection of multiple types of conflicts simultaneously."""
    detector = PrecedentConflictV8()

    precedent = {
        "violated_values": ["privacy"],
        "priority_order": ["dignity", "autonomy"]
    }

    deliberation = {
        "values_violated": ["autonomy"],  # Different violations
        "constitutional_priority_order": ["autonomy", "dignity"]  # Different priority
    }

    result = detector.detect(precedent, deliberation)

    assert result["conflict_detected"] is True
    assert len(result["reasons"]) >= 2  # Multiple conflicts detected


# ============================================================
# Precedent Drift Tests
# ============================================================

def test_drift_empty_scores():
    """Test drift computation with no historical scores."""
    detector = PrecedentDriftV8()

    result = detector.compute_drift([])

    assert result["drift_score"] == 0.0
    assert result["signal"] == "stable"


def test_drift_stable_high_alignment():
    """Test drift detection with stable high alignment scores."""
    detector = PrecedentDriftV8()

    # Scores close to 1.0 with low variance
    scores = [0.95, 0.97, 0.96, 0.98, 0.94, 0.96]

    result = detector.compute_drift(scores)

    assert result["drift_score"] < 0.2
    assert result["signal"] == "stable"


def test_drift_monitor_moderate_variance():
    """Test drift detection with moderate variance requiring monitoring."""
    detector = PrecedentDriftV8()

    # Scores that should produce monitor signal
    scores = [0.55, 0.65, 0.60, 0.70, 0.62, 0.68]

    result = detector.compute_drift(scores)

    # These scores have moderate deviation from 1.0, should get monitor or higher
    assert result["drift_score"] > 0.15
    assert result["signal"] in ["monitor", "drift_warning"]


def test_drift_warning_high_variance():
    """Test drift detection with high variance triggering warning."""
    detector = PrecedentDriftV8()

    # High variance scores
    scores = [0.1, 0.9, 0.2, 0.8, 0.15, 0.85]

    result = detector.compute_drift(scores)

    # High variance should produce elevated drift score
    assert result["drift_score"] > 0.3
    assert result["signal"] in ["monitor", "drift_warning"]


def test_drift_warning_low_alignment():
    """Test drift detection with consistently low alignment scores."""
    detector = PrecedentDriftV8()

    # Consistently low scores (far from 1.0)
    scores = [0.05, 0.08, 0.06, 0.10, 0.07, 0.09]

    result = detector.compute_drift(scores)

    # Mean is very far from 1.0, should have high drift score
    assert result["drift_score"] >= 0.4
    assert result["signal"] in ["monitor", "drift_warning"]


def test_drift_single_score():
    """Test drift computation with a single score."""
    detector = PrecedentDriftV8()

    result = detector.compute_drift([0.95])

    # Single score at 0.95 should be stable
    assert result["drift_score"] < 0.2
    assert result["signal"] == "stable"


def test_drift_perfect_alignment():
    """Test drift with perfect alignment scores."""
    detector = PrecedentDriftV8()

    result = detector.compute_drift([1.0, 1.0, 1.0, 1.0])

    assert result["drift_score"] == 0.0
    assert result["signal"] == "stable"


def test_drift_score_capped_at_one():
    """Test that drift score is capped at 1.0."""
    detector = PrecedentDriftV8()

    # Extremely poor alignment with high variance
    scores = [0.0, 0.0, 0.1, 0.0, 0.05]

    result = detector.compute_drift(scores)

    # Drift score should not exceed 1.0
    assert result["drift_score"] <= 1.0
    assert result["signal"] == "drift_warning"
