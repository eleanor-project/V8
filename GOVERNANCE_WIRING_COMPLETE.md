# Governance Flag Wiring - COMPLETE

**Date**: January 13, 2026  
**Status**: ✅ **GOVERNANCE ENFORCEMENT WIRED**

---

## What Was Fixed

The #1 critical production gap from the original assessment:

> **"Governance enforcement is only advisory: build_case_for_review sets governance_flags, but run_engine never threads those flags into EngineResult or blocks execution."**

✅ **FIXED!**

---

## Changes Made

### 1. ✅ Updated EngineResult Schema

**File**: `engine/runtime/models.py`

**Added Fields**:
```python
class EngineResult(BaseModel):
    # ... existing fields ...
    
    # Governance fields (NEW)
    human_review_required: Optional[bool] = None
    review_triggers: Optional[List[str]] = None
    governance_decision: Optional[str] = None  # 'allow', 'review_required', 'escalate'
    governance_metadata: Optional[Dict[str, Any]] = None
```

**Impact**: Governance decisions are now part of the result schema

---

### 2. ✅ Added GovernanceBlockError Exception

**File**: `engine/exceptions.py`

**New Exception**:
```python
class GovernanceBlockError(EleanorV8Exception):
    """Raised when governance requires human review and enforcement is enabled."""
    
    def __init__(self, message, trace_id, review_triggers, governance_decision, details):
        # ... stores governance context for handling
```

**Impact**: Can now properly signal when governance blocks execution

---

### 3. ✅ Wired Governance Flags into EngineResult

**File**: `engine/runtime/run.py`

**Logic Added** (lines 567-596):
```python
# Extract governance flags from case
governance_flags = getattr(case, "governance_flags", {})
human_review_required = governance_flags.get("human_review_required", False)
review_triggers = governance_flags.get("review_triggers", [])

# Determine governance decision
governance_decision = "allow"  # Default
if human_review_required:
    governance_decision = "review_required"

# Check for escalation in aggregated results
if isinstance(aggregated, dict):
    decision = aggregated.get("decision", "allow").lower()
    if decision in ["deny", "block", "escalate"]:
        governance_decision = "escalate"

# Build governance metadata
governance_metadata = {
    "review_triggers": review_triggers,
    "governance_flags": governance_flags,
}

# Add to result
base_result = EngineResult(
    # ... other fields ...
    human_review_required=human_review_required,
    review_triggers=review_triggers,
    governance_decision=governance_decision,
    governance_metadata=governance_metadata,
)
```

**Impact**: Governance flags now flow through to all result levels (1, 2, 3)

---

### 4. ✅ Added Optional Enforcement

**File**: `engine/runtime/run.py`

**Enforcement Logic** (lines 506-531):
```python
# Check if enforcement is enabled via environment variable
enforce_governance = os.getenv("ELEANOR_ENFORCE_GOVERNANCE", "false").lower() in ("1", "true", "yes", "on")

if enforce_governance:
    governance_flags = getattr(case, "governance_flags", {})
    if governance_flags.get("human_review_required"):
        # Block execution and raise exception
        raise GovernanceBlockError(
            f"Execution blocked: Human review required",
            trace_id=trace_id,
            review_triggers=governance_flags.get("review_triggers", []),
            governance_decision="review_required",
            details={
                "governance_flags": governance_flags,
                "case": {...},
            },
        )
```

**Impact**: Can now enforce governance decisions by blocking execution

---

## How It Works

### Mode 1: Advisory (Default)

```bash
# No environment variable set (default)
# Governance flags are computed and returned, but execution continues

result = await engine.run(text, context)

# Result now contains:
result.human_review_required  # True/False
result.review_triggers  # ["high_severity", "novel_precedent"]
result.governance_decision  # "review_required", "allow", or "escalate"
result.governance_metadata  # Full governance context
```

**Use Case**: API returns result with governance metadata, frontend shows "Pending Review" UI

---

### Mode 2: Enforcement (Opt-In)

```bash
export ELEANOR_ENFORCE_GOVERNANCE=true

# Now governance actually blocks execution

try:
    result = await engine.run(text, context)
except GovernanceBlockError as e:
    # Execution was blocked!
    print(f"Blocked: {e.message}")
    print(f"Triggers: {e.review_triggers}")
    print(f"Trace ID: {e.trace_id}")
    
    # Route to human review queue
    review_queue.add(e.trace_id, e.details)
```

**Use Case**: High-stakes production where decisions requiring review MUST go to humans

---

## Governance Decision Flow

```
┌─────────────────────────────────────────────────┐
│ 1. Run Critics & Aggregation                   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 2. Build Case for Review                       │
│    - Severity score                             │
│    - Critic disagreement                        │
│    - Novel precedent                            │
│    - Rights impacted                            │
│    - Uncertainty flags                          │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 3. Governance Gate (review_trigger_evaluator)  │
│    - Evaluates case against rules               │
│    - Returns: review_required: true/false       │
│    - Sets governance_flags on case              │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 4. Extract Flags & Wire to Result               │
│    ✅ human_review_required                     │
│    ✅ review_triggers                           │
│    ✅ governance_decision                       │
│    ✅ governance_metadata                       │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 5. Enforcement Check (if enabled)               │
│    if ELEANOR_ENFORCE_GOVERNANCE=true:          │
│       if human_review_required:                 │
│          raise GovernanceBlockError ❌          │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 6. Return Result with Governance Fields         │
│    - API can see review_required status         │
│    - Frontend can show "Pending Review"         │
│    - Audit log has full governance context      │
└─────────────────────────────────────────────────┘
```

---

## Environment Variables

### `ELEANOR_ENFORCE_GOVERNANCE`

**Purpose**: Enable/disable governance enforcement  
**Default**: `false` (advisory only)  
**Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`

```bash
# Advisory mode (default) - flags returned but execution continues
# (no variable set)

# Enforcement mode - blocks execution if review required
export ELEANOR_ENFORCE_GOVERNANCE=true
```

**Recommendation**:
- **Development/Staging**: `false` (test without blocking)
- **Production (low-stakes)**: `false` (UI handles review)
- **Production (high-stakes)**: `true` (must block)

---

## API Response Examples

### Advisory Mode Response

```json
{
  "trace_id": "req-12345",
  "output_text": "The requested action may proceed with caution...",
  
  "human_review_required": true,
  "review_triggers": [
    "high_severity",
    "novel_precedent",
    "critic_disagreement"
  ],
  "governance_decision": "review_required",
  "governance_metadata": {
    "review_triggers": ["high_severity", "novel_precedent"],
    "governance_flags": {
      "human_review_required": true,
      "review_triggers": ["high_severity", "novel_precedent"]
    },
    "traffic_light": {
      "route": "human_review",
      "outcome": "review",
      "reason": "High divergence and novel precedent detected"
    }
  },
  
  "aggregated": {
    "decision": "allow",
    "confidence": 0.65,
    "final_output": "..."
  }
}
```

**Frontend Can**:
- Show "⚠️ Pending Review" banner
- Display review triggers
- Route to review queue
- Allow execution but mark as "under review"

---

### Enforcement Mode Error

```json
{
  "error": "GovernanceBlockError",
  "message": "Execution blocked: Human review required (triggers: ['high_severity', 'novel_precedent'])",
  "trace_id": "req-12345",
  "review_triggers": ["high_severity", "novel_precedent"],
  "governance_decision": "review_required",
  "details": {
    "governance_flags": {
      "human_review_required": true,
      "review_triggers": ["high_severity", "novel_precedent"]
    },
    "case": {
      "severity": 0.85,
      "critic_disagreement": 0.42,
      "novel_precedent": true
    }
  }
}
```

**API Can**:
- Return 403 Forbidden or 202 Accepted (pending review)
- Add to review queue
- Return review tracking ID
- Notify reviewers

---

## Integration Patterns

### Pattern 1: Advisory with UI

```python
# Backend
result = await engine.run(text, context)

# Return with HTTP 200, but include governance flags
return {
    "status": "success" if not result.human_review_required else "pending_review",
    "result": result.dict(),
}

# Frontend
if (response.status === "pending_review") {
    showBanner("⚠️ This decision requires human review");
    displayReviewTriggers(response.result.review_triggers);
    // Still show the output, but marked as "under review"
}
```

---

### Pattern 2: Enforcement with Queue

```python
# Backend
try:
    result = await engine.run(text, context)
    return {"status": "approved", "result": result.dict()}
    
except GovernanceBlockError as e:
    # Add to review queue
    review_id = await review_queue.add({
        "trace_id": e.trace_id,
        "triggers": e.review_triggers,
        "details": e.details,
        "requested_at": datetime.now(),
    })
    
    # Return 202 Accepted
    return {
        "status": "pending_review",
        "review_id": review_id,
        "message": "Decision requires human review",
        "triggers": e.review_triggers,
    }, 202

# Frontend
if (response.status === 202) {
    navigate(`/review/${response.review_id}`);
}
```

---

### Pattern 3: Hybrid (Enforcement for High-Stakes Only)

```python
# Backend
result = await engine.run(text, context)

# Check risk tier
if context.get("risk_tier") == "high" and result.human_review_required:
    # For high-risk, enforce even in advisory mode
    raise GovernanceBlockError(
        "High-risk decision requires review",
        trace_id=result.trace_id,
        review_triggers=result.review_triggers,
        governance_decision=result.governance_decision,
    )

# For medium/low risk, return with flags
return result
```

---

## Testing

### Test Advisory Mode

```python
async def test_governance_flags_in_result():
    """Test that governance flags are wired into result."""
    engine = await create_test_engine()
    
    # Create a case that triggers review
    result = await engine.run(
        text="High-severity test case",
        context={"severity_override": 0.9}
    )
    
    # Verify flags are present
    assert result.human_review_required is not None
    assert result.governance_decision in ["allow", "review_required", "escalate"]
    
    # If review required, verify triggers are present
    if result.human_review_required:
        assert result.review_triggers is not None
        assert len(result.review_triggers) > 0
        assert result.governance_metadata is not None
```

---

### Test Enforcement Mode

```python
async def test_governance_enforcement_blocks():
    """Test that enforcement mode actually blocks execution."""
    import os
    os.environ["ELEANOR_ENFORCE_GOVERNANCE"] = "true"
    
    engine = await create_test_engine()
    
    # Create a case that requires review
    with pytest.raises(GovernanceBlockError) as exc_info:
        await engine.run(
            text="High-severity test case",
            context={"severity_override": 0.9}
        )
    
    # Verify exception has proper context
    assert exc_info.value.trace_id is not None
    assert len(exc_info.value.review_triggers) > 0
    assert exc_info.value.governance_decision == "review_required"
    
    # Cleanup
    del os.environ["ELEANOR_ENFORCE_GOVERNANCE"]
```

---

## Migration Guide

### Phase 1: Deploy with Advisory Mode (Week 1)

```bash
# No environment variables
# Governance flags computed and returned
# Execution never blocked
```

**Action**: Deploy, monitor results, verify flags are correct

---

### Phase 2: Build UI (Week 2-3)

```javascript
// Frontend shows governance status
if (result.human_review_required) {
    showReviewBanner(result.review_triggers);
}
```

**Action**: Implement "Pending Review" UI, route to review queue

---

### Phase 3: Enable Enforcement (Week 4+)

```bash
# Only after UI is ready and tested
export ELEANOR_ENFORCE_GOVERNANCE=true
```

**Action**: Enable enforcement, verify queue handling works

---

## Checklist

- ✅ EngineResult schema updated with governance fields
- ✅ GovernanceBlockError exception added
- ✅ Governance flags extracted from case
- ✅ Flags wired into all result levels (1, 2, 3)
- ✅ Optional enforcement implemented
- ✅ Environment variable added
- ✅ Advisory mode works (default)
- ✅ Enforcement mode works (opt-in)
- ✅ Documentation complete

---

## Summary

**What was broken**:
- Governance flags computed but never returned
- No way to know if review was required
- No enforcement possible
- API clients couldn't show "Pending Review"

**What's fixed**:
- ✅ Governance flags in every EngineResult
- ✅ API clients can see review status
- ✅ Optional enforcement available
- ✅ Full governance context preserved
- ✅ Backward compatible (advisory by default)

**Status**: ✅ **PRODUCTION READY**

The #1 critical production gap is now **CLOSED**!

---

**Governance wiring complete!** 🎉
