# Legacy Guardrail Applied - Critics.py

**Date**: January 13, 2026  
**Status**: ✅ **GUARDRAIL ACTIVE**

---

## Changes Applied to `engine/runtime/critics.py`

### 1. Module Documentation (Lines 1-11)

Added comprehensive module docstring:

```python
"""  
Legacy critic execution pipeline.

NOTE: This module is retained for backward compatibility and targeted testing,
but the production runtime should execute critics via OrchestratorV2
(engine.runtime.critic_infrastructure.run_critics_with_orchestrator).

To prevent accidental use in production, run_critics_parallel is guarded by
ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE (default: true). To explicitly allow
legacy execution (tests/dev only), set ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER=1.
"""
```

### 2. Import Additions (Lines 13-15)

Added required imports:
```python
import os
import warnings
```

### 3. Legacy Guardrail (Lines 378-395)

Added runtime guardrail to `run_critics_parallel()`:

```python
# --- Legacy guardrail ---
enforce = os.getenv("ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE", "true").strip().lower() in ("1", "true", "yes", "on")
allow = os.getenv("ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER", "").strip().lower() in ("1", "true", "yes", "on")

if enforce and not allow:
    raise RuntimeError(
        "Legacy critic runner invoked (engine.runtime.critics.run_critics_parallel). "
        "Production should use OrchestratorV2 via engine.runtime.critic_infrastructure.run_critics_with_orchestrator. "
        "Set ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER=1 to override (tests/dev only), or set "
        "ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE=false to disable this guard."
    )

warnings.warn(
    "engine.runtime.critics.run_critics_parallel is legacy; use run_critics_with_orchestrator instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

---

## How It Works

### Default Behavior (Production)

```python
# Environment: (no special variables set)
# ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE = "true" (default)
# ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER = "" (not set)

# Result: RuntimeError raised
❌ BLOCKS legacy path
✅ Forces use of OrchestratorV2
```

### Test/Dev Override

```python
# Environment:
# ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER=1

# Result: DeprecationWarning issued
⚠️ Warning shown
✅ Execution allowed
```

### Disable Guardrail (Not Recommended)

```python
# Environment:
# ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE=false

# Result: No error, only warning
⚠️ Warning shown
✅ Execution allowed
```

---

## Environment Variables

### `ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE`

**Purpose**: Enable/disable the guardrail  
**Default**: `"true"`  
**Values**: `"true"`, `"1"`, `"yes"`, `"on"` (enabled) | `"false"`, `"0"`, `"no"`, `"off"` (disabled)

**Recommendation**: Leave enabled in production

### `ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER`

**Purpose**: Explicitly allow legacy execution (bypass guardrail)  
**Default**: `""` (not set, blocking)  
**Values**: `"1"`, `"true"`, `"yes"`, `"on"` (allow legacy)

**Recommendation**: Only set for tests/dev

---

## Use Cases

### Use Case 1: Production Deployment

```bash
# No environment variables set
# Default behavior

Result:
- Guardrail ACTIVE
- Legacy path BLOCKED
- Must use OrchestratorV2
```

✅ **This is the desired production state**

### Use Case 2: Running Old Tests

```bash
# Test needs legacy runner
export ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER=1
pytest tests/test_old_critic_execution.py
```

Result:
- Guardrail bypassed
- DeprecationWarning shown
- Test can run

### Use Case 3: Local Development

```bash
# Developer testing legacy code path
export ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER=1
python test_script.py
```

Result:
- Legacy execution allowed
- Warning reminds developer to migrate

### Use Case 4: Temporary Rollback (Emergency)

```bash
# Production issue, need to rollback quickly
export ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE=false
```

Result:
- Guardrail disabled
- Legacy execution works
- Warning still shown

⚠️ **Only for emergencies - fix and re-enable ASAP**

---

## Impact Analysis

### What This Prevents

1. **Accidental Legacy Use**: Developer can't accidentally call old path
2. **Deployment Issues**: CI/CD will catch legacy usage
3. **Code Path Drift**: Ensures single execution path
4. **Testing Gaps**: Forces migration of tests to new path

### What This Allows

1. **Backward Compatibility**: Legacy tests still work (with flag)
2. **Gradual Migration**: Can enable per-test as needed
3. **Emergency Rollback**: Can disable guardrail if needed
4. **Clear Deprecation**: Warning informs developers

---

## Migration Strategy

### Phase 1: Guardrail Active (Current State)

```
Status: ✅ Complete
- Legacy path blocked by default
- Tests can opt-in with environment variable
- Production forced to use OrchestratorV2
```

### Phase 2: Update Tests (Next)

```
Goal: Remove ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER from tests
Action: Update tests to use OrchestratorV2

Example:
# Before:
@pytest.fixture
def env():
    os.environ["ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER"] = "1"

# After:
# (Remove the environment variable, test uses new path)
```

### Phase 3: Remove Legacy Code (Future)

```
When: After all tests migrated
Action: 
- Remove run_critics_parallel entirely
- Remove ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER checks
- Remove engine/runtime/critics.py or keep only helper functions
```

---

## Testing the Guardrail

### Test 1: Verify Guardrail Blocks

```python
# test_guardrail.py
import pytest
from engine.runtime.critics import run_critics_parallel

def test_legacy_path_blocked():
    """Verify legacy path raises error by default."""
    with pytest.raises(RuntimeError, match="Legacy critic runner invoked"):
        await run_critics_parallel(engine, ...)
```

### Test 2: Verify Override Works

```python
# test_legacy_override.py
import os
import pytest

@pytest.fixture
def allow_legacy():
    os.environ["ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER"] = "1"
    yield
    os.environ.pop("ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER", None)

def test_legacy_allowed_with_flag(allow_legacy):
    """Verify override allows legacy execution."""
    with pytest.warns(DeprecationWarning):
        result = await run_critics_parallel(engine, ...)
        assert result is not None
```

### Test 3: Verify New Path Works

```python
# test_orchestrator_path.py
from engine.runtime.critic_infrastructure import run_critics_with_orchestrator

async def test_orchestrator_path():
    """Verify new path works without flags."""
    result = await run_critics_with_orchestrator(engine, ...)
    assert result is not None
    # No error, no warning
```

---

## Error Messages

### Production Error (Guardrail Active)

```
RuntimeError: Legacy critic runner invoked (engine.runtime.critics.run_critics_parallel). 
Production should use OrchestratorV2 via engine.runtime.critic_infrastructure.run_critics_with_orchestrator. 
Set ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER=1 to override (tests/dev only), or set 
ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE=false to disable this guard.
```

**Fix**: Update code to use `run_critics_with_orchestrator`

### Deprecation Warning (Legacy Allowed)

```
DeprecationWarning: engine.runtime.critics.run_critics_parallel is legacy; 
use run_critics_with_orchestrator instead.
```

**Fix**: Migrate to new orchestrator path

---

## Checklist

- ✅ Module docstring added
- ✅ Imports added (`os`, `warnings`)
- ✅ Guardrail logic implemented
- ✅ Environment variable checks implemented
- ✅ RuntimeError for production violations
- ✅ DeprecationWarning for legacy use
- ✅ Clear error messages
- ✅ Documentation complete

---

## Verification Steps

### 1. Check Imports
```bash
grep -n "import os" engine/runtime/critics.py
grep -n "import warnings" engine/runtime/critics.py
```

Expected: Lines 13-14

### 2. Check Guardrail
```bash
grep -A 10 "Legacy guardrail" engine/runtime/critics.py
```

Expected: Guardrail code visible

### 3. Run Tests
```bash
# Should pass (mixins routes through orchestrator)
pytest tests/test_engine.py -v

# Should raise error (unless flag set)
# (if any tests call run_critics_parallel directly)
pytest tests/ -v
```

---

## Summary

The legacy guardrail is now **ACTIVE** and will:

1. ✅ **Block** accidental use of legacy path in production
2. ✅ **Allow** explicit opt-in for tests/dev
3. ✅ **Warn** users about deprecation
4. ✅ **Force** migration to OrchestratorV2
5. ✅ **Provide** clear error messages
6. ✅ **Enable** emergency rollback if needed

**Status**: ✅ **PRODUCTION SAFE**

The guardrail ensures all new code uses the orchestrator path while allowing backward compatibility for existing tests that need gradual migration.

---

**Diff successfully applied!**
