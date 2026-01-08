# Git Issues Review and Fixes

**Date**: January 8, 2025  
**Branch**: `eleanor/production-hardening`

## Summary

Reviewed all modified files in git and fixed all identified issues. All files now compile correctly and pass linting.

---

## Issues Found and Fixed

### 1. ✅ EventType None Check Issue (Fixed)

**Files Affected**:
- `engine/recorder/evidence_recorder.py`
- `engine/runtime/critics.py`
- `engine/runtime/routing.py`
- `engine/runtime/run.py`

**Issue**: When event bus import fails, `EventType` is set to `None`, but code accessed `EventType.EVIDENCE_RECORDED` (and similar) without checking if `EventType` is `None`, which would raise `AttributeError`.

**Fix**: Added `EventType is not None` check to all event bus availability checks.

**Before**:
```python
if EVENT_BUS_AVAILABLE and get_event_bus:
    event = Event(event_type=EventType.EVIDENCE_RECORDED, ...)  # ❌ Could raise AttributeError
```

**After**:
```python
if EVENT_BUS_AVAILABLE and get_event_bus and EventType is not None:
    event = Event(event_type=EventType.EVIDENCE_RECORDED, ...)  # ✅ Safe
```

---

### 2. ✅ Unreachable Code and Incorrect Auth Logic (Fixed)

**File**: `api/middleware/auth.py`

**Issue**: 
- The `elif token and not token.has_role(role)` branch at line 260 was:
  1. Unreachable when `config.enabled` is True (all cases handled in `if` block)
  2. Incorrectly enforcing role checks when `config.enabled` is False, contradicting the design intent where auth should be optional in development

**Fix**: Removed the problematic `elif` branch. When `config.enabled` is False, the function now allows access without role checks.

**Before**:
```python
if config.enabled:
    if token is None:
        raise HTTPException(...)  # 401
    if not token.has_role(role):
        raise HTTPException(...)  # 403
elif token and not token.has_role(role):  # ❌ Unreachable when enabled=True, wrong when enabled=False
    raise HTTPException(...)  # 403
```

**After**:
```python
if config.enabled:
    if token is None:
        raise HTTPException(...)  # 401
    if not token.has_role(role):
        raise HTTPException(...)  # 403
# When auth is disabled (development mode), allow access regardless of role
```

---

### 3. ✅ require_scope Type and Logic Issues (Fixed)

**File**: `api/middleware/auth.py`

**Issues**:
1. Type annotation was `token: TokenPayload` but `verify_token` returns `Optional[TokenPayload]`
2. Logic was inconsistent with `require_role` - didn't properly handle disabled auth
3. Missing 401 error when token is None but auth is enabled

**Fix**: 
- Changed type to `Optional[TokenPayload]`
- Made logic consistent with `require_role`
- Added proper 401 error handling when auth is enabled but token is None

**Before**:
```python
async def wrapper(*args, token: TokenPayload = Depends(verify_token), ...):  # ❌ Wrong type
    if config.enabled and token and not token.has_scope(scope):  # ❌ Missing None check
        raise HTTPException(...)
```

**After**:
```python
async def wrapper(*args, token: Optional[TokenPayload] = Depends(verify_token), ...):  # ✅ Correct type
    if config.enabled:
        if token is None:
            raise HTTPException(...)  # 401
        if not token.has_scope(scope):
            raise HTTPException(...)  # 403
    # When auth is disabled, allow access
```

---

### 4. ✅ Trailing Whitespace (Fixed)

**File**: `api/middleware/auth.py`

**Issue**: Trailing whitespace on line 287

**Fix**: Removed trailing whitespace

---

## Verification

### ✅ All Files Compile
```bash
python3 -m py_compile api/middleware/auth.py engine/recorder/evidence_recorder.py \
  engine/runtime/critics.py engine/runtime/routing.py engine/runtime/run.py
# SUCCESS: All files compile without syntax errors
```

### ✅ All Imports Work
```bash
python3 -c "from api.middleware.auth import require_role, require_scope, get_auth_config; ..."
# SUCCESS: All imports work correctly
```

### ✅ No Linter Errors
All files pass linting with no errors.

### ✅ No Git Check Issues
```bash
git diff --check
# No whitespace or other issues
```

---

## Modified Files Summary

1. **api/middleware/auth.py**
   - Fixed unreachable code in `require_role`
   - Fixed type and logic issues in `require_scope`
   - Removed trailing whitespace

2. **engine/recorder/evidence_recorder.py**
   - Added `EventType is not None` check

3. **engine/runtime/critics.py**
   - Added `EventType is not None` check

4. **engine/runtime/routing.py**
   - Added `EventType is not None` check

5. **engine/runtime/run.py**
   - Added `EventType is not None` check

---

## Status

✅ **All Issues Fixed**  
✅ **All Files Verified**  
✅ **Ready for Commit**

---

## Next Steps

1. Review the changes: `git diff`
2. Stage the changes: `git add api/middleware/auth.py engine/recorder/evidence_recorder.py engine/runtime/critics.py engine/runtime/routing.py engine/runtime/run.py`
3. Commit: `git commit -m "Fix: EventType None checks and auth middleware logic issues"`
