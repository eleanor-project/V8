from __future__ import annotations

import json
import os
from types import ModuleType
from pathlib import Path
from typing import Optional

from engine.schemas.escalation import AuditRecord

_fcntl: Optional[ModuleType]
try:
    import fcntl as _fcntl  # type: ignore
except Exception:  # pragma: no cover - platform specific
    _fcntl = None

fcntl: Optional[ModuleType] = _fcntl


def _audit_path() -> Path:
    raw = os.getenv("ELEANOR_EXEC_AUDIT_PATH", "logs/execution_audit.jsonl")
    return Path(raw)


def store_audit_record(record: AuditRecord, path: Optional[Path] = None) -> Optional[str]:
    if os.getenv("ELEANOR_DISABLE_EXEC_AUDIT", "").lower() in ("1", "true", "yes"):
        return None

    target = path or _audit_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = record.model_dump(mode="json") if hasattr(record, "model_dump") else record.dict()
    line = json.dumps(payload, ensure_ascii=True)

    with target.open("a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line + "\n")
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_UN)

    return str(target)


__all__ = ["store_audit_record"]
