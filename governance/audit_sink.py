from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """Append a single JSON object to a JSONL file.

    Dependency-free, single-write append. Creates parent dirs if needed.
    """
    if not path:
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def append_jsonl_safely(path: str, obj: Dict[str, Any], *, swallow_errors: bool = True) -> Optional[str]:
    """Best-effort append helper.

    Returns an error string if swallow_errors=True and an exception occurs.
    """
    try:
        append_jsonl(path, obj)
        return None
    except Exception as exc:
        if swallow_errors:
            return str(exc)
        raise
