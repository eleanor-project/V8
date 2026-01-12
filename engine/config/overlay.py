from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional

import logging

logger = logging.getLogger(__name__)


def resolve_overlay_path() -> Optional[Path]:
    path = os.getenv("ELEANOR_CONFIG_OVERLAY_PATH")
    if path:
        return Path(path)

    base = os.getenv("ELEANOR_CONFIG_PATH") or os.getenv("ELEANOR_CONFIG")
    if not base:
        return None

    base_path = Path(base)
    if base_path.suffix.lower() in (".yml", ".yaml"):
        return base_path.with_suffix(f".overlay{base_path.suffix}")

    return Path(f"{base}.overlay.yaml")


def _canonical_json(payload: Any) -> str:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def overlay_hash(payload: Dict[str, Any]) -> Optional[str]:
    if not payload:
        return None
    digest = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def load_overlay_payload() -> Dict[str, Any]:
    path = resolve_overlay_path()
    if path is None or not path.exists() or not path.is_file():
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for config overlay support.") from exc

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            logger.warning("overlay_payload_invalid", extra={"path": str(path)})
            return {}
        return payload
    except Exception as exc:
        logger.warning("overlay_payload_load_failed", extra={"path": str(path), "error": str(exc)})
        return {}


def write_overlay_payload(payload: Dict[str, Any]) -> Path:
    path = resolve_overlay_path()
    if path is None:
        raise RuntimeError("Config overlay path is not configured.")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for config overlay support.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True, default_flow_style=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    return path


__all__ = [
    "resolve_overlay_path",
    "overlay_hash",
    "load_overlay_payload",
    "write_overlay_payload",
]
