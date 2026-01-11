from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from engine.config import ConfigManager
from engine.config.settings import EleanorSettings
from engine.logging_config import get_logger
from engine.runtime.config import EngineConfig
from engine.security.ledger import LedgerRecord, get_ledger_reader, get_ledger_writer
from engine.version import ELEANOR_VERSION

logger = get_logger(__name__)


class PreviewValidationError(Exception):
    def __init__(
        self,
        *,
        code: str,
        status_code: int,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message
        self.details = details or {}


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _hash_payload(payload: Any) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _hash_with_prefix(payload: Any) -> str:
    return f"sha256:{_hash_payload(payload)}"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _environment(settings: Optional[EleanorSettings] = None) -> str:
    if settings is not None:
        return settings.environment
    return os.getenv("ELEANOR_ENVIRONMENT") or os.getenv("ELEANOR_ENV") or "development"


def _deep_merge(base: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (changes or {}).items():
        if value is None:
            merged.pop(key, None)
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged.get(key, {}), value)
        else:
            merged[key] = value
    return merged


def _diff_dict(
    base: Dict[str, Any],
    candidate: Dict[str, Any],
    prefix: str = "",
) -> Tuple[List[str], int, int, int]:
    changed_keys: List[str] = []
    additions = removals = modifications = 0
    for key in sorted(set(base.keys()) | set(candidate.keys()), key=str):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in base:
            additions += 1
            changed_keys.append(path)
            continue
        if key not in candidate:
            removals += 1
            changed_keys.append(path)
            continue
        base_val = base.get(key)
        cand_val = candidate.get(key)
        if isinstance(base_val, dict) and isinstance(cand_val, dict):
            sub_keys, sub_add, sub_rem, sub_mod = _diff_dict(base_val, cand_val, path)
            changed_keys.extend(sub_keys)
            additions += sub_add
            removals += sub_rem
            modifications += sub_mod
            continue
        if base_val != cand_val:
            modifications += 1
            changed_keys.append(path)
    return changed_keys, additions, removals, modifications


def _settings_dict(settings: EleanorSettings) -> Dict[str, Any]:
    return settings.model_dump(mode="json")


def _config_fingerprint(settings_dict: Dict[str, Any]) -> str:
    return _hash_with_prefix(settings_dict)


def _collect_continuity_hashes(engine: Any) -> Dict[str, Optional[str]]:
    policy_bundle_hash = (
        os.getenv("ELEANOR_OPA_POLICY_BUNDLE_HASH")
        or os.getenv("OPA_POLICY_BUNDLE_HASH")
        or None
    )
    router_policy = None
    router = getattr(engine, "router", None)
    if router is not None:
        router_policy = getattr(router, "policy", None)
    router_policy_hash = _hash_with_prefix(router_policy) if router_policy else None

    critic_models = getattr(engine, "critic_models", None)
    critic_snapshot = None
    if isinstance(critic_models, dict) and critic_models:
        critic_snapshot = {
            str(key): getattr(value, "model", None)
            or getattr(value, "name", None)
            or str(value)
            for key, value in critic_models.items()
        }
    critic_bindings_hash = _hash_with_prefix(critic_snapshot) if critic_snapshot else None

    precedent_index_hash = os.getenv("ELEANOR_PRECEDENT_INDEX_HASH") or None

    return {
        "policy_bundle_hash": policy_bundle_hash,
        "router_policy_hash": router_policy_hash,
        "critic_bindings_hash": critic_bindings_hash,
        "precedent_index_hash": precedent_index_hash,
    }


def _infer_reason_codes(changed_keys: List[str]) -> List[str]:
    reasons: List[str] = []
    for key in changed_keys:
        lowered = key.lower()
        if "opa" in lowered or "policy" in lowered:
            reasons.append("OPA_POLICY_CHANGED")
        if "precedent" in lowered:
            reasons.append("PRECEDENT_SUPERSEDED")
        if "risk" in lowered or "threshold" in lowered:
            reasons.append("RISK_THRESHOLD_EXCEEDED")
    return sorted(set(reasons))


def _requires_precedent_ratify(changes: Dict[str, Any]) -> bool:
    def _scan(value: Any) -> bool:
        if isinstance(value, dict):
            for key, item in value.items():
                if "precedent" in str(key).lower():
                    return True
                if _scan(item):
                    return True
        elif isinstance(value, list):
            return any(_scan(item) for item in value)
        return False

    return _scan(changes or {})


def _generate_proposal_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = uuid4().hex[:4]
    return f"cprop-{timestamp}-{suffix}"


def _generate_artifact_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    suffix = uuid4().hex[:6]
    return f"preview-{timestamp}-{suffix}"


def _artifact_hash(artifact: Dict[str, Any]) -> str:
    payload = dict(artifact)
    payload.pop("artifact_hash", None)
    return _hash_with_prefix(payload)


def _iter_ledger_records() -> Iterable[LedgerRecord]:
    reader = get_ledger_reader()
    return reader.read_all()


def _find_latest_event(event: str, proposal_id: str) -> Optional[LedgerRecord]:
    latest: Optional[LedgerRecord] = None
    for record in _iter_ledger_records():
        if record.event != event:
            continue
        payload = record.payload or {}
        if payload.get("proposal_id") != proposal_id:
            continue
        latest = record
    return latest


def submit_proposal(
    *,
    proposal: Dict[str, Any],
    actor: str,
    settings: Optional[EleanorSettings] = None,
) -> Dict[str, Any]:
    proposal_id = _generate_proposal_id()
    payload = {
        "proposal_id": proposal_id,
        "proposal": proposal,
        "submitted_at": _utc_timestamp(),
        "actor": actor,
        "environment": _environment(settings),
    }
    writer = get_ledger_writer()
    writer.append("config_proposal_submitted", payload)
    return payload


def list_proposals() -> List[Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for record in _iter_ledger_records():
        if not record.event.startswith("config_proposal_"):
            continue
        payload = record.payload or {}
        proposal_id = payload.get("proposal_id")
        if not proposal_id:
            continue
        summary = summaries.setdefault(
            proposal_id,
            {
                "proposal_id": proposal_id,
                "title": None,
                "status": "submitted",
                "submitted_at": payload.get("submitted_at"),
                "last_preview": None,
                "last_apply": None,
            },
        )
        if record.event == "config_proposal_submitted":
            proposal = payload.get("proposal") or {}
            summary["title"] = proposal.get("title")
            summary["submitted_at"] = payload.get("submitted_at") or record.timestamp
        elif record.event == "config_proposal_previewed":
            summary["last_preview"] = payload.get("created_at") or record.timestamp
            summary["status"] = "previewed"
        elif record.event == "config_proposal_applied":
            summary["last_apply"] = payload.get("applied_at") or record.timestamp
            summary["status"] = "applied"
    items = list(summaries.values())
    items.sort(key=lambda item: item.get("submitted_at") or "")
    return items


def get_proposal(proposal_id: str) -> Optional[Dict[str, Any]]:
    record = _find_latest_event("config_proposal_submitted", proposal_id)
    if record is None:
        return None
    payload = record.payload or {}
    proposal = payload.get("proposal")
    if isinstance(proposal, dict):
        return proposal
    return None


def build_preview_artifact(
    *,
    proposal_id: str,
    proposal: Dict[str, Any],
    mode: str,
    window: Dict[str, Any],
    limits: Dict[str, Any],
    engine: Any,
) -> Dict[str, Any]:
    settings = ConfigManager().settings
    baseline_settings = _settings_dict(settings)
    changes = proposal.get("changes") or {}
    candidate_settings = _deep_merge(baseline_settings, changes)

    changed_keys, additions, removals, modifications = _diff_dict(
        baseline_settings, candidate_settings
    )

    artifact = {
        "$schema": "https://schemas.eleanor.ai/v8/preview-artifact.schema.json",
        "schema_version": 1,
        "artifact_id": _generate_artifact_id(),
        "proposal_id": proposal_id,
        "created_at": _utc_timestamp(),
        "environment": _environment(settings),
        "fingerprints": {
            "baseline": _config_fingerprint(baseline_settings),
            "candidate": _config_fingerprint(candidate_settings),
        },
        "preview_mode": mode,
        "window": window,
        "inputs": {
            "preview_mode": mode,
            "window": window,
            "limits": limits,
        },
        "diff_summary": {
            "changed_keys": changed_keys,
            "additions": additions,
            "removals": removals,
            "modifications": modifications,
        },
        "reason_codes": _infer_reason_codes(changed_keys),
        "warnings": [],
        "metrics": {
            "preview_duration_ms": 0,
            "trace_count": 0,
        },
        "signatures": {
            "preview_engine": ELEANOR_VERSION,
            "ledger_event": "config_proposal_previewed",
        },
        "continuity": _collect_continuity_hashes(engine),
    }

    if mode == "policy_only":
        artifact["warnings"].append(
            "POLICY_ONLY_PREVIEW: Full replay was not executed for this preview."
        )

    artifact["artifact_hash"] = _artifact_hash(artifact)
    return artifact


def store_preview_artifact(
    *,
    proposal_id: str,
    artifact: Dict[str, Any],
    actor: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "proposal_id": proposal_id,
        "artifact_hash": artifact.get("artifact_hash"),
        "artifact": artifact,
        "created_at": artifact.get("created_at"),
        "environment": artifact.get("environment"),
    }
    if actor:
        payload["actor"] = actor
    writer = get_ledger_writer()
    writer.append("config_proposal_previewed", payload)
    return payload


def load_preview_payload(proposal_id: str) -> Optional[Dict[str, Any]]:
    latest = _find_latest_event("config_proposal_previewed", proposal_id)
    if latest is None:
        return None
    return latest.payload or {}


def apply_candidate_settings(
    *,
    engine: Any,
    candidate_settings: EleanorSettings,
) -> None:
    from engine.config import settings as settings_module
    from engine.config.manager import ConfigManager as Manager

    settings_module._settings = candidate_settings
    manager = Manager()
    manager._settings = candidate_settings

    engine.settings = candidate_settings
    engine.config = EngineConfig.from_settings(candidate_settings)

    try:
        from engine.core.feature_integration import integrate_optional_features

        integrate_optional_features(engine, candidate_settings)
    except Exception as exc:
        logger.warning("optional_feature_refresh_failed", extra={"error": str(exc)})


def compute_candidate_settings(
    *,
    proposal: Dict[str, Any],
) -> Tuple[EleanorSettings, Dict[str, Any]]:
    settings = ConfigManager().settings
    baseline_settings = _settings_dict(settings)
    changes = proposal.get("changes") or {}
    candidate_settings = _deep_merge(baseline_settings, changes)
    try:
        candidate_model = EleanorSettings.model_validate(candidate_settings)
    except Exception as exc:
        raise PreviewValidationError(
            code="INVALID_PROPOSAL",
            status_code=400,
            message="Proposal changes failed validation against settings schema.",
            details={"error": str(exc)},
        ) from exc
    return candidate_model, candidate_settings


def apply_proposal(
    *,
    proposal_id: str,
    artifact_hash: str,
    engine: Any,
    actor: str,
) -> Dict[str, Any]:
    proposal = get_proposal(proposal_id)
    if proposal is None:
        raise PreviewValidationError(
            code="PROPOSAL_NOT_FOUND",
            status_code=404,
            message="Proposal not found",
        )

    preview_payload = load_preview_payload(proposal_id)
    if preview_payload is None:
        raise PreviewValidationError(
            code="PREVIEW_REQUIRED",
            status_code=428,
            message="Preview artifact not found",
        )
    latest_hash = preview_payload.get("artifact_hash")
    if latest_hash != artifact_hash:
        raise PreviewValidationError(
            code="INVALID_PREVIEW_ARTIFACT",
            status_code=400,
            message="Preview artifact hash does not match latest preview",
            details={"latest": latest_hash, "provided": artifact_hash},
        )

    artifact = preview_payload.get("artifact")
    if not isinstance(artifact, dict):
        raise PreviewValidationError(
            code="INVALID_PREVIEW_ARTIFACT",
            status_code=400,
            message="Preview artifact payload is missing or malformed",
        )

    recalculated = _artifact_hash(artifact)
    if recalculated != artifact_hash:
        raise PreviewValidationError(
            code="INVALID_PREVIEW_ARTIFACT",
            status_code=400,
            message="Preview artifact hash mismatch",
            details={"expected": artifact_hash, "computed": recalculated},
        )

    settings = ConfigManager().settings
    current_env = _environment(settings)
    if artifact.get("environment") != current_env:
        raise PreviewValidationError(
            code="ENV_MISMATCH",
            status_code=409,
            message="Preview artifact environment mismatch",
            details={"artifact": artifact.get("environment"), "current": current_env},
        )

    baseline_fingerprint = artifact.get("fingerprints", {}).get("baseline")
    current_fingerprint = _config_fingerprint(_settings_dict(settings))
    if baseline_fingerprint and baseline_fingerprint != current_fingerprint:
        raise PreviewValidationError(
            code="BASELINE_DIVERGED",
            status_code=412,
            message="Runtime config changed since preview; re-run preview before applying.",
            details={
                "artifact_baseline": baseline_fingerprint,
                "current_runtime": current_fingerprint,
            },
        )

    candidate_model, candidate_dict = compute_candidate_settings(proposal=proposal)
    candidate_fingerprint = artifact.get("fingerprints", {}).get("candidate")
    if candidate_fingerprint and candidate_fingerprint != _config_fingerprint(candidate_dict):
        raise PreviewValidationError(
            code="CANDIDATE_MISMATCH",
            status_code=409,
            message="Candidate fingerprint mismatch; proposal contents have changed.",
        )

    continuity = artifact.get("continuity") or {}
    current_continuity = _collect_continuity_hashes(engine)
    for key, previous in continuity.items():
        if previous is None:
            continue
        current = current_continuity.get(key)
        if current != previous:
            raise PreviewValidationError(
                code="PREVIEW_REQUIRED",
                status_code=428,
                message=f"Continuity check failed for {key}; re-run preview.",
                details={"key": key, "previous": previous, "current": current},
            )

    if _requires_precedent_ratify(proposal.get("changes") or {}):
        if not _truthy(os.getenv("ELEANOR_ENABLE_PRECEDENT_AUTO_RATIFY")):
            raise PreviewValidationError(
                code="PRECEDENT_RATIFY_DISABLED",
                status_code=403,
                message="Precedent auto-ratify is disabled.",
            )

    apply_candidate_settings(engine=engine, candidate_settings=candidate_model)

    payload = {
        "proposal_id": proposal_id,
        "artifact_hash": artifact_hash,
        "applied_at": _utc_timestamp(),
        "actor": actor,
        "environment": current_env,
        "flags": {
            "ELEANOR_ENABLE_ADMIN_WRITE": _truthy(os.getenv("ELEANOR_ENABLE_ADMIN_WRITE")),
            "ELEANOR_ENABLE_PRECEDENT_AUTO_RATIFY": _truthy(
                os.getenv("ELEANOR_ENABLE_PRECEDENT_AUTO_RATIFY")
            ),
        },
    }
    writer = get_ledger_writer()
    ledger_record = writer.append("config_proposal_applied", payload)

    return {
        "proposal_id": proposal_id,
        "status": "applied",
        "applied_at": payload["applied_at"],
        "fingerprints": {
            "previous": baseline_fingerprint,
            "current": _config_fingerprint(_settings_dict(candidate_model)),
        },
        "ledger": {
            "event": "config_proposal_applied",
            "event_id": ledger_record.event_id if ledger_record else None,
            "artifact_hash": artifact_hash,
        },
    }
