from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from engine.config import ConfigManager
from engine.config.overlay import (
    load_overlay_payload,
    overlay_hash,
    resolve_overlay_path,
    write_overlay_payload,
)
from engine.config.settings import EleanorSettings
from engine.logging_config import get_logger
from engine.runtime.config import EngineConfig
from engine.security.ledger import LedgerRecord, get_ledger_reader, get_ledger_writer
from engine.version import ELEANOR_VERSION
from api.rest.services.deliberation_utils import normalize_engine_result
from engine.core import build_eleanor_engine_v8

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


def _parse_duration(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    raw = raw.strip().lower()
    if not raw:
        return None
    unit = raw[-1]
    if unit not in ("s", "m", "h", "d"):
        return None
    try:
        value = int(raw[:-1])
    except ValueError:
        return None
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return value * multipliers[unit]


def _timestamp_to_epoch(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            candidate = value
            if candidate.endswith("Z"):
                candidate = candidate.replace("Z", "+00:00")
            return datetime.fromisoformat(candidate).timestamp()
        except ValueError:
            return None
    return None


def _load_replay_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path:
        return records
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except FileNotFoundError:
        return []
    return records


def _select_replay_window(
    records: List[Dict[str, Any]],
    window: Dict[str, Any],
    limits: Dict[str, Any],
) -> List[Dict[str, Any]]:
    window_type = (window or {}).get("type") or "time"
    max_traces = limits.get("max_traces") if isinstance(limits, dict) else None
    selected = records

    if window_type == "count":
        limit = window.get("limit") if isinstance(window, dict) else None
        limit = limit or max_traces
        if isinstance(limit, int) and limit > 0:
            selected = selected[-limit:]
    else:
        duration = _parse_duration(window.get("duration") if isinstance(window, dict) else None)
        if duration is None:
            duration = 4 * 3600
        cutoff = datetime.now(timezone.utc).timestamp() - duration
        filtered: List[Dict[str, Any]] = []
        for record in selected:
            ts = _timestamp_to_epoch(record.get("timestamp"))
            if ts is None or ts >= cutoff:
                filtered.append(record)
        selected = filtered
        if isinstance(max_traces, int) and max_traces > 0 and len(selected) > max_traces:
            selected = selected[-max_traces:]

    return selected


def _decision_label(raw: Optional[str]) -> str:
    value = (raw or "allow").lower()
    mapping = {
        "allow": "ALLOW",
        "constrained_allow": "ALLOW_WITH_CONSTRAINTS",
        "deny": "DENY",
        "escalate": "ESCALATE",
        "abstain": "ABSTAIN",
    }
    return mapping.get(value, value.upper())


def _extract_score(result: Dict[str, Any]) -> Optional[float]:
    aggregated = result.get("aggregator_output") or {}
    for key in ("final_score", "score", "aggregate_score"):
        value = aggregated.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def build_preview_engine(
    *,
    settings: EleanorSettings,
    constitutional_config: Optional[Dict[str, Any]],
    evidence_path: Optional[str],
) -> Any:
    return build_eleanor_engine_v8(
        constitutional_config=constitutional_config,
        evidence_path=evidence_path,
        settings_override=settings,
    )


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


def _config_fingerprint(settings_dict: Dict[str, Any], overlay_digest: Optional[str]) -> str:
    payload = dict(settings_dict)
    if overlay_digest:
        payload["config_overlay_hash"] = overlay_digest
    return _hash_with_prefix(payload)


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


def _find_applied_event(proposal_id: str, artifact_hash: str) -> Optional[LedgerRecord]:
    latest = _find_latest_event("config_proposal_applied", proposal_id)
    if latest is None:
        return None
    payload = latest.payload or {}
    if payload.get("artifact_hash") == artifact_hash:
        return latest
    return None


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

    baseline_overlay = load_overlay_payload()
    candidate_overlay = _deep_merge(baseline_overlay, changes)
    baseline_overlay_hash = overlay_hash(baseline_overlay)
    candidate_overlay_hash = overlay_hash(candidate_overlay)

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
            "baseline": _config_fingerprint(baseline_settings, baseline_overlay_hash),
            "candidate": _config_fingerprint(candidate_settings, candidate_overlay_hash),
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


async def run_full_replay_preview(
    *,
    proposal_id: str,
    proposal: Dict[str, Any],
    window: Dict[str, Any],
    limits: Dict[str, Any],
    engine: Any,
) -> Dict[str, Any]:
    settings = ConfigManager().settings
    replay_path = settings.replay_log_path
    records = _load_replay_records(replay_path)
    selected = _select_replay_window(records, window, limits)

    artifact = build_preview_artifact(
        proposal_id=proposal_id,
        proposal=proposal,
        mode="full_replay",
        window=window,
        limits=limits,
        engine=engine,
    )
    base_reason_codes = artifact.get("reason_codes") or []

    if not selected:
        artifact["warnings"].append("REPLAY_EMPTY: No replay traces found for preview.")
        return artifact

    candidate_settings, _ = compute_candidate_settings(proposal=proposal)
    candidate_engine = build_preview_engine(
        settings=candidate_settings,
        constitutional_config=None,
        evidence_path=settings.evidence.jsonl_path,
    )

    if hasattr(candidate_engine, "__aenter__"):
        await candidate_engine.__aenter__()
    elif hasattr(candidate_engine, "_setup_resources"):
        await candidate_engine._setup_resources()

    changed_traces: List[Dict[str, Any]] = []
    net_effect = {
        "allow_to_deny": 0,
        "deny_to_allow": 0,
        "allow_to_review": 0,
        "review_to_allow": 0,
        "review_to_deny": 0,
    }
    warnings: List[str] = []

    try:
        for record in selected:
            input_text = record.get("input") or ""
            context = record.get("context") or {}
            trace_id = record.get("trace_id") or record.get("request_id") or uuid4().hex

            try:
                baseline_result = await engine.run(
                    input_text,
                    context=context,
                    trace_id=trace_id,
                    detail_level=3,
                )
                baseline_norm = normalize_engine_result(
                    baseline_result, input_text, trace_id, context
                )
            except Exception as exc:
                warnings.append("REPLAY_TRACE_ERROR: baseline_run_failed")
                logger.warning(
                    "preview_baseline_run_failed",
                    extra={"trace_id": trace_id, "error": str(exc)},
                )
                continue

            try:
                candidate_result = await candidate_engine.run(
                    input_text,
                    context=context,
                    trace_id=trace_id,
                    detail_level=3,
                )
                candidate_norm = normalize_engine_result(
                    candidate_result, input_text, trace_id, context
                )
            except Exception as exc:
                warnings.append("REPLAY_TRACE_ERROR: candidate_run_failed")
                logger.warning(
                    "preview_candidate_run_failed",
                    extra={"trace_id": trace_id, "error": str(exc)},
                )
                continue

            baseline_decision = _decision_label(
                (baseline_norm.get("aggregator_output") or {}).get("decision")
            )
            candidate_decision = _decision_label(
                (candidate_norm.get("aggregator_output") or {}).get("decision")
            )

            if baseline_decision != candidate_decision:
                baseline_score = _extract_score(baseline_norm)
                candidate_score = _extract_score(candidate_norm)
                delta_score = 1.0
                if baseline_score is not None and candidate_score is not None:
                    delta_score = abs(candidate_score - baseline_score)

                changed_traces.append(
                    {
                        "trace_id": trace_id,
                        "request_type": "decision",
                        "subject_type": "replay_trace",
                        "baseline_outcome": baseline_decision,
                        "candidate_outcome": candidate_decision,
                        "delta_score": delta_score,
                        "primary_reason_codes": [
                            *base_reason_codes,
                            "HEURISTIC_ATTRIBUTION",
                        ],
                        "why_changed": {
                            "attribution_kind": "heuristic",
                            "summary": "Outcome changed under candidate configuration.",
                            "evidence_refs": [],
                        },
                        "diff_snippet": {
                            "baseline_score": baseline_score,
                            "candidate_score": candidate_score,
                        },
                    }
                )

                if baseline_decision == "ALLOW" and candidate_decision == "DENY":
                    net_effect["allow_to_deny"] += 1
                elif baseline_decision == "DENY" and candidate_decision == "ALLOW":
                    net_effect["deny_to_allow"] += 1
                elif baseline_decision == "ALLOW" and candidate_decision == "ESCALATE":
                    net_effect["allow_to_review"] += 1
                elif baseline_decision == "ESCALATE" and candidate_decision == "ALLOW":
                    net_effect["review_to_allow"] += 1
                elif baseline_decision == "ESCALATE" and candidate_decision == "DENY":
                    net_effect["review_to_deny"] += 1

        changed_traces.sort(key=lambda item: item.get("delta_score", 0), reverse=True)
        max_changed = limits.get("max_changed_traces") if isinstance(limits, dict) else None
        if isinstance(max_changed, int) and max_changed > 0:
            changed_traces = changed_traces[:max_changed]

        artifact["warnings"].extend(sorted(set(warnings)))
        artifact["metrics"]["trace_count"] = len(selected)
        artifact["metrics"]["changed_trace_count"] = len(changed_traces)
        artifact["outcome_summary"] = {
            "trace_count": len(selected),
            "changed_trace_count": len(changed_traces),
            "net_effect": net_effect,
        }
        artifact["top_changed_traces"] = changed_traces
        return artifact
    finally:
        if hasattr(candidate_engine, "shutdown"):
            try:
                await candidate_engine.shutdown()
            except Exception as exc:
                logger.warning("preview_candidate_shutdown_failed", extra={"error": str(exc)})
        elif hasattr(candidate_engine, "__aexit__"):
            try:
                await candidate_engine.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("preview_candidate_exit_failed", extra={"error": str(exc)})


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

    baseline_overlay = load_overlay_payload()
    baseline_overlay_hash = overlay_hash(baseline_overlay)

    existing_apply = _find_applied_event(proposal_id, artifact_hash)
    if existing_apply:
        applied_payload = existing_apply.payload or {}
        current_fingerprint = _config_fingerprint(
            _settings_dict(settings),
            baseline_overlay_hash,
        )
        return {
            "proposal_id": proposal_id,
            "status": "already_applied",
            "applied_at": applied_payload.get("applied_at") or existing_apply.timestamp,
            "fingerprints": {
                "previous": artifact.get("fingerprints", {}).get("baseline"),
                "current": current_fingerprint,
            },
            "ledger": {
                "event": "config_proposal_applied",
                "event_id": existing_apply.event_id,
                "artifact_hash": artifact_hash,
            },
        }

    baseline_fingerprint = artifact.get("fingerprints", {}).get("baseline")
    current_fingerprint = _config_fingerprint(
        _settings_dict(settings),
        baseline_overlay_hash,
    )
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
    candidate_overlay = _deep_merge(baseline_overlay, proposal.get("changes") or {})
    candidate_overlay_hash = overlay_hash(candidate_overlay)
    candidate_fingerprint = artifact.get("fingerprints", {}).get("candidate")
    if candidate_fingerprint and candidate_fingerprint != _config_fingerprint(
        candidate_dict,
        candidate_overlay_hash,
    ):
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

    if resolve_overlay_path() is None:
        raise PreviewValidationError(
            code="CONFIG_OVERLAY_PATH_REQUIRED",
            status_code=500,
            message="Config overlay path is not configured.",
        )

    write_overlay_payload(candidate_overlay)

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
            "current": _config_fingerprint(
                _settings_dict(candidate_model),
                candidate_overlay_hash,
            ),
        },
        "ledger": {
            "event": "config_proposal_applied",
            "event_id": ledger_record.event_id if ledger_record else None,
            "artifact_hash": artifact_hash,
        },
    }
