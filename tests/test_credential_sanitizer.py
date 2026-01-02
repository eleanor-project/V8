import pytest

from engine.security.sanitizer import CredentialSanitizer
from engine.recorder.evidence_recorder import EvidenceRecorder


def test_sanitize_text_redacts_openai_key():
    raw = "token sk-1234567890ABCDEF1234567890ABCDEF"
    sanitized = CredentialSanitizer.sanitize_text(raw)
    assert "sk-" not in sanitized
    assert "[OPENAI_API_KEY_REDACTED]" in sanitized


def test_sanitize_dict_redacts_sensitive_keys():
    payload = {
        "api_key": "sk-1234567890ABCDEF1234567890ABCDEF",
        "nested": {"token": "Bearer abcdefghijklmnop", "note": "ok"},
    }
    sanitized = CredentialSanitizer.sanitize_dict(payload)
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["token"] == "[REDACTED]"
    assert sanitized["nested"]["note"] == "ok"


@pytest.mark.asyncio
async def test_evidence_recorder_sanitizes_payloads():
    recorder = EvidenceRecorder(jsonl_path=None)
    record = await recorder.record(
        critic="test",
        rule_id="rule",
        severity="INFO",
        violation_description="Uses token sk-1234567890ABCDEF1234567890ABCDEF",
        confidence=0.5,
        context={"auth_token": "Bearer abcdefghijklmnop"},
        raw_text="sk-1234567890ABCDEF1234567890ABCDEF",
    )

    assert "[OPENAI_API_KEY_REDACTED]" in (record.violation_description or "")
    assert record.context.get("auth_token") == "[REDACTED]"
    assert "[OPENAI_API_KEY_REDACTED]" in (record.raw_text or "")
