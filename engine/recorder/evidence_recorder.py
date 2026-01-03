import os
import asyncio
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from .db_sink import EvidenceDBSink
from engine.security.sanitizer import CredentialSanitizer


# ---------------------------------------------------------
# Evidence Record Model
# ---------------------------------------------------------


class EvidenceRecord(BaseModel):
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    trace_id: str
    request_id: Optional[str] = None

    model_name: Optional[str] = None
    model_version: Optional[str] = None

    critic: Optional[str] = None
    rule_id: Optional[str] = None
    redundancy_group: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None

    violation_description: Optional[str] = None
    mitigation: Optional[str] = None

    detector_metadata: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

    uncertainty_flags: Dict[str, Any] = Field(default_factory=dict)

    precedent_sources: List[str] = Field(default_factory=list)
    precedent_candidates: List[str] = Field(default_factory=list)

    raw_text: Optional[str] = None


# ---------------------------------------------------------
# Evidence Recorder V8 — Tri-sink implementation
# ---------------------------------------------------------


class EvidenceRecorder:
    """
    EvidenceRecorder supports:
        - In-Memory Circular Buffer
        - JSONL Sink
        - Async Database Sink (pluggable)
    """

    def __init__(
        self,
        jsonl_path: Optional[str] = None,
        db_sink: Optional[EvidenceDBSink] = None,
        buffer_size: Optional[int] = None,
        flush_interval: Optional[float] = None,
    ):
        # Configurable buffer size
        env_override = os.getenv("ELEANOR_EVIDENCE_BUFFER_SIZE")
        if env_override is not None:
            try:
                buffer_size = int(env_override)
            except ValueError:
                pass

        self.buffer_size = buffer_size or 10_000
        self.buffer: List[EvidenceRecord] = []
        self.flush_interval = flush_interval or 5.0
        self._pending: List[EvidenceRecord] = []
        self._flush_task: Optional[asyncio.Task] = None
        self._shutdown = False

        self.jsonl_path = jsonl_path
        self.db_sink = db_sink

        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize recorder resources."""
        if self.jsonl_path and self._flush_task is None:
            self._shutdown = False
            self._flush_task = asyncio.create_task(self._periodic_flush())

    async def _periodic_flush(self) -> None:
        """Periodically flush pending evidence to disk."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logging.getLogger(__name__).error(
                    "evidence_flush_failed", extra={"error": str(exc)}
                )

    # -----------------------------------------------------
    # Internal helper: Add to in-memory buffer
    # -----------------------------------------------------
    def _add_to_buffer(self, record: EvidenceRecord):
        self.buffer.append(record)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    # -----------------------------------------------------
    # JSONL sink
    # -----------------------------------------------------
    async def _write_jsonl(self, record: EvidenceRecord):
        if not self.jsonl_path:
            return
        line = record.json() + "\n"
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._append_file, self.jsonl_path, line)

    @staticmethod
    def _append_file(path: str, text: str):
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)

    # -----------------------------------------------------
    # DB sink (async)
    # -----------------------------------------------------
    async def _write_db(self, record: EvidenceRecord):
        if self.db_sink:
            await self.db_sink.write(record)

    # -----------------------------------------------------
    # Public API: Record an evidence event
    # -----------------------------------------------------
    async def record(
        self,
        *,
        critic: str,
        rule_id: str,
        severity: str,
        violation_description: str,
        confidence: float,
        redundancy_group: Optional[str] = None,
        mitigation: Optional[str] = None,
        detector_metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        trace_id: Optional[str] = None,
        request_id: Optional[str] = None,
        raw_text: Optional[str] = None,
        uncertainty_flags: Optional[Dict[str, Any]] = None,
        precedent_sources: Optional[List[str]] = None,
        precedent_candidates: Optional[List[str]] = None,
    ) -> EvidenceRecord:
        trace_id = trace_id or str(uuid.uuid4())

        sanitized_violation_description = CredentialSanitizer.sanitize_text(violation_description)
        sanitized_mitigation = (
            CredentialSanitizer.sanitize_text(mitigation) if mitigation is not None else None
        )
        sanitized_detector_metadata = CredentialSanitizer.sanitize_dict(detector_metadata or {})
        sanitized_context = CredentialSanitizer.sanitize_dict(context or {})
        sanitized_raw_text = (
            CredentialSanitizer.sanitize_text(raw_text) if raw_text is not None else None
        )
        sanitized_uncertainty_flags = CredentialSanitizer.sanitize_dict(uncertainty_flags or {})
        sanitized_precedent_sources = [
            CredentialSanitizer.sanitize_text(source) for source in (precedent_sources or [])
        ]
        sanitized_precedent_candidates = [
            CredentialSanitizer.sanitize_text(candidate)
            for candidate in (precedent_candidates or [])
        ]

        record = EvidenceRecord(
            trace_id=trace_id,
            request_id=request_id,
            model_name=model_name,
            model_version=model_version,
            critic=critic,
            rule_id=rule_id,
            redundancy_group=redundancy_group,
            severity=severity,
            confidence=confidence,
            violation_description=sanitized_violation_description,
            mitigation=sanitized_mitigation,
            detector_metadata=sanitized_detector_metadata,
            context=sanitized_context,
            uncertainty_flags=sanitized_uncertainty_flags,
            precedent_sources=sanitized_precedent_sources,
            precedent_candidates=sanitized_precedent_candidates,
            raw_text=sanitized_raw_text,
        )

        # In-memory buffer sink
        self._add_to_buffer(record)

        # JSONL sink
        if self.jsonl_path:
            self._pending.append(record)
            if len(self._pending) >= self.buffer_size or self.flush_interval <= 0:
                await self.flush()

        # DB sink
        await self._write_db(record)

        return record

    # -----------------------------------------------------
    # Retrieve last N evidence records
    # -----------------------------------------------------
    def latest(self, n: int = 100) -> List[EvidenceRecord]:
        return self.buffer[-n:]

    # -----------------------------------------------------
    # Flush buffer to JSONL (optional utility)
    # -----------------------------------------------------
    async def flush_to_jsonl(self):
        if not self.jsonl_path:
            return
        await self.flush()

    async def flush(self) -> None:
        """Flush pending evidence to JSONL."""
        if not self.jsonl_path or not self._pending:
            return
        pending = list(self._pending)
        self._pending.clear()
        for record in pending:
            await self._write_jsonl(record)

    async def close(self) -> None:
        """Shutdown recorder and flush remaining evidence."""
        self._shutdown = True
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        await self.flush()
