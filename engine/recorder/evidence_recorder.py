import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from .db_sink import EvidenceDBSink


# ---------------------------------------------------------
# Evidence Record Model
# ---------------------------------------------------------

class EvidenceRecord(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
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

        self.jsonl_path = jsonl_path
        self.db_sink = db_sink

        self._lock = asyncio.Lock()

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
            violation_description=violation_description,
            mitigation=mitigation,
            detector_metadata=detector_metadata or {},
            context=context or {},
            uncertainty_flags=uncertainty_flags or {},
            precedent_sources=precedent_sources or [],
            precedent_candidates=precedent_candidates or [],
            raw_text=raw_text,
        )

        # In-memory buffer sink
        self._add_to_buffer(record)

        # JSONL sink
        await self._write_jsonl(record)

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
        for record in self.buffer:
            await self._write_jsonl(record)
