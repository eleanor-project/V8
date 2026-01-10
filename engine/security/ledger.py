from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

LEDGER_IMPL_ID = "stone_tablet_ledger"

try:
    import boto3 as _boto3  # type: ignore[import-not-found]
    from botocore.exceptions import ClientError as _ClientError  # type: ignore[import-not-found]
except Exception:
    _boto3 = None
    _ClientError = None

boto3 = _boto3
ClientError = _ClientError


class LedgerBackend(str, Enum):
    STONE_TABLET_LEDGER = "stone_tablet_ledger"
    S3_OBJECT_LOCK = "s3_object_lock"
    POSTGRES_APPEND_ONLY = "postgres_append_only"
    QLDB = "qldb"
    DISABLED = "disabled"


def ledger_impl_id() -> str:
    return LEDGER_IMPL_ID


def _resolve_backend_raw() -> str:
    raw = os.getenv("ELEANOR_LEDGER_BACKEND") or os.getenv("ELEANOR_LEDGER_IMPL")
    raw = (raw or LEDGER_IMPL_ID).strip().lower()
    aliases = {
        "file_jsonl": LEDGER_IMPL_ID,
        "jsonl": LEDGER_IMPL_ID,
        "stone_tablet": LEDGER_IMPL_ID,
        "stone": LEDGER_IMPL_ID,
    }
    return aliases.get(raw, raw)


def ledger_backend_id() -> str:
    return _resolve_backend_raw()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _hash_payload(payload: Any) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _verification_failure(
    index: int,
    record: LedgerRecord,
    count: int,
    backend: str,
    error: str,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "records": count,
        "backend": backend,
        "index": index,
        "error": error,
        "trace_id": record.trace_id,
        "record_hash": record.record_hash,
    }


def _verify_record_chain(records: Iterable[LedgerRecord], backend: str) -> Dict[str, Any]:
    count = 0
    last_hash = ""
    for index, record in enumerate(records, start=1):
        expected_payload_hash = _hash_payload(record.payload)
        if record.payload_hash != expected_payload_hash:
            return _verification_failure(
                index,
                record,
                count,
                backend,
                f"payload_hash mismatch ({record.payload_hash} != {expected_payload_hash})",
            )

        record_core = {
            "event_id": record.event_id,
            "timestamp": record.timestamp,
            "event": record.event,
            "trace_id": record.trace_id,
            "actor_id": record.actor_id,
            "payload": record.payload,
            "payload_hash": record.payload_hash,
            "prev_hash": record.prev_hash,
        }
        expected_record_hash = _hash_payload(record_core)
        if record.record_hash != expected_record_hash:
            return _verification_failure(
                index,
                record,
                count,
                backend,
                f"record_hash mismatch ({record.record_hash} != {expected_record_hash})",
            )

        if record.prev_hash != last_hash:
            return _verification_failure(
                index,
                record,
                count,
                backend,
                f"prev_hash mismatch ({record.prev_hash} != {last_hash})",
            )

        last_hash = record.record_hash
        count += 1

    return {
        "ok": True,
        "records": count,
        "backend": backend,
        "last_hash": last_hash,
    }


def _normalize_prefix(prefix: str) -> str:
    prefix = (prefix or "").strip("/")
    return f"{prefix}/" if prefix else ""


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _is_conditional_check_failed(exc: Exception) -> bool:
    if ClientError and isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException"
    return "ConditionalCheckFailedException" in str(exc)


def _extract_trace_and_actor(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    trace_id = payload.get("trace_id") or payload.get("request_id")
    actor = payload.get("user") or payload.get("user_id") or payload.get("actor")
    return (str(trace_id) if trace_id else None, str(actor) if actor else None)


@dataclass
class LedgerRecord:
    event_id: str
    timestamp: str
    event: str
    payload: Dict[str, Any]
    payload_hash: str
    prev_hash: str
    record_hash: str
    trace_id: Optional[str] = None
    actor_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event": self.event,
            "trace_id": self.trace_id,
            "actor_id": self.actor_id,
            "payload": self.payload,
            "payload_hash": self.payload_hash,
            "prev_hash": self.prev_hash,
            "record_hash": self.record_hash,
        }


class LedgerWriter:
    def append(self, event: str, payload: Dict[str, Any]) -> Optional[LedgerRecord]:
        raise NotImplementedError


class LedgerReader:
    def read_all(self) -> List[LedgerRecord]:
        raise NotImplementedError

    def verify_chain(self) -> Dict[str, Any]:
        raise NotImplementedError


class NullLedgerWriter(LedgerWriter):
    def append(self, event: str, payload: Dict[str, Any]) -> Optional[LedgerRecord]:
        return None


class NullLedgerReader(LedgerReader):
    def read_all(self) -> List[LedgerRecord]:
        return []

    def verify_chain(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "records": 0,
            "backend": LedgerBackend.DISABLED.value,
            "message": "Ledger backend disabled.",
        }


class JsonlLedgerWriter(LedgerWriter):
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._load_last_hash()

    def append(self, event: str, payload: Dict[str, Any]) -> Optional[LedgerRecord]:
        payload = dict(payload or {})
        trace_id, actor_id = _extract_trace_and_actor(payload)
        payload_hash = _hash_payload(payload)
        prev_hash = self._last_hash
        record = {
            "event_id": uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event": event,
            "trace_id": trace_id,
            "actor_id": actor_id,
            "payload": payload,
            "payload_hash": payload_hash,
            "prev_hash": prev_hash,
        }
        record_hash = _hash_payload(record)
        ledger_record = LedgerRecord(**{**record, "record_hash": record_hash})
        line = json.dumps(ledger_record.to_dict(), ensure_ascii=True, default=str)

        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._last_hash = record_hash

        return ledger_record

    def _load_last_hash(self) -> str:
        if not self.path.exists():
            return ""
        last_hash = ""
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    candidate = item.get("record_hash")
                    if candidate:
                        last_hash = str(candidate)
        except Exception as exc:
            logger.warning("ledger_load_failed", extra={"path": str(self.path), "error": str(exc)})
        return last_hash


class JsonlLedgerReader(LedgerReader):
    def __init__(self, path: str):
        self.path = Path(path)

    def read_all(self) -> List[LedgerRecord]:
        return list(self._iter_records(strict=False))

    def verify_chain(self) -> Dict[str, Any]:
        try:
            return _verify_record_chain(
                self._iter_records(strict=True),
                LedgerBackend.STONE_TABLET_LEDGER.value,
            )
        except Exception as exc:
            return {
                "ok": False,
                "records": 0,
                "backend": LedgerBackend.STONE_TABLET_LEDGER.value,
                "error": str(exc),
            }

    def _iter_records(self, strict: bool) -> Iterable[LedgerRecord]:
        if not self.path.exists():
            return iter(())

        def generator() -> Iterable[LedgerRecord]:
            with self.path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        if strict:
                            raise ValueError(f"Invalid JSON at line {line_number}") from exc
                        logger.warning(
                            "ledger_json_invalid",
                            extra={"path": str(self.path), "line": line_number},
                        )
                        continue

                    try:
                        record = self._record_from_dict(data)
                    except ValueError as exc:
                        if strict:
                            raise
                        logger.warning(
                            "ledger_record_invalid",
                            extra={"path": str(self.path), "line": line_number, "error": str(exc)},
                        )
                        continue
                    yield record

        return generator()

    @staticmethod
    def _record_from_dict(data: Dict[str, Any]) -> LedgerRecord:
        required = ("event_id", "timestamp", "event", "payload", "payload_hash", "prev_hash", "record_hash")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing fields: {', '.join(missing)}")

        return LedgerRecord(
            event_id=str(data["event_id"]),
            timestamp=str(data["timestamp"]),
            event=str(data["event"]),
            payload=dict(data.get("payload") or {}),
            payload_hash=str(data["payload_hash"]),
            prev_hash=str(data["prev_hash"]),
            record_hash=str(data["record_hash"]),
            trace_id=str(data.get("trace_id")) if data.get("trace_id") is not None else None,
            actor_id=str(data.get("actor_id")) if data.get("actor_id") is not None else None,
        )


def _resolve_s3_settings() -> Tuple[str, str, str]:
    bucket = os.getenv("ELEANOR_LEDGER_S3_BUCKET")
    region = (
        os.getenv("ELEANOR_LEDGER_S3_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
    )
    prefix = _normalize_prefix(os.getenv("ELEANOR_LEDGER_S3_PREFIX", "audit-ledger"))
    missing = []
    if not bucket:
        missing.append("ELEANOR_LEDGER_S3_BUCKET")
    if not region:
        missing.append("ELEANOR_LEDGER_S3_REGION or AWS_REGION/AWS_DEFAULT_REGION")
    if missing:
        raise RuntimeError(f"Missing required S3 ledger settings: {', '.join(missing)}")
    return bucket, prefix, region


def _resolve_dynamodb_settings(s3_region: Optional[str]) -> Optional[Tuple[str, str, str]]:
    table = os.getenv("ELEANOR_LEDGER_DDB_TABLE")
    if not table:
        return None
    region = os.getenv("ELEANOR_LEDGER_DDB_REGION") or s3_region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError("Missing required DynamoDB region (ELEANOR_LEDGER_DDB_REGION or AWS_REGION)")
    ledger_id = os.getenv("ELEANOR_LEDGER_DDB_LEDGER_ID", "default")
    return table, region, ledger_id


def _resolve_inventory_settings() -> Optional[Tuple[str, str]]:
    if not _truthy(os.getenv("ELEANOR_LEDGER_S3_INVENTORY_ENABLED")):
        return None
    bucket = os.getenv("ELEANOR_LEDGER_S3_INVENTORY_BUCKET")
    prefix = os.getenv("ELEANOR_LEDGER_S3_INVENTORY_PREFIX")
    if not prefix:
        raise RuntimeError("ELEANOR_LEDGER_S3_INVENTORY_PREFIX is required when inventory is enabled.")
    prefix = _normalize_prefix(prefix)
    return bucket, prefix


class DynamoLedgerIndex:
    def __init__(self, table_name: str, region: str, ledger_id: str):
        if boto3 is None:
            raise RuntimeError("boto3 required for DynamoDB ledger index. Install with: pip install boto3")
        self.table_name = table_name
        self.region = region
        self.ledger_id = ledger_id
        self._resource = boto3.resource("dynamodb", region_name=region)
        self._table = self._resource.Table(table_name)

    @property
    def _meta_key(self) -> Dict[str, Any]:
        return {"ledger_id": self.ledger_id, "seq": "META"}

    def get_meta(self) -> Dict[str, Any]:
        response = self._table.get_item(Key=self._meta_key, ConsistentRead=True)
        return response.get("Item") or {}

    def reserve_sequence(self, prev_hash: str, new_hash: str) -> int:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        response = self._table.update_item(
            Key=self._meta_key,
            UpdateExpression="SET last_hash = :new_hash, updated_at = :ts ADD seq_counter :inc",
            ConditionExpression="attribute_not_exists(last_hash) OR last_hash = :prev_hash",
            ExpressionAttributeValues={
                ":new_hash": new_hash,
                ":prev_hash": prev_hash,
                ":ts": timestamp,
                ":inc": 1,
            },
            ReturnValues="UPDATED_NEW",
        )
        seq_value = response.get("Attributes", {}).get("seq_counter")
        return int(seq_value) if seq_value is not None else 0

    def rollback_last_hash(self, prev_hash: str, new_hash: str) -> None:
        try:
            self._table.update_item(
                Key=self._meta_key,
                UpdateExpression="SET last_hash = :prev_hash",
                ConditionExpression="last_hash = :new_hash",
                ExpressionAttributeValues={":prev_hash": prev_hash, ":new_hash": new_hash},
            )
        except Exception as exc:
            logger.warning("ledger_ddb_rollback_failed", extra={"error": str(exc)})

    def store_record(self, seq: int, record: LedgerRecord, s3_key: str) -> None:
        item = {
            "ledger_id": self.ledger_id,
            "seq": f"SEQ#{seq:020d}",
            "s3_key": s3_key,
            "event_id": record.event_id,
            "timestamp": record.timestamp,
            "event": record.event,
            "trace_id": record.trace_id,
            "actor_id": record.actor_id,
            "payload_hash": record.payload_hash,
            "prev_hash": record.prev_hash,
            "record_hash": record.record_hash,
        }
        self._table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(ledger_id) AND attribute_not_exists(seq)",
        )

    def iter_entries(self) -> Iterable[Dict[str, Any]]:
        try:
            from boto3.dynamodb.conditions import Key as _Key
        except Exception as exc:
            raise RuntimeError("boto3 dynamodb conditions unavailable") from exc

        last_key = None
        while True:
            kwargs = {
                "KeyConditionExpression": _Key("ledger_id").eq(self.ledger_id) & _Key("seq").begins_with("SEQ#"),
                "ScanIndexForward": True,
            }
            if last_key:
                kwargs["ExclusiveStartKey"] = last_key
            response = self._table.query(**kwargs)
            for item in response.get("Items", []) or []:
                yield item
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break


def _load_inventory_keys(client: Any, bucket: str, prefix: str, ledger_prefix: str) -> List[str]:
    manifest_key = _find_latest_inventory_manifest(client, bucket, prefix)
    if not manifest_key:
        raise RuntimeError("No inventory manifest found for configured prefix.")
    manifest = _load_manifest(client, bucket, manifest_key)
    keys: List[str] = []
    for file_info in manifest.get("files", []):
        key = file_info.get("key")
        if not key:
            continue
        keys.extend(_load_inventory_file_keys(client, bucket, key, ledger_prefix))
    keys.sort()
    return keys


def _find_latest_inventory_manifest(client: Any, bucket: str, prefix: str) -> Optional[str]:
    keys = _list_s3_keys(client, bucket, prefix)
    candidates = [key for key in keys if key.endswith("manifest.json")]
    return candidates[-1] if candidates else None


def _load_manifest(client: Any, bucket: str, key: str) -> Dict[str, Any]:
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


def _load_inventory_file_keys(client: Any, bucket: str, key: str, ledger_prefix: str) -> List[str]:
    response = client.get_object(Bucket=bucket, Key=key)
    raw = response["Body"].read()
    if key.endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            data = gz.read().decode("utf-8")
    else:
        data = raw.decode("utf-8")

    keys: List[str] = []
    reader = csv.reader(io.StringIO(data))
    for row in reader:
        if len(row) < 2:
            continue
        candidate = row[1]
        if candidate.startswith(ledger_prefix):
            keys.append(candidate)
    return keys


def _list_s3_keys(client: Any, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        params = {"Bucket": bucket, "Prefix": prefix}
        if token:
            params["ContinuationToken"] = token
        response = client.list_objects_v2(**params)
        for item in response.get("Contents", []) or []:
            key = item.get("Key")
            if key:
                keys.append(key)
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
    keys.sort()
    return keys


class S3ObjectLockLedgerWriter(LedgerWriter):
    def __init__(self, bucket: str, prefix: str, region: str):
        if boto3 is None:
            raise RuntimeError("boto3 required for s3_object_lock ledger. Install with: pip install boto3")
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._lock = threading.Lock()
        self._client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=os.getenv("ELEANOR_LEDGER_S3_ENDPOINT_URL"),
        )
        self._index = None
        ddb_settings = _resolve_dynamodb_settings(region)
        if ddb_settings:
            table, ddb_region, ledger_id = ddb_settings
            self._index = DynamoLedgerIndex(table_name=table, region=ddb_region, ledger_id=ledger_id)
        else:
            logger.warning(
                "ledger_ddb_index_disabled",
                extra={"backend": LedgerBackend.S3_OBJECT_LOCK.value},
            )
        self._retention_days = int(os.getenv("ELEANOR_LEDGER_S3_RETENTION_DAYS", "0") or 0)
        self._lock_mode = os.getenv("ELEANOR_LEDGER_S3_OBJECT_LOCK_MODE", "COMPLIANCE").upper()
        if self._lock_mode not in {"COMPLIANCE", "GOVERNANCE"}:
            raise ValueError("ELEANOR_LEDGER_S3_OBJECT_LOCK_MODE must be COMPLIANCE or GOVERNANCE")
        self._kms_key_id = os.getenv("ELEANOR_LEDGER_S3_KMS_KEY_ID") or None
        self._last_hash = self._load_last_hash_from_index() if self._index else self._load_last_hash()

    def append(self, event: str, payload: Dict[str, Any]) -> Optional[LedgerRecord]:
        if self._index:
            return self._append_indexed(event, payload)
        return self._append_unindexed(event, payload)

    def _append_unindexed(self, event: str, payload: Dict[str, Any]) -> Optional[LedgerRecord]:
        payload = dict(payload or {})
        timestamp = datetime.now(timezone.utc)
        ledger_record = self._build_record(event, payload, timestamp, self._last_hash)
        body = json.dumps(ledger_record.to_dict(), ensure_ascii=True, default=str).encode("utf-8")
        key = self._object_key(timestamp, ledger_record.event_id, None)

        self._put_object(key, body, timestamp)
        self._last_hash = ledger_record.record_hash
        return ledger_record

    def _append_indexed(self, event: str, payload: Dict[str, Any]) -> Optional[LedgerRecord]:
        payload = dict(payload or {})
        attempts = 0
        while attempts < 5:
            prev_hash = ""
            try:
                meta = self._index.get_meta() if self._index else {}
                prev_hash = str(meta.get("last_hash") or "")
                timestamp = datetime.now(timezone.utc)
                ledger_record = self._build_record(event, payload, timestamp, prev_hash)
                seq = self._index.reserve_sequence(prev_hash, ledger_record.record_hash)
            except Exception as exc:
                if _is_conditional_check_failed(exc):
                    attempts += 1
                    continue
                raise

            body = json.dumps(ledger_record.to_dict(), ensure_ascii=True, default=str).encode("utf-8")
            key = self._object_key(timestamp, ledger_record.event_id, seq)
            try:
                self._put_object(key, body, timestamp)
            except Exception:
                if self._index:
                    self._index.rollback_last_hash(prev_hash, ledger_record.record_hash)
                raise

            if self._index:
                try:
                    self._index.store_record(seq, ledger_record, key)
                except Exception as exc:
                    logger.error(
                        "ledger_ddb_record_store_failed",
                        extra={"error": str(exc), "seq": seq, "s3_key": key},
                    )
                    raise

            self._last_hash = ledger_record.record_hash
            return ledger_record

        raise RuntimeError("Failed to append ledger record after retries.")

    def _build_record(
        self,
        event: str,
        payload: Dict[str, Any],
        timestamp: datetime,
        prev_hash: str,
    ) -> LedgerRecord:
        trace_id, actor_id = _extract_trace_and_actor(payload)
        payload_hash = _hash_payload(payload)
        record = {
            "event_id": uuid4().hex,
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "event": event,
            "trace_id": trace_id,
            "actor_id": actor_id,
            "payload": payload,
            "payload_hash": payload_hash,
            "prev_hash": prev_hash,
        }
        record_hash = _hash_payload(record)
        return LedgerRecord(**{**record, "record_hash": record_hash})

    def _object_key(self, timestamp: datetime, event_id: str, seq: Optional[int]) -> str:
        if seq is not None:
            return f"{self.prefix}{seq:020d}-{event_id}.json"
        date_prefix = timestamp.strftime("%Y/%m/%d/%H/%M/%S")
        micros = timestamp.strftime("%f")
        return f"{self.prefix}{date_prefix}-{micros}-{event_id}.json"

    def _put_object(self, key: str, body: bytes, timestamp: datetime) -> None:
        put_params: Dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": body,
            "ContentType": "application/json",
        }
        if self._kms_key_id:
            put_params["ServerSideEncryption"] = "aws:kms"
            put_params["SSEKMSKeyId"] = self._kms_key_id
        if self._retention_days > 0:
            put_params["ObjectLockMode"] = self._lock_mode
            put_params["ObjectLockRetainUntilDate"] = timestamp + timedelta(days=self._retention_days)

        with self._lock:
            self._client.put_object(**put_params)

    def _load_last_hash(self) -> str:
        try:
            keys = _list_s3_keys(self._client, self.bucket, self.prefix)
            if not keys:
                return ""
            last_key = keys[-1]
            record = self._read_record(last_key)
            return record.record_hash if record else ""
        except Exception as exc:
            logger.warning(
                "ledger_s3_load_failed",
                extra={"bucket": self.bucket, "prefix": self.prefix, "error": str(exc)},
            )
            return ""

    def _load_last_hash_from_index(self) -> str:
        try:
            meta = self._index.get_meta() if self._index else {}
            return str(meta.get("last_hash") or "")
        except Exception as exc:
            logger.warning("ledger_ddb_meta_load_failed", extra={"error": str(exc)})
            return ""

    def _read_record(self, key: str) -> Optional[LedgerRecord]:
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"].read().decode("utf-8")
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict):
            try:
                return JsonlLedgerReader._record_from_dict(data)
            except ValueError:
                return None
        return None


class S3ObjectLockLedgerReader(LedgerReader):
    def __init__(self, bucket: str, prefix: str, region: str):
        if boto3 is None:
            raise RuntimeError("boto3 required for s3_object_lock ledger. Install with: pip install boto3")
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=os.getenv("ELEANOR_LEDGER_S3_ENDPOINT_URL"),
        )
        self._index = None
        ddb_settings = _resolve_dynamodb_settings(region)
        if ddb_settings:
            table, ddb_region, ledger_id = ddb_settings
            self._index = DynamoLedgerIndex(table_name=table, region=ddb_region, ledger_id=ledger_id)
        self._inventory = _resolve_inventory_settings()

    def read_all(self) -> List[LedgerRecord]:
        return list(self._iter_records(strict=False))

    def verify_chain(self) -> Dict[str, Any]:
        try:
            return _verify_record_chain(self._iter_records(strict=True), LedgerBackend.S3_OBJECT_LOCK.value)
        except Exception as exc:
            return {
                "ok": False,
                "records": 0,
                "backend": LedgerBackend.S3_OBJECT_LOCK.value,
                "error": str(exc),
            }

    def _iter_records(self, strict: bool) -> Iterable[LedgerRecord]:
        keys = self._resolve_keys()

        def generator() -> Iterable[LedgerRecord]:
            for key in keys:
                response = self._client.get_object(Bucket=self.bucket, Key=key)
                body = response["Body"].read().decode("utf-8")
                for record in self._parse_payload(body, key, strict):
                    yield record

        return generator()

    def _resolve_keys(self) -> List[str]:
        if self._index:
            keys: List[str] = []
            for entry in self._index.iter_entries():
                s3_key = entry.get("s3_key")
                if s3_key:
                    keys.append(str(s3_key))
            return keys

        if self._inventory:
            inventory_bucket, inventory_prefix = self._inventory
            bucket = inventory_bucket or self.bucket
            return _load_inventory_keys(self._client, bucket, inventory_prefix, self.prefix)

        return _list_s3_keys(self._client, self.bucket, self.prefix)

    def _parse_payload(self, body: str, key: str, strict: bool) -> Iterable[LedgerRecord]:
        items: List[Any] = []
        try:
            parsed = json.loads(body)
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                items = [parsed]
            else:
                raise ValueError("Unexpected JSON payload")
        except json.JSONDecodeError:
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            if not lines:
                return []
            for line in lines:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    if strict:
                        raise ValueError(f"Invalid JSON in {key}") from exc
                    logger.warning("ledger_s3_json_invalid", extra={"key": key})
        except ValueError as exc:
            if strict:
                raise
            logger.warning("ledger_s3_payload_invalid", extra={"key": key, "error": str(exc)})
            return []

        records: List[LedgerRecord] = []
        for item in items:
            if not isinstance(item, dict):
                if strict:
                    raise ValueError(f"Invalid record payload in {key}")
                logger.warning("ledger_s3_record_invalid", extra={"key": key})
                continue
            try:
                records.append(JsonlLedgerReader._record_from_dict(item))
            except ValueError as exc:
                if strict:
                    raise
                logger.warning(
                    "ledger_s3_record_invalid",
                    extra={"key": key, "error": str(exc)},
                )
        return records


_writer: Optional[LedgerWriter] = None
_reader: Optional[LedgerReader] = None


def get_ledger_writer() -> LedgerWriter:
    global _writer
    if _writer is not None:
        return _writer

    backend = _resolve_backend_raw()
    if backend == LedgerBackend.DISABLED.value:
        _writer = NullLedgerWriter()
        return _writer

    if backend == LedgerBackend.STONE_TABLET_LEDGER.value:
        path = os.getenv("ELEANOR_LEDGER_PATH", "logs/stone_tablet_ledger.jsonl")
        _writer = JsonlLedgerWriter(path=path)
        return _writer

    if backend == LedgerBackend.S3_OBJECT_LOCK.value:
        if boto3 is None:
            raise RuntimeError("boto3 required for s3_object_lock ledger. Install with: pip install boto3")
        bucket, prefix, region = _resolve_s3_settings()
        _writer = S3ObjectLockLedgerWriter(bucket=bucket, prefix=prefix, region=region)
        return _writer

    if backend in {LedgerBackend.POSTGRES_APPEND_ONLY.value, LedgerBackend.QLDB.value}:
        raise RuntimeError(f"Ledger backend '{backend}' is not implemented yet.")

    logger.warning("unknown_ledger_backend", extra={"backend": backend})
    _writer = NullLedgerWriter()
    return _writer


def get_ledger_reader(path: Optional[str] = None) -> LedgerReader:
    global _reader
    if path is not None:
        return JsonlLedgerReader(path=path)

    if _reader is not None:
        return _reader

    backend = _resolve_backend_raw()
    if backend == LedgerBackend.DISABLED.value:
        _reader = NullLedgerReader()
        return _reader

    if backend == LedgerBackend.STONE_TABLET_LEDGER.value:
        path = os.getenv("ELEANOR_LEDGER_PATH", "logs/stone_tablet_ledger.jsonl")
        _reader = JsonlLedgerReader(path=path)
        return _reader

    if backend == LedgerBackend.S3_OBJECT_LOCK.value:
        if boto3 is None:
            raise RuntimeError("boto3 required for s3_object_lock ledger. Install with: pip install boto3")
        bucket, prefix, region = _resolve_s3_settings()
        _reader = S3ObjectLockLedgerReader(bucket=bucket, prefix=prefix, region=region)
        return _reader

    if backend in {LedgerBackend.POSTGRES_APPEND_ONLY.value, LedgerBackend.QLDB.value}:
        raise RuntimeError(f"Ledger backend '{backend}' is not implemented yet.")

    logger.warning("unknown_ledger_backend", extra={"backend": backend})
    _reader = NullLedgerReader()
    return _reader


def verify_ledger(path: Optional[str] = None) -> Dict[str, Any]:
    try:
        reader = get_ledger_reader(path=path)
        return reader.verify_chain()
    except NotImplementedError as exc:
        return {
            "ok": False,
            "records": 0,
            "backend": ledger_backend_id(),
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "ok": False,
            "records": 0,
            "backend": ledger_backend_id(),
            "error": str(exc),
        }


__all__ = [
    "LedgerBackend",
    "LedgerRecord",
    "LedgerReader",
    "get_ledger_writer",
    "get_ledger_reader",
    "ledger_backend_id",
    "ledger_impl_id",
    "verify_ledger",
]
