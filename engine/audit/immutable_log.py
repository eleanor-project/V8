"""
ELEANOR V8 — Immutable Audit Log
---------------------------------

Cryptographically signed audit log for compliance and integrity.
"""

import hashlib
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

try:
    import cryptography
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    rsa = None
    padding = None


@dataclass
class AuditRecord:
    """Audit record with cryptographic signature."""
    timestamp: str
    trace_id: str
    event_type: str
    data: Dict[str, Any]
    previous_hash: Optional[str] = None
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)


class ImmutableAuditLog:
    """
    Cryptographically signed audit log.
    
    Provides:
    - Append-only log structure
    - Cryptographic signatures for integrity
    - Chain of hashes for tamper detection
    - Immutable record storage
    """
    
    def __init__(
        self,
        log_path: str,
        private_key_path: Optional[str] = None,
        public_key_path: Optional[str] = None,
    ):
        """
        Initialize immutable audit log.
        
        Args:
            log_path: Path to audit log file
            private_key_path: Path to private key for signing
            public_key_path: Path to public key for verification
        """
        self.log_path = log_path
        self.private_key = None
        self.public_key = None
        self._last_hash: Optional[str] = None
        
        if CRYPTO_AVAILABLE and private_key_path:
            self._load_keys(private_key_path, public_key_path)
        else:
            logger.warning(
                "cryptography_not_available_or_no_key",
                extra={"crypto_available": CRYPTO_AVAILABLE},
            )
    
    def _load_keys(self, private_key_path: str, public_key_path: Optional[str]) -> None:
        """Load cryptographic keys."""
        try:
            from cryptography.hazmat.primitives import serialization
            
            # Load private key
            with open(private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend(),
                )
            
            # Load public key if provided
            if public_key_path:
                with open(public_key_path, "rb") as f:
                    self.public_key = serialization.load_pem_public_key(
                        f.read(),
                        backend=default_backend(),
                    )
            else:
                # Derive public key from private key
                self.public_key = self.private_key.public_key()
        
        except Exception as exc:
            logger.error(
                "failed_to_load_audit_keys",
                extra={"error": str(exc)},
                exc_info=True,
            )
            raise
    
    def _sign(self, record: AuditRecord) -> str:
        """
        Sign audit record.
        
        Args:
            record: Audit record to sign
        
        Returns:
            Base64-encoded signature
        """
        if not self.private_key:
            # Fallback to hash-based signature if no key
            record_json = record.to_json()
            return hashlib.sha256(record_json.encode()).hexdigest()
        
        try:
            import base64
            
            record_json = record.to_json()
            signature = self.private_key.sign(
                record_json.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode()
        
        except Exception as exc:
            logger.error(
                "audit_record_signing_failed",
                extra={"error": str(exc)},
                exc_info=True,
            )
            # Fallback to hash
            record_json = record.to_json()
            return hashlib.sha256(record_json.encode()).hexdigest()
    
    def _hash_record(self, record: AuditRecord) -> str:
        """Calculate hash of record."""
        record_json = record.to_json()
        return hashlib.sha256(record_json.encode()).hexdigest()
    
    async def append(self, record: AuditRecord) -> str:
        """
        Append record with cryptographic signature.
        
        Args:
            record: Audit record to append
        
        Returns:
            Hash of appended record
        """
        # Set previous hash for chain
        record.previous_hash = self._last_hash
        
        # Sign record
        record.signature = self._sign(record)
        
        # Calculate hash
        record_hash = self._hash_record(record)
        
        # Write to log (append-only)
        try:
            import aiofiles
            
            async with aiofiles.open(self.log_path, "a") as f:
                await f.write(record.to_json() + "\n")
        except ImportError:
            # Fallback to sync I/O
            with open(self.log_path, "a") as f:
                f.write(record.to_json() + "\n")
        
        # Update last hash
        self._last_hash = record_hash
        
        logger.debug(
            "audit_record_appended",
            extra={
                "trace_id": record.trace_id,
                "event_type": record.event_type,
                "hash": record_hash,
            },
        )
        
        return record_hash
    
    def verify(self, record: AuditRecord) -> bool:
        """
        Verify record signature.
        
        Args:
            record: Audit record to verify
        
        Returns:
            True if signature is valid
        """
        if not record.signature:
            return False
        
        if not self.public_key:
            # Fallback: verify hash
            expected_hash = self._hash_record(record)
            return record.signature == expected_hash
        
        try:
            import base64
            
            record_json = record.to_json()
            signature_bytes = base64.b64decode(record.signature)
            
            self.public_key.verify(
                signature_bytes,
                record_json.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        
        except Exception as exc:
            logger.warning(
                "audit_record_verification_failed",
                extra={"error": str(exc)},
            )
            return False
    
    def verify_chain(self, records: list[AuditRecord]) -> bool:
        """
        Verify chain of records for tamper detection.
        
        Args:
            records: List of audit records
        
        Returns:
            True if chain is valid
        """
        if not records:
            return True
        
        previous_hash = None
        for record in records:
            # Verify signature
            if not self.verify(record):
                return False
            
            # Verify chain
            if previous_hash and record.previous_hash != previous_hash:
                return False
            
            previous_hash = self._hash_record(record)
        
        return True


__all__ = ["AuditRecord", "ImmutableAuditLog"]
