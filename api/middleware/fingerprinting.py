"""
ELEANOR V8 — Request Fingerprinting
------------------------------------

Fingerprint requests for anomaly detection and security.
"""

import hashlib
import logging
from typing import Dict, Optional
from fastapi import Request

logger = logging.getLogger(__name__)


class RequestFingerprinter:
    """
    Fingerprint requests for anomaly detection.
    
    Creates unique fingerprints based on request characteristics
    to detect suspicious patterns.
    """
    
    def __init__(self):
        self._fingerprint_cache: Dict[str, str] = {}
    
    def fingerprint(self, request: Request) -> str:
        """
        Create request fingerprint.
        
        Args:
            request: FastAPI request
        
        Returns:
            SHA256 hash of request fingerprint
        """
        # Get components for fingerprinting
        user_agent = request.headers.get("user-agent", "")
        client_host = request.client.host if request.client else "unknown"
        path = request.url.path
        method = request.method
        
        # Get authentication info if available
        auth_header = request.headers.get("authorization", "")
        auth_hash = hashlib.sha256(auth_header.encode()).hexdigest()[:8] if auth_header else "none"
        
        # Combine components
        components = [
            method,
            path,
            user_agent,
            client_host,
            auth_hash,
        ]
        
        fingerprint_string = "|".join(components)
        fingerprint_hash = hashlib.sha256(fingerprint_string.encode()).hexdigest()
        
        return fingerprint_hash
    
    def fingerprint_with_body(self, request: Request, body_hash: Optional[str] = None) -> str:
        """
        Create fingerprint including request body.
        
        Args:
            request: FastAPI request
            body_hash: Optional pre-computed body hash
        
        Returns:
            SHA256 hash of request fingerprint
        """
        base_fingerprint = self.fingerprint(request)
        
        if body_hash:
            combined = f"{base_fingerprint}:{body_hash}"
            return hashlib.sha256(combined.encode()).hexdigest()
        
        return base_fingerprint
    
    def get_fingerprint_components(self, request: Request) -> Dict[str, str]:
        """
        Get fingerprint components for analysis.
        
        Args:
            request: FastAPI request
        
        Returns:
            Dictionary of fingerprint components
        """
        return {
            "method": request.method,
            "path": request.url.path,
            "user_agent": request.headers.get("user-agent", ""),
            "client_host": request.client.host if request.client else "unknown",
            "has_auth": bool(request.headers.get("authorization")),
            "fingerprint": self.fingerprint(request),
        }


__all__ = ["RequestFingerprinter"]
