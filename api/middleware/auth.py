"""
ELEANOR V8 — Authentication Middleware
--------------------------------------

JWT-based authentication for the ELEANOR V8 API.

Configuration via environment variables:
- JWT_SECRET: Secret key for JWT signing (required in production)
- JWT_ALGORITHM: Algorithm for JWT signing (default: HS256)
- AUTH_ENABLED: Enable/disable authentication (default: true in production)
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List, cast
from types import ModuleType
from dataclasses import dataclass
from functools import wraps
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# JWT library - graceful import
_jwt: Optional[ModuleType]
try:
    import jwt as _jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    _jwt = None

jwt: Optional[ModuleType] = _jwt

logger = logging.getLogger(__name__)

# Session management
_sessions: Dict[str, Dict[str, Any]] = {}
_session_cleanup_interval = 300  # 5 minutes
_last_cleanup = time.time()


@dataclass
class AuthConfig:
    """Authentication configuration."""

    secret: str
    algorithm: str = "HS256"
    enabled: bool = True
    token_expiry_seconds: int = 3600

    @classmethod
    def from_env(cls) -> "AuthConfig":
        """Load configuration from environment variables."""
        env = (
            os.getenv("ELEANOR_ENVIRONMENT") or os.getenv("ELEANOR_ENV") or "development"
        ).lower()
        is_dev = env in ("development", "dev", "local")

        default_enabled = "false" if is_dev else "true"
        enabled = os.getenv("AUTH_ENABLED", default_enabled).lower() == "true"

        if not is_dev and not enabled:
            raise ValueError("AUTH_ENABLED cannot be false in production environments")

        # In development, auth can be disabled
        if is_dev and not enabled:
            return cls(secret="dev-secret", enabled=False)

        secret = os.getenv("JWT_SECRET")
        if enabled and not secret:
            if is_dev:
                secret = "dev-secret"
            else:
                raise ValueError(
                    "JWT_SECRET environment variable is required when AUTH_ENABLED=true"
                )
        if not is_dev and secret == "dev-secret":
            raise ValueError("JWT_SECRET must be set to a non-default value in production")

        return cls(
            secret=secret or "dev-secret",
            algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            enabled=enabled,
            token_expiry_seconds=int(os.getenv("TOKEN_EXPIRY_SECONDS", "3600")),
        )


# Global config instance
_auth_config: Optional[AuthConfig] = None


def get_auth_config() -> AuthConfig:
    """Get or create the auth configuration."""
    global _auth_config
    if _auth_config is None:
        _auth_config = AuthConfig.from_env()
    return _auth_config


# Security scheme
security = HTTPBearer(auto_error=False)


class TokenPayload:
    """Validated token payload."""

    def __init__(self, payload: Dict[str, Any]):
        self.sub = payload.get("sub")  # Subject (user ID)
        self.exp = payload.get("exp")  # Expiration
        self.iat = payload.get("iat")  # Issued at
        self.roles = payload.get("roles", [])
        self.scopes = payload.get("scopes", [])
        self.raw = payload

    @property
    def user_id(self) -> Optional[str]:
        return self.sub

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes


def _audit_log_auth_event(
    event_type: str,
    user_id: Optional[str] = None,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log authentication events for audit purposes.
    
    Args:
        event_type: Type of event (login, logout, token_refresh, token_validation, etc.)
        user_id: User ID if available
        success: Whether the operation succeeded
        details: Additional event details
    """
    logger.info(
        "auth_audit_event",
        extra={
            "event_type": event_type,
            "user_id": user_id,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(details or {}),
        },
    )


def decode_token(token: str, config: AuthConfig) -> TokenPayload:
    """Decode and validate a JWT token."""
    if not JWT_AVAILABLE:
        _audit_log_auth_event("token_validation", success=False, details={"error": "jwt_not_available"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT library not installed. Run: pip install PyJWT",
        )
    assert jwt is not None

    try:
        payload = jwt.decode(token, config.secret, algorithms=[config.algorithm])
        user_id = payload.get("sub")

        # Check expiration
        if "exp" in payload and payload["exp"] < time.time():
            _audit_log_auth_event("token_validation", user_id=user_id, success=False, details={"error": "token_expired"})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token_payload = TokenPayload(payload)
        _audit_log_auth_event("token_validation", user_id=user_id, success=True)
        return token_payload

    except jwt.ExpiredSignatureError:
        _audit_log_auth_event("token_validation", success=False, details={"error": "expired_signature"})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        _audit_log_auth_event("token_validation", success=False, details={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[TokenPayload]:
    """
    Verify the JWT token from the Authorization header.

    Returns None if auth is disabled, otherwise returns the validated payload.
    """
    config = get_auth_config()

    # Auth disabled - allow all requests
    if not config.enabled:
        return None

    # Auth enabled but no credentials provided
    if credentials is None:
        _audit_log_auth_event("token_verification", success=False, details={"error": "no_credentials"})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_payload = decode_token(credentials.credentials, config)
    if token_payload:
        _audit_log_auth_event("token_verification", user_id=token_payload.user_id, success=True)
    return token_payload


async def get_current_user(token: Optional[TokenPayload] = Depends(verify_token)) -> Optional[str]:
    """
    Get the current user ID from the token.
    
    In production, this will always return a user ID or raise an exception.
    In development, it may return None if auth is disabled.
    """
    config = get_auth_config()
    
    # In production, authentication is mandatory
    if config.enabled and token is None:
        _audit_log_auth_event("get_current_user", success=False, details={"error": "no_token"})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if token is None:
        return None
    
    user_id = token.user_id
    _audit_log_auth_event("get_current_user", user_id=user_id, success=True)
    return user_id


async def require_authenticated_user(token: Optional[TokenPayload] = Depends(verify_token)) -> str:
    """
    Require an authenticated user. Raises 401 if not authenticated.
    
    Use this for endpoints that MUST have authentication in all environments.
    In development, if authentication is disabled, this will allow access with a default user.
    In production, if authentication is disabled, this raises 401 to indicate authentication is required.
    """
    config = get_auth_config()
    
    if not config.enabled:
        # In development, allow access with default user when auth is disabled
        # In production, raise 401 to indicate authentication is required (not a server error)
        import os
        import logging
        logger = logging.getLogger(__name__)
        # Use ELEANOR_ENVIRONMENT or ELEANOR_ENV to match the rest of the application
        env = os.getenv("ELEANOR_ENVIRONMENT") or os.getenv("ELEANOR_ENV") or "development"
        if env == "development":
            logger.warning(
                "auth_disabled_but_required",
                extra={"endpoint": "require_authenticated_user"},
            )
            return "dev-user"  # Allow access in development
        else:
            # In production, raise 401 (not 500) to indicate authentication is required
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required for this endpoint",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = token.user_id
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_id


def require_role(role: str):
    """
    Decorator to require a specific role.
    
    In production, authentication is mandatory and role is enforced.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, token: Optional[TokenPayload] = Depends(verify_token), **kwargs):
            config = get_auth_config()
            
            # In production, authentication is mandatory
            if config.enabled:
                if token is None:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                if not token.has_role(role):
                    _audit_log_auth_event(
                        "role_check",
                        user_id=token.user_id,
                        success=False,
                        details={"required_role": role, "user_roles": token.roles}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN, 
                        detail=f"Role '{role}' required"
                    )
            # When auth is disabled (development mode), allow access regardless of role
            # This makes authentication optional in development as intended
            if config.enabled and token:
                _audit_log_auth_event(
                    "role_check",
                    user_id=token.user_id,
                    success=True,
                    details={"required_role": role, "user_roles": token.roles}
                )
            # When auth is disabled (development mode), allow access regardless of role
            # This makes authentication optional in development as intended
            
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_scope(scope: str):
    """Decorator to require a specific scope."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, token: Optional[TokenPayload] = Depends(verify_token), **kwargs):
            config = get_auth_config()
            # When auth is enabled, require token and scope
            if config.enabled:
                if token is None:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                if not token.has_scope(scope):
                    _audit_log_auth_event(
                        "scope_check",
                        user_id=token.user_id,
                        success=False,
                        details={"required_scope": scope, "user_scopes": token.scopes}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Scope '{scope}' required"
                    )
            # When auth is disabled (development mode), allow access regardless of scope
            if config.enabled and token:
                _audit_log_auth_event(
                    "scope_check",
                    user_id=token.user_id,
                    success=True,
                    details={"required_scope": scope, "user_scopes": token.scopes}
                )
            # When auth is disabled (development mode), allow access regardless of scope
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def create_token(
    user_id: str,
    roles: Optional[List[str]] = None,
    scopes: Optional[List[str]] = None,
    config: Optional[AuthConfig] = None,
) -> str:
    """
    Create a new JWT token.

    This is a utility function for testing and token generation.
    In production, tokens should be issued by an identity provider.
    """
    if not JWT_AVAILABLE:
        raise RuntimeError("JWT library not installed. Run: pip install PyJWT")
    assert jwt is not None

    config = config or get_auth_config()

    payload = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + config.token_expiry_seconds,
        "roles": roles or [],
        "scopes": scopes or [],
    }

    token = cast(str, jwt.encode(payload, config.secret, algorithm=config.algorithm))
    _audit_log_auth_event("token_created", user_id=user_id, success=True, details={"roles": roles, "scopes": scopes})
    return token


def refresh_token(
    token: str,
    config: Optional[AuthConfig] = None,
) -> str:
    """
    Refresh a JWT token by creating a new one with extended expiration.
    
    Args:
        token: Existing JWT token to refresh
        config: Optional auth config (uses default if not provided)
    
    Returns:
        New JWT token with extended expiration
    
    Raises:
        HTTPException: If token is invalid or expired
    """
    if not JWT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT library not installed. Run: pip install PyJWT",
        )
    assert jwt is not None

    config = config or get_auth_config()
    
    try:
        # Decode without expiration check to allow refresh of soon-to-expire tokens
        payload = jwt.decode(token, config.secret, algorithms=[config.algorithm], options={"verify_exp": False})
        user_id = payload.get("sub")
        
        # Check if token is too old to refresh (e.g., expired more than 1 hour ago)
        exp = payload.get("exp", 0)
        if exp < time.time() - 3600:  # 1 hour grace period
            _audit_log_auth_event("token_refresh", user_id=user_id, success=False, details={"error": "token_too_old"})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token is too old to refresh",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create new token with extended expiration
        new_payload = {
            "sub": user_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + config.token_expiry_seconds,
            "roles": payload.get("roles", []),
            "scopes": payload.get("scopes", []),
        }
        
        new_token = cast(str, jwt.encode(new_payload, config.secret, algorithm=config.algorithm))
        _audit_log_auth_event("token_refresh", user_id=user_id, success=True)
        return new_token
        
    except jwt.InvalidTokenError as e:
        _audit_log_auth_event("token_refresh", success=False, details={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_session(
    user_id: str,
    session_data: Optional[Dict[str, Any]] = None,
    ttl_seconds: int = 3600,
) -> str:
    """
    Create a session for long-running operations.
    
    Args:
        user_id: User ID for the session
        session_data: Optional session data to store
        ttl_seconds: Session time-to-live in seconds
    
    Returns:
        Session ID
    """
    import uuid
    session_id = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    
    _sessions[session_id] = {
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": expires_at.isoformat(),
        "data": session_data or {},
    }
    
    _audit_log_auth_event("session_created", user_id=user_id, success=True, details={"session_id": session_id})
    _cleanup_expired_sessions()
    
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get session data by session ID.
    
    Args:
        session_id: Session ID
    
    Returns:
        Session data or None if not found/expired
    """
    _cleanup_expired_sessions()
    
    if session_id not in _sessions:
        return None
    
    session = _sessions[session_id]
    expires_at = datetime.fromisoformat(session["expires_at"])
    
    if datetime.now(timezone.utc) > expires_at:
        del _sessions[session_id]
        return None
    
    return session


def delete_session(session_id: str) -> bool:
    """
    Delete a session.
    
    Args:
        session_id: Session ID to delete
    
    Returns:
        True if session was deleted, False if not found
    """
    if session_id in _sessions:
        user_id = _sessions[session_id].get("user_id")
        del _sessions[session_id]
        _audit_log_auth_event("session_deleted", user_id=user_id, success=True, details={"session_id": session_id})
        return True
    return False


def _cleanup_expired_sessions() -> None:
    """Clean up expired sessions (called periodically)."""
    global _last_cleanup
    current_time = time.time()
    
    # Only cleanup every N seconds to avoid overhead
    if current_time - _last_cleanup < _session_cleanup_interval:
        return
    
    _last_cleanup = current_time
    now = datetime.now(timezone.utc)
    expired = []
    
    for session_id, session in _sessions.items():
        expires_at = datetime.fromisoformat(session["expires_at"])
        if now > expires_at:
            expired.append(session_id)
    
    for session_id in expired:
        del _sessions[session_id]
    
    if expired:
        logger.debug("expired_sessions_cleaned", extra={"count": len(expired)})
