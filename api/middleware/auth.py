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
from typing import Optional, Dict, Any, List, cast
from types import ModuleType
from dataclasses import dataclass
from functools import wraps

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


def decode_token(token: str, config: AuthConfig) -> TokenPayload:
    """Decode and validate a JWT token."""
    if not JWT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT library not installed. Run: pip install PyJWT",
        )
    assert jwt is not None

    try:
        payload = jwt.decode(token, config.secret, algorithms=[config.algorithm])

        # Check expiration
        if "exp" in payload and payload["exp"] < time.time():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return TokenPayload(payload)

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decode_token(credentials.credentials, config)


async def get_current_user(token: Optional[TokenPayload] = Depends(verify_token)) -> Optional[str]:
    """
    Get the current user ID from the token.
    
    In production, this will always return a user ID or raise an exception.
    In development, it may return None if auth is disabled.
    """
    config = get_auth_config()
    
    # In production, authentication is mandatory
    if config.enabled and token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if token is None:
        return None
    return token.user_id


async def require_authenticated_user(token: Optional[TokenPayload] = Depends(verify_token)) -> str:
    """
    Require an authenticated user. Raises 401 if not authenticated.
    
    Use this for endpoints that MUST have authentication in all environments.
    """
    config = get_auth_config()
    
    if not config.enabled:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication is disabled but required for this endpoint",
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
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN, 
                        detail=f"Role '{role}' required"
                    )
            elif token and not token.has_role(role):
                # In development with auth enabled but wrong role
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, 
                    detail=f"Role '{role}' required"
                )
            
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_scope(scope: str):
    """Decorator to require a specific scope."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, token: TokenPayload = Depends(verify_token), **kwargs):
            config = get_auth_config()
            if config.enabled and token and not token.has_scope(scope):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail=f"Scope '{scope}' required"
                )
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

    return cast(str, jwt.encode(payload, config.secret, algorithm=config.algorithm))
