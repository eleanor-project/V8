"""
ELEANOR V8 — Per-User Rate Limiting
------------------------------------

Rate limiting per authenticated user for fine-grained control.
"""

import logging
from typing import Tuple, Dict, Optional
from fastapi import Request

from api.middleware.rate_limit import TokenBucketRateLimiter, RateLimitConfig

logger = logging.getLogger(__name__)


class UserRateLimiter:
    """
    Per-user rate limiting.
    
    Tracks rate limits per authenticated user, allowing different
    limits for different users or user roles.
    """
    
    def __init__(self, base_config: RateLimitConfig):
        self.base_config = base_config
        self._limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._user_configs: Dict[str, RateLimitConfig] = {}
    
    def set_user_limit(
        self,
        user_id: str,
        requests_per_window: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ) -> None:
        """
        Set custom rate limit for specific user.
        
        Args:
            user_id: User identifier
            requests_per_window: Custom requests per window (None = use base)
            window_seconds: Custom window in seconds (None = use base)
        """
        config = RateLimitConfig(
            enabled=self.base_config.enabled,
            requests_per_window=requests_per_window or self.base_config.requests_per_window,
            window_seconds=window_seconds or self.base_config.window_seconds,
            max_clients=self.base_config.max_clients,
            client_ttl_seconds=self.base_config.client_ttl_seconds,
            redis_url=self.base_config.redis_url,
            key_prefix=f"{self.base_config.key_prefix}user:",
        )
        self._user_configs[user_id] = config
    
    def _get_limiter(self, user_id: str) -> TokenBucketRateLimiter:
        """Get or create rate limiter for user."""
        if user_id not in self._limiters:
            config = self._user_configs.get(user_id, self.base_config)
            # Use user-specific prefix
            if user_id in self._user_configs:
                config.key_prefix = f"{self.base_config.key_prefix}user:"
            self._limiters[user_id] = TokenBucketRateLimiter(config)
        return self._limiters[user_id]
    
    async def check_user_limit(
        self,
        user_id: str,
        request: Request,
        endpoint: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Check if user has exceeded rate limit.
        
        Args:
            user_id: User identifier
            request: FastAPI request
            endpoint: Optional endpoint name for per-endpoint limits
        
        Returns:
            Tuple of (allowed: bool, headers: dict)
        """
        # Create endpoint-specific key if provided
        if endpoint:
            # Modify request to include endpoint in client ID
            original_client = request.client
            if hasattr(request, '_client_id'):
                request._client_id = f"{user_id}:{endpoint}"
        
        limiter = self._get_limiter(user_id)
        return await limiter.check(request)
    
    def reset_user(self, user_id: str) -> None:
        """Reset rate limit for user."""
        if user_id in self._limiters:
            self._limiters[user_id].reset(user_id)


__all__ = ["UserRateLimiter"]
