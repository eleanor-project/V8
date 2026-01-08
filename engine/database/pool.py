"""
ELEANOR V8 — Database Connection Pool Manager
----------------------------------------------

Async database connection pool with proper lifecycle management.
"""

import asyncio
import logging
from typing import Optional, Any, TYPE_CHECKING
from contextlib import asynccontextmanager

if TYPE_CHECKING:
    from asyncpg import Pool, Connection

logger = logging.getLogger(__name__)

try:
    import asyncpg
    from asyncpg import Pool, Connection
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    Pool = None
    Connection = None


class DatabasePool:
    """
    Async database connection pool manager.
    
    Provides connection pooling with configurable limits,
    timeout handling, and graceful shutdown.
    """

    def __init__(self, config: Any):
        """
        Initialize database pool.
        
        Args:
            config: DatabaseConfig instance with pool settings
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is required for database pooling. "
                "Install with: pip install asyncpg"
            )
        
        self.config = config
        self._pool: Optional[Pool] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection pool."""
        if self._initialized:
            return
        
        try:
            # Parse connection URL
            url = self.config.url
            # Extract connection parameters from URL
            # Format: postgresql+asyncpg://user:password@host:port/dbname
            if "+asyncpg" in url:
                url = url.replace("+asyncpg", "")
            
            self._pool = await asyncpg.create_pool(
                url,
                min_size=self.config.pool_size,
                max_size=self.config.pool_size + self.config.max_overflow,
                timeout=self.config.pool_timeout,
                max_inactive_connection_lifetime=self.config.pool_recycle,
                command_timeout=self.config.pool_timeout,
            )
            
            self._initialized = True
            logger.info(
                "database_pool_initialized",
                extra={
                    "min_size": self.config.pool_size,
                    "max_size": self.config.pool_size + self.config.max_overflow,
                    "timeout": self.config.pool_timeout,
                },
            )
        except Exception as exc:
            logger.error(
                "database_pool_init_failed",
                extra={"error": str(exc)},
                exc_info=True,
            )
            raise

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire connection from pool.
        
        Usage:
            async with pool.acquire() as conn:
                result = await conn.fetch("SELECT ...")
        """
        if not self._pool:
            raise RuntimeError("Pool not initialized. Call initialize() first.")
        
        conn = await self._pool.acquire()
        try:
            yield conn
        finally:
            await self._pool.release(conn)

    async def execute(self, query: str, *args) -> Any:
        """Execute query using pool."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list:
        """Fetch rows using pool."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[dict]:
        """Fetch single row using pool."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch single value using pool."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    def get_stats(self) -> dict:
        """Get pool statistics."""
        if not self._pool:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "size": self._pool.get_size(),
            "idle_size": self._pool.get_idle_size(),
            "min_size": self.config.pool_size,
            "max_size": self.config.pool_size + self.config.max_overflow,
        }

    async def close(self) -> None:
        """Close all connections in pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("database_pool_closed")

    async def __aenter__(self) -> "DatabasePool":
        """
        Async context manager entry.
        
        Initializes the pool and returns self.
        Raises exception if initialization fails.
        """
        try:
            await self.initialize()
            return self
        except Exception as exc:
            logger.error(
                "database_pool_init_failed_in_context",
                extra={"error": str(exc), "error_type": type(exc).__name__},
                exc_info=True,
            )
            # Ensure cleanup on failure
            try:
                await self.close()
            except Exception:
                pass
            raise

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """
        Async context manager exit.
        
        Properly handles exceptions and ensures pool cleanup.
        
        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred
        
        Returns:
            False to propagate exceptions, True to suppress them
        """
        try:
            await self.close()
        except Exception as close_exc:
            logger.error(
                "database_pool_close_failed",
                extra={
                    "error": str(close_exc),
                    "error_type": type(close_exc).__name__,
                    "original_exception": str(exc_val) if exc_val else None,
                },
                exc_info=True,
            )
            # Don't suppress original exception if close fails
            if exc_val:
                raise exc_val from close_exc
        
        # Return False to propagate any exceptions that occurred
        return False


__all__ = ["DatabasePool"]
