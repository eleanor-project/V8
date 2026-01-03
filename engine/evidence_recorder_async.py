"""
Async Evidence Recorder for ELEANOR V8

Provides:
- Async file I/O with buffering
- Periodic automatic flushing
- Graceful shutdown with data integrity
- Resource management via context protocol
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiofiles  # type: ignore[import-untyped]

    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

logger = logging.getLogger(__name__)


class AsyncEvidenceRecorder:
    """Evidence recorder with async resource management"""

    def __init__(
        self,
        jsonl_path: str,
        buffer_size: int = 1000,
        flush_interval: float = 5.0,
    ):
        """
        Args:
            jsonl_path: Path to JSONL evidence file
            buffer_size: Max records before auto-flush
            flush_interval: Seconds between periodic flushes
        """
        self.jsonl_path = Path(jsonl_path)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self.buffer: List[Dict[str, Any]] = []
        self._file_handle = None
        self._flush_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._initialized = False
        self._lock = asyncio.Lock()

        if not AIOFILES_AVAILABLE:
            logger.warning(
                "aiofiles not installed, falling back to sync I/O. "
                "Install with: pip install aiofiles"
            )

    async def __aenter__(self):
        """Context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()
        return False

    async def initialize(self):
        """Initialize recorder and start periodic flush task"""
        if self._initialized:
            return

        logger.info(
            "evidence_recorder_initializing",
            extra={
                "path": str(self.jsonl_path),
                "buffer_size": self.buffer_size,
                "flush_interval": self.flush_interval,
            },
        )

        # Ensure directory exists
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file handle
        if AIOFILES_AVAILABLE:
            self._file_handle = await aiofiles.open(
                self.jsonl_path,
                mode="a",
                encoding="utf-8",
            )
        else:
            # Fallback to sync file handle (will block event loop)
            self._file_handle = open(
                self.jsonl_path,
                mode="a",
                encoding="utf-8",
            )

        # Start periodic flush task
        self._flush_task = asyncio.create_task(self._periodic_flush())
        self._initialized = True

        logger.info("evidence_recorder_initialized")

    async def _periodic_flush(self):
        """Background task to periodically flush buffer to disk"""
        logger.debug("periodic_flush_task_started")

        while not self._shutdown:
            try:
                await asyncio.sleep(self.flush_interval)

                if self.buffer and not self._shutdown:
                    await self.flush()

            except asyncio.CancelledError:
                logger.debug("periodic_flush_task_cancelled")
                break
            except Exception as exc:
                logger.error(
                    "periodic_flush_failed",
                    extra={"error": str(exc)},
                    exc_info=True,
                )

    async def record(self, **kwargs):
        """
        Record evidence entry to buffer.
        Auto-flushes when buffer is full.

        Args:
            **kwargs: Evidence record fields
        """
        if self._shutdown:
            logger.warning("record_called_after_shutdown")
            return

        should_flush = False
        async with self._lock:
            self.buffer.append(kwargs)
            should_flush = len(self.buffer) >= self.buffer_size

        if should_flush:
            await self.flush()

    async def flush(self):
        """Flush buffer to disk"""
        if not self.buffer or not self._file_handle:
            return

        async with self._lock:
            buffer_size = len(self.buffer)

            try:
                if AIOFILES_AVAILABLE:
                    # Async file I/O
                    for record in self.buffer:
                        line = json.dumps(record, ensure_ascii=False) + "\n"
                        await self._file_handle.write(line)
                    await self._file_handle.flush()
                else:
                    # Fallback sync I/O (blocks event loop)
                    for record in self.buffer:
                        line = json.dumps(record, ensure_ascii=False) + "\n"
                        self._file_handle.write(line)
                    self._file_handle.flush()

                self.buffer.clear()

                logger.debug("evidence_buffer_flushed", extra={"records": buffer_size})

            except Exception as exc:
                logger.error(
                    "evidence_flush_failed",
                    extra={
                        "error": str(exc),
                        "buffer_size": buffer_size,
                    },
                    exc_info=True,
                )
                raise

    async def close(self):
        """Close recorder and cleanup resources"""
        if not self._initialized:
            return

        logger.info("evidence_recorder_closing")
        self._shutdown = True

        # Cancel periodic flush task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush to ensure no data loss
        try:
            await self.flush()
            logger.debug("final_flush_complete")
        except Exception as exc:
            logger.error(f"final_flush_failed: {exc}", exc_info=True)

        # Close file handle
        if self._file_handle:
            try:
                if AIOFILES_AVAILABLE:
                    await self._file_handle.close()
                else:
                    self._file_handle.close()
                logger.debug("file_handle_closed")
            except Exception as exc:
                logger.error(f"file_handle_close_failed: {exc}", exc_info=True)
            finally:
                self._file_handle = None

        self._initialized = False
        logger.info("evidence_recorder_closed")
