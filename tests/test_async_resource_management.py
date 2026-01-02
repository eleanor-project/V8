"""
Tests for Async Resource Management (Issue #19)
"""

import asyncio
import pytest
import tempfile
import json
from pathlib import Path

from engine.evidence_recorder_async import AsyncEvidenceRecorder
from engine.resource_manager import ResourceManager, ShutdownHandler, TimeoutProtection


class TestAsyncEvidenceRecorder:
    """Test AsyncEvidenceRecorder functionality"""
    
    @pytest.mark.asyncio
    async def test_initialize_and_close(self):
        """Test recorder initialization and cleanup"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evidence.jsonl"
            
            recorder = AsyncEvidenceRecorder(str(path))
            await recorder.initialize()
            
            assert recorder._initialized
            assert recorder._file_handle is not None
            assert recorder._flush_task is not None
            
            await recorder.close()
            
            assert not recorder._initialized
            assert recorder._file_handle is None
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test context manager protocol"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evidence.jsonl"
            
            async with AsyncEvidenceRecorder(str(path)) as recorder:
                assert recorder._initialized
                await recorder.record(test="data")
            
            # Should be closed after exit
            assert not recorder._initialized
    
    @pytest.mark.asyncio
    async def test_record_and_flush(self):
        """Test recording and flushing evidence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evidence.jsonl"
            
            async with AsyncEvidenceRecorder(str(path), buffer_size=10) as recorder:
                # Record multiple entries
                for i in range(5):
                    await recorder.record(index=i, data=f"test_{i}")
                
                assert len(recorder.buffer) == 5
                
                # Manual flush
                await recorder.flush()
                assert len(recorder.buffer) == 0
            
            # Verify data written to file
            with open(path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 5
                
                # Verify first line
                first = json.loads(lines[0])
                assert first['index'] == 0
                assert first['data'] == 'test_0'
    
    @pytest.mark.asyncio
    async def test_auto_flush_on_buffer_full(self):
        """Test automatic flush when buffer is full"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evidence.jsonl"
            
            async with AsyncEvidenceRecorder(str(path), buffer_size=3) as recorder:
                # Record exactly buffer_size entries
                await recorder.record(index=0)
                await recorder.record(index=1)
                await recorder.record(index=2)  # Should trigger flush
                
                # Buffer should be empty after auto-flush
                assert len(recorder.buffer) == 0
            
            # Verify 3 lines written
            with open(path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3
    
    @pytest.mark.asyncio
    async def test_final_flush_on_close(self):
        """Test that data is flushed on close even if buffer not full"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "evidence.jsonl"
            
            async with AsyncEvidenceRecorder(str(path), buffer_size=100) as recorder:
                # Record fewer than buffer_size
                await recorder.record(test="data")
                assert len(recorder.buffer) == 1
            
            # After close, buffer should be flushed
            with open(path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1


class TestResourceManager:
    """Test ResourceManager functionality"""
    
    @pytest.mark.asyncio
    async def test_shutdown_signal(self):
        """Test shutdown signaling"""
        manager = ResourceManager()
        assert not manager.is_shutdown()
        
        manager.signal_shutdown()
        assert manager.is_shutdown()
    
    @pytest.mark.asyncio
    async def test_task_tracking_and_cleanup(self):
        """Test task tracking and cancellation"""
        manager = ResourceManager()
        
        async def long_task():
            await asyncio.sleep(10)
        
        # Create and track tasks
        task1 = asyncio.create_task(long_task())
        task2 = asyncio.create_task(long_task())
        
        manager.track_task(task1)
        manager.track_task(task2)
        
        assert len(manager._cleanup_tasks) == 2
        
        # Cleanup should cancel tasks
        await manager.cleanup_tasks(timeout=1.0)
        
        assert task1.cancelled() or task1.done()
        assert task2.cancelled() or task2.done()
        assert len(manager._cleanup_tasks) == 0


class TestTimeoutProtection:
    """Test TimeoutProtection utility"""
    
    @pytest.mark.asyncio
    async def test_operation_completes_within_timeout(self):
        """Test successful operation completion"""
        async def fast_operation():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await TimeoutProtection.with_timeout(
            fast_operation(),
            timeout=1.0,
            operation="fast_op"
        )
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_operation_timeout_raises(self):
        """Test timeout raises error when configured"""
        async def slow_operation():
            await asyncio.sleep(2.0)
            return "success"
        
        with pytest.raises(asyncio.TimeoutError):
            await TimeoutProtection.with_timeout(
                slow_operation(),
                timeout=0.1,
                operation="slow_op",
                raise_on_timeout=True
            )
    
    @pytest.mark.asyncio
    async def test_operation_timeout_returns_none(self):
        """Test timeout returns None when configured"""
        async def slow_operation():
            await asyncio.sleep(2.0)
            return "success"
        
        result = await TimeoutProtection.with_timeout(
            slow_operation(),
            timeout=0.1,
            operation="slow_op",
            raise_on_timeout=False
        )
        
        assert result is None


class TestShutdownHandler:
    """Test ShutdownHandler for signal management"""
    
    @pytest.mark.asyncio
    async def test_shutdown_callback_called(self):
        """Test shutdown callback is invoked"""
        called = asyncio.Event()
        
        async def on_shutdown():
            called.set()
        
        handler = ShutdownHandler(on_shutdown)
        
        # Manually trigger shutdown
        await handler._handle_shutdown(signal.SIGTERM)
        
        # Wait briefly for callback
        await asyncio.sleep(0.1)
        assert called.is_set()
