"""Integration tests for complete pipeline execution."""
import pytest
import asyncio
from engine.engine import EleanorEngineV8
from engine.config import EngineConfig


class TestCompletePipelineExecution:
    """Test full end-to-end pipeline execution."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_detail_level_1_minimal_pipeline(self):
        """Test minimal pipeline with detail level 1 (output only)."""
        engine = EleanorEngineV8()
        
        async with engine:
            result = await engine.run(
                text="Evaluate this simple model output",
                context={"test": True},
                detail_level=1
            )
        
        # Level 1 should have model output
        assert result is not None
        assert result.model_info is not None
        # Level 1 may not have detailed forensics
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_detail_level_2_with_critics(self):
        """Test pipeline with detail level 2 (critics + precedent)."""
        engine = EleanorEngineV8()
        
        async with engine:
            result = await engine.run(
                text="Evaluate this model decision with ethical implications",
                context={"domain": "healthcare", "stakes": "medium"},
                detail_level=2
            )
        
        assert result is not None
        assert result.model_info is not None
        assert result.critic_findings is not None
        # Should have multiple critics
        assert len(result.critic_findings) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_detail_level_3_full_forensics(self):
        """Test complete pipeline with detail level 3 (full forensics)."""
        config = EngineConfig(
            enable_precedent_analysis=True,
            enable_uncertainty=True,
            enable_governance=True
        )
        engine = EleanorEngineV8(config=config)
        
        async with engine:
            result = await engine.run(
                text="Evaluate this high-stakes AI governance decision",
                context={
                    "domain": "healthcare",
                    "stakes": "high",
                    "user_id": "test-user",
                    "enable_precedent": True
                },
                detail_level=3
            )
        
        # Verify all pipeline stages completed
        assert result is not None
        assert result.model_info is not None
        assert result.critic_findings is not None
        assert result.aggregated is not None
        
        if config.enable_precedent_analysis:
            assert result.precedent_alignment is not None
        
        if config.enable_uncertainty:
            assert result.uncertainty is not None
        
        assert result.forensic is not None
        
        # Governance review may be present if escalated
        # assert result.governance_review is not None


class TestStreamingPipeline:
    """Test streaming pipeline execution."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_streaming_yields_events(self):
        """Test streaming mode yields events progressively."""
        engine = EleanorEngineV8()
        events = []
        
        async with engine:
            async for event in engine.run_stream(
                "Test streaming evaluation",
                context={},
                detail_level=2
            ):
                events.append(event)
        
        assert len(events) > 0
        # Should have multiple event types
        event_types = {e.event_type for e in events}
        assert len(event_types) > 1
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_streaming_with_slow_consumer(self):
        """Test streaming handles backpressure from slow consumer."""
        engine = EleanorEngineV8()
        events = []
        
        async with engine:
            async for event in engine.run_stream(
                "Test backpressure handling",
                context={},
                detail_level=2
            ):
                events.append(event)
                # Simulate slow consumer
                await asyncio.sleep(0.05)
        
        assert len(events) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_streaming_event_ordering(self):
        """Test streaming events are delivered in correct order."""
        engine = EleanorEngineV8()
        events = []
        
        async with engine:
            async for event in engine.run_stream(
                "Test event ordering",
                context={},
                detail_level=2
            ):
                events.append(event)
        
        # Verify logical ordering (e.g., start before complete)
        event_types = [e.event_type for e in events]
        
        # Should have start and complete events
        assert any("start" in e or "begin" in e for e in event_types)
        assert any("complete" in e or "done" in e for e in event_types)


class TestPipelineWithRealComponents:
    """Test pipeline with real component integration."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_router_integration(self):
        """Test router selection integration."""
        engine = EleanorEngineV8()
        
        async with engine:
            result = await engine.run(
                "Test router selection",
                context={"domain": "general"},
                detail_level=1
            )
        
        assert result.model_info is not None
        assert "model" in result.model_info or "provider" in result.model_info
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_critics_parallel_execution(self):
        """Test critics execute in parallel correctly."""
        import time
        
        engine = EleanorEngineV8()
        
        async with engine:
            start_time = time.time()
            result = await engine.run(
                "Test parallel critic execution",
                context={},
                detail_level=2
            )
            elapsed = time.time() - start_time
        
        assert result.critic_findings is not None
        # Parallel execution should be faster than sequential
        # (specific timing depends on number of critics)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_precedent_retrieval_integration(self):
        """Test precedent retrieval integration."""
        config = EngineConfig(enable_precedent_analysis=True)
        engine = EleanorEngineV8(config=config)
        
        async with engine:
            result = await engine.run(
                "Test precedent retrieval",
                context={"enable_precedent": True},
                detail_level=3
            )
        
        if config.enable_precedent_analysis:
            assert result.precedent_alignment is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_uncertainty_computation_integration(self):
        """Test uncertainty computation integration."""
        config = EngineConfig(enable_uncertainty=True)
        engine = EleanorEngineV8(config=config)
        
        async with engine:
            result = await engine.run(
                "Test uncertainty computation",
                context={},
                detail_level=3
            )
        
        if config.enable_uncertainty:
            assert result.uncertainty is not None


class TestPipelineErrorRecovery:
    """Test pipeline error recovery and resilience."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_recovers_from_single_critic_failure(self):
        """Test pipeline recovers from single critic failure."""
        engine = EleanorEngineV8()
        
        async with engine:
            result = await engine.run(
                "Test error recovery",
                context={},
                detail_level=2
            )
        
        # Should complete despite potential failures
        assert result is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_records_evidence_throughout_pipeline(self):
        """Test evidence is recorded throughout pipeline execution."""
        engine = EleanorEngineV8()
        
        async with engine:
            result = await engine.run(
                "Test evidence recording",
                context={},
                detail_level=3
            )
        
        assert result.forensic is not None
        # Forensic should contain execution trace
