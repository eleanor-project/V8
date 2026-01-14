"""Comprehensive integration tests for full engine pipeline."""
import pytest
import asyncio
from engine import EleanorEngineV8
from engine.config import EngineConfig


@pytest.mark.integration
class TestFullPipelineIntegration:
    """Integration tests for complete engine pipeline."""

    @pytest.mark.asyncio
    async def test_complete_pipeline_detail_level_1(self):
        """Test complete pipeline with detail level 1 (output only)."""
        engine = EleanorEngineV8()
        
        async with engine:
            result = await engine.run(
                text="Evaluate this model output",
                context={"domain": "general"},
                detail_level=1
            )
        
        assert result is not None
        assert result.model_info is not None
        # Detail level 1 returns minimal info
        assert result.critic_findings is None or len(result.critic_findings) == 0

    @pytest.mark.asyncio
    async def test_complete_pipeline_detail_level_2(self):
        """Test complete pipeline with detail level 2 (critics + precedent)."""
        engine = EleanorEngineV8(
            config=EngineConfig(
                enable_precedent_analysis=True,
                enable_uncertainty=True
            )
        )
        
        async with engine:
            result = await engine.run(
                text="Evaluate this constitutional AI decision",
                context={"domain": "healthcare"},
                detail_level=2
            )
        
        assert result is not None
        assert result.model_info is not None
        assert result.critic_findings is not None
        assert len(result.critic_findings) > 0
        assert result.aggregated is not None

    @pytest.mark.asyncio
    async def test_complete_pipeline_detail_level_3(self):
        """Test complete pipeline with detail level 3 (full forensics)."""
        engine = EleanorEngineV8(
            config=EngineConfig(
                enable_precedent_analysis=True,
                enable_uncertainty=True,
                enable_governance=True
            )
        )
        
        async with engine:
            result = await engine.run(
                text="Evaluate high-stakes AI governance decision",
                context={
                    "domain": "healthcare",
                    "stakes": "high",
                    "patient_impact": True
                },
                detail_level=3
            )
        
        # Verify all pipeline stages completed
        assert result is not None
        assert result.model_info is not None
        assert result.critic_findings is not None
        assert len(result.critic_findings) >= 5  # Should have multiple critics
        assert result.aggregated is not None
        assert result.uncertainty is not None
        assert result.precedent_alignment is not None
        assert result.forensic is not None
        assert result.governance_review is not None

    @pytest.mark.asyncio
    async def test_streaming_mode_complete_pipeline(self):
        """Test streaming mode with complete pipeline."""
        engine = EleanorEngineV8()
        events = []
        
        async with engine:
            async for event in engine.run_stream(
                text="Test input for streaming",
                context={"domain": "test"},
                detail_level=2
            ):
                events.append(event)
        
        assert len(events) > 0
        
        # Verify event types
        event_types = [e.event_type for e in events]
        assert "router_selected" in event_types
        assert "critic_complete" in event_types or "critics_complete" in event_types
        assert "aggregation_complete" in event_types

    @pytest.mark.asyncio
    async def test_streaming_with_backpressure(self):
        """Test streaming mode under backpressure (slow consumer)."""
        engine = EleanorEngineV8()
        events = []
        
        async with engine:
            async for event in engine.run_stream(
                text="Test input",
                context={},
                detail_level=2
            ):
                events.append(event)
                # Simulate slow consumer
                await asyncio.sleep(0.1)
        
        assert len(events) > 0
        # Should handle backpressure without errors

    @pytest.mark.asyncio
    async def test_precedent_analysis_integration(self):
        """Test precedent analysis integration in pipeline."""
        engine = EleanorEngineV8(
            config=EngineConfig(enable_precedent_analysis=True)
        )
        
        async with engine:
            result = await engine.run(
                text="Decision requiring precedent analysis",
                context={"domain": "legal"},
                detail_level=3
            )
        
        assert result.precedent_alignment is not None
        # Should have alignment score
        assert hasattr(result.precedent_alignment, 'alignment_score')

    @pytest.mark.asyncio
    async def test_uncertainty_quantification_integration(self):
        """Test uncertainty quantification integration."""
        engine = EleanorEngineV8(
            config=EngineConfig(enable_uncertainty=True)
        )
        
        async with engine:
            result = await engine.run(
                text="Decision with uncertainty",
                context={},
                detail_level=2
            )
        
        assert result.uncertainty is not None
        assert hasattr(result.uncertainty, 'overall_uncertainty')

    @pytest.mark.asyncio
    async def test_governance_review_gate_integration(self):
        """Test governance review gate integration."""
        engine = EleanorEngineV8(
            config=EngineConfig(enable_governance=True)
        )
        
        async with engine:
            result = await engine.run(
                text="High-stakes decision requiring review",
                context={"stakes": "critical"},
                detail_level=3
            )
        
        assert result.governance_review is not None

    @pytest.mark.asyncio
    async def test_evidence_recording_throughout_pipeline(self):
        """Test evidence recording at each pipeline stage."""
        engine = EleanorEngineV8()
        
        async with engine:
            result = await engine.run(
                text="Test evidence recording",
                context={},
                detail_level=3
            )
        
        # Check that evidence was recorded
        assert result.forensic is not None
        assert result.forensic.trace_id is not None

    @pytest.mark.asyncio
    async def test_context_propagation_through_pipeline(self):
        """Test context propagates correctly through pipeline."""
        engine = EleanorEngineV8()
        
        test_context = {
            "domain": "healthcare",
            "patient_id": "12345",
            "urgency": "high"
        }
        
        async with engine:
            result = await engine.run(
                text="Test context propagation",
                context=test_context,
                detail_level=2
            )
        
        # Context should be preserved in result
        assert result.context == test_context

    @pytest.mark.asyncio
    async def test_trace_id_propagation(self):
        """Test trace ID propagation through pipeline."""
        engine = EleanorEngineV8()
        custom_trace_id = "custom-trace-123"
        
        async with engine:
            result = await engine.run(
                text="Test trace propagation",
                context={},
                detail_level=2,
                trace_id=custom_trace_id
            )
        
        assert result.forensic.trace_id == custom_trace_id

    @pytest.mark.asyncio
    async def test_pipeline_with_all_critics(self):
        """Test pipeline executes all configured critics."""
        engine = EleanorEngineV8()
        
        async with engine:
            result = await engine.run(
                text="Test all critics",
                context={},
                detail_level=2
            )
        
        # Should have results from all critics
        expected_critics = ["truth", "fairness", "risk", "pragmatics", "autonomy", "rights"]
        
        assert result.critic_findings is not None
        for critic_name in expected_critics:
            assert critic_name in result.critic_findings
