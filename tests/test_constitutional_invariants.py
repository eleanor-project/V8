import pytest

@pytest.mark.asyncio
async def test_rights_critic_has_priority(engine):
    result = await engine.run("hello", detail_level=3)
    rights_findings = result.critic_findings.get("rights")
    if rights_findings and rights_findings.violations:
        assert "rights" in result.aggregated.get("critical_domains", [])
