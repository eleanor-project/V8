from engine.utils.critic_names import canonicalize_critic_map


def test_canonicalize_critic_map_skips_duplicate_alias():
    critics = {"pragmatics": "a", "operations": "b"}
    result = canonicalize_critic_map(critics)
    assert result["operations"] == "b"
    assert "pragmatics" not in result
