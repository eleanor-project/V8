from __future__ import annotations

from typing import Any, Dict

CRITIC_ALIASES = {
    "pragmatics": "operations",
}


def canonical_critic_name(name: str) -> str:
    key = name.lower()
    return CRITIC_ALIASES.get(key, key)


def canonicalize_critic_map(critics: Dict[str, Any]) -> Dict[str, Any]:
    canonical: Dict[str, Any] = {}
    for name, data in critics.items():
        key = canonical_critic_name(name)
        if key in canonical and name.lower() != key:
            continue
        canonical[key] = data
    return canonical
