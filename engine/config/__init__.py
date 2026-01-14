"""
ELEANOR V8 — Configuration Management

Hierarchical configuration system with clear precedence:
1. Command-line arguments (highest)
2. Environment variables (ELEANOR_*)
3. Environment files (.env, .env.production)
4. YAML configuration (legacy support)
5. Default values (lowest)
"""

from .settings import EleanorSettings, get_settings  # noqa: F401
from .manager import ConfigManager  # noqa: F401
from engine.runtime.config import EngineConfig  # noqa: F401

__all__ = [
    "EleanorSettings",
    "get_settings",
    "ConfigManager",
    "EngineConfig",
]
