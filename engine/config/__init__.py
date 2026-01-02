"""
ELEANOR V8 — Configuration Management

Hierarchical configuration system with clear precedence:
1. Command-line arguments (highest)
2. Environment variables (ELEANOR_*)
3. Environment files (.env, .env.production)
4. YAML configuration (legacy support)
5. Default values (lowest)
"""

from .settings import EleanorSettings, get_settings
from .manager import ConfigManager

__all__ = [
    "EleanorSettings",
    "get_settings",
    "ConfigManager",
]
