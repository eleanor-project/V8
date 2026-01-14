"""
ELEANOR V8 — Constitutional AI Governance Engine
"""

__version__ = "8.0.0"

# Public API
from engine.engine import EleanorEngineV8, create_engine  # noqa: F401
from engine.runtime.config import EngineConfig  # noqa: F401

__all__ = [
    "EleanorEngineV8",
    "create_engine",
    "EngineConfig",
    "__version__",
]
