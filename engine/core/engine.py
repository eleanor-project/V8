"""
Compatibility wrapper for the canonical runtime engine in engine/engine.py.

This module avoids duplicate implementations while preserving legacy imports.
"""

from engine.engine import EleanorEngineV8, create_engine

__all__ = ["EleanorEngineV8", "create_engine"]
