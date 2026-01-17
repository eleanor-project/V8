"""Compatibility layer for legacy engine.types imports.

Use engine.schemas.constitutional_types for canonical constitutional contracts.
"""

import warnings

from engine.schemas.constitutional_types import *  # noqa: F401,F403

warnings.warn(
    "engine.types is deprecated. Use engine.schemas.constitutional_types instead.",
    DeprecationWarning,
    stacklevel=2,
)
