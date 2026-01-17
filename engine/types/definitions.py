"""Deprecated legacy TypedDict definitions.

Use engine.schemas.legacy_types instead.
"""

import warnings

from engine.schemas.legacy_types import *  # noqa: F401,F403

warnings.warn(
    "engine.types.definitions is deprecated. Use engine.schemas.legacy_types instead.",
    DeprecationWarning,
    stacklevel=2,
)
