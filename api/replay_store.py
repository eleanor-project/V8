"""
Compatibility shim: api.replay_store now aliases engine.replay_store.

This keeps import paths stable while the implementation lives in the engine
package so installed wheels contain the module.
"""

import sys
import engine.replay_store as _replay_store

sys.modules[__name__] = _replay_store
