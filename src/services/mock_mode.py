"""AgentCore Platform v1.0"""

# Single source of truth for mock-mode resolution. Every service and node that
# gates real-vs-mock behaviour imports mock_mode_enabled() from here — never
# re-derive the flag locally.
#
# Two aliases are accepted:
#   USE_MOCK        canonical flag (what tests and local runs set)
#   STG_MOCK_MODE   what the scaffold deploy-stg CI job exports to uvicorn

from __future__ import annotations

import os

MOCK_MODE_ENV_VARS: tuple[str, ...] = ("USE_MOCK", "STG_MOCK_MODE")
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes"})


def mock_mode_enabled() -> bool:
    """Return True when any accepted mock-mode env var is set to a truthy value.

    Read per call (not cached at import) so tests and the process environment
    can change the flag at runtime.
    """
    return any(os.environ.get(name, "").strip().lower() in _TRUTHY for name in MOCK_MODE_ENV_VARS)
