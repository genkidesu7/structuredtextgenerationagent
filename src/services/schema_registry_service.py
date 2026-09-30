"""AgentCore Platform v1.0"""

# Service layer: schema allowlist enforcement and versioning.
# Validates caller-supplied target_schema is registered and not revoked.
# PB-4: Level-0 SDK (agenticstar) imports are prohibited in this file.

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from src.services.mock_data import stub_schema_allowlisted
from src.services.mock_mode import mock_mode_enabled

_REGISTRY_MOCK_ENV = "USE_MOCK_SCHEMA_REGISTRY"
_TRUTHY = frozenset({"1", "true", "yes"})


def schema_registry_mock_enabled() -> bool:
    """True under global mock mode, or when the registry alone is pinned to its stub.

    The schema registry has no backing service in this template, so a real-LLM
    run that still needs every schema accepted sets USE_MOCK_SCHEMA_REGISTRY
    without turning the LLM back to mock.
    """
    return mock_mode_enabled() or os.environ.get(_REGISTRY_MOCK_ENV, "").strip().lower() in _TRUTHY


class SchemaRegistryService:
    """Schema allowlist enforcement and stable schema identity.

    Validates that the caller-supplied target_schema is registered in the
    allowlist and computes a stable schema_id (SHA-256 of canonical JSON).

    Mock mode: when schema_registry_mock_enabled() is true (USE_MOCK,
    STG_MOCK_MODE or USE_MOCK_SCHEMA_REGISTRY), validate_allowlist() accepts
    every schema. Otherwise only schemas added via register_schema() pass.
    """

    def __init__(self) -> None:
        # Allowlist: schema_ids (SHA-256 hashes) registered and not revoked.
        # Starts empty — outside mock mode a schema must be registered first.
        self._allowlist: set[str] = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate_allowlist(self, schema: dict[str, Any]) -> bool:
        """Return True if schema is registered in the allowlist.

        In mock mode, always returns True.

        Args:
            schema: The JSON schema dict to validate.

        Returns:
            True if schema is allowlisted, False otherwise.
        """
        # Check flag per-call to support dynamic env var changes in tests
        if schema_registry_mock_enabled():
            return stub_schema_allowlisted(schema)
        schema_id = self.compute_schema_id(schema)
        return schema_id in self._allowlist

    def compute_schema_id(self, schema: dict[str, Any]) -> str:
        """Compute a stable schema_id as SHA-256 of canonical JSON.

        Canonical form uses sorted keys so that schema_id is deterministic
        regardless of the order fields appear in the input dict.

        Args:
            schema: The JSON schema dict to identify.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def register_schema(self, schema: dict[str, Any]) -> str:
        """Register a schema in the allowlist and return its schema_id.

        Adds the schema's id to the in-process allowlist.

        Args:
            schema: The JSON schema dict to register.

        Returns:
            The schema_id (SHA-256 hash) of the registered schema.
        """
        schema_id = self.compute_schema_id(schema)
        self._allowlist.add(schema_id)
        return schema_id
