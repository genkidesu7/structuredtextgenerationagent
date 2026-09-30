"""AgentCore Platform v1.0"""

# Stub data served through the sanctioned mock-mode gate (mock_mode_enabled()).
# Nothing here is reachable unless the caller has checked the gate first.

from __future__ import annotations

from typing import Any

STUB_TOKEN_UNITS: int = 42
_MAX_STUB_DEPTH = 5


def _stub_value(prop: dict[str, Any], name: str, depth: int) -> Any:
    """Build a value that satisfies a single JSON-schema property."""
    if "const" in prop:
        return prop["const"]
    if prop.get("enum"):
        return prop["enum"][0]
    field_type = prop.get("type", "string")
    if isinstance(field_type, list):
        field_type = next((t for t in field_type if t != "null"), "null")
    if field_type == "string":
        return f"mock_{name}"
    if field_type in ("integer", "number"):
        return prop.get("minimum", 0)
    if field_type == "boolean":
        return True
    if field_type == "array":
        return []
    if field_type == "object":
        return stub_structured_output(prop, depth + 1)
    return None


def stub_structured_output(schema: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """Return a minimal dict that conforms to ``schema`` (objects only).

    Every declared property receives a type-correct value so that optional
    properties never break validation; nested objects are filled recursively
    up to a bounded depth.
    """
    properties: dict[str, Any] = schema.get("properties") or {}
    required: list[str] = schema.get("required") or []
    if depth > _MAX_STUB_DEPTH:
        return {}
    names = list(dict.fromkeys([*required, *properties]))
    if not names:
        return {"mock_field": "mock_value", "generated": True}
    return {name: _stub_value(properties.get(name, {}), name, depth) for name in names}


def stub_generate_structured(schema: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Stub for LLMProviderService.generate_structured(): (output, token units)."""
    return stub_structured_output(schema), STUB_TOKEN_UNITS


def stub_schema_allowlisted(schema: dict[str, Any]) -> bool:
    """Stub for SchemaRegistryService.validate_allowlist(): allow every schema."""
    return True
