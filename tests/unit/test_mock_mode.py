"""mock_mode_enabled() — resolved under every accepted alias, individually."""

import pytest

from src.services.mock_mode import MOCK_MODE_ENV_VARS, mock_mode_enabled
from src.services.schema_registry_service import SchemaRegistryService, schema_registry_mock_enabled


@pytest.fixture(autouse=True)
def _no_flags(clear_mock_flags):
    """Every test starts with all aliases absent."""


def test_mock_mode_enabled_via_use_mock(monkeypatch):
    monkeypatch.setenv("USE_MOCK", "true")
    assert mock_mode_enabled() is True


def test_mock_mode_enabled_via_stg_mock_mode_alias(monkeypatch):
    """Regression: deploy-stg sets STG_MOCK_MODE, never USE_MOCK."""
    monkeypatch.setenv("STG_MOCK_MODE", "true")
    assert mock_mode_enabled() is True


def test_mock_mode_disabled_when_neither_set():
    assert mock_mode_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " Yes "])
@pytest.mark.parametrize("name", MOCK_MODE_ENV_VARS)
def test_truthy_values_enable_mock_mode(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert mock_mode_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
@pytest.mark.parametrize("name", MOCK_MODE_ENV_VARS)
def test_falsy_values_do_not_enable_mock_mode(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert mock_mode_enabled() is False


def test_registry_follows_global_mock_mode_under_stg_alias(monkeypatch):
    monkeypatch.setenv("STG_MOCK_MODE", "true")
    assert schema_registry_mock_enabled() is True
    assert SchemaRegistryService().validate_allowlist({"type": "object"}) is True


def test_registry_can_be_pinned_to_stub_while_llm_is_real(monkeypatch):
    monkeypatch.setenv("USE_MOCK_SCHEMA_REGISTRY", "true")
    assert mock_mode_enabled() is False
    assert schema_registry_mock_enabled() is True


def test_registry_rejects_unregistered_schema_outside_mock_mode():
    service = SchemaRegistryService()
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert service.validate_allowlist(schema) is False
    service.register_schema(schema)
    assert service.validate_allowlist(schema) is True
