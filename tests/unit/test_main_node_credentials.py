"""MainNode credential boundary: mock mode skips require(); real mode fails fast."""

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.secrets.base import MissingSecret
from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider
from src.nodes.main_node import MainNode
from src.services import llm_provider_service
from tests.conftest import SIMPLE_SCHEMA

SECRETS = {
    "AZURE_OPENAI_API_KEY": "mock-api-key-for-testing",
    "AZURE_OPENAI_ENDPOINT": "https://example.services.ai.azure.com",
    "AZURE_OPENAI_DEPLOYMENT": "mock-deployment",
}


@pytest.fixture
def main_state(base_agent_state):
    return {**base_agent_state, "rendered_prompt": "What is the capital of France?", "schema_id": "id"}


@pytest.mark.parametrize("alias", ["USE_MOCK", "STG_MOCK_MODE"])
def test_succeeds_with_no_secrets_bound_in_mock_mode(monkeypatch, clear_mock_flags, main_state, alias):
    """A fresh CI checkout has no env/ secret files — mock mode must not need any.

    Deliberately runs WITHOUT bound_secrets(): that is the real CI condition.
    """
    monkeypatch.setenv(alias, "true")
    result = MainNode().execute(main_state)
    assert result["status"] == AgentStatus.SUCCESS
    assert result["result"] == {"answer": "mock_answer"}


def test_real_path_raises_missing_secret_when_every_alias_cleared(clear_mock_flags, main_state):
    """Negative proof-of-boundary: no mock alias + no secrets => fail fast, not a silent retry."""
    with pytest.raises(MissingSecret):
        MainNode().execute(main_state)


@pytest.mark.parametrize("missing", list(SECRETS))
def test_real_path_requires_each_declared_secret(clear_mock_flags, main_state, missing):
    provider = InMemoryProvider({k: v for k, v in SECRETS.items() if k != missing})
    with bound_secrets(provider), pytest.raises(MissingSecret):
        MainNode().execute(main_state)


def test_real_path_calls_llm_with_bound_secrets(monkeypatch, clear_mock_flags, main_state):
    captured: dict = {}

    def _fake_call_llm(messages, credential_handle, **kwargs):
        captured["handle"] = credential_handle
        return {"content": json.dumps({"answer": "Paris"}), "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    monkeypatch.setattr(llm_provider_service, "call_llm", _fake_call_llm)
    with bound_secrets(InMemoryProvider(SECRETS)):
        result = MainNode().execute(main_state)
    assert result["status"] == AgentStatus.SUCCESS
    assert result["result"] == {"answer": "Paris"}
    assert result["llm_usage"] == 5
    assert captured["handle"].require("AZURE_OPENAI_DEPLOYMENT") == "mock-deployment"


def test_unparseable_reply_retries_with_feedback(monkeypatch, clear_mock_flags, main_state):
    monkeypatch.setattr(llm_provider_service, "call_llm", lambda *a, **k: {"content": "not json"})
    with bound_secrets(InMemoryProvider(SECRETS)):
        result = MainNode().execute(main_state)
    assert result["status"] == AgentStatus.RETRY
    assert result["retry_count"] == 1
    assert "not valid JSON" in result["last_validation_error"]


def test_schema_violation_retries_then_errors(monkeypatch, clear_mock_flags, main_state):
    monkeypatch.setattr(
        llm_provider_service, "call_llm", lambda *a, **k: {"content": json.dumps({"wrong": 1})}
    )
    with bound_secrets(InMemoryProvider(SECRETS)):
        first = MainNode().execute(main_state)
        exhausted = MainNode().execute({**main_state, "retry_count": 3})
    assert first["status"] == AgentStatus.RETRY
    assert exhausted["status"] == AgentStatus.ERROR
    assert main_state["target_schema"] == SIMPLE_SCHEMA
