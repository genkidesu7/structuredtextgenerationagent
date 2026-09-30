"""service.call_llm — the single AzureOpenAIClient construction point."""

import pytest

from framework.secrets.base import MissingSecret
from shared.secrets.inmemory_provider import InMemoryProvider
from src.services import service

SECRETS = {
    "AZURE_OPENAI_API_KEY": "mock-api-key-for-testing",
    "AZURE_OPENAI_ENDPOINT": "https://example.services.ai.azure.com",
    "AZURE_OPENAI_DEPLOYMENT": "mock-deployment",
}


def test_call_llm_builds_client_from_secrets_and_returns_completion(monkeypatch):
    seen: dict = {}

    class FakeClient:
        def __init__(self, config):
            seen["config"] = config

        def complete(self, messages):
            seen["messages"] = messages
            return {"content": "{}", "tool_calls": [], "model": "mock-deployment"}

    monkeypatch.setattr("shared.services.llm.azure_openai_client.AzureOpenAIClient", FakeClient)
    result = service.call_llm(
        [{"role": "user", "content": "hi"}],
        InMemoryProvider(SECRETS),
        response_format={"type": "json_object"},
        max_tokens=10,
        temperature=0.0,
    )
    assert result["content"] == "{}"
    assert seen["config"] == {
        "api_key": SECRETS["AZURE_OPENAI_API_KEY"],
        "azure_endpoint": SECRETS["AZURE_OPENAI_ENDPOINT"],
        "azure_deployment": SECRETS["AZURE_OPENAI_DEPLOYMENT"],
        "response_format": {"type": "json_object"},
        "max_tokens": 10,
        "temperature": 0.0,
    }
    assert seen["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("missing", list(SECRETS))
def test_call_llm_fails_fast_on_missing_secret(missing):
    provider = InMemoryProvider({k: v for k, v in SECRETS.items() if k != missing})
    with pytest.raises(MissingSecret):
        service.call_llm([{"role": "user", "content": "hi"}], provider)
