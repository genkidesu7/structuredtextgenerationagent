"""Real Azure OpenAI path — runs only with USE_MOCK=false and AZURE_OPENAI_* exported.

Only the LLM call is real. Every other service (schema registry, ...) still
returns mock data, as pinned by the `_pin_mock_mode` fixture in conftest.py.
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

from framework.secrets.context import bound_secrets
from shared.secrets.inmemory_provider import InMemoryProvider
from src.graph.graph import Graph
from tests.conftest import AZURE_SECRET_KEYS, SIMPLE_SCHEMA

pytestmark = pytest.mark.real_llm

PROMPT = "What is the capital of France? Answer with just the city name in the `answer` field."


def _secrets() -> InMemoryProvider:
    return InMemoryProvider({key: os.environ[key] for key in AZURE_SECRET_KEYS})


@pytest.mark.parametrize("provider", ["fallback", "openai"])
def test_graph_invoke_generates_schema_conforming_output(provider):
    agent = Graph(config={"max_retry": 3})
    agent.compile()
    agent.provision_secrets(_secrets())
    from framework.schemas.invocation_context import InvocationContext
    from framework.schemas.trust_level import TrustLevel

    with bound_secrets(_secrets()):
        ctx = InvocationContext(session_id="real-llm", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="t")
        result = agent.invoke(
            PROMPT,
            ctx=ctx,
            input_context={"target_schema": SIMPLE_SCHEMA, "provider": provider, "model_config": {"max_tokens": 64}},
        )
    assert result["status"] == "success", result
    data = result["output"]["data"]
    assert isinstance(data["answer"], str) and data["answer"]
    assert result["output"]["llm_usage"] > 0, "real provider must report token usage"
    assert "mock_answer" not in data["answer"]


@pytest.mark.asyncio
async def test_http_route_generates_real_output(monkeypatch):
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", "mock-invoke-token-for-testing")
    from src.api.server import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/invoke",
            headers={"Authorization": "Bearer mock-invoke-token-for-testing"},
            json={"input": PROMPT, "target_schema": SIMPLE_SCHEMA, "model_config": {"max_tokens": 64}},
        )
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "success", body
    assert body["output"]["llm_usage"] > 0
