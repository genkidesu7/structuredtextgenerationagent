"""Pytest configuration for all test suites.

Sets up mock environment flags so that tests run without real LLM or
schema registry credentials. Also provides shared fixtures.
"""
import os
import pytest


# ---------------------------------------------------------------------------
# Mock mode — USE_MOCK is the canonical flag resolved by
# src.services.mock_mode.mock_mode_enabled(); it also drives the schema
# registry stub.
#
# Every test is pinned to mock mode unless it is marked `real_llm` AND the
# caller explicitly exported USE_MOCK=false (+ the AZURE_OPENAI_* secrets).
# That is how `USE_MOCK=false run-local/test_pytest.sh` exercises the real
# Azure OpenAI path while every other service keeps returning mock data.
# Tests that probe the non-mock path on purpose clear every alias with the
# `clear_mock_flags` fixture below.
# ---------------------------------------------------------------------------
REAL_LLM_REQUESTED: bool = os.environ.get("USE_MOCK", "").strip().lower() in ("0", "false", "no")
AZURE_SECRET_KEYS = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_llm: calls the real Azure OpenAI deployment; runs only with USE_MOCK=false and AZURE_OPENAI_* exported",
    )


@pytest.fixture(autouse=True)
def _pin_mock_mode(request, monkeypatch):
    if request.node.get_closest_marker("real_llm"):
        if not REAL_LLM_REQUESTED:
            pytest.skip("real-LLM test: export USE_MOCK=false (and AZURE_OPENAI_*) to run")
        missing = [key for key in AZURE_SECRET_KEYS if not os.environ.get(key)]
        if missing:
            pytest.fail(f"USE_MOCK=false requested but {missing} not exported — refusing to skip silently")
        monkeypatch.setenv("USE_MOCK", "false")
        monkeypatch.delenv("STG_MOCK_MODE", raising=False)
        # Only the LLM is real; the schema registry has no backing service.
        monkeypatch.setenv("USE_MOCK_SCHEMA_REGISTRY", "true")
    else:
        monkeypatch.setenv("USE_MOCK", "true")


# ---------------------------------------------------------------------------
# Shared state fixtures
# ---------------------------------------------------------------------------

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel


SIMPLE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
}


@pytest.fixture
def clear_mock_flags(monkeypatch):
    """Remove every accepted mock-mode alias so the real (non-mock) path runs."""
    from src.services.mock_mode import MOCK_MODE_ENV_VARS

    for name in (*MOCK_MODE_ENV_VARS, "USE_MOCK_SCHEMA_REGISTRY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def base_agent_state():
    """Minimal base state compatible with all AgentState fields."""
    return {
        "user_input": "What is the capital of France?",
        "input_context": {},
        "target_schema": SIMPLE_SCHEMA,
        "provider": "fallback",
        "model_config": {"model": "test-model", "max_retry": 3},
        "context_vars": {},
        "session_id": "s1",
        "correlation_id": "c1",
        "trace_id": "t1",
        "thread_id": "th1",
        "schema_version": "1.0",
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "caller_id": "test-caller",
        "hitl_allowed": True,
        "status": AgentStatus.PENDING.value,
        "retry_count": 0,
        "node_history": [],
        "error_log": [],
        "execution_time": {},
    }
