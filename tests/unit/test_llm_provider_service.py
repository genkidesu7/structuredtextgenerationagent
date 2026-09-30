"""LLMProviderService real (non-mock) path — request shaping, parsing, usage."""

import json

import pytest

from src.services import llm_provider_service
from src.services.llm_provider_service import LLMProviderService, StructuredOutputParseError

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


@pytest.fixture
def fake_call_llm(monkeypatch, clear_mock_flags):
    """Replace the AzureOpenAIClient boundary; record what the service sends."""
    calls: list[dict] = []
    reply = {"content": json.dumps({"answer": "Paris"}), "usage": {"prompt_tokens": 7, "completion_tokens": 5}}

    def _fake(messages, credential_handle, response_format=None, max_tokens=None, temperature=None):
        calls.append(
            {
                "messages": messages,
                "credential_handle": credential_handle,
                "response_format": response_format,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return reply

    monkeypatch.setattr(llm_provider_service, "call_llm", _fake)
    return calls, reply


STRICT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_openai_provider_uses_strict_json_schema_for_closed_schema(fake_call_llm):
    calls, _ = fake_call_llm
    output, units = LLMProviderService().generate_structured("prompt", STRICT_SCHEMA, "openai", {}, "handle")
    assert output == {"answer": "Paris"}
    assert units == 12
    fmt = calls[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == STRICT_SCHEMA
    assert calls[0]["credential_handle"] == "handle"


def test_openai_provider_falls_back_to_non_strict_for_open_schema(fake_call_llm):
    """An open schema would be rejected by strict mode on every attempt."""
    calls, _ = fake_call_llm
    open_schema = {"$schema": "http://json-schema.org/draft-07/schema#", **SCHEMA}
    LLMProviderService().generate_structured("prompt", open_schema, "openai", {}, "handle")
    fmt = calls[0]["response_format"]["json_schema"]
    assert fmt["strict"] is False
    assert "$schema" not in fmt["schema"]


@pytest.mark.parametrize("provider", ["anthropic", "gemini", "fallback", "something-else"])
def test_other_providers_use_json_object_with_schema_in_prompt(fake_call_llm, provider):
    calls, _ = fake_call_llm
    LLMProviderService().generate_structured("the prompt", SCHEMA, provider, {}, "handle")
    assert calls[0]["response_format"] == {"type": "json_object"}
    system, user = calls[0]["messages"]
    assert system["role"] == "system" and json.dumps(SCHEMA, sort_keys=True) in system["content"]
    assert user == {"role": "user", "content": "the prompt"}


def test_model_config_limits_are_forwarded(fake_call_llm):
    calls, _ = fake_call_llm
    LLMProviderService().generate_structured("p", SCHEMA, "fallback", {"max_tokens": 64, "temperature": 0.5}, "h")
    assert calls[0]["max_tokens"] == 64
    assert calls[0]["temperature"] == 0.5


def test_unreported_usage_counts_as_zero(fake_call_llm):
    _, reply = fake_call_llm
    reply.pop("usage")
    _, units = LLMProviderService().generate_structured("p", SCHEMA, "fallback", {}, "h")
    assert units == 0


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '"text"', ""])
def test_non_object_reply_raises_parse_error(fake_call_llm, content):
    _, reply = fake_call_llm
    reply["content"] = content
    with pytest.raises(StructuredOutputParseError):
        LLMProviderService().generate_structured("p", SCHEMA, "fallback", {}, "h")


def test_mock_mode_never_calls_the_llm(monkeypatch):
    monkeypatch.setenv("USE_MOCK", "true")

    def _boom(*args, **kwargs):
        raise AssertionError("call_llm must not run in mock mode")

    monkeypatch.setattr(llm_provider_service, "call_llm", _boom)
    output, _ = LLMProviderService().generate_structured("p", SCHEMA, "openai", {}, "mock")
    assert output == {"answer": "mock_answer"}
