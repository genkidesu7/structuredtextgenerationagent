"""AgentCore Platform v1.0"""

# Service layer: structured-output request shaping and response parsing.
# Every real LLM call goes through AzureOpenAIClient (see service.call_llm);
# the `provider` field selects how the request is constrained:
#   openai     -> response_format json_schema, strict mode
#   anthropic / gemini / fallback
#              -> response_format json_object + the schema embedded in the prompt
# PB-4: Level-0 SDK (agenticstar) imports are prohibited in this file.
# Credentials must come via credential_handle — not os.environ or State.

from __future__ import annotations

import json
from typing import Any

from src.services.mock_data import stub_generate_structured
from src.services.mock_mode import mock_mode_enabled
from src.services.service import call_llm

_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.0


def _is_strict_compatible(schema: Any) -> bool:
    """True when every object in ``schema`` is closed and lists all its properties as required."""
    if isinstance(schema, list):
        return all(_is_strict_compatible(item) for item in schema)
    if not isinstance(schema, dict):
        return True
    if schema.get("type") == "object" or "properties" in schema:
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is not False:
            return False
        if set(schema.get("required") or []) != set(properties):
            return False
    return all(_is_strict_compatible(value) for value in schema.values())


class StructuredOutputParseError(ValueError):
    """The model reply was not a single JSON object."""


class LLMProviderService:
    """Constrained structured generation behind one provider-agnostic interface.

    Accepts a ``credential_handle`` (opaque SecretProvider, or the literal
    ``"mock"`` when mock mode is on); the service never reads credentials from
    os.environ or State fields.

    Mock mode: when ``mock_mode_enabled()`` is true, generate_structured()
    returns a schema-conforming stub and never touches the credential handle.
    """

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        provider: str,
        model_config: dict[str, Any],
        credential_handle: Any,
    ) -> tuple[dict[str, Any], int]:
        """Generate structured output conforming to schema.

        Args:
            prompt: Rendered prompt string.
            schema: JSON schema dict the output must conform to.
            provider: Constraint style: "openai", "anthropic", "gemini" or
                "fallback" (any other value behaves as "fallback").
            model_config: Generation settings (``temperature``, ``max_tokens``).
            credential_handle: Opaque SecretProvider from InvocationContext.
                The service does NOT resolve or log this value outside call_llm.

        Returns:
            Tuple of (parsed_output_dict, token_units_used).

        Raises:
            StructuredOutputParseError: The reply was not a JSON object.
            MissingSecret: A required AZURE_OPENAI_* secret is absent.
        """
        if mock_mode_enabled():
            return stub_generate_structured(schema)

        messages, response_format = self._build_request(prompt, schema, provider)
        response = call_llm(
            messages,
            credential_handle,
            response_format=response_format,
            max_tokens=int(model_config.get("max_tokens") or _DEFAULT_MAX_TOKENS),
            temperature=float(model_config.get("temperature", _DEFAULT_TEMPERATURE)),
        )
        return self._parse_object(response.get("content", "")), self._token_units(response)

    @staticmethod
    def _build_request(
        prompt: str, schema: dict[str, Any], provider: str
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Return (messages, response_format) for the requested constraint style."""
        schema_text = json.dumps(schema, sort_keys=True)
        system = (
            "Reply with a single JSON object and nothing else. " f"It must conform to this JSON Schema: {schema_text}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if provider.lower() == "openai":
            # Strict mode only accepts schemas that close every object
            # (additionalProperties: false, all properties required); any other schema
            # would be rejected on every attempt, so it is sent best-effort instead.
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": {k: v for k, v in schema.items() if k != "$schema"},
                    "strict": _is_strict_compatible(schema),
                },
            }
        else:
            response_format = {"type": "json_object"}
        return messages, response_format

    @staticmethod
    def _parse_object(content: str) -> dict[str, Any]:
        """Parse the model reply into a dict or raise StructuredOutputParseError."""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise StructuredOutputParseError(f"model reply is not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise StructuredOutputParseError(f"model reply is {type(parsed).__name__}, expected a JSON object")
        return parsed

    @staticmethod
    def _token_units(response: dict[str, Any]) -> int:
        """Total tokens (prompt + completion) reported by the provider; 0 if unreported."""
        usage = response.get("usage") or {}
        return int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
