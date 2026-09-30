"""AgentCore Platform v1.0"""

# Service layer: the one place that constructs AzureOpenAIClient.
# Credentials arrive as an opaque SecretProvider handle (ctx.secrets) and are
# resolved with .require() — never os.environ, never State.

from __future__ import annotations

from typing import Any

LLM_SECRET_KEYS: tuple[str, ...] = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
)


def call_llm(
    messages: list[dict[str, str]],
    credential_handle: Any,
    response_format: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Send ``messages`` through AzureOpenAIClient and return its canonical response.

    Args:
        messages: Chat messages (``{"role": ..., "content": ...}``).
        credential_handle: SecretProvider (``ctx.secrets``) holding the three
            ``AZURE_OPENAI_*`` secrets.
        response_format: Optional OpenAI ``response_format`` (json_object / json_schema).
        max_tokens: Optional completion-token cap.
        temperature: Optional sampling temperature.

    Returns:
        Canonical ``BaseLLM.complete()`` dict: ``content``, ``tool_calls``, ``model``
        and (when the provider reports it) ``usage``.

    Raises:
        MissingSecret: A required ``AZURE_OPENAI_*`` secret is not provisioned.
    """
    from shared.services.llm.azure_openai_client import AzureOpenAIClient  # lazy: heavy import

    config: dict[str, Any] = {
        "api_key": credential_handle.require("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": credential_handle.require("AZURE_OPENAI_ENDPOINT"),
        "azure_deployment": credential_handle.require("AZURE_OPENAI_DEPLOYMENT"),
    }
    if response_format is not None:
        config["response_format"] = response_format
    if max_tokens is not None:
        config["max_tokens"] = max_tokens
    if temperature is not None:
        config["temperature"] = temperature
    client = AzureOpenAIClient(config=config)
    response: dict[str, Any] = client.complete(messages)
    return response
