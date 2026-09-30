"""AgentCore Platform v1.0"""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.

import operator
from typing import Annotated, Any

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """StructuredTextGenerationAgent state.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, retry_count, etc.) are inherited from AgentState.

    Constraints (ADR-005):
    - Flat TypedDict only — primitives + JSON-serializable types
    - No credentials, JWT, API keys in State (checkpoint leakage)
    - No Pydantic models, dataclasses, or arbitrary Python objects
    - input_context is read-only — never written by nodes
    """

    # --- Input fields (set by caller, read-only after pre_process) ---
    target_schema: dict[str, Any]  # JSON schema dict the output must conform to
    provider: str  # "anthropic" | "openai" | "gemini" | "fallback"
    model_config: dict[str, Any]  # model name, temperature, max_tokens, etc.
    context_vars: dict[str, Any]  # Jinja2 template variables for prompt rendering

    # --- pre_process outputs ---
    validated_input: str  # sanitized user_input
    rendered_prompt: str  # Jinja2-rendered prompt string
    schema_id: str  # SHA-256 of canonical schema JSON
    schema_version: str  # caller-supplied version tag

    # --- main outputs ---
    result: dict[str, Any]  # parsed LLM response dict
    llm_usage: int  # total LLM tokens consumed in this invocation
    attempt_count: int  # number of generation attempts (1 + retries)
    last_validation_error: str  # last jsonschema validation error (retry feedback)

    # --- post_process outputs ---
    formatted_output: dict[str, Any]  # final shaped response with data + metadata

    # --- Accumulator fields ---
    validation_errors: Annotated[list[str], operator.add]  # all validation errors across retries
