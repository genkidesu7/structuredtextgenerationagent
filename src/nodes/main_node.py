"""AgentCore Platform v1.0"""

# Node contract (agents_layer_design.md §1):
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Credentials ONLY via InvocationContext — never State or env
#  - Never import from mediator/, api/, or other agents
#  - No agenticstar imports — PB-4

from __future__ import annotations

from typing import Any, ClassVar

import jsonschema
import jsonschema.exceptions

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.utils.audit_logger import emit_trace_event

from src.services.llm_provider_service import LLMProviderService
from src.services.mock_mode import mock_mode_enabled
from src.services.service import LLM_SECRET_KEYS

# Hard ceiling on retries — regardless of max_retry config (design §MainNode)
MAX_RETRY_CEILING = 10

_llm_service = LLMProviderService()


class MainNode(FunctionNode):
    """Route to provider-native structured generation, validate, route status.

    Reads: rendered_prompt, target_schema, provider, model_config,
           schema_id, schema_version, retry_count, last_validation_error
    Writes: result, status, tokens_used, attempt_count, last_validation_error,
            validation_errors (accumulator)
    """

    # S-1: callers must present VERIFIED_EXTERNAL trust
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Generating structured output...",
            metadata={"stage": "main", "mock_mode": mock_mode_enabled()},
        )
        rendered_prompt: str = state.get("rendered_prompt") or ""
        target_schema: dict[str, Any] = state.get("target_schema") or {}
        provider: str = state.get("provider") or "fallback"
        model_config: dict[str, Any] = state.get("model_config") or {}
        schema_id: str = state.get("schema_id") or ""
        schema_version: str = state.get("schema_version") or ""
        retry_count: int = state.get("retry_count") or 0
        attempt_count: int = (state.get("attempt_count") or 0) + 1
        llm_usage: int = state.get("llm_usage") or 0

        # Hard ceiling: if retry_count has hit the ceiling, force ERROR
        if retry_count >= MAX_RETRY_CEILING:
            return {
                "status": AgentStatus.ERROR,
                "attempt_count": attempt_count,
                "error_log": [f"MainNode: MAX_RETRY_CEILING ({MAX_RETRY_CEILING}) reached"],
            }

        # Obtain credential handle from InvocationContext (S-4 handle-based).
        # Mock mode never touches a credential, so it must not demand one;
        # outside mock mode a missing secret fails fast (MissingSecret) here,
        # before the retry loop could mistake it for a transient API error.
        ctx = InvocationContext.from_state(state)
        credential_handle: Any
        if mock_mode_enabled():
            credential_handle = "mock"
        else:
            for secret_key in LLM_SECRET_KEYS:
                ctx.secrets.require(secret_key)
            credential_handle = ctx.secrets

        # Call LLM provider service
        try:
            raw_output, call_units = _llm_service.generate_structured(
                prompt=rendered_prompt,
                schema=target_schema,
                provider=provider,
                model_config=model_config,
                credential_handle=credential_handle,
            )
            llm_usage += call_units
        except Exception as exc:
            # Transient API error → RETRY (bounded by max_retry)
            max_retry: int = model_config.get("max_retry") or 3
            new_status = AgentStatus.RETRY if retry_count < max_retry else AgentStatus.ERROR
            emit_trace_event(
                "structured_generation_attempt",
                {
                    "schema_id": schema_id,
                    "schema_version": schema_version,
                    "provider": provider,
                    "attempt_count": attempt_count,
                    "llm_usage": llm_usage,
                    "outcome": "api_error",
                    "error": str(exc),
                },
                state,
            )
            return {
                "status": new_status,
                # The framework router only reads retry_count; this node must advance it,
                # or a RETRY loops until LangGraph's recursion limit.
                "retry_count": retry_count + 1 if new_status == AgentStatus.RETRY else retry_count,
                "attempt_count": attempt_count,
                "llm_usage": llm_usage,
                "last_validation_error": str(exc),
                "validation_errors": [str(exc)],
                "error_log": [f"MainNode: LLM API error — {exc}"],
            }

        # Validate raw_output against target_schema
        validation_error: str = ""
        try:
            jsonschema.validate(instance=raw_output, schema=target_schema)
        except jsonschema.exceptions.ValidationError as exc:
            validation_error = exc.message

        # Determine routing status
        max_retry = model_config.get("max_retry") or 3
        if not validation_error:
            outcome = "success"
            new_status = AgentStatus.SUCCESS
        elif retry_count < max_retry:
            outcome = "retry"
            new_status = AgentStatus.RETRY
        else:
            outcome = "error_max_retry"
            new_status = AgentStatus.ERROR

        # S-4 audit: emit on every attempt
        emit_trace_event(
            "structured_generation_attempt",
            {
                "schema_id": schema_id,
                "schema_version": schema_version,
                "provider": provider,
                "model": model_config.get("model", ""),
                "attempt_count": attempt_count,
                "llm_usage": llm_usage,
                "outcome": outcome,
                "validation_error": validation_error,
            },
            state,
        )

        result: dict[str, Any] = {
            "result": raw_output,
            "status": new_status,
            "llm_usage": llm_usage,
            "attempt_count": attempt_count,
        }
        if new_status == AgentStatus.RETRY:
            result["retry_count"] = retry_count + 1
        if validation_error:
            result["last_validation_error"] = validation_error
            result["validation_errors"] = [validation_error]
        if new_status == AgentStatus.ERROR and validation_error:
            result["error_log"] = [
                f"MainNode: schema validation failed after {attempt_count} attempt(s) — {validation_error}"
            ]

        return result
