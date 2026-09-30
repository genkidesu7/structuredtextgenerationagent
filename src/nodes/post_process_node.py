"""AgentCore Platform v1.0"""

# Node contract (agents_layer_design.md §1):
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Never import from mediator/, api/, or other agents
#  - No agenticstar imports — PB-4

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, ClassVar

from framework.errors import SecurityViolationError
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.utils.audit_logger import emit_trace_event

# Regex patterns for common credential-like strings (belt-and-suspenders S-3)
_CREDENTIAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[A-Za-z0-9]{20,}", re.IGNORECASE),  # OpenAI key
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),  # Bearer token
    re.compile(r"eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+"),  # JWT
    re.compile(r"AIza[A-Za-z0-9\-_]{35}", re.IGNORECASE),  # Google API key
    re.compile(r"AKIA[A-Z0-9]{16}"),  # AWS access key
]


class PostProcessNode(FunctionNode):
    """Shape final structured output, apply S-3 output gate, emit audit record.

    Reads: result, schema_id, schema_version, tokens_used, attempt_count,
           validation_errors
    Writes: formatted_output
    """

    # S-1: explicit by design — this node does no privileged work; MainNode
    # (the LLM call) is the node that requires VERIFIED_EXTERNAL.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Shaping final output...",
            metadata={"stage": "post_process"},
        )
        result = state.get("result")
        schema_id: str = state.get("schema_id") or ""
        schema_version: str = state.get("schema_version") or ""
        llm_usage: int = state.get("llm_usage") or 0
        attempt_count: int = state.get("attempt_count") or 0
        model: str = (state.get("model_config") or {}).get("model", "")

        # Handle malformed result (not a dict) gracefully
        if not isinstance(result, dict):
            formatted_output: dict[str, Any] = {
                "data": None,
                "schema_id": schema_id,
                "schema_version": schema_version,
                "attempt_count": attempt_count,
                "llm_usage": llm_usage,
                "error": "PostProcessNode: result is not a dict — malformed LLM output",
            }
            emit_trace_event(
                "generation_audit",
                {
                    "schema_id": schema_id,
                    "schema_version": schema_version,
                    "model": model,
                    "attempt_count": attempt_count,
                    "llm_usage": llm_usage,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "outcome": "malformed_result",
                },
                state,
            )
            return {
                "formatted_output": formatted_output,
                "status": AgentStatus.SUCCESS,
            }

        # Assemble formatted_output (S-5 audit record shape)
        formatted_output = {
            "data": result,
            "schema_id": schema_id,
            "schema_version": schema_version,
            "attempt_count": attempt_count,
            "llm_usage": llm_usage,
        }

        # S-5 generation audit record
        emit_trace_event(
            "generation_audit",
            {
                "schema_id": schema_id,
                "schema_version": schema_version,
                "model": model,
                "attempt_count": attempt_count,
                "llm_usage": llm_usage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "outcome": "success",
            },
            state,
        )

        return {
            "formatted_output": formatted_output,
            "status": AgentStatus.SUCCESS,
        }

    # ------------------------------------------------------------------
    # S-3 extension gate
    # ------------------------------------------------------------------

    def _extra_security_gate_output(self, result: dict[str, Any]) -> dict[str, Any]:
        """S-3 domain extension: scan result dict values for credential patterns.

        Belt-and-suspenders check on top of the framework's automatic
        _security_gate_output() credential scan. Raises SecurityViolationError
        if any value in formatted_output.data contains a credential pattern.
        """
        formatted_output = result.get("formatted_output")
        if not isinstance(formatted_output, dict):
            return result
        data = formatted_output.get("data")
        if not isinstance(data, dict):
            return result
        for key, value in data.items():
            if not isinstance(value, str):
                continue
            for pattern in _CREDENTIAL_PATTERNS:
                if pattern.search(value):
                    raise SecurityViolationError(
                        f"PostProcessNode S-3: credential pattern detected in " f"output field '{key}' — blocked"
                    )
        return result
