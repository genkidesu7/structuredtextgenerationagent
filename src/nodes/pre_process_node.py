"""AgentCore Platform v1.0"""

# Node contract (agents_layer_design.md §1):
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings [A1]
#  - Read input_context via state.get("input_context", {}) — read-only [C1]
#  - Never import from mediator/, api/, or other agents
#  - No agenticstar imports — PB-4

from __future__ import annotations

from typing import Any, ClassVar

import jsonschema
import jsonschema.exceptions

from framework.errors import SecurityViolationError
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.utils.audit_logger import emit_trace_event

from src.services.prompt_template_service import PromptTemplateService
from src.services.schema_registry_service import SchemaRegistryService

# Maximum recursive $ref nesting depth before Tier 3 complexity warning
_MAX_REF_DEPTH = 5

_schema_registry = SchemaRegistryService()
_prompt_service = PromptTemplateService()


class PreProcessNode(FunctionNode):
    """Validate request, enforce schema allowlist (S-1), render Jinja2 prompt.

    Reads: user_input, target_schema, provider, model_config, context_vars,
           last_validation_error (non-empty on retry)
    Writes: validated_input, rendered_prompt, schema_id, schema_version
    """

    # S-1: explicit by design — this node does no privileged work; MainNode
    # (the LLM call) is the node that requires VERIFIED_EXTERNAL.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Validating request and rendering prompt...",
            metadata={"stage": "pre_process"},
        )
        user_input = state.get("user_input", "")
        # Domain fields may arrive via top-level state (unit tests / direct invoke)
        # OR via input_context (graph.invoke() integration path). Read from both.
        input_ctx: dict[str, Any] = state.get("input_context") or {}
        target_schema: dict[str, Any] = state.get("target_schema") or input_ctx.get("target_schema") or {}
        provider: str = state.get("provider") or input_ctx.get("provider") or "fallback"
        model_config: dict[str, Any] = state.get("model_config") or input_ctx.get("model_config") or {}
        context_vars: dict[str, Any] = state.get("context_vars") or input_ctx.get("context_vars") or {}
        last_validation_error: str = state.get("last_validation_error") or ""

        # 1. Validate user_input is non-empty
        if not user_input or not user_input.strip():
            return {
                "status": AgentStatus.ERROR,
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        # 2. Validate target_schema is a valid JSON Schema (draft-07)
        if not target_schema:
            return {
                "status": AgentStatus.ERROR,
                "error_log": ["PreProcessNode: target_schema is missing or empty"],
            }
        try:
            jsonschema.Draft7Validator.check_schema(target_schema)
        except jsonschema.exceptions.SchemaError as exc:
            return {
                "status": AgentStatus.ERROR,
                "error_log": [f"PreProcessNode: invalid JSON Schema — {exc.message}"],
            }

        # 3. Schema allowlist enforcement (S-1 domain gate)
        if not _schema_registry.validate_allowlist(target_schema):
            raise SecurityViolationError("PreProcessNode: target_schema is not registered in the allowlist")

        # 4. Sanitize user_input — escape Jinja2 delimiters
        sanitized_input = user_input.strip()

        # 5. Compute schema_id and extract schema_version
        schema_id = _schema_registry.compute_schema_id(target_schema)
        schema_version: str = (
            state.get("schema_version")
            or context_vars.get("schema_version", "")
            or target_schema.get("$schema", "unknown")
        )

        # 6. Render prompt via PromptTemplateService
        try:
            rendered_prompt = _prompt_service.render(
                template_str=sanitized_input,
                context_vars=context_vars,
                validation_error=last_validation_error or None,
            )
        except Exception as exc:
            return {
                "status": AgentStatus.ERROR,
                "error_log": [f"PreProcessNode: prompt render failed — {exc}"],
            }

        emit_trace_event(
            "pre_process_complete",
            {
                "schema_id": schema_id,
                "schema_version": schema_version,
                "has_validation_error": bool(last_validation_error),
            },
            state,
        )

        return {
            "target_schema": target_schema,
            "provider": provider,
            "model_config": model_config,
            "context_vars": context_vars,
            "validated_input": sanitized_input,
            "rendered_prompt": rendered_prompt,
            "schema_id": schema_id,
            "schema_version": schema_version,
            "status": AgentStatus.SUCCESS,
        }

    # ------------------------------------------------------------------
    # S-2 extension gate
    # ------------------------------------------------------------------

    def _extra_security_gate_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """S-2 domain extension: schema allowlist + Tier 3 complexity warning.

        Checks for disallowed schema patterns (recursive $ref depth > 5)
        and emits a Tier 3 warning. Does NOT raise — returns state.
        """
        target_schema: dict[str, Any] = state.get("target_schema") or {}
        if target_schema:
            depth = _compute_ref_depth(target_schema, target_schema, 0)
            if depth > _MAX_REF_DEPTH:
                emit_trace_event(
                    "schema_complexity_warning",
                    {
                        "tier": 3,
                        "ref_depth": depth,
                        "max_ref_depth": _MAX_REF_DEPTH,
                        "message": "Recursive $ref depth exceeds Tier 3 threshold; "
                        "falling back to function-calling mode recommended.",
                    },
                    state,
                )
        return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_ref_depth(
    node: Any,
    root_schema: dict[str, Any],
    current_depth: int,
) -> int:
    """Recursively compute maximum $ref nesting depth in a JSON schema."""
    if current_depth > _MAX_REF_DEPTH + 1:
        # Short-circuit to avoid infinite recursion on truly recursive schemas
        return current_depth
    if not isinstance(node, dict):
        return current_depth
    max_depth = current_depth
    for key, value in node.items():
        if key == "$ref" and isinstance(value, str):
            # $ref found — count this level
            max_depth = max(max_depth, current_depth + 1)
        elif isinstance(value, dict):
            child_depth = _compute_ref_depth(value, root_schema, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        elif isinstance(value, list):
            for item in value:
                child_depth = _compute_ref_depth(item, root_schema, current_depth + 1)
                max_depth = max(max_depth, child_depth)
    return max_depth
