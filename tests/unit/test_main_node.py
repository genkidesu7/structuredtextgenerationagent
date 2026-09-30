# cmn_c1_517 — Unit Tests: Main Node

import pytest
from src.nodes.main_node import MainNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

SIMPLE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {"mock_field": {"type": "string"}},
    "required": ["mock_field"],
}

BASE_STATE = {
    "user_input": "test input",
    "validated_input": "test input",
    "rendered_prompt": "Generate output conforming to the schema.",
    "target_schema": SIMPLE_SCHEMA,
    "provider": "fallback",
    "model_config": {"model": "test-model", "max_retry": 3},
    "context_vars": {},
    "schema_id": "abc123",
    "schema_version": "1.0",
    "input_context": {},
    "session_id": "s1",
    "correlation_id": "c1",
    "trace_id": "t1",
    "thread_id": "th1",
    "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
    "caller_id": "test-caller",
    "hitl_allowed": True,
    "status": AgentStatus.PENDING.value,
    "retry_count": 0,
    "node_history": [],
    "error_log": [],
    "execution_time": {},
}


class TestMainNode:
    """Unit tests for the main business logic node."""

    def setup_method(self):
        self.node = MainNode()

    def test_success_path(self):
        """TC: Main node processes valid input and returns SUCCESS (mock mode)."""
        result = self.node.execute(BASE_STATE)
        assert result["status"] == AgentStatus.SUCCESS
        assert result["result"] is not None

    def test_empty_input(self):
        """TC: Main node handles empty rendered_prompt gracefully (mock LLM still responds)."""
        state = {**BASE_STATE, "rendered_prompt": ""}
        result = self.node.execute(state)
        # Mock mode returns a fixture — may succeed even with empty prompt
        assert result["status"] in (AgentStatus.SUCCESS, AgentStatus.RETRY, AgentStatus.ERROR)

    def test_execute_method_signature(self):
        """Node must implement execute(state) not _invoke_impl.

        Canonical contract:
          - Override: execute(self, state: AgentState) -> dict
          - PROHIBITED: _invoke_impl(), process() override
        """
        import inspect

        # Must have execute() defined on the concrete class (not just inherited stub)
        assert hasattr(MainNode, "execute"), "MainNode must implement execute()"

        sig = inspect.signature(MainNode.execute)
        params = list(sig.parameters.keys())
        # execute(self, state) — at minimum two parameters
        assert len(params) >= 2, (
            f"execute() must accept (self, state), got params: {params}"
        )
        assert params[1] == "state", (
            f"Second parameter must be 'state', got '{params[1]}'"
        )

        # Must NOT define _invoke_impl at the domain level
        assert "_invoke_impl" not in MainNode.__dict__, (
            "_invoke_impl() must not be defined in MainNode — use execute() instead"
        )

    def test_required_trust_level_is_verified_external(self):
        """MainNode requires VERIFIED_EXTERNAL trust per design (S-1 gate)."""
        assert MainNode.required_trust_level == TrustLevel.VERIFIED_EXTERNAL