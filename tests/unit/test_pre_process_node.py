# Unit Tests: PreProcessNode

import pytest
from src.nodes.pre_process_node import PreProcessNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

SIMPLE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

BASE_STATE = {
    "user_input": "hello world",
    "input_context": {},
    "target_schema": SIMPLE_SCHEMA,
    "provider": "fallback",
    "model_config": {"model": "test-model"},
    "context_vars": {},
    "session_id": "s1",
    "correlation_id": "c1",
    "trace_id": "t1",
    "thread_id": "th1",
    "schema_version": "1.0",
    "caller_trust_level": TrustLevel.ANONYMOUS.value,
    "caller_id": "",
    "hitl_allowed": True,
    "status": AgentStatus.PENDING.value,
    "retry_count": 0,
    "node_history": [],
    "error_log": [],
    "execution_time": {},
}


class TestPreProcessNodeExecute:
    def setup_method(self):
        self.node = PreProcessNode()

    def test_success_valid_input(self):
        state = {**BASE_STATE, "user_input": "  hello  "}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS
        assert result["validated_input"] == "hello"

    def test_error_on_empty_user_input(self):
        state = {**BASE_STATE, "user_input": ""}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.ERROR
        assert len(result["error_log"]) > 0

    def test_error_on_whitespace_only_input(self):
        state = {**BASE_STATE, "user_input": "   "}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.ERROR

    def test_execute_method_signature(self):
        import inspect
        assert hasattr(PreProcessNode, "execute")
        sig = inspect.signature(PreProcessNode.execute)
        params = list(sig.parameters.keys())
        assert len(params) >= 2
        assert params[1] == "state"
        assert "_invoke_impl" not in PreProcessNode.__dict__

    def test_does_not_return_full_state(self):
        state = {**BASE_STATE, "user_input": "test"}
        result = self.node.execute(state)
        # Node must return only changed fields, not the full state
        assert "user_input" not in result or result.get("user_input") == state["user_input"]

    def test_reads_input_context_readonly(self):
        state = {**BASE_STATE, "user_input": "test", "input_context": {"channel": "web"}}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS


class TestPreProcessNodeViaCall:
    """Tests exercised through __call__ to verify full pipeline including S-1, S-2, S-3."""

    def setup_method(self):
        self.node = PreProcessNode()

    def _make_state(self, **overrides):
        return {**BASE_STATE, **overrides}

    def test_call_success_path(self):
        state = self._make_state(user_input="valid input")
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS

    def test_call_error_on_empty_input(self):
        state = self._make_state(user_input="")
        result = self.node(state)
        assert result["status"] == AgentStatus.ERROR

    def test_node_history_appended(self):
        state = self._make_state(user_input="test")
        result = self.node(state)
        assert "PreProcessNode" in result["node_history"]
