# Unit Tests: PostProcessNode

import pytest
from src.nodes.post_process_node import PostProcessNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel


BASE_STATE = {
    "user_input": "test",
    "result": "some result",
    "input_context": {},
    "session_id": "s1",
    "correlation_id": "c1",
    "trace_id": "t1",
    "thread_id": "th1",
    "schema_version": "1.0",
    "caller_trust_level": TrustLevel.ANONYMOUS.value,
    "caller_id": "",
    "hitl_allowed": True,
    "status": AgentStatus.SUCCESS.value,
    "retry_count": 0,
    "node_history": [],
    "error_log": [],
    "execution_time": {},
}


class TestPostProcessNodeExecute:
    def setup_method(self):
        self.node = PostProcessNode()

    def test_success_formats_result(self):
        state = {**BASE_STATE, "result": {"answer": "Paris"}}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS
        assert isinstance(result["formatted_output"], dict)
        assert result["formatted_output"]["data"] == {"answer": "Paris"}

    def test_empty_result_handled(self):
        state = {**BASE_STATE, "result": ""}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS
        assert "formatted_output" in result

    def test_missing_result_field_handled(self):
        state = {k: v for k, v in BASE_STATE.items() if k != "result"}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS
        assert "formatted_output" in result

    def test_execute_method_signature(self):
        import inspect
        assert hasattr(PostProcessNode, "execute")
        sig = inspect.signature(PostProcessNode.execute)
        params = list(sig.parameters.keys())
        assert len(params) >= 2
        assert params[1] == "state"
        assert "_invoke_impl" not in PostProcessNode.__dict__


class TestPostProcessNodeViaCall:
    def setup_method(self):
        self.node = PostProcessNode()

    def test_call_success_path(self):
        result = self.node(BASE_STATE)
        assert result["status"] == AgentStatus.SUCCESS
        assert "PostProcessNode" in result["node_history"]
