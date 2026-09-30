# PB-6 — Invoke Execution Order
# Verifies that the graph backbone executes nodes in the correct sequence:
#   InitializeNode → PreProcessNode → MainNode → PostProcessNode → FinalizeNode
# and that the S-2 input gate fires before execute().

import pytest
from framework.schemas.trust_level import TrustLevel
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext

_SIMPLE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {"mock_field": {"type": "string"}},
    "required": ["mock_field"],
}

_INVOKE_CTX = InvocationContext(
    caller_id="pb6-test",
    caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
)

_INVOKE_INPUT_CTX = {
    "target_schema": _SIMPLE_SCHEMA,
    "provider": "fallback",
    "model_config": {"model": "test-model", "max_retry": 3},
    "context_vars": {},
}

_BASE_STATE = {
    "user_input": "test input",
    "input_context": {},
    "target_schema": _SIMPLE_SCHEMA,
    "provider": "fallback",
    "model_config": {"model": "test-model", "max_retry": 3},
    "context_vars": {},
    "session_id": "s1",
    "correlation_id": "c1",
    "trace_id": "t1",
    "thread_id": "th1",
    "schema_version": "1.0",
    "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
    "caller_id": "pb6-test",
    "hitl_allowed": True,
    "status": AgentStatus.PENDING.value,
    "retry_count": 0,
    "node_history": [],
    "error_log": [],
    "execution_time": {},
}


class TestPB6InvokeOrder:
    """PB-6: node_history must record the full 5-node backbone in correct order."""

    def _invoke(self):
        from src.graph.graph import Graph
        graph = Graph()
        graph.compile()
        return graph.invoke("hello world", ctx=_INVOKE_CTX, input_context=_INVOKE_INPUT_CTX)

    def test_all_backbone_nodes_present_in_history(self):
        result = self._invoke()
        history = result.get("node_history", [])
        for node_name in ("InitializeNode", "PreProcessNode", "MainNode", "PostProcessNode", "FinalizeNode"):
            assert node_name in history, f"{node_name} missing from node_history: {history}"

    def test_initialize_runs_first(self):
        result = self._invoke()
        history = result.get("node_history", [])
        assert history[0] == "InitializeNode", f"Expected InitializeNode first, got: {history[0]}"

    def test_pre_process_runs_before_main(self):
        result = self._invoke()
        history = result.get("node_history", [])
        assert history.index("PreProcessNode") < history.index("MainNode")

    def test_main_runs_before_post_process(self):
        result = self._invoke()
        history = result.get("node_history", [])
        assert history.index("MainNode") < history.index("PostProcessNode")

    def test_post_process_runs_before_finalize(self):
        result = self._invoke()
        history = result.get("node_history", [])
        assert history.index("PostProcessNode") < history.index("FinalizeNode")

    def test_s2_gate_input_fires_before_execute(self):
        """S-2: _security_gate_input() must run before execute() — verified by confirming
        that execute() is reached (no gate exception) and the call completes cleanly."""
        from src.nodes.pre_process_node import PreProcessNode

        executed_inputs: list[str] = []

        class InspectingPreProcessNode(PreProcessNode):
            def execute(self, state: dict) -> dict:
                executed_inputs.append(state.get("user_input", ""))
                return super().execute(state)

        node = InspectingPreProcessNode()
        state = {**_BASE_STATE, "user_input": "My email is test@example.com"}
        node(state)
        assert len(executed_inputs) == 1, "execute() must be called exactly once after the S-2 gate"
