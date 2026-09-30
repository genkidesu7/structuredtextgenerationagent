# Integration Tests: Full graph pipeline

import pytest
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.schemas.invocation_context import InvocationContext

SIMPLE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {"mock_field": {"type": "string"}},
    "required": ["mock_field"],
}

INVOKE_KWARGS = {
    "input_context": {
        "target_schema": SIMPLE_SCHEMA,
        "provider": "fallback",
        "model_config": {"model": "test-model", "max_retry": 3},
        "context_vars": {},
    },
    "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL,
}


class TestGraphCompilation:
    def test_graph_compiles_without_error(self):
        from src.graph.graph import Graph
        graph = Graph()
        graph.compile()
        assert graph._compiled is not None

    def test_graph_name(self):
        from src.graph.graph import Graph
        graph = Graph()
        assert isinstance(graph.name, str)
        assert len(graph.name) > 0

    def test_graph_state_schema(self):
        from src.graph.graph import Graph
        from src.schemas.state import State
        graph = Graph()
        assert graph.state_schema is State

    def test_graph_registers_all_required_nodes(self):
        from src.graph.graph import Graph
        graph = Graph()
        graph.register_nodes()
        assert graph._nodes.get("pre_process") is not None
        assert graph._nodes.get("main") is not None
        assert graph._nodes.get("post_process") is not None
        assert graph._nodes.get("initialize") is not None
        assert graph._nodes.get("finalize") is not None

    def test_super_register_nodes_called(self):
        """super().register_nodes() must be called first — initialize/finalize are framework defaults."""
        from src.graph.graph import Graph
        from framework.nodes.defaults.initialize_node import InitializeNode
        from framework.nodes.defaults.finalize_node import FinalizeNode
        graph = Graph()
        graph.register_nodes()
        assert isinstance(graph._nodes["initialize"], InitializeNode)
        assert isinstance(graph._nodes["finalize"], FinalizeNode)


class TestGraphInvoke:
    def setup_method(self):
        from src.graph.graph import Graph
        self.graph = Graph()
        self.graph.compile()
        # Context required for tests that exercise the full pipeline
        self._ctx = InvocationContext(
            caller_id="test-caller",
            caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
        )
        self._input_ctx = {
            "target_schema": SIMPLE_SCHEMA,
            "provider": "fallback",
            "model_config": {"model": "test-model", "max_retry": 3},
            "context_vars": {},
        }

    def test_full_pipeline_success(self):
        result = self.graph.invoke(
            "Generate a mock answer.",
            ctx=self._ctx,
            input_context=self._input_ctx,
        )
        assert result["status"] == AgentStatus.SUCCESS
        assert result.get("output") is not None

    def test_full_pipeline_error_on_empty_input(self):
        result = self.graph.invoke("", ctx=self._ctx, input_context=self._input_ctx)
        assert result["status"] in (AgentStatus.ERROR, AgentStatus.CANCELLED)

    def test_output_contains_required_keys(self):
        result = self.graph.invoke(
            "test query", ctx=self._ctx, input_context=self._input_ctx
        )
        assert "status" in result
        assert "output" in result
        assert "node_history" in result

    def test_node_history_all_five_nodes(self):
        result = self.graph.invoke(
            "test", ctx=self._ctx, input_context=self._input_ctx
        )
        history = result.get("node_history", [])
        for expected in (
            "InitializeNode",
            "PreProcessNode",
            "MainNode",
            "PostProcessNode",
            "FinalizeNode",
        ):
            assert expected in history, f"Expected {expected} in node_history: {history}"

    def test_invoke_with_session_id(self):
        result = self.graph.invoke(
            "test", session_id="my-session-123", ctx=self._ctx, input_context=self._input_ctx
        )
        assert result["status"] == AgentStatus.SUCCESS

    def test_invoke_with_invocation_context(self):
        ctx = InvocationContext(
            caller_id="test-caller",
            caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
        )
        result = self.graph.invoke(
            "test query", ctx=ctx, input_context=self._input_ctx
        )
        assert result["status"] == AgentStatus.SUCCESS

    def test_invoke_with_input_context(self):
        result = self.graph.invoke(
            "test", ctx=self._ctx, input_context={**self._input_ctx, "channel": "api"}
        )
        assert result["status"] == AgentStatus.SUCCESS

    def test_graph_is_not_run_method(self):
        """Verify .invoke() exists and .run() is NOT used (framework contract)."""
        assert hasattr(self.graph, "invoke")
        assert not hasattr(self.graph, "run"), (
            "Graph must use .invoke(), not .run()"
        )

    def test_add_edges_not_overridden(self):
        """Graph must not override add_edges() — framework owns the 5-node backbone."""
        from src.graph.graph import Graph
        from framework.graph.agent_base_graph import AgentBaseGraph
        assert "add_edges" not in Graph.__dict__, (
            "Graph must not override add_edges() — framework owns edge wiring"
        )

    def test_route_not_overridden(self):
        """Graph must not override route() — framework owns RETRY/SUCCESS/ERROR routing."""
        from src.graph.graph import Graph
        assert "route" not in Graph.__dict__, (
            "Graph must not override route() — framework owns routing"
        )


class TestGraphBusinessLogic:
    """BL-01, BL-02: Business logic tests."""

    def setup_method(self):
        from src.graph.graph import Graph
        self.graph = Graph()
        self.graph.compile()
        self._ctx = InvocationContext(
            caller_id="test-caller",
            caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
        )
        self._input_ctx = {
            "target_schema": SIMPLE_SCHEMA,
            "provider": "fallback",
            "model_config": {"model": "test-model", "max_retry": 3},
            "context_vars": {},
        }

    def test_bl01_non_empty_input_produces_result(self):
        """BL-01: A valid non-empty user_input produces a non-None output."""
        result = self.graph.invoke(
            "What is the capital of France?",
            ctx=self._ctx,
            input_context=self._input_ctx,
        )
        assert result["status"] == AgentStatus.SUCCESS
        assert result.get("output") is not None

    def test_bl02_whitespace_only_input_produces_error(self):
        """BL-02: Whitespace-only user_input is treated as empty and rejected."""
        result = self.graph.invoke(
            "   ", ctx=self._ctx, input_context=self._input_ctx
        )
        assert result["status"] in (AgentStatus.ERROR, AgentStatus.CANCELLED)


class TestRetryTerminates:
    """Regression: a persistently failing LLM must stop after max_retry, not loop forever."""

    @pytest.mark.parametrize("reply", [{"content": "not json"}, {"content": '{"wrong": 1}'}])
    def test_persistent_failure_is_bounded(self, monkeypatch, clear_mock_flags, reply):
        from framework.schemas.invocation_context import InvocationContext
        from framework.schemas.trust_level import TrustLevel
        from framework.secrets.context import bound_secrets
        from shared.secrets.inmemory_provider import InMemoryProvider
        from src.graph.graph import Graph
        from src.services import llm_provider_service

        calls: list[int] = []

        def _failing_llm(*args, **kwargs):
            calls.append(1)
            return reply

        monkeypatch.setattr(llm_provider_service, "call_llm", _failing_llm)
        monkeypatch.setenv("USE_MOCK_SCHEMA_REGISTRY", "true")
        secrets = InMemoryProvider(
            {
                "AZURE_OPENAI_API_KEY": "mock-api-key-for-testing",
                "AZURE_OPENAI_ENDPOINT": "https://example.services.ai.azure.com",
                "AZURE_OPENAI_DEPLOYMENT": "mock-deployment",
            }
        )
        agent = Graph(config={"max_retry": 3})
        agent.compile()
        agent.provision_secrets(secrets)
        with bound_secrets(secrets):
            ctx = InvocationContext(session_id="s", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="t")
            result = agent.invoke(
                "question",
                ctx=ctx,
                input_context={
                    "target_schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    }
                },
            )
        assert result["status"] in ("error", "retry")
        assert len(calls) == 3, "the router stops once retry_count reaches max_retry=3"
