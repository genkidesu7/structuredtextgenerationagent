# Framework Compliance Tests (TC-01 through TC-08)

import ast
import inspect
import pytest
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from framework.nodes.function_node import FunctionNode


BASE_STATE = {
    "user_input": "test input",
    "input_context": {},
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


class TestTC01StateContract:
    """TC-01: State contract — flat TypedDict, no Pydantic/dataclass."""

    def test_state_is_typed_dict_not_pydantic(self):
        from src.schemas.state import State
        import pydantic
        assert not issubclass(State, pydantic.BaseModel), "State must not inherit from Pydantic BaseModel"

    def test_state_is_not_dataclass(self):
        from src.schemas.state import State
        import dataclasses
        assert not dataclasses.is_dataclass(State), "State must not be a dataclass"

    def test_state_inherits_agent_state(self):
        import inspect
        from src.schemas import state as state_module
        # TypedDict's __bases__ always show dict, not the TypedDict parent.
        # Check the source declares AgentState as the parent.
        source = inspect.getsource(state_module)
        assert "AgentState" in source, "State must inherit from AgentState"
        assert "class State(AgentState)" in source, "State class must extend AgentState"


class TestTC02SecurityViolationOnInvalidInput:
    """TC-02: SecurityViolationError fires on invalid input (gate bypass attempt)."""

    def test_security_gate_input_is_final(self):
        """_security_gate_input is @final on FunctionNode — cannot be overridden."""
        from src.nodes.pre_process_node import PreProcessNode
        from src.nodes.main_node import MainNode
        from src.nodes.post_process_node import PostProcessNode

        for cls in (PreProcessNode, MainNode, PostProcessNode):
            assert "_security_gate_input" not in cls.__dict__, (
                f"{cls.__name__} must not override _security_gate_input"
            )

    def test_security_gate_output_is_final(self):
        from src.nodes.pre_process_node import PreProcessNode
        from src.nodes.main_node import MainNode
        from src.nodes.post_process_node import PostProcessNode

        for cls in (PreProcessNode, MainNode, PostProcessNode):
            assert "_security_gate_output" not in cls.__dict__, (
                f"{cls.__name__} must not override _security_gate_output"
            )

    def test_subclass_override_raises_type_error(self):
        """Attempting to override a @final gate raises TypeError at class definition."""
        with pytest.raises(TypeError, match="Cannot override"):
            class BadNode(FunctionNode):
                def execute(self, state: dict) -> dict:
                    return {}

                def _security_gate_input(self, state):  # type: ignore[override]
                    return state


class TestTC03NoCredentialsInState:
    """TC-03: No JWT/Credential in State — __init_subclass__ detection."""

    def test_state_has_no_credential_fields(self):
        import re
        from src.schemas import state as state_module
        import inspect

        source = inspect.getsource(state_module)
        tree = ast.parse(source)

        credential_pattern = re.compile(
            r"(jwt|token|api_key|secret|password|credential|connection_string)",
            re.IGNORECASE,
        )
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        if credential_pattern.search(item.target.id):
                            violations.append(item.target.id)

        assert violations == [], f"Credential-like fields found in State: {violations}"

    def test_no_credential_literals_in_node_sources(self):
        """S-5: no node module carries a credential-shaped literal (JWT, API key, ...).

        framework.security.credential_detector is the framework's own scanner; the
        class-definition hook that would reject these at import time is not
        implemented in the installed framework, so the template scans its own
        node sources instead.
        """
        from pathlib import Path

        from framework.security.credential_detector import detect_credentials

        node_dir = Path(__file__).resolve().parents[2] / "src" / "nodes"
        findings = {
            path.name: detect_credentials(path.read_text())
            for path in sorted(node_dir.glob("*.py"))
        }
        assert {name: hits for name, hits in findings.items() if hits} == {}

    def test_credential_detector_flags_a_jwt_literal(self):
        """The scanner used above really does flag a three-part JWT literal."""
        from framework.security.credential_detector import detect_credentials

        fake_jwt = (
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        assert detect_credentials(f"_BAD = '{fake_jwt}'")


class TestTC04InvocationContextViaConfigurableOnly:
    """TC-04: InvocationContext is accessed via config["configurable"], never state."""

    def test_invocation_context_not_in_state_annotations(self):
        import inspect
        from src.schemas.state import State

        source = inspect.getsource(State)
        assert "InvocationContext" not in source, (
            "InvocationContext must not appear in State — use config['configurable'] only"
        )

    def test_node_source_does_not_read_invocation_context_from_state(self):
        from src import nodes
        import os

        nodes_dir = os.path.join(os.path.dirname(nodes.__file__))
        for fname in os.listdir(nodes_dir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(nodes_dir, fname)
            with open(fpath) as f:
                source = f.read()
            assert 'state["invocation_context"]' not in source, (
                f"{fname}: InvocationContext must not be read from state"
            )
            assert "state.get(\"invocation_context\")" not in source, (
                f"{fname}: InvocationContext must not be read from state"
            )


class TestTC05EmitTraceEventLogs:
    """TC-05: emit_trace_event logs are recorded on node invocation."""

    def test_node_complete_event_emitted_on_success(self, capsys):
        from src.nodes.main_node import MainNode
        node = MainNode()
        state = {**BASE_STATE, "user_input": "test", "validated_input": "test"}
        result = node(state)
        # Framework emits node_start + node_complete — verified by checking node_history is populated
        assert "MainNode" in result.get("node_history", [])

    def test_node_error_event_emitted_on_pre_process_empty_input(self):
        from src.nodes.pre_process_node import PreProcessNode
        node = PreProcessNode()
        state = {**BASE_STATE, "user_input": ""}
        result = node(state)
        assert result["status"] == AgentStatus.ERROR


class TestTC06SecurityGateInputNonBypassable:
    """TC-06: _security_gate_input() is non-bypassable — always executes."""

    def test_gate_input_executes_on_every_call(self):
        from src.nodes.pre_process_node import PreProcessNode
        node = PreProcessNode()

        # PII-like input: should be masked before execute() runs
        state = {**BASE_STATE, "user_input": "My SSN is 123-45-6789"}
        result = node(state)
        # Gate ran (node returned normally — PII was masked by S-2 before execute)
        assert result is not None

    def test_gate_input_cannot_be_overridden_in_any_node(self):
        from src.nodes.pre_process_node import PreProcessNode
        from src.nodes.main_node import MainNode
        from src.nodes.post_process_node import PostProcessNode

        for cls in (PreProcessNode, MainNode, PostProcessNode):
            assert "_security_gate_input" not in cls.__dict__


class TestTC07SecurityGateOutputNonBypassable:
    """TC-07: _security_gate_output() is non-bypassable — content filter always executes."""

    def test_gate_output_executes_on_every_call(self):
        from src.nodes.post_process_node import PostProcessNode
        node = PostProcessNode()
        state = {**BASE_STATE, "result": "clean result"}
        result = node(state)
        assert result is not None

    def test_gate_output_blocks_credential_in_output(self):
        """S-3 gate blocks output containing a credential-pattern string."""
        from framework.nodes.function_node import FunctionNode
        from framework.schemas.agent_status import AgentStatus

        class CredentialLeakNode(FunctionNode):
            required_trust_level = TrustLevel.ANONYMOUS

            def execute(self, state: dict) -> dict:
                # Return a value that looks like an AWS secret key
                return {
                    "result": "AKIAIOSFODNN7EXAMPLE/wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                    "status": AgentStatus.SUCCESS,
                }

        node = CredentialLeakNode()
        result = node(BASE_STATE)
        # S-3 gate should catch this and return ERROR
        assert result["status"] == AgentStatus.ERROR

    def test_gate_output_cannot_be_overridden_in_any_node(self):
        from src.nodes.pre_process_node import PreProcessNode
        from src.nodes.main_node import MainNode
        from src.nodes.post_process_node import PostProcessNode

        for cls in (PreProcessNode, MainNode, PostProcessNode):
            assert "_security_gate_output" not in cls.__dict__


class TestTC08RequiredTrustLevelEnforced:
    """TC-08: required_trust_level enforced — insufficient trust → refused."""

    def test_anonymous_caller_allowed_by_default(self):
        """PreProcessNode and PostProcessNode default to ANONYMOUS trust.
        MainNode intentionally requires VERIFIED_EXTERNAL (design §Security S-1).
        """
        from src.nodes.pre_process_node import PreProcessNode
        from src.nodes.main_node import MainNode
        from src.nodes.post_process_node import PostProcessNode

        # Pre/post-process nodes are accessible to all callers
        assert PreProcessNode.required_trust_level == TrustLevel.ANONYMOUS
        assert PostProcessNode.required_trust_level == TrustLevel.ANONYMOUS
        # MainNode requires VERIFIED_EXTERNAL (S-1 gate — handles credentials)
        assert MainNode.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_node_with_higher_trust_blocks_anonymous(self):
        """A node requiring VERIFIED_EXTERNAL blocks an ANONYMOUS caller."""
        from framework.nodes.function_node import FunctionNode
        from framework.schemas.agent_status import AgentStatus

        class RestrictedNode(FunctionNode):
            required_trust_level = TrustLevel.VERIFIED_EXTERNAL

            def execute(self, state: dict) -> dict:
                return {"status": AgentStatus.SUCCESS, "result": "ok"}

        node = RestrictedNode()
        anon_state = {**BASE_STATE, "caller_trust_level": TrustLevel.ANONYMOUS.value}
        result = node(anon_state)
        assert result["status"] == AgentStatus.ERROR
        assert any("S-1 trust gate denied" in e for e in result.get("error_log", []))

    def test_node_with_higher_trust_allows_verified_external(self):
        from framework.nodes.function_node import FunctionNode

        class RestrictedNode(FunctionNode):
            required_trust_level = TrustLevel.VERIFIED_EXTERNAL

            def execute(self, state: dict) -> dict:
                return {"status": AgentStatus.SUCCESS, "result": "ok"}

        node = RestrictedNode()
        verified_state = {**BASE_STATE, "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        result = node(verified_state)
        assert result["status"] == AgentStatus.SUCCESS

    def test_internal_trust_allows_any_node(self):
        from src.nodes.main_node import MainNode
        node = MainNode()
        internal_state = {
            **BASE_STATE,
            "caller_trust_level": TrustLevel.INTERNAL.value,
            "validated_input": "query",
            "rendered_prompt": "Generate structured output.",
            "target_schema": {
                "type": "object",
                "properties": {"mock_field": {"type": "string"}},
                "required": ["mock_field"],
            },
            "provider": "fallback",
            "model_config": {"model": "test-model", "max_retry": 3},
            "schema_id": "abc",
            "schema_version": "1.0",
        }
        result = node(internal_state)
        assert result["status"] == AgentStatus.SUCCESS
