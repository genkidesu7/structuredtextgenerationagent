"""AgentCore Platform v1.0"""

# Cat 1 — Single technical capability: provider-agnostic structured text generation.
# Framework: AgentBaseGraph (fixed 5-node pipeline, no autonomous loop needed).
# Pipeline: START → initialize → pre_process → main → {route} → post_process → finalize → END
#                                                          ↓ (RETRY, up to max_retry=3)
#                                                       pre_process
#
# Rules:
#   - super().register_nodes() MUST be called first (injects initialize + finalize)
#   - add_edges() MUST NOT be overridden (framework owns the 5-node backbone)
#   - route() MUST NOT be overridden (framework reads state["status"])
#   - .invoke() ONLY — never .run()
#   - No agenticstar imports — PB-4

from framework.graph.agent_base_graph import AgentBaseGraph

from src.nodes.post_process_node import PostProcessNode
from src.nodes.main_node import MainNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State


class StructuredTextGenerationAgent(AgentBaseGraph):
    """Cat 1 agent: provider-agnostic structured text generation with auto-retry."""

    @property
    def name(self) -> str:
        return "cmn_c1_517"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        # super() injects "initialize" (sets session_id, correlation_id,
        # caller_trust_level, schema_version) and "finalize" (builds
        # response_metadata, total_time_ms) automatically.
        super().register_nodes()

        # Domain pipeline slots (required by AgentBaseGraph.compile())
        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = MainNode()
        self._nodes["post_process"] = PostProcessNode()


# Conventional alias — the tests and the deploy tooling import `Graph`.
Graph = StructuredTextGenerationAgent
