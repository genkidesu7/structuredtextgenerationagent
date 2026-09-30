"""AGENTIC STAR Marketplace entrypoint — one-shot Pod process.

Referenced by agenticstar-deploy/Dockerfile's `CMD ["python", "cli.py"]`.
Compiles the agent, provisions its secrets, then hands off to
shared.bootstrap.marketplace_app for the Marketplace lifecycle (identity,
input, events, terminal delivery, exit).
"""

from pathlib import Path

from framework.utils.config_loader import load_agent_config
from shared.bootstrap.marketplace_app import run_agent_marketplace
from src.graph.graph import StructuredTextGenerationAgent

# Add config overrides here if needed, instead of editing config/config.yaml.
extend_config: dict = {}

if __name__ == "__main__":
    run_agent_marketplace(
        StructuredTextGenerationAgent,
        agent_name="CMN-C1-517",  # matches config/agent.yaml `id`
        namespace="cmn",  # matches config/agent.yaml `namespace`
        config={**load_agent_config(Path(__file__).resolve().parent), **extend_config},
    )
