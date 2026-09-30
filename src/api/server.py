"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# For platform-level routing, AgentGateway calls agent.invoke() directly.

import asyncio
import os
import secrets
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ConfigDict, Field

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.utils.config_loader import load_config
from shared.bootstrap.marketplace_app import build_marketplace_secret_provider
from src.graph.graph import StructuredTextGenerationAgent

app = FastAPI(title="StructuredTextGenerationAgent")

# Same config_dir / "config.yaml" convention as AgentRegistry._compile_and_cache()
# — absent config.yaml is tolerated, matching the registry's own guard. Without
# this the adapter would run with config={} and max_retry etc. would never reach
# the graph on this path.
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_config = load_config(str(_CONFIG_PATH)) if _CONFIG_PATH.exists() else {}

# Same EnvProvider-first / DotenvProvider-fallback chain cli.py's
# run_agent_marketplace() builds, so a secret injected as a process env var
# (Marketplace "User-defined Environment Variables") resolves on this path too.
# namespace / agent_name match config/agent.yaml.
_secrets_provider = build_marketplace_secret_provider(agent_name="StructuredTextGenerationAgent", namespace="cmn")

agent = StructuredTextGenerationAgent(config=_config)

# Mirror AgentRegistry._compile_and_cache()'s conditional checkpointer — hitl.enabled
# or memory_enabled needs one, or interrupt()/memory silently no-ops on this path.
# This process runs exactly one agent, so a private MemorySaver per process is the
# standalone-path equivalent of the registry's shared checkpointer.
_hitl_enabled = agent.config.get("hitl", {}).get("enabled", False)
_needs_checkpointer = agent.config.get("memory_enabled") or _hitl_enabled
agent.compile(checkpointer=MemorySaver() if _needs_checkpointer else None)
agent.provision_secrets(_secrets_provider)


class InvokeRequest(BaseModel):
    # `model_config` is reserved by pydantic, so the request key of that name is
    # exposed as `model_settings` via its alias.
    model_config = ConfigDict(populate_by_name=True)

    input: str
    session_id: str = ""
    target_schema: dict[str, Any] = Field(default_factory=dict)
    provider: str = "fallback"
    model_settings: dict[str, Any] = Field(default_factory=dict, alias="model_config")
    context_vars: dict[str, Any] = Field(default_factory=dict)


def _bearer_matches(supplied: str, expected: str) -> bool:
    """Constant-time bearer comparison that is safe for non-ASCII header input."""
    return secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode())


def _resolve_standalone_trust(
    current: TrustLevel, authorization: str, invoke_auth_token: str | None, internal_runner_token: str | None
) -> TrustLevel:
    """Authenticate standalone callers without allowing external-token elevation.

    STG_INTERNAL_RUNNER_TOKEN is a distinct, CI-generated deployment credential.
    It is considered only for an anonymous caller and maps exactly to INTERNAL;
    INVOKE_AUTH_TOKEN remains VERIFIED_EXTERNAL. Middleware-established trust is
    never changed.
    """
    if current is not TrustLevel.ANONYMOUS:
        return current
    if internal_runner_token and _bearer_matches(authorization, internal_runner_token):
        return TrustLevel.INTERNAL
    if invoke_auth_token and _bearer_matches(authorization, invoke_auth_token):
        return TrustLevel.VERIFIED_EXTERNAL
    if internal_runner_token or invoke_auth_token:
        raise HTTPException(status_code=401, detail="Token is invalid or expired.")
    return TrustLevel.ANONYMOUS


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> dict[str, Any]:
    # This adapter is the entry-point auth boundary (standalone equivalent of
    # platform AuthMiddleware). Both values are deployment-level caller credentials,
    # not agent secrets: no InvocationContext exists before this boundary, so
    # ctx.secrets cannot apply.
    trust = _resolve_standalone_trust(
        getattr(request.state, "trust_level", TrustLevel.ANONYMOUS),
        request.headers.get("authorization", ""),
        os.environ.get("INVOKE_AUTH_TOKEN"),
        os.environ.get("STG_INTERNAL_RUNNER_TOKEN"),
    )
    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        # agent.invoke() is synchronous and performs a blocking LLM call; running it
        # on the event loop would stall every other request and breaks any node that
        # uses asyncio.run(). to_thread copies the current contextvars.Context, so the
        # bound_secrets() binding stays visible in the worker thread.
        return await asyncio.to_thread(
            agent.invoke,
            req.input,
            ctx=ctx,
            input_context={
                "target_schema": req.target_schema,
                "provider": req.provider,
                "model_config": req.model_settings,
                "context_vars": req.context_vars,
            },
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "StructuredTextGenerationAgent"}
