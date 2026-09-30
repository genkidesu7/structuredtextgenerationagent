"""HTTP-level tests: the real ASGI app on a real event loop, and the CI repro end-to-end."""

import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import app
from src.nodes import main_node
from tests.conftest import SIMPLE_SCHEMA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVOKE_TOKEN = "mock-invoke-token-for-testing"


@pytest.fixture
def verified_caller(monkeypatch):
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", INVOKE_TOKEN)
    monkeypatch.delenv("STG_INTERNAL_RUNNER_TOKEN", raising=False)
    return {"Authorization": f"Bearer {INVOKE_TOKEN}"}


@pytest.mark.asyncio
async def test_invoke_succeeds_over_http_in_mock_mode(monkeypatch, verified_caller):
    """agent.invoke() is sync; the route must not run it on the event loop.

    Regression for the deploy-stg bug class: a 200 response with
    status "error" hides a sync/async boundary failure.
    """
    monkeypatch.setenv("USE_MOCK", "true")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/invoke",
            headers=verified_caller,
            json={"input": "What is the capital of France?", "target_schema": SIMPLE_SCHEMA},
        )
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "success", body
    assert body["output"]["data"] == {"answer": "mock_answer"}


@pytest.mark.asyncio
async def test_invoke_succeeds_under_stg_mock_mode_alias(monkeypatch, clear_mock_flags, verified_caller):
    monkeypatch.setenv("STG_MOCK_MODE", "true")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/invoke", headers=verified_caller, json={"input": "hello", "target_schema": SIMPLE_SCHEMA}
        )
    assert resp.json()["status"] == "success"


@pytest.mark.asyncio
async def test_model_config_request_key_reaches_the_llm_service(monkeypatch, verified_caller):
    """`model_config` is reserved by pydantic; the request key must still be honoured."""
    monkeypatch.setenv("USE_MOCK", "true")
    seen: dict = {}
    real = main_node._llm_service.generate_structured

    def _spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(main_node._llm_service, "generate_structured", _spy)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/invoke",
            headers=verified_caller,
            json={
                "input": "hello",
                "target_schema": SIMPLE_SCHEMA,
                "provider": "openai",
                "model_config": {"temperature": 0.25},
            },
        )
    assert resp.json()["status"] == "success"
    assert seen["model_config"] == {"temperature": 0.25}
    assert seen["provider"] == "openai"


@pytest.mark.asyncio
async def test_bad_token_is_rejected(monkeypatch, verified_caller):
    monkeypatch.setenv("USE_MOCK", "true")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/invoke",
            headers={"Authorization": "Bearer wrong"},
            json={"input": "hello", "target_schema": SIMPLE_SCHEMA},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.json() == {"status": "ok", "agent": "StructuredTextGenerationAgent"}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_deploy_stg_repro_passes_with_only_stg_mock_mode(tmp_path):
    """Boots uvicorn exactly as deploy-stg does: clean env, only STG_MOCK_MODE + a token.

    No .env, no env/ secret tree, no USE_MOCK — the condition a fresh CI runner has.
    """
    port = _free_port()
    token = secrets.token_urlsafe(32)
    python_bin_dir = str(Path(sys.executable).parent)
    server_env = {
        "PATH": f"{python_bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "STG_MOCK_MODE": "true",
        "INVOKE_AUTH_TOKEN": token,
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.server:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=PROJECT_ROOT,
        env=server_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.5)
        else:
            pytest.fail("uvicorn did not become healthy")

        payload = json.loads((PROJECT_ROOT / "deploy" / "invoke_payload.json").read_text())
        resp = httpx.post(
            f"http://127.0.0.1:{port}/invoke",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success", resp.json()

        out = tmp_path / "evidence.json"
        subprocess.run(
            [
                sys.executable,
                "scripts/stg_invoke_evidence.py",
                "--mode",
                "provisional",
                "--url",
                f"http://127.0.0.1:{port}",
                "--out",
                str(out),
            ],
            cwd=PROJECT_ROOT,
            env={**server_env, "PATH": server_env["PATH"]},
            check=False,
            capture_output=True,
        )
        evidence = json.loads(out.read_text())
        failed = [a["name"] for a in evidence["assertions"] if not a["passed"]]
        assert evidence["overall"] == "PASS", failed
    finally:
        server.terminate()
        server.wait(timeout=10)
