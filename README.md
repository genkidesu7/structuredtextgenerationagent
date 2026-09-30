# CMN-C1-517 — StructuredTextGenerationAgent

> **Category**: Cat 1 (Single technical capability, use-case-agnostic)
> **Industry**: CMN (Common / Cross-Industry)

## Overview

Provider-agnostic structured text generation with schema validation and automatic retry.

Given a prompt and a JSON Schema (draft-07), the agent asks the LLM for a JSON object, validates the
reply against the schema, and — when validation fails — feeds the validation error back into the
prompt and retries (3 attempts by default, hard ceiling of 10). Every result carries the `schema_id`
(SHA-256 of the canonical schema), `schema_version`, attempt count and token usage.

- **Schema-constrained output**: the `provider` request field selects how the model is constrained —
  `openai` uses `response_format` `json_schema` (strict); `anthropic`, `gemini` and `fallback` use
  JSON-object mode with the schema embedded in the prompt. All calls are served by the Azure OpenAI
  deployment configured for the agent.
- **Validation and retry**: `jsonschema` validation on every attempt, with the last validation error
  injected into the next prompt.
- **Schema allowlist**: callers may only generate against registered schemas (S-1 gate).
- **Security gates**: caller trust check, PII and credential scanning on node input and output, and
  audit events for every generation attempt.
- **Progress events**: every node reports a progress update while it runs.

Request fields (`input` is the prompt; `target_schema` is required):

| Field | Purpose |
|---|---|
| `input` | Prompt text; may contain Jinja2 placeholders filled from `context_vars` |
| `target_schema` | JSON Schema the output must conform to |
| `provider` | Constraint style: `openai`, `anthropic`, `gemini` or `fallback` (default) |
| `model_config` | Generation settings: `temperature`, `max_tokens` |
| `context_vars` | Values for the prompt template |

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| **Azure OpenAI deployment** | Provide `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` (bare resource endpoint, no `/openai` path) and `AZURE_OPENAI_DEPLOYMENT` as agent secrets. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent raises
`PlatformRequired` during graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

The test suite runs in mock mode (`USE_MOCK=true`, set automatically): the LLM call returns a stub that
conforms to the requested schema, so no Azure OpenAI secrets are needed. Tests marked `real_llm` call
the real deployment and run only when you export `USE_MOCK=false` together with the three
`AZURE_OPENAI_*` variables.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent manifest (agent.yaml) and runtime settings (config.yaml)
docs/         design and test specification
cli.py        AGENTIC STAR Marketplace entry point
agenticstar-deploy/   container image build toolchain
```

See `docs/02_design.md` and `docs/03_test_spec.md` for the design and the test specification.

## Customising

1. Adjust `config/` for your own environment and policies (`max_retry`, `timeout_s`, required trust level).
2. Register the schemas your callers are allowed to use with the schema registry service
   (`src/services/schema_registry_service.py`) and replace the sample schemas in the tests.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.

---
