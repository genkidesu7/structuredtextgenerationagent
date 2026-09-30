# Template Design Specification

**Status**: completed
**Date**: 2026-06-29
**Reviewer**: TBD

---

## Position in AgentCore Architecture

- **Agent Class**: StructuredTextGenerationAgent
- **L1 Base**: AgentBaseGraph
- **Three-Layer Separation**:
  - State: flat TypedDict composition (no Pydantic — msgpack incompatible)
  - Node: L1 inheritance (Template Method: `execute(self, state: dict) -> dict` override only)
  - Graph: composition (`register_nodes()` for node substitution)

---

## Architecture Overview

StructuredTextGenerationAgent is a **Cat 1** single-capability agent that generates LLM output conforming to a caller-supplied JSON schema. It uses provider-native constrained decoding (Anthropic `output_config`, OpenAI `json_schema` strict mode, Google Gemini schema mode, with fallback to function-calling for local models) and automatically retries with validation error feedback when schema validation fails.

### Graph Type Selection

**Selected: `AgentBaseGraph`**

Rationale: This is a fixed request/response pipeline with no autonomous reasoning loop. The agent receives a prompt + schema, calls an LLM, validates the output, and returns — all in a deterministic 5-node backbone. `AutonomousBaseGraph` is not required.

### Node Configuration

| Node | Responsibility | Input State | Output State | Inherits |
|------|---------------|-------------|--------------|----------|
| initialize | Set session_id, correlation_id, caller_trust_level, schema_version | (framework) | (framework) | InitializeNode (default) |
| pre_process | Validate request: check `user_input`, `target_schema`, `provider`; render Jinja2 prompt; enforce schema allowlist (S-1); sanitize context_vars | `user_input`, `target_schema`, `provider`, `model_config`, `context_vars` | `validated_input`, `rendered_prompt`, `schema_id`, `schema_version` | FunctionNode |
| main | Route to provider-native structured generation; validate JSON output against schema; set RETRY on validation failure; emit S-4 audit | `rendered_prompt`, `target_schema`, `provider`, `model_config`, `schema_id`, `schema_version`, `retry_count`, `last_validation_error` | `result`, `status`, `tokens_used`, `attempt_count`, `last_validation_error` | FunctionNode |
| post_process | Shape final response; scrub any PII from output; emit generation audit record (S-4) | `result`, `schema_id`, `schema_version`, `tokens_used`, `attempt_count` | `formatted_output` | FunctionNode |
| finalize | Build response_metadata, total_time_ms | (framework) | (framework) | FinalizeNode (default) |

### Data Flow

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                            ↓ (RETRY, up to max_retry=3)
                                          pre_process
```

Routing logic (AgentBaseGraph.route() reads `state["status"]`):
- `SUCCESS` → `post_process`
- `RETRY` → `pre_process` (up to `max_retry` times; `last_validation_error` injected into re-render)
- `ERROR` → `finalize`

---

## State Schema

**File**: `src/schemas/state.py`

```python
import operator
from typing import Annotated
from framework.schemas.agent_state import AgentState


class State(AgentState):
    # --- Input fields (set by caller, read-only after pre_process) ---
    target_schema: dict           # JSON schema dict the output must conform to
    provider: str                 # "anthropic" | "openai" | "gemini" | "fallback"
    model_config: dict            # model name, temperature, max_tokens, etc.
    context_vars: dict            # Jinja2 template variables for prompt rendering

    # --- pre_process outputs ---
    validated_input: str          # sanitized user_input
    rendered_prompt: str          # Jinja2-rendered prompt string
    schema_id: str                # stable identifier for the schema (hash or registry key)
    schema_version: str           # caller-supplied version tag

    # --- main outputs ---
    tokens_used: int              # total tokens consumed in this invocation
    attempt_count: int            # number of generation attempts (1 + retries)
    last_validation_error: str    # last jsonschema validation error message (for retry feedback)

    # --- Accumulator fields ---
    validation_errors: Annotated[list[str], operator.add]  # all validation errors across retries
```

**State Constraints (mandatory):**
- Flat TypedDict only (primitives + JSON-serializable types)
- No JWT, API keys, credentials in State (checkpoint DB leakage)
- `InvocationContext` via `config["configurable"]` only (not in State)
- No Pydantic models, dataclass, arbitrary Python objects (msgpack incompatible)
- `input_context` is read-only — never written by nodes

---

## Nodes Definition

**File layout**: one node per file under `src/nodes/`.

### PreProcessNode (`src/nodes/pre_process_node.py`)

- **Purpose**: Validate the incoming request, enforce the schema allowlist (S-1 domain gate), render the Jinja2 prompt template with `context_vars`, and extract schema identity fields.
- **Input**: `state["user_input"]`, `state["target_schema"]`, `state["provider"]`, `state["model_config"]`, `state["context_vars"]`, `state["last_validation_error"]` (non-empty on retry)
- **Output**: `validated_input`, `rendered_prompt`, `schema_id`, `schema_version`
- **Logic**:
  1. Verify `target_schema` is a valid JSON Schema (draft-07 meta-schema check via `jsonschema`). Raise `SecurityViolationError` if schema is not on allowlist.
  2. Sanitize `user_input` — escape Jinja2 delimiters to prevent prompt injection.
  3. Render prompt via `PromptTemplateService`: substitute `context_vars` and, if `last_validation_error` is set, append the error as a correction instruction.
  4. Compute `schema_id` (SHA-256 of canonical schema JSON) and extract `schema_version` from `input_context` or `context_vars`.
  5. Emit `_extra_security_gate_input` check for disallowed schema patterns (e.g. recursive `$ref` depth > 5 → Tier 3 warning).
- **Error Cases**:
  - Invalid JSON Schema → `SecurityViolationError` → `status = ERROR`
  - Schema not on allowlist → `SecurityViolationError` → `status = ERROR`
  - Jinja2 template render failure → log error → `status = ERROR`
- **Dependencies**: `PromptTemplateService`, `SchemaRegistryService`

### MainNode (`src/nodes/main_node.py`)

- **Purpose**: Route the generation request to the correct provider-native structured output API, validate the response against `target_schema`, and emit an S-4 audit event.
- **Input**: `state["rendered_prompt"]`, `state["target_schema"]`, `state["provider"]`, `state["model_config"]`, `state["schema_id"]`, `state["schema_version"]`, `state["retry_count"]`
- **Output**: `result` (parsed dict), `status` (`SUCCESS` / `RETRY` / `ERROR`), `tokens_used`, `attempt_count`, `last_validation_error`
- **Logic**:
  1. Obtain `InvocationContext` via `config["configurable"]["invocation_context"]`; get credential handle for the LLM provider.
  2. Delegate to `LLMProviderService.generate_structured(...)`:
     - `anthropic` → `output_config` with JSON schema
     - `openai` → `response_format = {"type": "json_schema", "json_schema": {...}, "strict": True}`
     - `gemini` → `generation_config.response_schema`
     - `fallback` → function-calling with schema as function parameter spec (XGrammar / Cohere)
  3. Parse raw LLM response to dict.
  4. Validate against `target_schema` using `jsonschema.validate()`.
     - Pass → `status = AgentStatus.SUCCESS`
     - Fail and `retry_count < max_retry` → set `last_validation_error`, `status = AgentStatus.RETRY`
     - Fail and `retry_count >= max_retry` → `status = AgentStatus.ERROR`
  5. Accumulate `tokens_used`; increment `attempt_count`.
  6. Emit `emit_trace_event("structured_generation_attempt", {...}, state)`.
- **Error Cases**:
  - LLM API error (rate limit, timeout) → transient → `status = RETRY` (bounded by `max_retry`)
  - Permanent API error (auth, bad request) → `status = ERROR`
  - `retry_count >= MAX_RETRY_CEILING (10)` → force `status = ERROR`
- **Dependencies**: `LLMProviderService`, `SchemaRegistryService`, `InvocationContext`

### PostProcessNode (`src/nodes/post_process_node.py`)

- **Purpose**: Shape the final structured output, apply output-layer PII scrubbing (`_extra_security_gate_output`), and emit the generation audit record.
- **Input**: `state["result"]`, `state["schema_id"]`, `state["schema_version"]`, `state["tokens_used"]`, `state["attempt_count"]`, `state["validation_errors"]`
- **Output**: `formatted_output` (dict with `data`, `schema_id`, `schema_version`, `attempt_count`, `tokens_used`)
- **Logic**:
  1. Apply `_extra_security_gate_output`: scan result dict values for credential-pattern strings (belt-and-suspenders, S-3 framework gate already ran).
  2. Assemble `formatted_output`:
     ```python
     {
         "data": state["result"],
         "schema_id": state["schema_id"],
         "schema_version": state["schema_version"],
         "attempt_count": state["attempt_count"],
         "tokens_used": state["tokens_used"],
     }
     ```
  3. Emit `emit_trace_event("generation_audit", {schema_id, schema_version, model, attempt_count, tokens_used, timestamp}, state)` — S-5 audit record.
- **Error Cases**: Malformed `result` (not dict) → log and return error-shaped `formatted_output`.
- **Dependencies**: `AuditSink`

---

## Graph Composition

**File**: `src/graph/graph.py`

```python
from framework.graph.agent_base_graph import AgentBaseGraph
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode
from src.schemas.state import State


class StructuredTextGenerationAgent(AgentBaseGraph):

    @property
    def name(self) -> str:
        return "cmn_c1_517"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        super().register_nodes()  # injects InitializeNode + FinalizeNode
        self._nodes["pre_process"]  = PreProcessNode()
        self._nodes["main"]         = MainNode()
        self._nodes["post_process"] = PostProcessNode()
```

**Rules followed:**
- `super().register_nodes()` called first
- `add_edges()` NOT overridden (framework owns the 5-node backbone)
- `route()` NOT overridden (AgentBaseGraph.route() reads `status`)
- `.invoke()` used, never `.run()`

---

## Services

**File layout**: `src/services/`

### LLMProviderService (`src/services/llm_provider_service.py`)

- **Purpose**: Structured output request shaping and response parsing. Every real call goes through `AzureOpenAIClient` (via `src/services/service.py::call_llm`); the `provider` field selects the constraint style — `openai` uses `response_format` `json_schema` (strict), while `anthropic`, `gemini` and `fallback` use JSON-object mode with the schema embedded in the prompt.
- **Interface**: `generate_structured(prompt, schema, provider, model_config, credential_handle) -> (dict, int)`  — returns parsed dict + tokens used.
- **Mock mode**: `mock_mode_enabled()` (`src/services/mock_mode.py`, true when `USE_MOCK` or `STG_MOCK_MODE` is set) returns a schema-conforming stub from `src/services/mock_data.py` and never reads a credential.
- **No credentials in service**: accepts `credential_handle` (opaque); runtime resolves value.

### SchemaRegistryService (`src/services/schema_registry_service.py`)

- **Purpose**: Schema allowlist enforcement and versioning. Validates that the caller-supplied `target_schema` is registered and not revoked. Computes stable `schema_id`.
- **Interface**: `validate_allowlist(schema) -> bool`, `compute_schema_id(schema) -> str`.
- **Mock mode**: allows any schema through when `mock_mode_enabled()` is true or `USE_MOCK_SCHEMA_REGISTRY` is set; otherwise only schemas added with `register_schema()` pass.

### PromptTemplateService (`src/services/prompt_template_service.py`)

- **Purpose**: Jinja2-based prompt rendering with input sanitization. Escapes Jinja2 delimiters in `context_vars` values to prevent prompt injection.
- **Interface**: `render(template_str, context_vars, validation_error=None) -> str`.
- **No external dependencies beyond Jinja2.**

---

## Framework Utilization

### Shared Components Used
- [x] `InvocationContext` (via `config["configurable"]["invocation_context"]`) — credential handle, trust level
- [x] `AuditSink` / `emit_trace_event` — S-4 generation audit and attempt logging
- [x] `SecurityViolationError` — schema allowlist violation, prompt injection detection
- [x] `_security_gate_input()` (PII detection — automatic via FunctionNode, mandatory)
- [x] `_security_gate_output()` (credential pattern scan — automatic via FunctionNode, mandatory)
- [x] `_extra_security_gate_input()` — domain extension: schema allowlist + Tier 3 complexity warning
- [x] `_extra_security_gate_output()` — domain extension: belt-and-suspenders credential pattern scan on generated JSON

### Composition Pattern

- **Pattern**: Standalone
- **Composition target**: N/A (Cat 1 primitive; composed BY Cat 2/3 agents via `RemoteAgentNode` or `GraphNode`)
- **Error propagation strategy**: N/A (standalone; caller receives `ERROR` status in `formatted_output`)

---

## Security Design

| Layer | Implementation | Detail |
|-------|---------------|--------|
| S-1 | `required_trust_level = TrustLevel.VERIFIED_EXTERNAL` on MainNode | Callers must present a verified external credential; anonymous calls refused |
| S-2 | `_security_gate_input()` (automatic, FunctionNode) + `_extra_security_gate_input()` (schema allowlist) | PII detection on `user_input`/`validated_input`; schema allowlist enforcement |
| S-3 | `_security_gate_output()` (automatic, FunctionNode) + `_extra_security_gate_output()` (credential scan on JSON values) | Credential pattern scan on generated dict values |
| S-4 | `emit_trace_event("structured_generation_attempt", ...)` in MainNode; `emit_trace_event("generation_audit", ...)` in PostProcessNode | Every attempt logged; full audit record on completion |
| S-5 | `__init_subclass__` (framework automatic) | JWT pattern detection at import time |

**Prompt injection mitigation**: `PromptTemplateService` escapes `{{` / `}}` Jinja2 delimiters in all `context_vars` values before rendering.

**Schema complexity tiers** (documented, auto-detected):
- Tier 1: flat object, ≤10 properties → native constrained decoding, all providers
- Tier 2: nested object, depth ≤3 → native constrained decoding, warn on Gemini
- Tier 3: recursive `$ref`, `anyOf`/`oneOf` depth > 5 → warning logged; fallback to function-calling mode

---

## Error Handling

| Scenario | Node | Action |
|----------|------|--------|
| Invalid JSON Schema supplied | PreProcessNode | `SecurityViolationError` → `status = ERROR` → skip to finalize |
| Schema not on allowlist | PreProcessNode | `SecurityViolationError` → `status = ERROR` |
| Jinja2 render failure | PreProcessNode | log error → `status = ERROR` |
| LLM validation failure (schema mismatch) | MainNode | inject error into prompt, `status = RETRY` (up to max_retry) |
| LLM API transient error | MainNode | `status = RETRY` |
| LLM API permanent error | MainNode | `status = ERROR` |
| max_retry exhausted | MainNode | `status = ERROR`, surface `last_validation_error` in `formatted_output` |

---

## Performance Requirements

| Metric | Target |
|--------|--------|
| P95 latency (single attempt, Tier 1 schema) | < 5 s |
| P95 latency (with 1 retry) | < 10 s |
| Max retries | 3 (configurable via `agent.yaml config.max_retry`) |
| Max token budget | Enforced via `model_config.max_tokens` (caller-supplied) |
| Schema compliance rate | ≥ 99.8% (native constrained decoding path) |

---

## Import Isolation Confirmation
- [x] Template does not import agenticstar-platform SDK (Level 0)
- [x] Import targets: `framework/` and `shared/` only
- [x] Services have no `agenticstar` imports
- [x] No cross-template imports

---

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed deterministic pipeline; no autonomous reasoning loop needed |
| Retry mechanism | Custom loop in MainNode | Framework RETRY routing | **Framework RETRY routing** | `status = RETRY` routes back to pre_process, re-rendering prompt with error feedback — uses built-in `max_retry` ceiling |
| Provider routing | Single provider only | Multi-provider routing service | **Multi-provider LLMProviderService** | Proposal requirement; single interface for Anthropic/OpenAI/Gemini/fallback |
| Schema validation | Prompt-only JSON mode | Native constrained decoding + jsonschema | **Native constrained decoding + jsonschema** | Industry standard 2026; 99.8%+ compliance vs 85-92% prompt-only |
| Composition pattern | Standalone | GraphNode (subgraph) | **Standalone** | Cat 1 primitive designed to be composed BY other agents, not to contain a subgraph |
| Schema allowlist | Optional | Required (S-1 domain gate) | **Required** | Prevents injection of arbitrary recursive schemas that could cause excessive LLM calls or parser failures |
| Credential handling | `os.environ` direct read | `InvocationContext.secrets` | **InvocationContext.secrets** | Framework mandate, prevents credential leakage to State/logs |
