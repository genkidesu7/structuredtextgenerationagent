"""Stub data served through the mock-mode gate must satisfy the schema it was built for."""

import jsonschema
import pytest

from src.services.mock_data import STUB_TOKEN_UNITS, stub_generate_structured, stub_structured_output

SCHEMAS = [
    {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
    {"type": "object", "properties": {"n": {"type": "integer", "minimum": 3}, "ok": {"type": "boolean"}}},
    {
        "type": "object",
        "properties": {
            "kind": {"enum": ["a", "b"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "nested": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
            },
        },
        "required": ["kind", "nested"],
    },
    {"type": "object"},
]


@pytest.mark.parametrize("schema", SCHEMAS)
def test_stub_output_conforms_to_schema(schema):
    jsonschema.validate(instance=stub_structured_output(schema), schema=schema)


def test_stub_generate_structured_returns_units():
    output, units = stub_generate_structured(SCHEMAS[0])
    assert output == {"answer": "mock_answer"}
    assert units == STUB_TOKEN_UNITS
