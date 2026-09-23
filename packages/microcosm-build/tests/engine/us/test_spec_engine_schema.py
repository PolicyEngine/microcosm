"""Tests split from packages/microcosm-build/tests/test_spec_engine_schema.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.spec_engine_schema import *


def test_every_object_schema_is_closed_or_has_typed_map_values() -> None:
    """Refuse catch-all records that could make a normative field untyped."""
    __import__("policyengine_us")
    catalog = load_schema_registry()
    violations: list[str] = []

    def visit(value: object, *, location: str) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                additional = value.get("additionalProperties")
                unevaluated = value.get("unevaluatedProperties")
                closed = additional is False or unevaluated is False
                typed_map = isinstance(additional, dict) and bool(additional)
                if not closed and not typed_map:
                    violations.append(location)
            for key, child in value.items():
                visit(child, location=f"{location}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, location=f"{location}/{index}")

    for schema_id, schema in sorted(catalog.schemas.items()):
        visit(schema, location=schema_id)

    assert violations == []
