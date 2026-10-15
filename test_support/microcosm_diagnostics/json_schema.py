"""Check calibration-diagnostics documents the way a JSON Schema intake would."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping

from jsonschema import Draft202012Validator, ValidationError

from microcosm import diagnostics


def _refuse_non_finite_literal(literal: str) -> float:
    raise ValueError(f"strict JSON has no {literal} literal")


def load_strict_json(text: str | bytes) -> object:
    """Decode RFC 8259 JSON, refusing the NaN and Infinity tokens Python accepts."""

    return json.loads(text, parse_constant=_refuse_non_finite_literal)


def _leaf_errors(errors: Iterable[ValidationError]) -> Iterator[ValidationError]:
    """Descend into ``anyOf`` branches so each error names its innermost path."""

    for error in errors:
        if error.context:
            yield from _leaf_errors(error.context)
        else:
            yield error


def json_schema_errors(document: object, schema: Mapping[str, object]) -> list[str]:
    """Return every ``<JSON path>: <message>`` error, sorted for stable output."""

    validator = Draft202012Validator(schema)
    return sorted(
        f"{error.json_path}: {error.message}"
        for error in _leaf_errors(validator.iter_errors(document))
    )


def published_json_schema_errors(document: object) -> list[str]:
    """Return the errors against the committed schema-8 file consumers read."""

    return json_schema_errors(
        document, diagnostics.load_calibration_diagnostics_v8_json_schema()
    )
