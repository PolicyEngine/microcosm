"""The JSON Schema generated for calibration diagnostics schema 8.

The pydantic models stay the authority. These tests pin what the generated
schema accepts and refuses, and keep its list of unexpressed model rules
complete: every validator in the schema-8 model tree must be listed.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from functools import cache
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from microcosm import diagnostics
from test_support.microcosm_diagnostics.json_schema import (
    json_schema_errors,
    load_strict_json,
)
from test_support.microcosm_diagnostics.payloads import (
    schema_8_payload,
    uk_schema_8_payload,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-diagnostics")
DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
#: The documented location consumers such as the collector read.
PUBLISHED_SCHEMA_PATH = (
    "packages/microcosm-diagnostics/src/microcosm/diagnostics/schemas/"
    "calibration_diagnostics.v8.schema.json"
)

DocumentBuilder = Callable[[], object]
DocumentMutation = Callable[[dict], None]


@cache
def _generated_schema_text() -> str:
    return json.dumps(diagnostics.calibration_diagnostics_v8_json_schema())


def _generated_schema() -> dict:
    """A fresh copy per test, so no test can change another's schema."""

    return json.loads(_generated_schema_text())


def _set_value(path: tuple[object, ...], value: object) -> DocumentMutation:
    def mutate(document: dict) -> None:
        owner = document
        for part in path[:-1]:
            owner = owner[part]
        owner[path[-1]] = value

    return mutate


def _delete_value(path: tuple[object, ...]) -> DocumentMutation:
    def mutate(document: dict) -> None:
        owner = document
        for part in path[:-1]:
            owner = owner[part]
        del owner[path[-1]]

    return mutate


def _document(builder: DocumentBuilder, mutation: DocumentMutation | None) -> object:
    document = builder()
    if mutation is not None:
        mutation(document)
    return document


def _payload_with_every_optional_block() -> dict:
    payload = schema_8_payload()
    target = payload["targets"][0]
    target["period"] = "2024"
    target["filter"] = {"kind": "callable", "name": "is_adult"}
    target["metadata"] = {"ledger": {"values": [1, 2.5, None, True, "text"]}}
    target["hierarchy"]["dimensions"] = [
        {"id": "sex", "label": "Sex", "value_id": "female", "value_label": "Female"}
    ]
    target.update(
        {
            "registry": {
                "family": "census_population",
                "se": None,
                "signed": False,
                "notes": "",
            },
            "target_loss_weight": 1.0,
            "target_loss_weight_share": 1.0,
            "target_loss_scale": 2.0,
            "final_capped_scaled_error": 0.0,
            "final_loss_contribution": 0.0,
        }
    )
    payload["target_registry"]["n_specs"] = 2
    payload["l0_lambda"] = None
    payload["loss_trajectory"] = [0.1, None, 0.01]
    payload["skipped"] = [{"name": "ghost@2024", "reason": "measure is missing"}]
    payload["diagnostic_warnings"] = [
        {
            "code": "target_loss_attribution_withheld",
            "severity": "warning",
            "message": "Attribution could not be validated.",
        }
    ]
    payload["past_cap_census"] = {
        "cap": 1.0,
        "scale_basis": "target",
        "n_targets": 1,
        "initial_past_cap": 1,
        "final_past_cap": 0,
        "escaped": 1,
        "frozen": 0,
        "pushed_out": 0,
        "pushed_out_rows": [
            {"name": "population@2024", "init_rel": 1.5, "final_rel": 0.0}
        ],
    }
    payload["target_loss_basis"] = {
        "formula": "sum(weight * min(scaled_error, cap) ** 2)",
        "cap": 1.0,
        "target_count": 1,
        "total_target_weight": 1.0,
        "weight_kind": "provided",
        "scale_kind": "target",
        "hash_algorithm": "sha256",
        "sha256": "b" * 64,
    }
    payload["build"] = {"release_id": "fixture", "timing": {"seconds": 1.5}}
    return payload


def _uk_payload_with_skipped_holdout() -> dict:
    payload = uk_schema_8_payload()
    payload["uk_diagnostics"]["rotated_holdout"] = {"skipped": True}
    return payload


def _legacy_schema_6_payload() -> dict:
    """A document the parser accepts through its schema-6 compatibility branch."""

    payload = schema_8_payload()
    payload["schema_version"] = 6
    del payload["targets"][0]["hierarchy"]
    return payload


def _array_document() -> list:
    return []


def test_generated_schema_declares_draft_2020_12_and_passes_its_metaschema() -> None:
    schema = _generated_schema()

    assert schema["$schema"] == DRAFT_2020_12
    assert schema["$id"] == "calibration_diagnostics.v8.schema.json"
    assert schema["title"] == "Microcosm calibration diagnostics, schema 8"
    assert diagnostics.CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_ID == schema["$id"]
    Draft202012Validator.check_schema(schema)


def test_generated_schema_pins_version_8_and_forbids_undeclared_fields() -> None:
    schema = _generated_schema()
    object_definitions = {
        name: definition
        for name, definition in schema["$defs"].items()
        if definition.get("type") == "object"
    }

    assert schema["properties"]["schema_version"]["const"] == 8
    assert schema["additionalProperties"] is False
    assert "UKDiagnosticsV1" in object_definitions
    assert all(
        definition["additionalProperties"] is False
        for definition in object_definitions.values()
    )


@pytest.mark.parametrize(
    "builder",
    [
        pytest.param(schema_8_payload, id="minimal-us-registry"),
        pytest.param(_payload_with_every_optional_block, id="every-optional-block"),
        pytest.param(uk_schema_8_payload, id="complete-uk-extension"),
        pytest.param(_uk_payload_with_skipped_holdout, id="skipped-uk-holdout"),
    ],
)
def test_generated_schema_accepts_schema_8_documents(
    builder: DocumentBuilder,
) -> None:
    document = load_strict_json(json.dumps(builder(), allow_nan=False))
    diagnostics.CalibrationDiagnosticsV8.model_validate(document)

    assert json_schema_errors(document, _generated_schema()) == []


@pytest.mark.parametrize(
    ("builder", "mutation", "error_path"),
    [
        pytest.param(
            schema_8_payload,
            _delete_value(("targets",)),
            "$",
            id="missing-required-field",
        ),
        pytest.param(
            schema_8_payload,
            _set_value(("unexpected",), 1),
            "$",
            id="undeclared-top-level-field",
        ),
        pytest.param(
            uk_schema_8_payload,
            _set_value(("uk_diagnostics", "weights", "unexpected"), 1),
            "$.uk_diagnostics.weights",
            id="undeclared-uk-field",
        ),
        pytest.param(
            schema_8_payload,
            _set_value(("n_records",), "2"),
            "$.n_records",
            id="string-for-integer",
        ),
        pytest.param(
            schema_8_payload,
            _set_value(("n_nonzero",), True),
            "$.n_nonzero",
            id="boolean-for-integer",
        ),
        pytest.param(
            schema_8_payload,
            _set_value(("final_loss",), "0.01"),
            "$.final_loss",
            id="string-for-number",
        ),
        pytest.param(
            schema_8_payload,
            _set_value(("n_nonzero",), -1),
            "$.n_nonzero",
            id="negative-count",
        ),
        pytest.param(
            schema_8_payload,
            _set_value(("target_registry", "n_specs"), 0),
            "$.target_registry.n_specs",
            id="zero-for-positive-count",
        ),
        pytest.param(
            schema_8_payload,
            _set_value(("schema_version",), 7),
            "$.schema_version",
            id="schema-version-7",
        ),
        pytest.param(
            schema_8_payload,
            _set_value(("target_surface", "sha256"), "A" * 64),
            "$.target_surface.sha256",
            id="uppercase-digest",
        ),
        pytest.param(
            schema_8_payload,
            _set_value(("targets", 0, "measure", "kind"), "expression"),
            "$.targets[0].measure.kind",
            id="unknown-selector-kind",
        ),
        pytest.param(
            uk_schema_8_payload,
            _set_value(
                (
                    "uk_diagnostics",
                    "target_pass_rates_by_geography_level",
                    0,
                    "geography_level",
                ),
                "ward",
            ),
            "$.uk_diagnostics.target_pass_rates_by_geography_level[0].geography_level",
            id="unknown-uk-geography-level",
        ),
        pytest.param(
            uk_schema_8_payload,
            _set_value(("uk_diagnostics", "weights", "ess_fraction"), 1.5),
            "$.uk_diagnostics.weights.ess_fraction",
            id="fraction-above-one",
        ),
        pytest.param(
            schema_8_payload,
            _set_value(("weight_entity",), None),
            "$.weight_entity",
            id="null-for-required-text",
        ),
        pytest.param(
            uk_schema_8_payload,
            _set_value(("uk_diagnostics", "zero_weight_rows_by_stratum"), []),
            "$.uk_diagnostics.zero_weight_rows_by_stratum",
            id="empty-required-list",
        ),
        pytest.param(
            uk_schema_8_payload,
            _set_value(("uk_diagnostics", "rotated_holdout", "n_folds"), 1),
            "$.uk_diagnostics.rotated_holdout.n_folds",
            id="too-few-holdout-folds",
        ),
        pytest.param(
            schema_8_payload,
            _set_value(("targets", 0, "period"), 2024.5),
            "$.targets[0].period",
            id="fractional-period",
        ),
        pytest.param(
            schema_8_payload,
            _set_value(("loss_trajectory",), ["0.1"]),
            "$.loss_trajectory[0]",
            id="string-in-loss-trajectory",
        ),
        pytest.param(
            _legacy_schema_6_payload,
            None,
            "$.schema_version",
            id="legacy-schema-6-document",
        ),
        pytest.param(_array_document, None, "$", id="array-document"),
    ],
)
def test_generated_schema_and_models_reject_invalid_documents(
    builder: DocumentBuilder,
    mutation: DocumentMutation | None,
    error_path: str,
) -> None:
    document = _document(builder, mutation)

    errors = json_schema_errors(document, _generated_schema())

    assert any(error.startswith(f"{error_path}: ") for error in errors), errors
    with pytest.raises(ValidationError):
        diagnostics.CalibrationDiagnosticsV8.model_validate(document)


_DUPLICATED_DIMENSION = {
    "id": "sex",
    "label": "Sex",
    "value_id": "female",
    "value_label": "Female",
}

#: One or more documents per documented gap: the generated schema accepts
#: each, and the models refuse it with the given message.
GAP_CASES = [
    pytest.param(
        "finite-numbers",
        schema_8_payload,
        _set_value(("final_loss",), float("nan")),
        "must be finite or null",
        id="finite-numbers-top-level-scalar",
    ),
    pytest.param(
        "finite-numbers",
        schema_8_payload,
        _set_value(("options", "nested"), [{"value": float("inf")}]),
        "must be finite or null",
        id="finite-numbers-nested-solver-option",
    ),
    pytest.param(
        "strict-integers",
        schema_8_payload,
        _set_value(("n_records",), 2.0),
        "valid integer",
        id="strict-integers",
    ),
    pytest.param(
        "non-blank-text",
        schema_8_payload,
        _set_value(("targets", 0, "source"), " "),
        "non-whitespace",
        id="non-blank-text",
    ),
    pytest.param(
        "digest-pattern-anchoring",
        schema_8_payload,
        _set_value(("target_surface", "sha256"), "a" * 64 + "\n"),
        "should match pattern",
        id="digest-pattern-anchoring",
    ),
    pytest.param(
        "hierarchy-identity",
        schema_8_payload,
        _set_value(
            ("targets", 0, "hierarchy", "category", "provider_id"),
            "another-provider",
        ),
        "provider_id must match provider id",
        id="hierarchy-category-provider",
    ),
    pytest.param(
        "hierarchy-identity",
        schema_8_payload,
        _set_value(
            ("targets", 0, "hierarchy", "dimensions"),
            [_DUPLICATED_DIMENSION, _DUPLICATED_DIMENSION],
        ),
        "dimension ids must be unique",
        id="hierarchy-dimension-ids",
    ),
    pytest.param(
        "target-hierarchy",
        schema_8_payload,
        _set_value(("targets", 0, "hierarchy", "target", "id"), "wrong-target"),
        "hierarchy target id must match target_name",
        id="target-hierarchy",
    ),
    pytest.param(
        "target-surface-shape",
        schema_8_payload,
        _set_value(("target_surface", "constraint_matrix", "rows"), 2),
        "rows must equal n_targets",
        id="target-surface-shape",
    ),
    pytest.param(
        "diagnostics-counts",
        schema_8_payload,
        _set_value(("n_nonzero",), 3),
        "n_nonzero cannot exceed n_records",
        id="diagnostics-counts",
    ),
    pytest.param(
        "uk-extension-presence",
        uk_schema_8_payload,
        _delete_value(("uk_diagnostics",)),
        "require a uk_diagnostics block",
        id="uk-registry-without-uk-extension",
    ),
    pytest.param(
        "uk-extension-presence",
        uk_schema_8_payload,
        _set_value(("target_registry", "country"), "us"),
        "only valid when target_registry.country is 'uk'",
        id="uk-extension-on-another-registry",
    ),
    pytest.param(
        "uk-weight-summary",
        uk_schema_8_payload,
        _set_value(("uk_diagnostics", "weights", "ess_fraction"), 0.25),
        "ess_fraction must equal",
        id="uk-weight-summary",
    ),
    pytest.param(
        "uk-zero-weight-strata",
        uk_schema_8_payload,
        _set_value(("uk_diagnostics", "zero_weight_rows_by_stratum", 0, "rows"), 3),
        "stratum row counts do not reconcile",
        id="uk-zero-weight-strata",
    ),
    pytest.param(
        "uk-geography-pass-rates",
        uk_schema_8_payload,
        _set_value(
            (
                "uk_diagnostics",
                "target_pass_rates_by_geography_level",
                4,
                "geography_level",
            ),
            "national",
        ),
        "levels must be unique",
        id="uk-geography-pass-rates",
    ),
    pytest.param(
        "uk-local-fit",
        uk_schema_8_payload,
        _delete_value(("uk_diagnostics", "weakest_areas_by_fit")),
        "require both family and area summaries",
        id="uk-local-fit",
    ),
    pytest.param(
        "uk-rotated-holdout",
        uk_schema_8_payload,
        _set_value(("uk_diagnostics", "rotated_holdout", "fold_losses"), [0.1, 0.3]),
        "fold rows must match fold_losses",
        id="uk-rotated-holdout",
    ),
    pytest.param(
        "uk-consumption-drift",
        uk_schema_8_payload,
        _set_value(
            (
                "uk_diagnostics",
                "consumption_drift",
                "totals",
                "petrol_spending",
                "change",
            ),
            3.0,
        ),
        "final minus design",
        id="uk-consumption-drift",
    ),
]


@pytest.mark.parametrize(
    ("gap_key", "builder", "mutation", "model_error"),
    GAP_CASES,
)
def test_generated_schema_accepts_documents_only_the_models_reject(
    gap_key: str,
    builder: DocumentBuilder,
    mutation: DocumentMutation,
    model_error: str,
) -> None:
    documented_keys = {
        gap.key for gap in diagnostics.CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_GAPS
    }
    document = _document(builder, mutation)

    assert gap_key in documented_keys
    assert json_schema_errors(document, _generated_schema()) == []
    with pytest.raises(ValidationError, match=model_error):
        diagnostics.CalibrationDiagnosticsV8.model_validate(document)


def test_every_documented_gap_has_a_characterizing_document() -> None:
    documented_keys = [
        gap.key for gap in diagnostics.CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_GAPS
    ]
    characterized_keys = {case.values[0] for case in GAP_CASES}

    assert len(documented_keys) == len(set(documented_keys))
    assert set(documented_keys) == characterized_keys


@pytest.mark.parametrize(
    ("mutation", "error_path"),
    [
        pytest.param(
            _set_value(("uk_diagnostics", "rotated_holdout"), {"skipped": 1}),
            "$.uk_diagnostics.rotated_holdout",
            id="one-for-true",
        ),
        pytest.param(
            _set_value(("uk_diagnostics", "schema_version"), True),
            "$.uk_diagnostics.schema_version",
            id="true-for-one",
        ),
    ],
)
def test_generated_schema_requires_the_literal_values_the_writer_emits(
    mutation: DocumentMutation,
    error_path: str,
) -> None:
    document = _document(uk_schema_8_payload, mutation)

    errors = json_schema_errors(document, _generated_schema())
    model = diagnostics.CalibrationDiagnosticsV8.model_validate(document)
    rewritten = json.loads(model.model_dump_json(exclude_defaults=True))

    assert any(error.startswith(f"{error_path}: ") for error in errors), errors
    assert json_schema_errors(rewritten, _generated_schema()) == []


def _validator_qualified_names(core_schema: object) -> set[str]:
    """Every validator function pydantic runs for a model, by qualified name."""

    names: set[str] = set()
    pending = [core_schema]
    visited: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, dict):
            if str(node.get("type", "")).startswith("function-"):
                names.add(node["function"]["function"].__qualname__)
            pending.extend(node.values())
        elif isinstance(node, (list, tuple)):
            pending.extend(node)
    return names


def test_every_model_validator_is_documented_as_a_schema_gap() -> None:
    model_validators = _validator_qualified_names(
        diagnostics.CalibrationDiagnosticsV8.__pydantic_core_schema__
    )
    documented_validators = {
        validator
        for gap in diagnostics.CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_GAPS
        for validator in gap.validators
    }

    assert "CalibrationDiagnosticsV8.reconcile_counts" in model_validators
    assert documented_validators == model_validators


def test_generated_schema_description_lists_every_gap() -> None:
    description = _generated_schema()["description"]

    assert "microcosm.diagnostics.parse_calibration_diagnostics" in description
    assert diagnostics.CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_LITERAL_NOTE in (
        description
    )
    for gap in diagnostics.CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_GAPS:
        assert gap.rule in description, gap.key


def test_rendered_schema_is_deterministic_json_text() -> None:
    rendered = diagnostics.render_calibration_diagnostics_v8_json_schema()

    assert rendered == diagnostics.render_calibration_diagnostics_v8_json_schema()
    assert rendered.endswith("}\n")
    assert json.loads(rendered) == _generated_schema()
    assert list(json.loads(rendered))[:5] == [
        "$schema",
        "$id",
        "$comment",
        "title",
        "description",
    ]


def _load_generator_tool():
    path = (
        _TEST_PATHS.repository
        / "tools"
        / "generate_calibration_diagnostics_json_schema.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_calibration_diagnostics_json_schema", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_committed_schema_matches_the_generated_schema() -> None:
    committed = diagnostics.committed_calibration_diagnostics_v8_json_schema_path()

    assert committed.read_bytes() == (
        diagnostics.render_calibration_diagnostics_v8_json_schema().encode("utf-8")
    )
    assert diagnostics.load_calibration_diagnostics_v8_json_schema() == (
        diagnostics.calibration_diagnostics_v8_json_schema()
    )
    diagnostics.assert_calibration_diagnostics_v8_json_schema_current()


def test_committed_schema_lives_at_its_documented_path() -> None:
    committed = diagnostics.committed_calibration_diagnostics_v8_json_schema_path()
    package_readme = (_TEST_PATHS.package / "README.md").read_text(encoding="utf-8")

    assert committed.resolve() == (
        (_TEST_PATHS.repository / PUBLISHED_SCHEMA_PATH).resolve()
    )
    assert committed.name == diagnostics.CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_ID
    assert f"`{PUBLISHED_SCHEMA_PATH}`" in package_readme


def test_generator_tool_writes_and_checks_the_committed_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_generator_tool()
    committed = diagnostics.committed_calibration_diagnostics_v8_json_schema_path()
    written = tmp_path / committed.name

    assert tool.main(["--check"]) == 0
    assert tool.main(["--out", str(written)]) == 0
    assert written.read_bytes() == committed.read_bytes()

    written.write_text("{}\n", encoding="utf-8")
    assert tool.main(["--check", "--out", str(written)]) == 1
    assert "regenerate it with" in capsys.readouterr().err
    with pytest.raises(ValueError, match="differs from the schema the models generate"):
        diagnostics.assert_calibration_diagnostics_v8_json_schema_current(written)
