"""The JSON Schema generated for calibration diagnostics schema 9.

Schema 9 is what the writer publishes. Its JSON Schema is generated from
``CalibrationDiagnosticsV9`` the way schema 8's is from
``CalibrationDiagnosticsV8``; the schema-8 file stays committed for documents
already published. The models remain the authority, so every model rule the
generated schema cannot express is listed, with a document that shows it.
"""

from __future__ import annotations

import copy
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
    published_json_schema_errors,
)
from test_support.microcosm_diagnostics.payloads import (
    schema_8_payload,
    uk_schema_8_payload,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-diagnostics")
DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
PUBLISHED_V9_SCHEMA_PATH = (
    "packages/microcosm-diagnostics/src/microcosm/diagnostics/schemas/"
    "calibration_diagnostics.v9.schema.json"
)
#: Gaps that exist only because schema 9 refers to catalogs by ID.
CATALOG_GAP_KEYS = frozenset(
    {"hierarchy-catalog-identity", "hierarchy-references", "target-hierarchy"}
)

DocumentMutation = Callable[[dict], None]


@cache
def _generated_schema_text() -> str:
    return json.dumps(diagnostics.calibration_diagnostics_v9_json_schema())


def _generated_schema() -> dict:
    return json.loads(_generated_schema_text())


def _schema_9(payload: dict) -> dict:
    """The document the writer publishes for a schema-8 payload."""

    model = diagnostics.CalibrationDiagnosticsV8.model_validate(payload)
    return json.loads(
        diagnostics.catalog_calibration_diagnostics(model).model_dump_json(
            exclude_defaults=True
        )
    )


def _with_dimension(payload: dict) -> dict:
    payload = copy.deepcopy(payload)
    payload["targets"][0]["hierarchy"]["dimensions"] = [
        {
            "id": "age_band",
            "label": "Age band",
            "value_id": "0_17",
            "value_label": "Under 18",
        }
    ]
    return payload


def test_v9_schema_declares_draft_2020_12_and_passes_its_metaschema() -> None:
    schema = _generated_schema()

    assert schema["$schema"] == DRAFT_2020_12
    assert schema["$id"] == diagnostics.CALIBRATION_DIAGNOSTICS_V9_JSON_SCHEMA_ID
    Draft202012Validator.check_schema(schema)


def test_v9_schema_pins_version_9_requires_catalogs_and_forbids_undeclared_fields() -> (
    None
):
    schema = _generated_schema()

    assert schema["title"] == "Microcosm calibration diagnostics, schema 9"
    assert schema["properties"]["schema_version"]["const"] == 9
    assert "hierarchy_catalogs" in schema["required"]
    assert schema["additionalProperties"] is False
    assert all(
        definition.get("additionalProperties") is False
        for definition in schema["$defs"].values()
        if definition.get("type") == "object"
    )


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(schema_8_payload(), id="us"),
        pytest.param(_with_dimension(schema_8_payload()), id="us-with-dimension"),
        pytest.param(uk_schema_8_payload(), id="uk"),
    ],
)
def test_v9_schema_accepts_the_documents_the_writer_publishes(payload: dict) -> None:
    document = load_strict_json(json.dumps(_schema_9(payload), allow_nan=False))

    assert json_schema_errors(document, _generated_schema()) == []
    assert published_json_schema_errors(document) == []


def test_each_schema_refuses_the_other_version() -> None:
    schema_8 = schema_8_payload()
    schema_9 = _schema_9(schema_8)

    assert json_schema_errors(schema_8, _generated_schema())
    assert json_schema_errors(
        schema_9, diagnostics.calibration_diagnostics_v8_json_schema()
    )


def _row_hierarchy(document: dict) -> dict:
    return document["targets"][0]["hierarchy"]


def _set_category(document: dict) -> None:
    _row_hierarchy(document)["category_id"] = "census.households"


def _duplicate_geography(document: dict) -> None:
    geographies = document["hierarchy_catalogs"]["geographies"]
    geographies.append({**geographies[0], "label": "USA"})


def _misplace_value(document: dict) -> None:
    document["hierarchy_catalogs"]["dimensions"].append(
        {"id": "sex", "label": "Sex", "values": [{"id": "female", "label": "Female"}]}
    )
    _row_hierarchy(document)["dimensions"] = [{"id": "age_band", "value_id": "female"}]


def _repeat_row_dimension(document: dict) -> None:
    dimensions = _row_hierarchy(document)["dimensions"]
    dimensions.append(dict(dimensions[0]))


def _point_at_another_target(document: dict) -> None:
    _row_hierarchy(document)["target"]["id"] = "households"


GAP_CASES = [
    pytest.param(
        "hierarchy-references",
        _set_category,
        "category census.households is not in hierarchy_catalogs",
        id="unknown-category",
    ),
    pytest.param(
        "hierarchy-references",
        _misplace_value,
        "value female is not listed under dimension age_band",
        id="value-under-the-wrong-dimension",
    ),
    pytest.param(
        "hierarchy-references",
        _repeat_row_dimension,
        "hierarchy dimension ids must be unique",
        id="dimension-twice-on-a-row",
    ),
    pytest.param(
        "hierarchy-catalog-identity",
        _duplicate_geography,
        "hierarchy_catalogs.geographies lists 0100000US twice",
        id="duplicate-geography",
    ),
    pytest.param(
        "target-hierarchy",
        _point_at_another_target,
        "hierarchy target id must match",
        id="hierarchy-target-of-another-row",
    ),
]


@pytest.mark.parametrize(("gap_key", "mutation", "model_error"), GAP_CASES)
def test_v9_schema_accepts_documents_only_the_models_reject(
    gap_key: str,
    mutation: DocumentMutation,
    model_error: str,
) -> None:
    document = _schema_9(_with_dimension(schema_8_payload()))
    mutation(document)

    assert gap_key in {
        gap.key for gap in diagnostics.CALIBRATION_DIAGNOSTICS_V9_JSON_SCHEMA_GAPS
    }
    assert json_schema_errors(document, _generated_schema()) == []
    with pytest.raises(ValidationError, match=model_error):
        diagnostics.CalibrationDiagnosticsV9.model_validate(document)


def test_v9_gaps_are_schema_8_gaps_plus_the_catalog_rules() -> None:
    v8_keys = {
        gap.key for gap in diagnostics.CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_GAPS
    }
    v9_keys = [
        gap.key for gap in diagnostics.CALIBRATION_DIAGNOSTICS_V9_JSON_SCHEMA_GAPS
    ]

    assert len(v9_keys) == len(set(v9_keys))
    # Schema 8's row-hierarchy rule becomes the catalog rules.
    assert set(v9_keys) == (v8_keys - {"hierarchy-identity"}) | CATALOG_GAP_KEYS
    assert CATALOG_GAP_KEYS == {case.values[0] for case in GAP_CASES}


def _validator_qualified_names(core_schema: object) -> set[str]:
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


def test_every_v9_model_validator_is_documented_as_a_schema_gap() -> None:
    model_validators = _validator_qualified_names(
        diagnostics.CalibrationDiagnosticsV9.__pydantic_core_schema__
    )
    documented_validators = {
        validator
        for gap in diagnostics.CALIBRATION_DIAGNOSTICS_V9_JSON_SCHEMA_GAPS
        for validator in gap.validators
    }

    assert "CalibrationDiagnosticsV9.reconcile_counts" in model_validators
    assert documented_validators == model_validators


def test_v9_description_lists_every_gap() -> None:
    description = _generated_schema()["description"]

    assert "schema 9" in description
    for gap in diagnostics.CALIBRATION_DIAGNOSTICS_V9_JSON_SCHEMA_GAPS:
        assert gap.rule in description, gap.key


def test_committed_v9_schema_matches_the_generated_schema() -> None:
    committed = diagnostics.committed_calibration_diagnostics_v9_json_schema_path()

    assert committed.read_bytes() == (
        diagnostics.render_calibration_diagnostics_v9_json_schema().encode("utf-8")
    )
    assert diagnostics.load_calibration_diagnostics_v9_json_schema() == (
        diagnostics.calibration_diagnostics_v9_json_schema()
    )
    diagnostics.assert_calibration_diagnostics_v9_json_schema_current()


def test_committed_v9_schema_lives_at_its_documented_path() -> None:
    committed = diagnostics.committed_calibration_diagnostics_v9_json_schema_path()
    package_readme = (_TEST_PATHS.package / "README.md").read_text(encoding="utf-8")

    assert committed.resolve() == (
        (_TEST_PATHS.repository / PUBLISHED_V9_SCHEMA_PATH).resolve()
    )
    assert committed.name == diagnostics.CALIBRATION_DIAGNOSTICS_V9_JSON_SCHEMA_ID
    assert f"`{PUBLISHED_V9_SCHEMA_PATH}`" in package_readme


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


def test_generator_tool_writes_and_checks_the_committed_v9_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_generator_tool()
    committed = diagnostics.committed_calibration_diagnostics_v9_json_schema_path()
    written = tmp_path / committed.name

    assert tool.main(["--check"]) == 0
    assert tool.main(["--schema-version", "9", "--out", str(written)]) == 0
    assert written.read_bytes() == committed.read_bytes()

    written.write_text("{}\n", encoding="utf-8")
    assert tool.main(["--check", "--schema-version", "9", "--out", str(written)]) == 1
    assert "regenerate it with" in capsys.readouterr().err
    with pytest.raises(ValueError, match="differs from the schema the models generate"):
        diagnostics.assert_calibration_diagnostics_v9_json_schema_current(written)
