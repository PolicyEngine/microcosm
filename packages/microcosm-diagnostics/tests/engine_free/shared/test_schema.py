from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from microcosm.diagnostics import (
    CalibrationDiagnosticsV8,
    CalibrationDiagnosticsV9,
    calibration_diagnostics_json_schema,
    expand_calibration_diagnostics,
    parse_calibration_diagnostics,
    write_calibration_diagnostics,
)
from test_support.microcosm_diagnostics.payloads import (
    schema_8_payload,
    uk_schema_8_payload,
)


def _delete_path(payload: dict, path: tuple[object, ...]) -> None:
    owner = payload
    for part in path[:-1]:
        owner = owner[part]
    del owner[path[-1]]


def test_schema_8_round_trips_as_a_typed_model() -> None:
    model = parse_calibration_diagnostics(schema_8_payload())
    assert isinstance(model, CalibrationDiagnosticsV8)
    assert (
        json.loads(model.model_dump_json(exclude_defaults=True)) == schema_8_payload()
    )
    assert calibration_diagnostics_json_schema()["$defs"]


def test_schema_8_round_trips_the_complete_uk_extension() -> None:
    payload = uk_schema_8_payload()
    model = parse_calibration_diagnostics(payload)

    assert isinstance(model, CalibrationDiagnosticsV8)
    assert json.loads(model.model_dump_json(exclude_defaults=True)) == payload


def test_uk_area_fit_still_parses_legacy_support_figures() -> None:
    # Released diagnostics carry the floor flag in place of an area's support
    # figures (microcosm#1301); payloads written before still parse, and a
    # row may carry neither.
    payload = uk_schema_8_payload()
    (area,) = payload["uk_diagnostics"]["weakest_areas_by_fit"]["bottom_by_fit"]
    del area["support_below_floor"]
    bare = parse_calibration_diagnostics(payload)
    assert json.loads(bare.model_dump_json(exclude_defaults=True)) == payload

    area.update(
        nonzero_households=60,
        nonzero_source_households=55,
        effective_sample_size=52.5,
    )
    legacy = parse_calibration_diagnostics(payload)
    assert json.loads(legacy.model_dump_json(exclude_defaults=True)) == payload

    area["effective_sample_size"] = -1.0
    with pytest.raises(ValidationError):
        parse_calibration_diagnostics(payload)


def test_every_declared_uk_record_forbids_undeclared_fields() -> None:
    definitions = calibration_diagnostics_json_schema()["$defs"]
    uk_definitions = {
        name: definition
        for name, definition in definitions.items()
        if name.startswith("UK")
    }

    assert uk_definitions
    assert all(
        definition.get("additionalProperties") is False
        for definition in uk_definitions.values()
    )


@pytest.mark.parametrize(
    "field_path",
    [
        ("uk_diagnostics", "schema_version"),
        ("uk_diagnostics", "weights", "total_weight"),
        ("uk_diagnostics", "zero_weight_rows_by_stratum", 0, "stratum"),
        (
            "uk_diagnostics",
            "target_pass_rates_by_geography_level",
            0,
            "pass_rate",
        ),
        ("uk_diagnostics", "target_observation_basis"),
        ("uk_diagnostics", "weakest_families", 0, "loss_share"),
        ("uk_diagnostics", "weakest_areas_by_fit", "limit"),
        (
            "uk_diagnostics",
            "weakest_areas_by_fit",
            "bottom_by_fit",
            0,
            "area_code",
        ),
        (
            "uk_diagnostics",
            "weakest_areas_by_fit",
            "countries",
            0,
            "country",
        ),
        ("uk_diagnostics", "rotated_holdout", "target_loss_cap"),
        (
            "uk_diagnostics",
            "rotated_holdout",
            "folds",
            0,
            "holdout_target_indices",
        ),
        ("uk_diagnostics", "consumption_drift", "report_only"),
        (
            "uk_diagnostics",
            "consumption_drift",
            "totals",
            "petrol_spending",
            "relative_change",
        ),
    ],
)
def test_schema_8_rejects_missing_fields_across_the_uk_extension(
    field_path: tuple[object, ...],
) -> None:
    payload = uk_schema_8_payload()
    _delete_path(payload, field_path)

    with pytest.raises(ValidationError):
        parse_calibration_diagnostics(payload)


@pytest.mark.parametrize(
    "field_path",
    [
        ("uk_diagnostics",),
        ("uk_diagnostics", "weights"),
        ("uk_diagnostics", "zero_weight_rows_by_stratum", 0),
        ("uk_diagnostics", "target_pass_rates_by_geography_level", 0),
        ("uk_diagnostics", "weakest_families", 0),
        ("uk_diagnostics", "weakest_areas_by_fit"),
        ("uk_diagnostics", "weakest_areas_by_fit", "bottom_by_fit", 0),
        ("uk_diagnostics", "weakest_areas_by_fit", "countries", 0),
        ("uk_diagnostics", "rotated_holdout"),
        ("uk_diagnostics", "rotated_holdout", "folds", 0),
        ("uk_diagnostics", "consumption_drift"),
        (
            "uk_diagnostics",
            "consumption_drift",
            "shares",
            "fuel_car_households_without_fuel",
        ),
    ],
)
def test_schema_8_rejects_undeclared_fields_across_the_uk_extension(
    field_path: tuple[object, ...],
) -> None:
    payload = uk_schema_8_payload()
    owner = payload
    for part in field_path:
        owner = owner[part]
    owner["undeclared"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        parse_calibration_diagnostics(payload)


def test_schema_8_requires_the_typed_uk_extension_for_uk_registries() -> None:
    payload = uk_schema_8_payload()
    del payload["uk_diagnostics"]

    with pytest.raises(ValidationError, match="require a uk_diagnostics block"):
        parse_calibration_diagnostics(payload)


def test_schema_8_schematizes_an_explicitly_skipped_uk_holdout() -> None:
    payload = uk_schema_8_payload()
    payload["uk_diagnostics"]["rotated_holdout"] = {"skipped": True}

    model = parse_calibration_diagnostics(payload)

    assert isinstance(model, CalibrationDiagnosticsV8)
    assert model.uk_diagnostics is not None
    assert model.uk_diagnostics.rotated_holdout is not None
    assert model.uk_diagnostics.rotated_holdout.skipped is True


def test_schema_8_rejects_incomplete_hierarchy() -> None:
    payload = schema_8_payload()
    del payload["targets"][0]["hierarchy"]
    with pytest.raises(ValidationError, match="hierarchy"):
        parse_calibration_diagnostics(payload)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (
            ("targets", 0, "hierarchy", "category", "provider_id"),
            "another-provider",
            "category provider_id must match provider id",
        ),
        (
            ("targets", 0, "hierarchy", "target", "id"),
            "wrong-target",
            "hierarchy target id must match target_name",
        ),
    ],
)
def test_schema_8_rejects_inconsistent_hierarchy_identity(
    field_path: tuple[object, ...],
    value: str,
    message: str,
) -> None:
    payload = schema_8_payload()
    owner = payload
    for part in field_path[:-1]:
        owner = owner[part]
    owner[field_path[-1]] = value

    with pytest.raises(ValidationError, match=message):
        parse_calibration_diagnostics(payload)


@pytest.mark.parametrize(
    "field_path",
    [
        ("targets", 0, "hierarchy", "provider", "label"),
        ("targets", 0, "hierarchy", "dimensions", 0, "value_label"),
        ("targets", 0, "source"),
    ],
)
def test_schema_8_rejects_whitespace_only_text(field_path: tuple[object, ...]) -> None:
    payload = schema_8_payload()
    payload["targets"][0]["hierarchy"]["dimensions"] = [
        {"id": "sex", "label": "Sex", "value_id": "female", "value_label": "Female"}
    ]
    owner = payload
    for part in field_path[:-1]:
        owner = owner[part]
    owner[field_path[-1]] = " "

    with pytest.raises(ValidationError, match="non-whitespace"):
        parse_calibration_diagnostics(payload)


@pytest.mark.parametrize(
    ("section", "field"),
    [("target_surface", "n_targets"), ("target_registry", "n_specs")],
)
def test_schema_8_requires_positive_declared_target_counts(
    section: str,
    field: str,
) -> None:
    payload = schema_8_payload()
    payload[section][field] = 0

    with pytest.raises(ValidationError, match="greater than 0"):
        parse_calibration_diagnostics(payload)


def test_schema_8_reconciles_target_surface_matrix_shape() -> None:
    payload = schema_8_payload()
    payload["target_surface"]["constraint_matrix"]["rows"] = 2

    with pytest.raises(ValidationError, match="rows must equal n_targets"):
        parse_calibration_diagnostics(payload)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
@pytest.mark.parametrize(
    "location",
    [
        "top-level scalar",
        "target scalar",
        "loss trajectory",
        "nested solver option",
        "nested target metadata",
        "nested build provenance",
        "UK extension",
    ],
)
def test_schema_8_rejects_non_finite_numbers_recursively(
    location: str,
    value: float,
) -> None:
    payload = (
        uk_schema_8_payload() if location == "UK extension" else schema_8_payload()
    )
    if location == "top-level scalar":
        payload["final_loss"] = value
    elif location == "target scalar":
        payload["targets"][0]["relative_error"] = value
    elif location == "loss trajectory":
        payload["loss_trajectory"][0] = value
    elif location == "nested solver option":
        payload["options"]["nested"] = [{"value": value}]
    elif location == "nested target metadata":
        payload["targets"][0]["metadata"]["nested"] = [{"value": value}]
    elif location == "nested build provenance":
        payload["build"] = {"nested": [{"value": value}]}
    else:
        payload["uk_diagnostics"]["weights"]["total_weight"] = value

    with pytest.raises(ValidationError, match="must be finite or null"):
        parse_calibration_diagnostics(payload)


def test_writer_strict_serialization_rejects_a_non_finite_mutated_mapping(
    tmp_path: Path,
) -> None:
    payload = schema_8_payload()
    payload["build"] = {"nested": {"value": 1.0}}
    model = CalibrationDiagnosticsV8.model_validate(payload)
    assert model.build is not None
    nested = model.build["nested"]
    assert isinstance(nested, dict)
    nested["value"] = float("nan")
    path = tmp_path / "calibration_diagnostics.json"
    path.write_text("stale")

    outcome = write_calibration_diagnostics(model, path)

    # The writer validates the schema 9 document it builds from the model, so
    # the non-finite value is refused there, before serialization.
    assert outcome.status == "failed"
    assert outcome.error_code == "validation_error"
    assert not path.exists()


def test_writer_is_atomic_and_removes_stale_output_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "calibration_diagnostics.json"
    path.write_text("stale")
    model = CalibrationDiagnosticsV8.model_validate(schema_8_payload())

    def fail_write(self: Path, data: bytes) -> int:
        raise OSError("synthetic write failure")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    outcome = write_calibration_diagnostics(model, path)

    assert outcome.status == "failed"
    assert outcome.error_code == "io_error"
    assert not path.exists()


def test_unexpected_writer_failure_does_not_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "calibration_diagnostics.json"
    path.write_text("stale")
    model = CalibrationDiagnosticsV8.model_validate(schema_8_payload())

    def fail_write(self: Path, data: bytes) -> int:
        raise RuntimeError("synthetic unexpected failure")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    outcome = write_calibration_diagnostics(model, path)

    assert outcome.status == "failed"
    assert outcome.error_code == "unexpected_error"
    assert not path.exists()


def test_writer_round_trips_valid_output(tmp_path: Path) -> None:
    model = CalibrationDiagnosticsV8.model_validate(schema_8_payload())
    path = tmp_path / "calibration_diagnostics.json"
    outcome = write_calibration_diagnostics(model, path)

    assert outcome.status == "available"
    assert outcome.schema_version == 9
    written = parse_calibration_diagnostics(json.loads(path.read_text()))
    assert isinstance(written, CalibrationDiagnosticsV9)
    assert expand_calibration_diagnostics(written) == model


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"change": 3.0}, "final minus design"),
        ({"relative_change": None}, "undefined only at zero"),
    ],
)
def test_uk_consumption_drift_values_reconcile(change: dict, match: str) -> None:
    payload = uk_schema_8_payload()
    payload["uk_diagnostics"]["consumption_drift"]["totals"]["petrol_spending"].update(
        change
    )

    with pytest.raises(ValidationError, match=match):
        parse_calibration_diagnostics(payload)
