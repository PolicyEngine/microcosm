from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from microcosm.diagnostics import (
    CalibrationDiagnosticsV8,
    calibration_diagnostics_json_schema,
    parse_calibration_diagnostics,
    write_calibration_diagnostics,
)


def _payload() -> dict:
    digest = "a" * 64
    return {
        "schema_version": 8,
        "weight_entity": "household",
        "options": {"epochs": 1},
        "target_surface": {
            "schema_version": 1,
            "weight_entity": "household",
            "n_targets": 1,
            "n_records": 2,
            "constraint_matrix": {"rows": 1, "columns": 2, "nnz": 2},
            "sha256": digest,
            "names_sha256": digest,
            "values_sha256": digest,
        },
        "target_registry": {"country": "us", "version": digest, "n_specs": 1},
        "l0_lambda": 0.0,
        "n_nonzero": 2,
        "n_records": 2,
        "initial_loss": 0.1,
        "final_loss": 0.01,
        "fraction_within_10pct": 1.0,
        "effective_sample_size": 2.0,
        "realized_max_weight_ratio": 1.0,
        "top_1pct_weight_share": 0.5,
        "loss_trajectory": [0.1],
        "skipped": [],
        "past_cap_census": None,
        "diagnostic_warnings": [],
        "targets": [
            {
                "name": "population@2024",
                "target_name": "population",
                "period": 2024,
                "entity": "household",
                "measure": {"kind": "column", "name": "household_count"},
                "filter": None,
                "source": "Census fixture",
                "metadata": {},
                "target": 2.0,
                "compiled_target": 2.0,
                "initial_estimate": 1.8,
                "final_estimate": 2.0,
                "relative_error": 0.0,
                "within_tolerance": None,
                "hierarchy": {
                    "provider": {"id": "census", "label": "Census"},
                    "category": {
                        "id": "census.population",
                        "label": "Population",
                        "provider_id": "census",
                    },
                    "geography": {
                        "id": "0100000US",
                        "label": "United States",
                        "level": "country",
                    },
                    "dimensions": [],
                    "target": {"id": "population", "label": "Population"},
                },
            }
        ],
    }


def _uk_payload() -> dict:
    payload = _payload()
    payload["target_registry"]["country"] = "uk"
    payload["n_nonzero"] = 1
    payload["effective_sample_size"] = 1.0
    payload["top_1pct_weight_share"] = 1.0
    payload["uk_diagnostics"] = {
        "schema_version": 1,
        "weights": {
            "n_records": 2,
            "positive_weight_records": 1,
            "zero_weight_records": 1,
            "total_weight": 1.0,
            "effective_sample_size": 1.0,
            "ess_fraction": 0.5,
            "median_positive_weight": 1.0,
            "max_weight": 1.0,
            "max_to_median_positive_weight": 1.0,
            "top_1pct_weight_share": 1.0,
        },
        "zero_weight_rows_by_stratum": [
            {
                "stratum": {"source": "frs"},
                "rows": 2,
                "positive_weight_rows": 1,
                "zero_weight_rows": 1,
                "weight_sum": 1.0,
            }
        ],
        "target_pass_rates_by_geography_level": [
            {
                "geography_level": level,
                "n_targets": 1 if level == "national" else 0,
                "n_scored": 1 if level == "national" else 0,
                "n_skipped": 0,
                "n_within_10pct": 1 if level == "national" else 0,
                "pass_rate": 1.0 if level == "national" else None,
            }
            for level in (
                "national",
                "region",
                "country",
                "local_authority",
                "constituency",
            )
        ],
        "target_observation_basis": {"population@2024": "annual_stock"},
        "weakest_families": [
            {
                "family": "population",
                "n_targets": 1,
                "n_within_10pct": 1,
                "pass_rate": 1.0,
                "worst_target": "population@2024",
                "worst_abs_relative_error": 0.0,
                "loss_contribution": 0.0,
                "loss_share": 0.0,
            }
        ],
        "weakest_areas_by_fit": {
            "limit": 15,
            "n_areas_scored": 1,
            "bottom_by_fit": [
                {
                    "geography_level": "constituency",
                    "area_code": "E14000001",
                    "country": "England",
                    "n_targets": 1,
                    "n_within_10pct": 1,
                    "pass_rate": 1.0,
                    "worst_target": "population@2024",
                    "worst_abs_relative_error": 0.0,
                    "loss_contribution": 0.0,
                    "nonzero_households": 1,
                    "nonzero_source_households": 1,
                    "effective_sample_size": 1.0,
                }
            ],
            "countries": [
                {
                    "country": "England",
                    "geography_level": "constituency",
                    "n_areas": 1,
                    "n_targets": 1,
                    "n_within_10pct": 1,
                    "pass_rate": 1.0,
                    "worst_target": "population@2024",
                    "worst_abs_relative_error": 0.0,
                    "loss_contribution": 0.0,
                }
            ],
        },
        "rotated_holdout": {
            "report_only": True,
            "method": "rotated_folds",
            "target_loss_cap": 10.0,
            "loss_weight_scale": "held_local_grains_only",
            "target_weight_rule": "grain_equal",
            "population": "held_out_local_targets",
            "grains": ["constituency", "local_authority"],
            "n_folds": 2,
            "seed": 20260529,
            "solve_seed": 7,
            "mean_holdout_loss": 0.15,
            "worst_holdout_loss": 0.2,
            "fold_losses": [0.1, 0.2],
            "folds": [
                {
                    "fold": 0,
                    "n_train_targets": 1,
                    "n_holdout_targets": 1,
                    "holdout_target_indices": [0],
                    "training_national_rows": 1,
                    "holdout_loss": 0.1,
                },
                {
                    "fold": 1,
                    "n_train_targets": 1,
                    "n_holdout_targets": 1,
                    "holdout_target_indices": [1],
                    "training_national_rows": 1,
                    "holdout_loss": 0.2,
                },
            ],
        },
    }
    return payload


def _delete_path(payload: dict, path: tuple[object, ...]) -> None:
    owner = payload
    for part in path[:-1]:
        owner = owner[part]
    del owner[path[-1]]


def test_schema_8_round_trips_as_a_typed_model() -> None:
    model = parse_calibration_diagnostics(_payload())
    assert isinstance(model, CalibrationDiagnosticsV8)
    assert json.loads(model.model_dump_json(exclude_defaults=True)) == _payload()
    assert calibration_diagnostics_json_schema()["$defs"]


def test_schema_8_round_trips_the_complete_uk_extension() -> None:
    payload = _uk_payload()
    model = parse_calibration_diagnostics(payload)

    assert isinstance(model, CalibrationDiagnosticsV8)
    assert json.loads(model.model_dump_json(exclude_defaults=True)) == payload


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
    ],
)
def test_schema_8_rejects_missing_fields_across_the_uk_extension(
    field_path: tuple[object, ...],
) -> None:
    payload = _uk_payload()
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
    ],
)
def test_schema_8_rejects_undeclared_fields_across_the_uk_extension(
    field_path: tuple[object, ...],
) -> None:
    payload = _uk_payload()
    owner = payload
    for part in field_path:
        owner = owner[part]
    owner["undeclared"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        parse_calibration_diagnostics(payload)


def test_schema_8_requires_the_typed_uk_extension_for_uk_registries() -> None:
    payload = _uk_payload()
    del payload["uk_diagnostics"]

    with pytest.raises(ValidationError, match="require a uk_diagnostics block"):
        parse_calibration_diagnostics(payload)


def test_schema_8_schematizes_an_explicitly_skipped_uk_holdout() -> None:
    payload = _uk_payload()
    payload["uk_diagnostics"]["rotated_holdout"] = {"skipped": True}

    model = parse_calibration_diagnostics(payload)

    assert isinstance(model, CalibrationDiagnosticsV8)
    assert model.uk_diagnostics is not None
    assert model.uk_diagnostics.rotated_holdout is not None
    assert model.uk_diagnostics.rotated_holdout.skipped is True


def test_schema_8_rejects_incomplete_hierarchy() -> None:
    payload = _payload()
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
    payload = _payload()
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
    payload = _payload()
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
    payload = _payload()
    payload[section][field] = 0

    with pytest.raises(ValidationError, match="greater than 0"):
        parse_calibration_diagnostics(payload)


def test_schema_8_reconciles_target_surface_matrix_shape() -> None:
    payload = _payload()
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
    payload = _uk_payload() if location == "UK extension" else _payload()
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
    payload = _payload()
    payload["build"] = {"nested": {"value": 1.0}}
    model = CalibrationDiagnosticsV8.model_validate(payload)
    assert model.build is not None
    nested = model.build["nested"]
    assert isinstance(nested, dict)
    nested["value"] = float("nan")
    path = tmp_path / "calibration_diagnostics.json"
    path.write_text("stale")

    outcome = write_calibration_diagnostics(model, path)

    assert outcome.status == "failed"
    assert outcome.error_code == "serialization_error"
    assert not path.exists()


def test_writer_is_atomic_and_removes_stale_output_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "calibration_diagnostics.json"
    path.write_text("stale")
    model = CalibrationDiagnosticsV8.model_validate(_payload())

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
    model = CalibrationDiagnosticsV8.model_validate(_payload())

    def fail_write(self: Path, data: bytes) -> int:
        raise RuntimeError("synthetic unexpected failure")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    outcome = write_calibration_diagnostics(model, path)

    assert outcome.status == "failed"
    assert outcome.error_code == "unexpected_error"
    assert not path.exists()


def test_writer_round_trips_valid_output(tmp_path: Path) -> None:
    model = CalibrationDiagnosticsV8.model_validate(_payload())
    path = tmp_path / "calibration_diagnostics.json"
    outcome = write_calibration_diagnostics(model, path)

    assert outcome.status == "available"
    assert parse_calibration_diagnostics(json.loads(path.read_text())) == model
