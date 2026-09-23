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


def test_schema_8_round_trips_as_a_typed_model() -> None:
    model = parse_calibration_diagnostics(_payload())
    assert isinstance(model, CalibrationDiagnosticsV8)
    assert json.loads(model.model_dump_json(exclude_defaults=True)) == _payload()
    assert calibration_diagnostics_json_schema()["$defs"]


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
