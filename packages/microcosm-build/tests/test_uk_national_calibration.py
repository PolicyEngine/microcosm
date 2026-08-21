"""The UK national Ledger-backed calibration stage."""

from __future__ import annotations

import json
from importlib import resources as importlib_resources

import numpy as np
import pandas as pd
import pytest

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.uk_runtime.national_calibration import (
    UKNationalCalibrationStage,
)
from microcosm.build.uk_runtime.national_frame import validate_uk_national_frame
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

ACTIVE_REFERENCE_COUNT = 388


def _uc_reference(**overrides) -> LedgerTargetReference:
    values = {
        "name": "dwp.uc.households",
        "ledger_selector": {
            "source_name": "dwp",
            "source_concept": "dwp.uc_benefit_units",
            "geography_level": "country",
        },
        "entity": "benunit",
        "measure": "dwp/uc/households",
        "family": "dwp_uc",
        "period": 2025,
        "metadata": {"contract_target_id": "dwp.uc.households"},
    }
    values.update(overrides)
    return LedgerTargetReference(**values)


def _fact(
    *,
    concept: str = "dwp.uc_benefit_units",
    source_name: str = "dwp",
    value: float = 30.0,
) -> dict:
    return {
        "aggregate_fact_key": "ledger.aggregate_fact.v2:uc-fixture",
        "aggregation": {"method": "sum"},
        "assertion": "observation",
        "geography": {"level": "country", "id": "K02000001"},
        "observed_measure": {
            "source_name": source_name,
            "source_concept": concept,
            "source_measure_id": "total_units",
            "unit": "count",
        },
        "period": {"type": "month", "value": "2025-12"},
        "value": value,
    }


def _frame() -> Frame:
    ids = np.arange(4, dtype="int64")
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": ids,
                    "person_benunit_id": ids,
                    "person_household_id": ids,
                }
            ),
            "benunit": pd.DataFrame(
                {"benunit_id": ids, "universal_credit": [1.0, 1.0, 0.0, 0.0]}
            ),
            "household": pd.DataFrame({"household_id": ids}),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.full(4, 10.0), WeightKind.DESIGN)},
        metadata={"time_period": "2023"},
    )


def _nested_frame() -> Frame:
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(6, dtype="int64"),
                    "person_benunit_id": [0, 0, 1, 2, 3, 3],
                    "person_household_id": [0, 0, 0, 1, 2, 2],
                }
            ),
            "benunit": pd.DataFrame(
                {
                    "benunit_id": np.arange(4, dtype="int64"),
                    "universal_credit": [1.0, 0.0, 1.0, 1.0],
                }
            ),
            "household": pd.DataFrame(
                {"household_id": np.arange(3, dtype="int64")}
            ),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(np.array([10.0, 20.0, 30.0]), WeightKind.DESIGN)},
        metadata={"time_period": "2023"},
    )


def test_uc_calibration_compiles_and_moves_weighted_count_towards_fact() -> None:
    frame = _frame()
    stage = UKNationalCalibrationStage(
        [_fact()], references=[_uc_reference()], epochs=200, learning_rate=0.05
    )

    result = stage(frame)

    before = 20.0
    after = float(result.weights_for("household").values[:2].sum())
    assert abs(after - 30.0) < abs(before - 30.0)
    assert stage.manifest["activated_reference_count"] == 1
    assert stage.manifest["resolved_reference_count"] == 1
    assert stage.manifest["matrix_target_count"] == 1
    assert stage.diagnostics[0]["target"] == 30.0


def test_uc_calibration_stage_accepts_benunit_grain_reference_on_nested_frame() -> None:
    frame = _nested_frame()
    stage = UKNationalCalibrationStage(
        [_fact(value=60.0)], references=[_uc_reference()], epochs=5
    )

    result = stage(frame)

    assert stage.manifest["activated_reference_count"] == 1
    assert stage.manifest["resolved_reference_count"] == 1
    assert stage.manifest["matrix_target_count"] == 1
    assert stage.diagnostics[0]["target"] == 60.0
    assert stage.diagnostics[0]["estimate"] == pytest.approx(60.0)
    validate_uk_national_frame(result)


def test_activated_unresolvable_reference_aborts_loudly() -> None:
    stage = UKNationalCalibrationStage(
        [_fact(concept="different")], references=[_uc_reference()], epochs=1
    )

    with pytest.raises(ValueError, match="did not match a Ledger fact selector"):
        stage(_frame())


def test_chronicle_184_uc_and_obr_references_compile_fail_closed() -> None:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.ledger_targets import compile_ledger_target_references

    spec = load_country_spec("uk")
    assert len(spec.target_references) == ACTIVE_REFERENCE_COUNT
    reference_names = {reference.name for reference in spec.target_references}
    assert "obr.universal_credit_in_cap" in reference_names
    assert "dwp.uc.households" in reference_names

    membership = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("target_reference_membership.json")
        .read_text()
    )
    assert membership["targets"]["dwp.uc.households"]["status"] == "active"
    uc_reference = next(
        reference
        for reference in spec.target_references
        if reference.name == "dwp.uc.households"
    )
    assert uc_reference.value_operation == "calendar_year_average"

    references = tuple(
        reference
        for reference in spec.target_references
        if reference.name == "obr.universal_credit_in_cap"
    )
    registry = compile_ledger_target_references(
        [
            _fact(
                concept="obr.universal_credit_in_cap",
                source_name="obr",
                value=40_000_000_000,
            ),
        ],
        references,
        country="uk",
    )

    assert {spec.name for spec in registry.specs} == {"obr.universal_credit_in_cap"}


def test_calibration_preserves_entity_ids_and_national_integrity() -> None:
    frame = _frame()
    stage = UKNationalCalibrationStage(
        [_fact()], references=[_uc_reference()], epochs=5
    )

    result = stage(frame)

    for entity in frame.entities:
        id_column = f"{entity}_id"
        assert result.table(entity)[id_column].equals(frame.table(entity)[id_column])
    validate_uk_national_frame(result)


def test_checkpoint_metadata_round_trips_calibration_evidence() -> None:
    frame = _frame()
    stage = UKNationalCalibrationStage(
        [_fact()], references=[_uc_reference()], epochs=5
    )

    staged = stage(frame)
    metadata = json.loads(json.dumps(stage.checkpoint_metadata()))

    resumed = UKNationalCalibrationStage(
        [_fact()], references=[_uc_reference()], epochs=5
    )
    resumed.resume_from_checkpoint(metadata, staged)

    assert resumed.manifest == stage.manifest
    assert resumed.diagnostics == stage.diagnostics
    assert resumed.output_content_identity == metadata["output_content_identity"]

    drifted = UKNationalCalibrationStage(
        [_fact()], references=[_uc_reference()], epochs=5
    )
    with pytest.raises(RuntimeError, match="drifted record"):
        drifted.resume_from_checkpoint(metadata, frame)

    empty = UKNationalCalibrationStage(
        [_fact()], references=[_uc_reference()], epochs=5
    )
    with pytest.raises(RuntimeError, match="calibration counts"):
        empty.resume_from_checkpoint({}, staged)

    missing_count = dict(metadata)
    missing_count["calibration"] = {
        key: value
        for key, value in metadata["calibration"].items()
        if key != "activated_reference_count"
    }
    with pytest.raises(RuntimeError, match="calibration counts"):
        empty.resume_from_checkpoint(missing_count, staged)

    unrun = UKNationalCalibrationStage(
        [_fact()], references=[_uc_reference()], epochs=5
    )
    with pytest.raises(RuntimeError, match="has not run"):
        unrun.checkpoint_metadata()
