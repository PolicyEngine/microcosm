"""The UK national Ledger-backed calibration stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.build.uk_runtime.national_calibration import (
    UKNationalCalibrationStage,
)
from microcosm.build.uk_runtime.national_frame import validate_uk_national_frame
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


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


def test_activated_unresolvable_reference_aborts_loudly() -> None:
    stage = UKNationalCalibrationStage(
        [_fact(concept="different")], references=[_uc_reference()], epochs=1
    )

    with pytest.raises(ValueError, match="did not match a Ledger fact selector"):
        stage(_frame())


def test_chronicle_184_uc_and_obr_references_compile_fail_closed() -> None:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.ledger_targets import compile_ledger_target_references

    references = tuple(
        reference
        for reference in load_country_spec("uk").target_references
        if reference.name in {"dwp.uc.households", "obr.universal_credit_in_cap"}
    )
    registry = compile_ledger_target_references(
        [
            _fact(),
            _fact(
                concept="obr.universal_credit_in_cap",
                source_name="obr",
                value=40_000_000_000,
            ),
        ],
        references,
        country="uk",
    )

    assert {spec.name for spec in registry.specs} == {
        "dwp.uc.households",
        "obr.universal_credit_in_cap",
    }


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
