"""The UK stages' declared QRF sign-gate calibration (microcosm#1359)."""

from __future__ import annotations

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.qrf_gates import (
    declared_gate_calibration,
    gate_outcomes,
    pooled_gate_outcomes,
)

#: Every UK operation whose QRF draws a sign through the gate, as
#: ``(stage, operation kind)``; each declares the cross-fitted gate.
CROSS_FITTED_OPERATIONS = frozenset(
    {
        ("etb_services", "fit_weighted_qrf_chain"),
        ("hmrc_spi_income_spine", "fit_weighted_qrf_stage1"),
        ("hmrc_spi_income_spine", "fit_weighted_qrf_stage2"),
        ("lcfs_consumption", "fit_weighted_qrf_chain"),
        ("lcfs_consumption", "impute_prepayment_meter"),
        ("lcfs_consumption", "redraw_zero_road_fuel"),
        ("nts_bus_travel", "impute_bus_use_band"),
        ("salary_sacrifice", "fit_weighted_qrf"),
        ("spi_housing_shell", "impute_spi_housing_shell"),
        ("uc_reporter_redraw", "redraw_spi_reported_uc"),
        ("was_wealth", "fit_weighted_qrf_chain"),
    }
)


class _Fitted:
    def __init__(self, outcomes: dict[str, str | None]) -> None:
        self._outcomes = outcomes

    def gate_outcomes(self) -> dict[str, str | None]:
        return dict(self._outcomes)


def test_the_gated_uk_fits_declare_the_cross_fitted_gate() -> None:
    stages = load_country_spec("uk").sources.stage_map()
    declared = {
        (stage.stage, operation.kind): operation.parameters["gate_calibration"]
        for stage in stages.values()
        for operation in stage.operations
        if "gate_calibration" in operation.parameters
    }
    assert set(declared) == CROSS_FITTED_OPERATIONS
    assert set(declared.values()) == {"cross_fitted"}
    for stage, kind in CROSS_FITTED_OPERATIONS:
        assert declared_gate_calibration(stages[stage], kind) == "cross_fitted"


def test_an_undeclared_or_unmatched_operation_keeps_the_in_sample_gate() -> None:
    stage = load_country_spec("uk").sources.stage_map()["hmrc_spi_income_spine"]
    assert declared_gate_calibration(stage, "no_such_operation") == "in_sample"
    assert declared_gate_calibration(stage, "fit_weighted_qrf_stage2", seed=43) == (
        "cross_fitted"
    )
    assert declared_gate_calibration(stage, "fit_weighted_qrf_stage2", seed=0) == (
        "in_sample"
    )


def test_pooled_outcomes_name_one_outcome_or_count_each() -> None:
    folds = [
        _Fitted({"a": "cross_fitted", "b": "cross_fitted", "c": None}),
        _Fitted({"a": "cross_fitted", "b": "in_sample_thin_class", "c": None}),
        _Fitted({"a": "cross_fitted", "b": "cross_fitted", "c": None}),
    ]
    assert gate_outcomes("fit", folds[1]) == {
        "fit:a": "cross_fitted",
        "fit:b": "in_sample_thin_class",
    }
    assert pooled_gate_outcomes("fit", folds) == {
        "fit:a": "cross_fitted",
        "fit:b": "cross_fitted 2, in_sample_thin_class 1",
    }
    assert pooled_gate_outcomes("fit", []) == {}
