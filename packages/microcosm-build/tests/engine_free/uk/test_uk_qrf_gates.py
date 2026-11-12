"""The UK stages' declared QRF sign-gate calibration (microcosm#1359, #1276)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.qrf_gates import (
    declared_gate_calibration,
    declared_gate_level_columns,
    gate_level_reports,
    gate_level_settings_receipt,
    gate_levels,
    gate_outcomes,
    pooled_gate_level_reports,
    pooled_gate_levels,
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
#: The operations fitted on class-balanced weights (microcosm#1276).
BALANCED_OPERATIONS = frozenset({("hmrc_spi_income_spine", "fit_weighted_qrf_stage1")})
#: SPI stage 1's gates offset these predictors' levels (microcosm#1276).
SPI_STAGE1_GATE_LEVEL_COLUMNS = ("spi_age_band", "main_source", "has_pay", "has_trade")


class _Fitted:
    def __init__(
        self,
        outcomes: dict[str, str | None],
        levels: dict[str, tuple[str, ...]] | None = None,
        reports: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self._outcomes = outcomes
        self._levels = levels or {}
        self._reports = reports or {}

    def gate_outcomes(self) -> dict[str, str | None]:
        return dict(self._outcomes)

    def gate_levels(self) -> dict[str, tuple[str, ...]]:
        return dict(self._levels)

    def gate_level_reports(self) -> dict[str, dict[str, object]]:
        return dict(self._reports)


def _report(fit: str, levels, references, thin) -> dict[str, object]:
    return {
        "fit": fit,
        "levels": list(levels),
        "reference_levels": list(references),
        "thin_levels": thin,
    }


def test_the_gated_uk_fits_declare_the_cross_fitted_gate() -> None:
    stages = load_country_spec("uk").sources.stage_map()
    declared = {
        (stage.stage, operation.kind): operation.parameters["gate_calibration"]
        for stage in stages.values()
        for operation in stage.operations
        if "gate_calibration" in operation.parameters
    }
    assert set(declared) == CROSS_FITTED_OPERATIONS
    for stage, kind in CROSS_FITTED_OPERATIONS:
        expected = (
            "cross_fitted_balanced"
            if (stage, kind) in BALANCED_OPERATIONS
            else "cross_fitted"
        )
        assert declared[(stage, kind)] == expected
        assert declared_gate_calibration(stages[stage], kind) == expected


def test_only_spi_stage1_declares_gate_level_columns() -> None:
    stages = load_country_spec("uk").sources.stage_map()
    declared = {
        (stage.stage, operation.kind): tuple(operation.parameters["gate_level_columns"])
        for stage in stages.values()
        for operation in stage.operations
        if "gate_level_columns" in operation.parameters
    }
    assert declared == {
        ("hmrc_spi_income_spine", "fit_weighted_qrf_stage1"): (
            SPI_STAGE1_GATE_LEVEL_COLUMNS
        )
    }
    spine = stages["hmrc_spi_income_spine"]
    assert declared_gate_level_columns(spine, "fit_weighted_qrf_stage1") == (
        SPI_STAGE1_GATE_LEVEL_COLUMNS
    )
    assert declared_gate_level_columns(spine, "fit_weighted_qrf_stage2") == ()
    assert declared_gate_level_columns(spine, "no_such_operation") == ()


def test_declared_gate_level_columns_are_checked() -> None:
    def stage(columns: object) -> SimpleNamespace:
        operation = SimpleNamespace(
            kind="fit", parameters={"gate_level_columns": columns}
        )
        return SimpleNamespace(stage="toy", operations=(operation,))

    with pytest.raises(ValueError, match="repeats a column"):
        declared_gate_level_columns(stage(["a", "a"]), "fit")
    with pytest.raises(ValueError, match="as strings"):
        declared_gate_level_columns(stage(["a", 1]), "fit")


def test_gate_levels_name_each_target_s_offset_levels_or_count_them() -> None:
    folds = [
        _Fitted({}, {"a": ("x=1", "y=0"), "b": ()}),
        _Fitted({}, {"a": ("x=1",), "b": ("x=1",)}),
    ]
    assert gate_levels("fit", folds[0]) == {"fit:a": ["x=1", "y=0"]}
    assert pooled_gate_levels("fit", folds) == {
        "fit:a": {"x=1": 2, "y=0": 1},
        "fit:b": {"x=1": 1},
    }
    assert pooled_gate_levels("fit", []) == {}


def test_level_reports_name_each_fit_s_outcome_and_count_them_over_folds() -> None:
    folds = [
        _Fitted(
            {},
            reports={
                "a": _report("offsets", ["x=1", "y_b=1"], ["x=0", "y_a=1"], ["y_c=1"]),
                "b": _report("no_supported_level", [], ["x=0"], {"count": 400}),
            },
        ),
        _Fitted(
            {},
            reports={
                "a": _report("balance_not_met", ["x=1"], ["x=0", "y_a=1"], []),
                "b": _report("no_supported_level", [], ["x=0"], ["x=1"]),
            },
        ),
    ]
    assert gate_level_reports("fit", folds[0]) == {
        "fit:a": _report("offsets", ["x=1", "y_b=1"], ["x=0", "y_a=1"], ["y_c=1"]),
        "fit:b": _report("no_supported_level", [], ["x=0"], {"count": 400}),
    }
    assert pooled_gate_level_reports("fit", folds) == {
        "fit:a": {
            "fit": {"balance_not_met": 1, "offsets": 1},
            "levels": {"x=1": 2, "y_b=1": 1},
            "reference_levels": {"x=0": 2, "y_a=1": 2},
            "thin_levels": {"y_c=1": 1},
        },
        "fit:b": {
            "fit": {"no_supported_level": 2},
            "levels": {},
            "reference_levels": {"x=0": 2},
            "thin_levels": {"x=1": 1},
            "thin_levels_counted": 400,
        },
    }
    # A fit without level reports (an in-sample model, a test double) adds none.
    assert gate_level_reports("fit", SimpleNamespace()) == {}
    assert pooled_gate_level_reports("fit", []) == {}


def test_the_level_settings_receipt_is_the_fit_s() -> None:
    from microcosm.fit import gate_level_settings

    assert gate_level_settings_receipt() == gate_level_settings()
    assert set(gate_level_settings_receipt()) == {
        "min_rows_per_sign_class",
        "ridge",
        "balance_tolerance",
    }


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
