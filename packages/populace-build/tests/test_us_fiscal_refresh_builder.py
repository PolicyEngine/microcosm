import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from populace.calibrate import TargetRegistry, TargetSpec
from populace.frame import Frame, WeightKind


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _passing_critical_diagnostics(builder) -> tuple[SimpleNamespace, ...]:
    def diagnostic(name, target, final_estimate):
        return SimpleNamespace(
            name=f"{name}@{builder.PERIOD}",
            target=target,
            initial_estimate=target,
            final_estimate=final_estimate,
            relative_error=(final_estimate - target) / target,
        )

    return (
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_amount",
            2_105_345_646_000.0,
            2_067_762_165_736.424,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_returns",
            113_562_590.0,
            105_437_267.69738781,
        ),
        diagnostic(
            "ssa_supplement.cy2024.oasdi_ssi_payments."
            "social_security_benefits.payment_amount",
            1_471_195_000_000.0,
            1_541_540_768_722.367,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.ctc_amount",
            82_863_353_000.0,
            88_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.ctc_claims",
            38_068_980.0,
            40_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.actc_amount",
            33_857_960_000.0,
            35_300_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.actc_claims",
            17_691_400.0,
            17_100_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.eitc_amount",
            59_204_610_000.0,
            63_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.eitc_claims",
            23_692_200.0,
            23_800_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount",
            53_910_190_000.0,
            58_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_returns",
            7_841_370.0,
            8_200_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_amount",
            455_904_900_000.0,
            490_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_returns",
            24_475_100.0,
            26_000_000.0,
        ),
    )


def test_soi_component_amounts_use_source_specific_signs() -> None:
    builder = _load_builder_module()

    assert np.array_equal(
        builder._signed_component(np.array([-5.0, 0.0, 7.0]), "capital_gains_gross"),
        np.array([0.0, 0.0, 7.0]),
    )
    assert np.array_equal(
        builder._signed_component(np.array([-5.0, 0.0, 7.0]), "capital_gains_losses"),
        np.array([0.0, 0.0, 7.0]),
    )
    assert np.array_equal(
        builder._signed_component(np.array([-5.0, 0.0, 7.0]), "business_net_losses"),
        np.array([5.0, -0.0, -0.0]),
    )
    assert np.array_equal(
        builder._signed_component(np.array([-5.0, 7.0]), "adjusted_gross_income"),
        np.array([-5.0, 7.0]),
    )


def test_export_target_audit_is_opt_in(monkeypatch) -> None:
    builder = _load_builder_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
        ],
    )
    args = builder._parse_args()
    assert not args.audit_export_targets

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--audit-export-targets",
        ],
    )
    args = builder._parse_args()
    assert args.audit_export_targets


def test_soi_count_rows_count_positive_component_items() -> None:
    builder = _load_builder_module()

    assert np.array_equal(
        builder._soi_component_row(
            np.array([-5.0, 0.0, 7.0]),
            "capital_gains_gross",
            count=True,
        ),
        np.array([0.0, 0.0, 1.0]),
    )
    assert np.array_equal(
        builder._soi_component_row(
            np.array([-5.0, 0.0, 7.0]),
            "capital_gains_losses",
            count=True,
        ),
        np.array([0.0, 0.0, 1.0]),
    )
    assert np.array_equal(
        builder._soi_component_row(
            np.array([-5.0, 0.0, 7.0]),
            "business_net_losses",
            count=True,
        ),
        np.array([1.0, 0.0, 0.0]),
    )


def test_soi_eitc_child_count_filter_uses_ledger_filter_first() -> None:
    builder = _load_builder_module()

    assert (
        builder._soi_eitc_child_count_filter(
            {
                "ledger_filter_eitc_child_count": "2",
                "source_measure_id": "eitc_no_children_amount",
            }
        )
        == "2"
    )
    assert (
        builder._soi_eitc_child_count_filter(
            {"source_measure_id": "eitc_no_children_amount"}
        )
        == "0"
    )
    assert (
        builder._soi_eitc_child_count_filter(
            {"source_measure_id": "eitc_one_child_claims"}
        )
        == "1"
    )
    assert (
        builder._soi_eitc_child_count_filter(
            {"source_measure_id": "eitc_two_children_amount"}
        )
        == "2"
    )
    assert (
        builder._soi_eitc_child_count_filter(
            {"source_measure_id": "eitc_three_or_more_children_claims"}
        )
        == "3plus"
    )
    assert (
        builder._soi_eitc_child_count_filter(
            {
                "ledger_layout_record_set_id": (
                    "irs_soi.ty2022.table_2_5.eitc_by_agi_children."
                    "no_qualifying_children"
                ),
                "source_measure_id": "eitc_total",
            }
        )
        == "0"
    )
    assert (
        builder._soi_eitc_child_count_filter(
            {
                "ledger_layout_record_set_id": (
                    "irs_soi.ty2022.table_2_5.eitc_by_agi_children."
                    "three_or_more_qualifying_children"
                ),
                "source_measure_id": "eitc_total",
            }
        )
        == "3plus"
    )
    assert (
        builder._soi_eitc_child_count_filter({"source_measure_id": "eitc_total"})
        is None
    )


def test_unsupported_soi_ledger_filters_require_materializer_support() -> None:
    builder = _load_builder_module()

    assert (
        builder._unsupported_soi_ledger_filters(
            {
                "ledger_filter_income_range": "25k_to_30k",
                "ledger_filter_filing_status": "all",
                "ledger_filter_eitc_child_count": "1",
            }
        )
        == ()
    )
    assert (
        builder._unsupported_soi_ledger_filters(
            {
                "ledger_filter_new_dimension": "all",
            }
        )
        == ()
    )
    assert builder._unsupported_soi_ledger_filters(
        {
            "ledger_filter_new_dimension": "specific_slice",
        }
    ) == ("ledger_filter_new_dimension",)


def test_eitc_child_count_mask_supports_soi_child_groups() -> None:
    builder = _load_builder_module()
    counts = np.asarray([0, 1, 2, 3, 4], dtype=np.float64)

    assert np.array_equal(
        builder._eitc_child_count_mask(counts, "0"),
        np.asarray([True, False, False, False, False]),
    )
    assert np.array_equal(
        builder._eitc_child_count_mask(counts, "1"),
        np.asarray([False, True, False, False, False]),
    )
    assert np.array_equal(
        builder._eitc_child_count_mask(counts, "2"),
        np.asarray([False, False, True, False, False]),
    )
    assert np.array_equal(
        builder._eitc_child_count_mask(counts, "3plus"),
        np.asarray([False, False, False, True, True]),
    )
    assert np.array_equal(
        builder._eitc_child_count_mask(counts, "3+"),
        np.asarray([False, False, False, True, True]),
    )


def test_combined_household_values_unions_positive_person_support(small_frame) -> None:
    builder = _load_builder_module()

    variable_values = {
        "medicaid_enrolled": np.asarray([1.0, 1.0, 0.0, 0.0]),
        "chip_enrolled": np.asarray([1.0, 0.0, 1.0, 0.0]),
    }

    class FakeSimulation:
        def calculate(self, variable, *, period, map_to=None):
            assert period == builder.PERIOD
            assert map_to is None
            return variable_values[variable]

    person_entity = SimpleNamespace(key="person")
    system = SimpleNamespace(
        variables={
            variable: SimpleNamespace(entity=person_entity)
            for variable in variable_values
        }
    )

    values = builder._combined_household_values(
        frame=small_frame,
        simulation=FakeSimulation(),
        system=system,
        variables=("medicaid_enrolled", "chip_enrolled"),
        tax_unit_positions=np.asarray([], dtype=np.int64),
        positive_indicator=True,
    )
    assert np.array_equal(values, np.asarray([2.0, 1.0]))

    summed_values = builder._combined_household_values(
        frame=small_frame,
        simulation=FakeSimulation(),
        system=system,
        variables=("medicaid_enrolled", "chip_enrolled"),
        tax_unit_positions=np.asarray([], dtype=np.int64),
        positive_indicator=False,
    )
    assert np.array_equal(summed_values, np.asarray([3.0, 1.0]))


def test_combined_household_values_can_count_tax_unit_variable_on_people(
    small_frame,
) -> None:
    builder = _load_builder_module()

    mapped_values = {
        "assigned_aca_ptc": np.asarray([5_000.0, 5_000.0, 3_000.0, 0.0]),
        "is_aca_ptc_eligible": np.asarray([1.0, 0.0, 1.0, 1.0]),
    }

    class FakeSimulation:
        def calculate(self, variable, *, period, map_to=None):
            assert period == builder.PERIOD
            assert map_to == "person"
            return mapped_values[variable]

    system = SimpleNamespace(
        variables={
            "assigned_aca_ptc": SimpleNamespace(entity=SimpleNamespace(key="tax_unit")),
            "is_aca_ptc_eligible": SimpleNamespace(
                entity=SimpleNamespace(key="person")
            ),
        }
    )

    values = builder._combined_household_values(
        frame=small_frame,
        simulation=FakeSimulation(),
        system=system,
        variables=("assigned_aca_ptc",),
        tax_unit_positions=np.asarray([], dtype=np.int64),
        positive_indicator=True,
        map_to="person",
        filter_variable="is_aca_ptc_eligible",
    )

    assert np.array_equal(values, np.asarray([1.0, 1.0]))


def test_combined_household_values_threshold_count_keeps_domain_filter(
    small_frame,
) -> None:
    builder = _load_builder_module()

    mapped_values = {
        "selected_marketplace_plan_benchmark_ratio": np.asarray([0.8, 1.2, 0.7]),
        "assigned_aca_ptc": np.asarray([500.0, 500.0, 0.0]),
    }

    class FakeSimulation:
        def calculate(self, variable, *, period, map_to=None):
            assert period == builder.PERIOD
            assert map_to is None
            return mapped_values[variable]

    system = SimpleNamespace(
        variables={
            "selected_marketplace_plan_benchmark_ratio": SimpleNamespace(
                entity=SimpleNamespace(key="tax_unit")
            ),
            "assigned_aca_ptc": SimpleNamespace(entity=SimpleNamespace(key="tax_unit")),
        }
    )

    values = builder._combined_household_values(
        frame=small_frame,
        simulation=FakeSimulation(),
        system=system,
        variables=("selected_marketplace_plan_benchmark_ratio",),
        tax_unit_positions=np.asarray([0, 0, 1], dtype=np.int64),
        filter_variable="assigned_aca_ptc",
        less_than=1.0,
    )

    assert np.array_equal(values, np.asarray([1.0, 0.0]))


def test_release_gate_failures_are_not_unconditional() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder),
        initial_loss=10.0,
        final_loss=5.0,
    )

    assert builder._release_gate_failures(result, {"dropped_target_names": []}) == []

    assert builder._release_gate_failures(
        result,
        {"dropped_target_names": ["missing"]},
    ) == ["1 fiscal targets were not materialized."]

    skipped = SimpleNamespace(target=SimpleNamespace(name="skipped"), reason="bad")
    with_skipped = SimpleNamespace(
        skipped=(skipped,),
        diagnostics=_passing_critical_diagnostics(builder),
        initial_loss=10.0,
        final_loss=5.0,
    )
    assert builder._release_gate_failures(
        with_skipped,
        {"dropped_target_names": []},
    ) == ["1 fiscal targets were skipped by calibration."]

    worse = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder),
        initial_loss=5.0,
        final_loss=10.0,
    )
    assert builder._release_gate_failures(worse, {"dropped_target_names": []}) == [
        "Calibration final loss is worse than the initial loss (10.0 > 5.0)."
    ]


def test_release_gate_failures_include_target_profile_coverage() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder),
        initial_loss=10.0,
        final_loss=5.0,
    )
    target_profile_gate = builder.GateResult(
        name="target_profile_coverage",
        passed=False,
        failures=("medicaid_chip_enrollment: missing",),
    )

    assert builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
        target_profile_gate,
    ) == [
        "Target profile coverage failed: medicaid_chip_enrollment: missing",
    ]


def test_release_gate_failures_include_health_input_signal() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder),
        initial_loss=10.0,
        final_loss=5.0,
    )
    health_input_gate = builder.GateResult(
        name="health_input_signal",
        passed=False,
        failures=("takes_up_aca_if_eligible: constant",),
    )

    assert builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
        health_input_gate=health_input_gate,
    ) == [
        "Health input signal failed: takes_up_aca_if_eligible: constant",
    ]


def test_base_population_scale_gate_rejects_underweighted_base(small_frame) -> None:
    builder = _load_builder_module()

    gate = builder._base_population_scale_gate(small_frame)

    assert not gate.passed
    assert gate.name == "base_population_scale"
    assert gate.details["population"] == 6000.0
    assert "mass='conserve'" in gate.failures[0]


def test_base_population_scale_gate_accepts_national_scale_base(small_frame) -> None:
    builder = _load_builder_module()
    benchmark = builder.US_BASE_PERSON_POPULATION_BENCHMARK
    frame = small_frame.with_weights(
        "household",
        builder.Weights(
            values=np.asarray([benchmark / 4.0, benchmark / 4.0]),
            kind=WeightKind.DESIGN,
        ),
        mass=builder.MassChange(
            factor=benchmark / 6000.0,
            reason="test fixture national-scale base",
        ),
    )

    gate = builder._base_population_scale_gate(frame)

    assert gate.passed
    assert gate.details["population"] == benchmark
    assert gate.details["relative_error"] == 0.0


def test_release_gate_failures_reject_positive_zero_support_targets() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=(
            SimpleNamespace(
                name=f"nation/irs/zero@{builder.PERIOD}",
                target=1_000.0,
                initial_estimate=0.0,
                final_estimate=0.0,
            ),
            SimpleNamespace(
                name=f"nation/irs/nonzero@{builder.PERIOD}",
                target=1_000.0,
                initial_estimate=10.0,
                final_estimate=20.0,
            ),
            *_passing_critical_diagnostics(builder),
        ),
        initial_loss=10.0,
        final_loss=5.0,
    )

    assert builder._release_gate_failures(result, {"dropped_target_names": []}) == [
        "1 positive fiscal targets have zero materialized support "
        f"(examples: nation/irs/zero@{builder.PERIOD})."
    ]


def test_release_gate_failures_reject_bad_critical_target_fit() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=(
            SimpleNamespace(
                name=(
                    "irs_soi.ty2022.historic_table_2.us.all."
                    f"income_tax_liability_amount@{builder.PERIOD}"
                ),
                target=2_105_345_646_000.0,
                initial_estimate=2_000_000_000_000.0,
                final_estimate=735_173_331_468.564,
                relative_error=0.0,
            ),
            *_passing_critical_diagnostics(builder)[1:],
        ),
        initial_loss=10.0,
        final_loss=5.0,
    )

    failures = builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
    )

    assert len(failures) == 2
    assert "stale relative_error" in failures[0]
    assert "federal income tax liability amount" in failures[1]
    assert "relative_error=-0.650806" in failures[1]


def test_release_gate_failures_reject_missing_critical_targets() -> None:
    builder = _load_builder_module()
    result = SimpleNamespace(
        skipped=(),
        diagnostics=_passing_critical_diagnostics(builder)[1:],
        initial_loss=10.0,
        final_loss=5.0,
    )

    failures = builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
    )

    assert failures == [
        "Critical fiscal target "
        "'irs_soi.ty2022.historic_table_2.us.all."
        f"income_tax_liability_amount@{builder.PERIOD}' "
        "(federal income tax liability amount) is missing from calibration "
        "diagnostics."
    ]


def test_fiscal_target_loss_weights_ignore_roles_and_geography() -> None:
    builder = _load_builder_module()
    registry = TargetRegistry(
        (
            TargetSpec(
                name="national_critical_role",
                entity="household",
                value=100.0,
                source="fixture",
                metadata={"target_role": "federal_income_tax_total"},
            ),
            TargetSpec(
                name="state_role_row",
                entity="household",
                value=100.0,
                source="fixture",
                metadata={"state_fips": "06", "target_role": "tanf_total"},
            ),
            TargetSpec(
                name="ordinary_distribution_row",
                entity="household",
                value=100.0,
                source="fixture",
            ),
        ),
        country="us",
    )

    weights = builder._fiscal_target_loss_weights(registry)

    assert weights.shape == (3,)
    assert weights.mean() == 1.0
    assert np.array_equal(weights, np.ones(3))


def test_fiscal_target_loss_weights_scale_by_sqrt_value_within_basis() -> None:
    builder = _load_builder_module()
    registry = TargetRegistry(
        (
            TargetSpec(
                name="amount_small",
                entity="household",
                value=100.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="amount_large",
                entity="household",
                value=300.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="returns_small",
                entity="household",
                value=10.0,
                source="fixture",
                metadata={
                    "source_measure_id": "income_tax_liability_returns",
                    "count": "true",
                },
            ),
            TargetSpec(
                name="returns_large",
                entity="household",
                value=30.0,
                source="fixture",
                metadata={"source_measure_id": "ctc_claims", "count": "true"},
            ),
        ),
        country="us",
    )

    weights = builder._fiscal_target_loss_weights(registry)

    assert weights.mean() == 1.0
    assert np.isclose(weights[1] / weights[0], np.sqrt(3.0))
    assert np.isclose(weights[3] / weights[2], np.sqrt(3.0))
    assert weights[0] == weights[2]
    assert weights[1] == weights[3]


def test_fiscal_target_loss_weights_split_evenly_between_amount_and_count() -> None:
    builder = _load_builder_module()
    registry = TargetRegistry(
        (
            TargetSpec(
                name="amount_small",
                entity="household",
                value=100.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="amount_large",
                entity="household",
                value=300.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="returns",
                entity="household",
                value=10.0,
                source="fixture",
                metadata={"source_measure_id": "ctc_claims", "count": "true"},
            ),
        ),
        country="us",
    )

    weights = builder._fiscal_target_loss_weights(registry)
    bases = np.asarray(
        [builder._fiscal_target_value_basis(spec) for spec in registry.specs],
        dtype=object,
    )

    assert weights.mean() == 1.0
    assert weights[bases == "amount"].sum() == weights[bases == "count"].sum()
    assert np.isclose(weights[1] / weights[0], np.sqrt(3.0))


def test_fiscal_target_loss_weights_floor_zero_subunit_and_abs_values() -> None:
    builder = _load_builder_module()
    registry = TargetRegistry(
        (
            TargetSpec(
                name="zero",
                entity="household",
                value=0.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="subunit",
                entity="household",
                value=0.25,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="negative",
                entity="household",
                value=-9.0,
                source="fixture",
                signed=True,
                metadata={"source_measure_id": "payment_amount"},
            ),
        ),
        country="us",
    )

    weights = builder._fiscal_target_loss_weights(registry)

    assert weights.mean() == 1.0
    assert weights[0] == weights[1]
    assert np.isclose(weights[2] / weights[0], 3.0)


def test_fiscal_target_value_basis_uses_only_amount_and_count() -> None:
    builder = _load_builder_module()
    amount = TargetSpec(
        name="amount",
        entity="household",
        value=100.0,
        source="fixture",
        metadata={"source_measure_id": "payment_amount"},
    )
    return_count = TargetSpec(
        name="return_count",
        entity="household",
        value=100.0,
        source="fixture",
        metadata={"source_measure_id": "ctc_claims", "count": "true"},
    )
    person_count = TargetSpec(
        name="person_count",
        entity="household",
        value=100.0,
        source="fixture",
        metadata={
            "measure_mode": "positive_count",
            "source_measure_id": "aptc_recipients",
            "target_role": "aca_ptc_recipients",
            "count_map_to": "person",
        },
    )
    bronze_count = TargetSpec(
        name="bronze_count",
        entity="household",
        value=100.0,
        source="fixture",
        metadata={
            "measure_mode": "less_than_count",
            "source_measure_id": "bronze_aptc_consumers",
            "target_role": "aca_bronze_aptc_consumers",
        },
    )

    assert builder._fiscal_target_value_basis(amount) == "amount"
    assert builder._fiscal_target_value_basis(return_count) == "count"
    assert builder._fiscal_target_value_basis(person_count) == "count"
    assert builder._fiscal_target_value_basis(bronze_count) == "count"


def test_release_calibration_diagnostics_include_gate_failures(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    captured: dict[str, object] = {}

    def fake_write_calibration_diagnostics(result, path, *, target_registry, build):
        captured["result"] = result
        captured["path"] = path
        captured["target_registry"] = target_registry
        captured["build"] = build
        return path

    monkeypatch.setattr(builder, "_sha256", lambda path: "base-sha")
    monkeypatch.setattr(
        builder, "write_calibration_diagnostics", fake_write_calibration_diagnostics
    )
    result = SimpleNamespace()
    registry = TargetRegistry((), country="us")
    profile_gate = SimpleNamespace(passed=True, failures=(), details={"n": 1})
    health_gate = SimpleNamespace(passed=True, failures=(), details={"n": 2})
    base_population_gate = SimpleNamespace(
        passed=True,
        failures=(),
        details={"population": 334_200_000.0},
    )

    builder._write_release_calibration_diagnostics(
        result=result,
        release_dir=tmp_path,
        registry=registry,
        base_h5=tmp_path / "base.h5",
        compilation={"dropped_target_names": []},
        target_profile_gate=profile_gate,
        health_input_gate=health_gate,
        base_population_gate=base_population_gate,
        audit_export_targets=False,
        gate_failures=["ctc failed"],
    )

    assert captured["path"] == tmp_path / "calibration_diagnostics.json"
    build = captured["build"]
    assert build["base_dataset_sha256"] == "base-sha"
    assert build["target_loss_weighting"].endswith("_cap_100pct")
    assert build["target_loss_cap"] == 1.0
    assert build["release_gates"] == {
        "passed": False,
        "failures": ["ctc failed"],
    }
    assert build["health_input_signal"] == {
        "passed": True,
        "failures": [],
        "details": {"n": 2},
    }
    assert build["base_population_scale"] == {
        "passed": True,
        "failures": [],
        "details": {"population": 334_200_000.0},
    }


def test_main_writes_diagnostics_before_post_calibration_gate_failure(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    release_id = "populace-us-2024-gate-failure-test"
    base_h5 = tmp_path / "base.h5"
    facts = tmp_path / "facts.jsonl"
    out = tmp_path / "out"
    base_h5.write_bytes(b"h5")
    facts.write_text("{}\n")
    target_spec = TargetSpec(
        name="amount",
        entity="household",
        measure="income",
        value=100.0,
        source="fixture",
        metadata={"source_measure_id": "payment_amount"},
    )
    registry = TargetRegistry((target_spec,), country="us")
    result = SimpleNamespace(
        skipped=(),
        diagnostics=(),
        initial_loss=2.0,
        final_loss=1.0,
    )
    captured: dict[str, object] = {}

    class FakeFrame:
        pass

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--base-h5",
            str(base_h5),
            "--ledger-facts",
            str(facts),
            "--out",
            str(out),
            "--release-id",
            release_id,
        ],
    )
    monkeypatch.setattr(builder, "_git_dirty", lambda: False)
    monkeypatch.setattr(builder, "_sha256", lambda path: "base-sha")
    monkeypatch.setattr(builder, "_git_output", lambda *args: "commit")
    monkeypatch.setattr(builder, "_load_ledger_facts", lambda path: ({"fact": 1},))
    monkeypatch.setattr(
        builder,
        "compile_us_fiscal_target_registry",
        lambda facts, *, target_period: registry,
    )
    monkeypatch.setattr(
        builder,
        "target_profile_coverage_gate",
        lambda specs, requirements: builder.GateResult(
            name="target_profile_coverage",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(builder, "_load_frame", lambda path: FakeFrame())
    monkeypatch.setattr(
        builder,
        "_base_population_scale_gate",
        lambda frame: builder.GateResult(
            name="base_population_scale",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "_with_aca_marketplace_source_outputs",
        lambda frame, specs, *, seed: frame,
    )
    monkeypatch.setattr(
        builder,
        "_health_input_signal_gate",
        lambda frame: builder.GateResult(
            name="health_input_signal",
            passed=True,
            details={"checked": True},
        ),
    )
    monkeypatch.setattr(
        builder,
        "_materialize_target_frame",
        lambda frame, specs: (
            frame,
            registry,
            {"dropped_target_names": []},
        ),
    )

    def fake_calibrate(*args, **kwargs):
        captured["target_loss_weights"] = kwargs["target_loss_weights"]
        captured["target_loss_cap"] = kwargs["target_loss_cap"]
        return result

    def fake_write_diagnostics(**kwargs):
        captured["diagnostics"] = kwargs
        release_dir = kwargs["release_dir"]
        release_dir.mkdir(parents=True, exist_ok=True)
        (release_dir / "calibration_diagnostics.json").write_text("{}")
        return release_dir / "calibration_diagnostics.json"

    monkeypatch.setattr(builder, "calibrate", fake_calibrate)
    monkeypatch.setattr(
        builder,
        "_release_gate_failures",
        lambda *args: ["ctc failed"],
    )
    monkeypatch.setattr(
        builder,
        "_write_release_calibration_diagnostics",
        fake_write_diagnostics,
    )

    try:
        builder.main()
    except RuntimeError as exc:
        assert str(exc) == "Release gates failed: ctc failed"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected post-calibration gate failure.")

    release_dir = out / "releases" / release_id
    assert (release_dir / "calibration_diagnostics.json").exists()
    assert captured["diagnostics"]["gate_failures"] == ["ctc failed"]
    assert captured["target_loss_cap"] == 1.0
    assert np.array_equal(captured["target_loss_weights"], np.asarray([1.0]))


def test_release_gate_failures_reject_bad_national_credit_and_ss_fits() -> None:
    builder = _load_builder_module()
    cases = (
        (
            "irs_soi.ty2022.historic_table_2.us.all.ctc_amount",
            "Child Tax Credit amount",
            82_863_353_000.0,
            99_282_300_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.ctc_claims",
            "Child Tax Credit claims",
            38_068_980.0,
            43_994_700.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.eitc_amount",
            "Earned Income Tax Credit amount",
            59_204_610_000.0,
            70_208_900_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount",
            "Premium Tax Credit amount",
            53_910_190_000.0,
            84_823_800_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_returns",
            "Premium Tax Credit returns",
            7_841_370.0,
            11_637_100.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_amount",
            "taxable Social Security amount",
            455_904_900_000.0,
            540_351_000_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_returns",
            "taxable Social Security returns",
            24_475_100.0,
            31_887_700.0,
        ),
    )

    for target_name, label, target, final_estimate in cases:
        diagnostics = list(_passing_critical_diagnostics(builder))
        name = f"{target_name}@{builder.PERIOD}"
        index = next(
            i for i, diagnostic in enumerate(diagnostics) if diagnostic.name == name
        )
        diagnostics[index] = SimpleNamespace(
            name=name,
            target=target,
            initial_estimate=target,
            final_estimate=final_estimate,
            relative_error=(final_estimate - target) / target,
        )
        result = SimpleNamespace(
            skipped=(),
            diagnostics=tuple(diagnostics),
            initial_loss=10.0,
            final_loss=5.0,
        )

        failures = builder._release_gate_failures(
            result,
            {"dropped_target_names": []},
        )

        assert len(failures) == 1
        assert label in failures[0]
        assert "exceeding 0.1" in failures[0]


def test_critical_gate_allows_improvement_over_incumbent_diagnostics() -> None:
    builder = _load_builder_module()
    name = f"irs_soi.ty2022.historic_table_2.us.all.ctc_amount@{builder.PERIOD}"
    target = 82_863_353_000.0
    diagnostics = list(_passing_critical_diagnostics(builder))
    index = next(
        i for i, diagnostic in enumerate(diagnostics) if diagnostic.name == name
    )
    diagnostics[index] = SimpleNamespace(
        name=name,
        target=target,
        initial_estimate=99_315_000_000.0,
        final_estimate=99_282_300_000.0,
        relative_error=(99_282_300_000.0 - target) / target,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=tuple(diagnostics),
        initial_loss=10.0,
        final_loss=5.0,
    )
    incumbent = {
        name: {
            "target": target,
            "final_estimate": 134_904_000_000.0,
        }
    }

    assert (
        builder._release_gate_failures(
            result,
            {"dropped_target_names": []},
            incumbent_diagnostics=incumbent,
        )
        == []
    )


def test_critical_gate_rejects_miss_when_incumbent_is_better() -> None:
    builder = _load_builder_module()
    name = f"irs_soi.ty2022.historic_table_2.us.all.ctc_amount@{builder.PERIOD}"
    target = 82_863_353_000.0
    diagnostics = list(_passing_critical_diagnostics(builder))
    index = next(
        i for i, diagnostic in enumerate(diagnostics) if diagnostic.name == name
    )
    diagnostics[index] = SimpleNamespace(
        name=name,
        target=target,
        initial_estimate=99_315_000_000.0,
        final_estimate=99_282_300_000.0,
        relative_error=(99_282_300_000.0 - target) / target,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=tuple(diagnostics),
        initial_loss=10.0,
        final_loss=5.0,
    )
    incumbent = {
        name: {
            "target": target,
            "final_estimate": 90_000_000_000.0,
        }
    }

    failures = builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
        incumbent_diagnostics=incumbent,
    )

    assert len(failures) == 1
    assert "Child Tax Credit amount" in failures[0]
    assert "incumbent_relative_error=" in failures[0]


def test_health_input_signal_gate_rejects_degenerate_aca_inputs() -> None:
    builder = _load_builder_module()

    class FakeFrame:
        def table(self, name):
            assert name == "tax_unit"
            return pd.DataFrame(
                {
                    "takes_up_aca_if_eligible": [True, True, True],
                    "selected_marketplace_plan_benchmark_ratio": [1.0, 1.0, 1.0],
                }
            )

    gate = builder._health_input_signal_gate(FakeFrame())

    assert not gate.passed
    assert gate.name == "health_input_signal"
    assert len(gate.failures) == 2
    assert any("takes_up_aca_if_eligible" in failure for failure in gate.failures)
    assert any(
        "selected_marketplace_plan_benchmark_ratio" in failure
        for failure in gate.failures
    )


def test_health_input_signal_gate_accepts_varied_aca_inputs() -> None:
    builder = _load_builder_module()

    class FakeFrame:
        def table(self, name):
            assert name == "tax_unit"
            return pd.DataFrame(
                {
                    "takes_up_aca_if_eligible": [True, False, True],
                    "selected_marketplace_plan_benchmark_ratio": [1.0, 0.8, 1.2],
                }
            )

    gate = builder._health_input_signal_gate(FakeFrame())

    assert gate.passed
    assert gate.details["unique_counts"] == {
        "selected_marketplace_plan_benchmark_ratio": 3,
        "takes_up_aca_if_eligible": 2,
    }


def test_aca_source_runtime_refreshes_degenerate_release_inputs(monkeypatch) -> None:
    builder = _load_builder_module()
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 10, 20], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
            "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
            "person_marital_unit_id": np.asarray([10000, 10000, 20000], dtype="int64"),
            "has_marketplace_health_coverage_at_interview": [False, False, True],
        }
    )
    frame = Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2], dtype="int64"),
                    "state_fips": np.asarray([1, 1]),
                }
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": np.asarray([10, 20], dtype="int64"),
                    "stable_tax_unit_draw": [0.1, 0.2],
                    "takes_up_aca_if_eligible": [False, False],
                    "selected_marketplace_plan_benchmark_ratio": [1.0, 1.0],
                }
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": [100, 200]}),
            "family": pd.DataFrame({"family_id": [1000, 2000]}),
            "marital_unit": pd.DataFrame({"marital_unit_id": [10000, 20000]}),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([1.0, 1.0]),
                kind=WeightKind.DESIGN,
            )
        },
    )
    specs = (
        TargetSpec(
            name="cms_aca.oep2024.state_marketplace.al.aptc_recipients",
            entity="household",
            measure="takes_up_aca_if_eligible",
            value=3.0,
            source="CMS Marketplace OEP",
            family="cms_aca",
            metadata={"target_role": "aca_ptc_recipients", "state_fips": "01"},
        ),
    )
    values = {
        "is_aca_ptc_eligible": np.asarray([1.0, 1.0]),
        "is_aca_ptc_eligible:person": np.asarray([1.0, 1.0, 1.0]),
        "health_insurance_premiums_without_medicare_part_b": np.asarray(
            [400.0, 1200.0]
        ),
        "assigned_aca_ptc": np.asarray([0.0, 0.0]),
        "aca_ptc": np.asarray([100.0, 0.0]),
        "slcsp": np.asarray([1000.0, 1000.0]),
    }

    def fake_calculate_array(simulation, variable, *, map_to=None):
        assert simulation is fake_simulation
        assert map_to in {"tax_unit", "person"}
        if map_to == "person":
            return values[f"{variable}:person"]
        return values[variable]

    fake_simulation = object()
    monkeypatch.setattr(builder, "_calculate_array", fake_calculate_array)

    refreshed = builder._with_aca_marketplace_source_outputs(
        frame,
        specs,
        seed=42,
        simulation=fake_simulation,
    )

    tax_unit = refreshed.table("tax_unit")
    assigned = tax_unit.set_index("tax_unit_id")["takes_up_aca_if_eligible"]
    assert bool(assigned.loc[10]) is True
    assert bool(assigned.loc[20]) is False
    assert tax_unit["takes_up_aca_if_eligible"].nunique() == 2
    assert tax_unit["selected_marketplace_plan_benchmark_ratio"].nunique() == 2
    person_counts = person.assign(
        assigned=person["person_tax_unit_id"].map(assigned).fillna(False)
    )
    assert float(person_counts["assigned"].sum()) == 2.0
    assert builder._health_input_signal_gate(refreshed).passed
    assert frame.table("tax_unit")["takes_up_aca_if_eligible"].nunique() == 1
    assert (
        frame.table("tax_unit")["selected_marketplace_plan_benchmark_ratio"].nunique()
        == 1
    )


def test_aca_source_runtime_rejects_enrollment_only_fallback() -> None:
    builder = _load_builder_module()
    specs = (
        TargetSpec(
            name="cms_aca.oep2024.state_marketplace.al.marketplace_enrollment",
            entity="household",
            measure="has_marketplace_health_coverage_at_interview",
            value=2.0,
            source="CMS Marketplace OEP",
            family="cms_aca",
            metadata={"target_role": "aca_enrollment", "state_fips": "01"},
        ),
    )

    try:
        builder._with_aca_marketplace_source_outputs(
            object(),
            specs,
            seed=42,
            simulation=object(),
        )
    except RuntimeError as exc:
        assert "requires an APTC-recipient target" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected enrollment-only ACA source refresh to fail.")


def test_jct_materialization_collapses_reform_tax_units_and_clears_caches(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1, 2, 3], dtype="int64"),
                    "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
                    "person_tax_unit_id": np.asarray([10, 20, 30], dtype="int64"),
                    "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
                    "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
                    "person_marital_unit_id": np.asarray(
                        [10000, 20000, 30000], dtype="int64"
                    ),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2], dtype="int64"),
                    "state_fips": np.asarray([6, 36], dtype="int64"),
                }
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([10, 20, 30], dtype="int64")}
            ),
            "spm_unit": pd.DataFrame(
                {"spm_unit_id": np.asarray([100, 200], dtype="int64")}
            ),
            "family": pd.DataFrame(
                {"family_id": np.asarray([1000, 2000], dtype="int64")}
            ),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([10000, 20000, 30000], dtype="int64")}
            ),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([1.0, 1.0]), kind=WeightKind.DESIGN
            )
        },
    )
    target = TargetSpec(
        name=f"jct.mock_tax_expenditure@{builder.PERIOD}",
        entity="household",
        measure="jct_mock_tax_expenditure",
        value=-45.0,
        source="Mock JCT",
        family="jct",
        signed=True,
    )
    reform_spec = SimpleNamespace(
        neutralized_variable="mock_credit", measure="jct_mock_tax_expenditure"
    )
    datasets = []
    simulations = []

    class FakeVariable:
        entity = SimpleNamespace(key="tax_unit")

    class FakeSystem:
        variables = {
            "state_income_tax": FakeVariable(),
            "mock_credit": FakeVariable(),
        }

    class FakeMicrosimulation:
        def __init__(self, *, dataset, reform=None):
            self.dataset = dataset
            self.reform = reform
            self.cache_invalidations = 0
            simulations.append(self)

        def calculate(self, variable, *, period, **kwargs):
            assert period == builder.PERIOD
            if self.reform is not None:
                assert variable == "income_tax"
                assert kwargs == {}
                return np.asarray([90.0, 25.0, 40.0])
            arrays = {
                "income_tax": np.asarray([100.0, 30.0, 70.0]),
                "taxable_income": np.asarray([1000.0, 2000.0, 3000.0]),
                "adjusted_gross_income": np.asarray([1100.0, 2100.0, 3100.0]),
                "filing_status": np.asarray(["SINGLE", "SINGLE", "SINGLE"]),
                "state_income_tax": np.asarray([5.0, 6.0, 7.0]),
            }
            assert kwargs == {}
            return arrays[variable]

        def _invalidate_all_caches(self):
            self.cache_invalidations += 1

    def fake_dataset_from_frame(frame_arg, *, zero_variables=(), system=None):
        datasets.append((frame_arg, tuple(zero_variables), system))
        return {"zero_variables": tuple(zero_variables)}

    def fake_make_zero_variable_reform(system, variable_name):
        assert isinstance(system, FakeSystem)
        assert variable_name == "mock_credit"
        return object()

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=FakeSystem,
            Microsimulation=FakeMicrosimulation,
        ),
    )
    monkeypatch.setattr(builder, "_dataset_from_frame", fake_dataset_from_frame)
    monkeypatch.setattr(
        builder, "_make_zero_variable_reform", fake_make_zero_variable_reform
    )
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", (reform_spec,))
    monkeypatch.setattr(builder, "SOI_VARIABLE_MAP", {})

    target_frame, registry, dropped = builder._materialize_target_frame(
        frame, (target,)
    )

    household = target_frame.table("household")
    assert np.array_equal(household["income_tax"], np.asarray([130.0, 70.0]))
    assert np.array_equal(
        household["jct_mock_tax_expenditure"], np.asarray([-15.0, -30.0])
    )
    assert len(registry) == 1
    assert dropped["dropped_target_names"] == []
    assert [dataset[1] for dataset in datasets] == [(), ("mock_credit",)]
    assert len(simulations) == 2
    assert [simulation.cache_invalidations for simulation in simulations] == [1, 1]


def test_soi_eitc_child_targets_materialize_distinct_child_slices(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1, 2, 3], dtype="int64"),
                    "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
                    "person_tax_unit_id": np.asarray([10, 20, 30], dtype="int64"),
                    "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
                    "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
                    "person_marital_unit_id": np.asarray(
                        [10000, 20000, 30000], dtype="int64"
                    ),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2], dtype="int64"),
                    "state_fips": np.asarray([6, 6], dtype="int64"),
                }
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([10, 20, 30], dtype="int64")}
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
            "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([10000, 20000, 30000])}
            ),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([1.0, 1.0]), kind=WeightKind.DESIGN
            )
        },
    )

    def eitc_spec(name, measure, child_filter, *, count=False):
        metadata = {
            "variable": "eitc",
            "agi_lower_bound": "-inf",
            "agi_upper_bound": "inf",
            "filing_status": "All",
            "source_measure_id": "eitc_returns" if count else "eitc_total",
            "ledger_filter_eitc_child_count": child_filter,
        }
        if count:
            metadata["count"] = "true"
        return TargetSpec(
            name=name,
            entity="household",
            measure=measure,
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata=metadata,
        )

    targets = (
        eitc_spec("no_child_amount", "no_child_amount", "0"),
        eitc_spec("two_child_amount", "two_child_amount", "2"),
        eitc_spec("three_plus_amount", "three_plus_amount", "3plus"),
        eitc_spec("two_child_returns", "two_child_returns", "2", count=True),
    )

    class FakeVariable:
        entity = SimpleNamespace(key="tax_unit")

    class FakeSystem:
        variables = {
            name: FakeVariable()
            for name in (
                "income_tax",
                "taxable_income",
                "adjusted_gross_income",
                "filing_status",
                "state_income_tax",
                "eitc",
                "eitc_child_count",
            )
        }

    class FakeMicrosimulation:
        def __init__(self, *, dataset, reform=None):
            self.dataset = dataset
            self.reform = reform

        def calculate(self, variable, *, period, **kwargs):
            assert period == builder.PERIOD
            assert kwargs == {}
            arrays = {
                "income_tax": np.asarray([0.0, 0.0, 0.0]),
                "taxable_income": np.asarray([0.0, 0.0, 0.0]),
                "adjusted_gross_income": np.asarray([10_000.0, 20_000.0, 30_000.0]),
                "filing_status": np.asarray(["SINGLE", "SINGLE", "SINGLE"]),
                "state_income_tax": np.asarray([0.0, 0.0, 0.0]),
                "eitc": np.asarray([100.0, 200.0, 300.0]),
                "eitc_child_count": np.asarray([0.0, 2.0, 3.0]),
            }
            return arrays[variable]

        def _invalidate_all_caches(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=FakeSystem,
            Microsimulation=FakeMicrosimulation,
        ),
    )
    monkeypatch.setattr(builder, "_dataset_from_frame", lambda *args, **kwargs: {})
    monkeypatch.setattr(builder, "SOI_VARIABLE_MAP", {"eitc": "eitc"})
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", ())

    target_frame, registry, compilation = builder._materialize_target_frame(
        frame, targets
    )

    household = target_frame.table("household")
    assert np.array_equal(household["no_child_amount"], np.asarray([100.0, 0.0]))
    assert np.array_equal(household["two_child_amount"], np.asarray([200.0, 0.0]))
    assert np.array_equal(household["three_plus_amount"], np.asarray([0.0, 300.0]))
    assert np.array_equal(household["two_child_returns"], np.asarray([1.0, 0.0]))
    assert len(registry) == 4
    assert compilation["dropped_target_names"] == []


def test_soi_ctc_targets_materialize_nonrefundable_credit(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    assert builder.SOI_VARIABLE_MAP["ctc"] == "ctc"
    assert builder.SOI_VARIABLE_MAP["refundable_ctc"] == "refundable_ctc"
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1, 2, 3], dtype="int64"),
                    "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
                    "person_tax_unit_id": np.asarray([10, 20, 30], dtype="int64"),
                    "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
                    "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
                    "person_marital_unit_id": np.asarray(
                        [10000, 20000, 30000], dtype="int64"
                    ),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2], dtype="int64"),
                    "state_fips": np.asarray([6, 6], dtype="int64"),
                }
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([10, 20, 30], dtype="int64")}
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
            "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([10000, 20000, 30000])}
            ),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([1.0, 1.0]), kind=WeightKind.DESIGN
            )
        },
    )

    def soi_spec(name, measure, source_name, source_measure_id, *, count=False):
        metadata = {
            "variable": source_name,
            "agi_lower_bound": "-inf",
            "agi_upper_bound": "inf",
            "filing_status": "All",
            "source_measure_id": source_measure_id,
        }
        if count:
            metadata["count"] = "true"
        return TargetSpec(
            name=name,
            entity="household",
            measure=measure,
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata=metadata,
        )

    targets = (
        soi_spec("ctc_amount", "ctc_amount", "ctc", "ctc_amount"),
        soi_spec("ctc_claims", "ctc_claims", "ctc", "ctc_claims", count=True),
        soi_spec(
            "actc_amount",
            "actc_amount",
            "refundable_ctc",
            "actc_amount",
        ),
        soi_spec(
            "actc_claims",
            "actc_claims",
            "refundable_ctc",
            "actc_claims",
            count=True,
        ),
    )

    class FakeVariable:
        entity = SimpleNamespace(key="tax_unit")

    class FakeSystem:
        variables = {
            name: FakeVariable()
            for name in (
                "income_tax",
                "taxable_income",
                "adjusted_gross_income",
                "filing_status",
                "state_income_tax",
                "ctc",
                "ctc_limiting_tax_liability",
                "refundable_ctc",
            )
        }

    class FakeMicrosimulation:
        def __init__(self, *, dataset, reform=None):
            self.dataset = dataset
            self.reform = reform

        def calculate(self, variable, *, period, **kwargs):
            assert period == builder.PERIOD
            assert kwargs == {}
            arrays = {
                "income_tax": np.asarray([0.0, 0.0, 0.0]),
                "taxable_income": np.asarray([0.0, 0.0, 0.0]),
                "adjusted_gross_income": np.asarray([10_000.0, 20_000.0, 30_000.0]),
                "filing_status": np.asarray(["SINGLE", "SINGLE", "SINGLE"]),
                "state_income_tax": np.asarray([0.0, 0.0, 0.0]),
                "ctc": np.asarray([1_000.0, 2_000.0, 3_000.0]),
                "ctc_limiting_tax_liability": np.asarray([80.0, 0.0, 20.0]),
                "refundable_ctc": np.asarray([10.0, 30.0, 0.0]),
            }
            return arrays[variable]

        def _invalidate_all_caches(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=FakeSystem,
            Microsimulation=FakeMicrosimulation,
        ),
    )
    monkeypatch.setattr(builder, "_dataset_from_frame", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        builder,
        "SOI_VARIABLE_MAP",
        {
            "ctc": "ctc",
            "refundable_ctc": "refundable_ctc",
        },
    )
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", ())

    target_frame, registry, compilation = builder._materialize_target_frame(
        frame, targets
    )

    household = target_frame.table("household")
    assert np.array_equal(household["ctc_amount"], np.asarray([80.0, 20.0]))
    assert np.array_equal(household["ctc_claims"], np.asarray([1.0, 1.0]))
    assert np.array_equal(household["actc_amount"], np.asarray([40.0, 0.0]))
    assert np.array_equal(household["actc_claims"], np.asarray([2.0, 0.0]))
    assert len(registry) == 4
    assert compilation["dropped_target_names"] == []


def test_population_age_targets_materialize_person_age_counts(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
                    "person_household_id": np.asarray([1, 1, 2, 2], dtype="int64"),
                    "person_tax_unit_id": np.asarray([10, 20, 30, 30], dtype="int64"),
                    "person_spm_unit_id": np.asarray(
                        [100, 100, 200, 200], dtype="int64"
                    ),
                    "person_family_id": np.asarray(
                        [1000, 1000, 2000, 2000], dtype="int64"
                    ),
                    "person_marital_unit_id": np.asarray(
                        [10000, 20000, 30000, 30000], dtype="int64"
                    ),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2], dtype="int64"),
                    "state_fips": np.asarray([6, 12], dtype="int64"),
                }
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([10, 20, 30], dtype="int64")}
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
            "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([10000, 20000, 30000])}
            ),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([1.0, 1.0]), kind=WeightKind.DESIGN
            )
        },
    )

    def population_age_spec(name, lower, upper, *, state_fips=None):
        metadata = {
            "materializer": "population_age",
            "measure_mode": "count",
            "target_role": "population_age",
            "geography_scope": "state" if state_fips else "national",
            "age_lower_bound": str(lower),
            "age_upper_bound": str(upper),
        }
        if state_fips:
            metadata["state_fips"] = state_fips
        return TargetSpec(
            name=name,
            entity="household",
            measure=name,
            value=1.0,
            source="fixture",
            family="census_population",
            metadata=metadata,
        )

    targets = (
        population_age_spec("national_age_0_to_4", 0, 5),
        population_age_spec("ca_age_0_to_4", 0, 5, state_fips="06"),
        population_age_spec("ca_age_5_to_9", 5, 10, state_fips="06"),
    )

    class FakeVariable:
        def __init__(self, entity):
            self.entity = SimpleNamespace(key=entity)

    class FakeSystem:
        variables = {
            "income_tax": FakeVariable("tax_unit"),
            "taxable_income": FakeVariable("tax_unit"),
            "adjusted_gross_income": FakeVariable("tax_unit"),
            "filing_status": FakeVariable("tax_unit"),
            "state_income_tax": FakeVariable("tax_unit"),
            "age": FakeVariable("person"),
        }

    class FakeMicrosimulation:
        def __init__(self, *, dataset, reform=None):
            self.dataset = dataset
            self.reform = reform

        def calculate(self, variable, *, period, **kwargs):
            assert period == builder.PERIOD
            assert kwargs == {}
            arrays = {
                "income_tax": np.asarray([0.0, 0.0, 0.0]),
                "taxable_income": np.asarray([0.0, 0.0, 0.0]),
                "adjusted_gross_income": np.asarray([0.0, 0.0, 0.0]),
                "filing_status": np.asarray(["SINGLE", "SINGLE", "SINGLE"]),
                "state_income_tax": np.asarray([0.0, 0.0, 0.0]),
                "age": np.asarray([2.0, 7.0, 4.0, 11.0]),
            }
            return arrays[variable]

        def _invalidate_all_caches(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=FakeSystem,
            Microsimulation=FakeMicrosimulation,
        ),
    )
    monkeypatch.setattr(builder, "_dataset_from_frame", lambda *args, **kwargs: {})
    monkeypatch.setattr(builder, "SOI_VARIABLE_MAP", {})
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", ())

    target_frame, registry, compilation = builder._materialize_target_frame(
        frame, targets
    )

    household = target_frame.table("household")
    assert np.array_equal(household["national_age_0_to_4"], np.asarray([1.0, 1.0]))
    assert np.array_equal(household["ca_age_0_to_4"], np.asarray([1.0, 0.0]))
    assert np.array_equal(household["ca_age_5_to_9"], np.asarray([1.0, 0.0]))
    assert len(registry) == 3
    assert compilation["dropped_target_names"] == []


def test_unknown_ledger_filter_metadata_fails_closed() -> None:
    builder = _load_builder_module()
    target = TargetSpec(
        name="unknown_filter_target",
        entity="household",
        measure="income_tax",
        value=1.0,
        source="fixture",
        family="irs_soi",
        metadata={"ledger_filter_unmodeled_axis": "example"},
    )

    try:
        builder._assert_supported_ledger_filter_metadata((target,))
    except RuntimeError as exc:
        assert "ledger_filter_unmodeled_axis" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected unknown Ledger filter metadata to fail closed.")


def test_build_manifests_emits_policyengine_certifiable_release_manifest(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    release_id = "populace-us-2024-abcdef1-20260615"
    release_dir = tmp_path / "release" / release_id
    release_dir.mkdir(parents=True)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / builder.DATASET_FILENAME).write_bytes(b"h5")
    (artifact_root / builder.CALIBRATION_FILENAME).write_bytes(b"npz")
    (release_dir / "calibration_diagnostics.json").write_text("{}")
    (release_dir / "us_source_coverage.json").write_text("{}")

    monkeypatch.setattr(
        builder,
        "_runtime_versions",
        lambda: {
            "python": "3.14.0",
            "populace-data": "0.1.0",
            "policyengine-core": "3.26.11",
            "policyengine-us": "1.729.0",
        },
    )
    monkeypatch.setattr(
        builder,
        "_git_output",
        lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        builder,
        "diagnostics_payload",
        lambda result, target_registry: {
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "fraction_within_10pct": 1.0,
            "target_surface": {"sha256": "b" * 64, "n_targets": 1},
        },
    )

    result = SimpleNamespace(
        skipped=(),
        diagnostics=(
            SimpleNamespace(
                name=f"nation/cbo/individual_income_tax@{builder.PERIOD}",
                target=1.0,
                initial_estimate=1.0,
                final_estimate=1.0,
            ),
        ),
        initial_loss=2.0,
        final_loss=1.0,
    )

    class FakeRegistry:
        version = "registry-sha"

        def __len__(self):
            return 1

    registry = FakeRegistry()

    builder._build_manifests(
        release_id=release_id,
        release_dir=release_dir,
        artifact_root=artifact_root,
        result=result,
        registry=registry,
        dropped={"dropped_target_names": []},
        target_profile_gate=builder.GateResult(
            name="target_profile_coverage",
            passed=True,
            details={"requirements_checked": 1},
        ),
        base_population_gate=builder.GateResult(
            name="base_population_scale",
            passed=True,
            details={
                "population": 334_200_000.0,
                "benchmark": 334_200_000.0,
                "relative_error": 0.0,
            },
        ),
        health_input_gate=builder.GateResult(
            name="health_input_signal",
            passed=True,
            details={
                "unique_counts": {
                    "takes_up_aca_if_eligible": 2,
                    "selected_marketplace_plan_benchmark_ratio": 3,
                }
            },
        ),
    )

    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    assert build_manifest["gates"]["target_profile_coverage"]["passed"]
    assert (
        build_manifest["gates"]["target_profile_coverage"]["details"][
            "requirements_checked"
        ]
        == 1
    )
    assert build_manifest["gates"]["health_input_signal"]["passed"]
    assert build_manifest["gates"]["health_input_signal"]["details"][
        "unique_counts"
    ] == {
        "takes_up_aca_if_eligible": 2,
        "selected_marketplace_plan_benchmark_ratio": 3,
    }
    assert build_manifest["gates"]["base_population_scale"]["passed"]
    assert (
        build_manifest["gates"]["base_population_scale"]["details"]["relative_error"]
        == 0.0
    )
    assert manifest["data_package"] == {"name": "populace-data", "version": "0.1.0"}
    assert manifest["default_datasets"] == {"national": "populace_us_2024"}
    assert manifest["build"]["built_with_model_package"] == {
        "name": "policyengine-us",
        "version": "1.729.0",
    }
    assert manifest["compatible_core_packages"] == [
        {"name": "policyengine-core", "specifier": "==3.26.11"}
    ]
    assert manifest["compatible_model_packages"] == [
        {"name": "policyengine-us", "specifier": "==1.729.0"}
    ]
    for artifact in manifest["artifacts"].values():
        assert artifact["repo_id"] == builder.REPO_ID
        assert artifact["revision"] == release_id
        assert artifact["kind"]
        assert artifact["sha256"]


def test_build_manifests_uses_incumbent_aware_calibration_gate(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    release_id = "populace-us-2024-abcdef1-20260615"
    release_dir = tmp_path / "release" / release_id
    release_dir.mkdir(parents=True)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / builder.DATASET_FILENAME).write_bytes(b"h5")
    (artifact_root / builder.CALIBRATION_FILENAME).write_bytes(b"npz")
    (release_dir / "calibration_diagnostics.json").write_text("{}")
    (release_dir / "us_source_coverage.json").write_text("{}")

    monkeypatch.setattr(
        builder,
        "_runtime_versions",
        lambda: {
            "python": "3.14.0",
            "populace-data": "0.1.0",
            "policyengine-core": "3.26.11",
            "policyengine-us": "1.729.0",
        },
    )
    monkeypatch.setattr(
        builder,
        "_git_output",
        lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        builder,
        "diagnostics_payload",
        lambda result, target_registry: {
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "fraction_within_10pct": 1.0,
            "target_surface": {"sha256": "b" * 64, "n_targets": 1},
        },
    )

    name = f"irs_soi.ty2022.historic_table_2.us.all.ctc_amount@{builder.PERIOD}"
    target = 82_863_353_000.0
    diagnostics = list(_passing_critical_diagnostics(builder))
    index = next(
        i for i, diagnostic in enumerate(diagnostics) if diagnostic.name == name
    )
    diagnostics[index] = SimpleNamespace(
        name=name,
        target=target,
        initial_estimate=99_315_000_000.0,
        final_estimate=99_282_300_000.0,
        relative_error=(99_282_300_000.0 - target) / target,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=tuple(diagnostics),
        initial_loss=2.0,
        final_loss=1.0,
    )

    class FakeRegistry:
        version = "registry-sha"

        def __len__(self):
            return 1

    builder._build_manifests(
        release_id=release_id,
        release_dir=release_dir,
        artifact_root=artifact_root,
        result=result,
        registry=FakeRegistry(),
        dropped={"dropped_target_names": []},
        target_profile_gate=builder.GateResult(
            name="target_profile_coverage",
            passed=True,
            details={"requirements_checked": 1},
        ),
        incumbent_diagnostics={
            name: {
                "target": target,
                "final_estimate": 134_904_000_000.0,
            }
        },
    )

    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    assert build_manifest["gates"]["calibration"] == {
        "passed": True,
        "failures": [],
        "initial_loss": 2.0,
        "final_loss": 1.0,
        "fraction_within_10pct": 1.0,
    }


def test_export_frame_drops_formula_owned_columns(monkeypatch, small_frame) -> None:
    builder = _load_builder_module()

    class FakePolicyEngineUSEngine:
        def _engine_computed_columns(self, tables, *, period):
            assert period == builder.PERIOD
            assert "income" in tables["person"]
            return {"income"}

    monkeypatch.setattr(builder, "PolicyEngineUSEngine", FakePolicyEngineUSEngine)

    stripped = builder._strip_calibration_columns(
        small_frame,
        np.array([1000.0, 2000.0]),
    )

    assert "income" not in stripped.table("person")
    assert stripped.weights_for("household").kind == WeightKind.CALIBRATED


def test_export_frame_seeds_partnership_inputs_before_formula_drop(
    monkeypatch, small_frame
) -> None:
    builder = _load_builder_module()

    person = small_frame.table("person").copy()
    person["partnership_s_corp_income"] = np.asarray([100.0, -5.0, 0.0, 40.0])
    frame = Frame(
        {"person": person, "household": small_frame.table("household").copy()},
        small_frame.schema,
        {"household": small_frame.weights_for("household")},
    )

    class FakePolicyEngineUSEngine:
        def _engine_computed_columns(self, tables, *, period):
            assert period == builder.PERIOD
            assert "partnership_income" in tables["person"]
            assert "s_corp_income" in tables["person"]
            return {"partnership_s_corp_income"}

    monkeypatch.setattr(builder, "PolicyEngineUSEngine", FakePolicyEngineUSEngine)

    stripped = builder._drop_formula_owned_columns(frame)

    assert "partnership_s_corp_income" not in stripped.table("person")
    assert np.array_equal(
        stripped.table("person")["partnership_income"].to_numpy(),
        np.asarray([100.0, -5.0, 0.0, 40.0]),
    )
    assert np.array_equal(
        stripped.table("person")["s_corp_income"].to_numpy(),
        np.zeros(4),
    )


def test_post_export_sanity_checks_full_target_surface(monkeypatch, tmp_path) -> None:
    builder = _load_builder_module()

    class FakeWeights:
        values = np.asarray([1.0])

    class FakeFrame:
        def weights_for(self, entity):
            assert entity == "household"
            return FakeWeights()

    class FakeTarget:
        entity = "household"
        row_name = f"nation/cbo/individual_income_tax@{builder.PERIOD}"

        def __init__(self):
            self.observed = 2_000_000_000_000.0

        def achieved_value(self, frame, weights):
            assert isinstance(frame, FakeFrame)
            assert np.array_equal(weights, np.asarray([1.0]))
            return self.observed

    target = FakeTarget()

    class FakeRegistry:
        def to_target_set(self):
            return (target,)

    monkeypatch.setattr(builder, "_load_frame", lambda path: f"loaded:{path}")
    monkeypatch.setattr(
        builder,
        "_materialize_target_frame",
        lambda frame, target_specs: (
            FakeFrame(),
            FakeRegistry(),
            {"dropped_target_names": []},
        ),
    )

    result = SimpleNamespace(
        diagnostics=(
            SimpleNamespace(
                name=f"nation/cbo/individual_income_tax@{builder.PERIOD}",
                final_estimate=2_000_000_000_000.0,
            ),
        )
    )

    builder._assert_export_matches_calibration(tmp_path / "candidate.h5", result, ())

    target.observed = 2_000_900_000_000.0
    builder._assert_export_matches_calibration(tmp_path / "candidate.h5", result, ())

    target.observed = 1_990_000_000_000.0
    try:
        builder._assert_export_matches_calibration(
            tmp_path / "candidate.h5", result, ()
        )
    except RuntimeError as exc:
        assert "Post-export sanity failed" in str(exc)
        assert "nation/cbo/individual_income_tax@2024 exported value" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected post-export sanity failure.")


def test_post_export_sanity_rejects_dropped_export_targets(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(builder, "_load_frame", lambda path: object())
    monkeypatch.setattr(
        builder,
        "_materialize_target_frame",
        lambda frame, target_specs: (
            object(),
            object(),
            {"dropped_target_names": ["missing"]},
        ),
    )

    try:
        builder._assert_export_matches_calibration(
            tmp_path / "candidate.h5", SimpleNamespace(diagnostics=()), ()
        )
    except RuntimeError as exc:
        assert "1 fiscal targets were not materialized after export" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected dropped-target post-export sanity failure.")


def test_reviewed_exclusions_are_exact_for_fiscal_refresh() -> None:
    builder = _load_builder_module()

    exclusions = builder._reviewed_exclusions(builder.DIRECT_ACTIVE_ALIASES)

    assert tuple(exclusions) == builder.REVIEWED_EXCLUDED_ALIASES


def test_fiscal_refresh_uses_target_period_medicaid_source() -> None:
    builder = _load_builder_module()

    assert (
        "cms-medicaid-chip-monthly-enrollment-december-2024"
        in builder.DIRECT_ACTIVE_ALIASES
    )
    assert (
        "cms-medicaid-chip-monthly-enrollment-dataset"
        in builder.REVIEWED_EXCLUDED_ALIASES
    )


def test_reviewed_exclusions_fail_when_hard_target_surface_changes(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(
        builder,
        "hard_target_package_aliases",
        lambda: (*builder.DIRECT_ACTIVE_ALIASES, "new-hard-target"),
    )

    try:
        builder._reviewed_exclusions(builder.DIRECT_ACTIVE_ALIASES)
    except RuntimeError as exc:
        assert "Reviewed hard-target exclusion list is stale" in str(exc)
        assert "new-hard-target" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected stale reviewed-exclusion failure.")


def test_fiscal_target_source_provenance_covers_active_families() -> None:
    builder = _load_builder_module()
    specs = (
        TargetSpec(
            name="income_tax",
            entity="household",
            measure="income_tax",
            value=1,
            source="CBO source",
            family="cbo",
        ),
        TargetSpec(
            name="salt",
            entity="household",
            measure="salt",
            value=1,
            source="JCT source",
            family="jct",
            metadata={"reference_url": "https://example.org/jct"},
        ),
        TargetSpec(
            name="agi",
            entity="household",
            measure="agi",
            value=1,
            source="SOI source",
            family="irs_soi",
        ),
        TargetSpec(
            name="state_income_tax",
            entity="household",
            measure="state_income_tax",
            value=1,
            source="Census source",
            family="state_income_tax",
            metadata={"reference_url": "https://example.org/stc"},
        ),
    )

    provenance = builder._fiscal_target_source_provenance(specs)

    assert set(provenance) == {"cbo", "irs_soi", "jct", "state_income_tax"}
    assert provenance["cbo"]["target_count"] == 1
    assert provenance["jct"]["target_count"] == 1
    assert provenance["irs_soi"]["sources"]
    assert provenance["state_income_tax"]["reference_urls"]


def test_us_release_id_guard() -> None:
    builder = _load_builder_module()

    builder._assert_us_release_id("populace-us-2024-base-commit-20260615T000000Z")

    try:
        builder._assert_us_release_id("populace-uk-2024-base-commit-20260615T000000Z")
    except ValueError as exc:
        assert "must start with 'populace-us-'" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected non-US release id to fail.")
