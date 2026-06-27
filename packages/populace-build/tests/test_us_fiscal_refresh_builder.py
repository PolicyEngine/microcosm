import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

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


def test_runtime_versions_use_local_workspace_package_version(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    package = tmp_path / "packages" / "populace-data"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "populace-data"\nversion = "0.1.0"\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        builder.importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(
            builder.importlib.metadata.PackageNotFoundError(name)
        ),
    )

    versions = builder._runtime_versions()

    assert versions["populace-data"] == "0.1.0"


def test_reviewed_exclusions_do_not_report_opted_in_cd_sources() -> None:
    builder = _load_builder_module()
    acs_cd_alias = "census-acs-s0101-congressional-district-age-2024"
    soi_cd_alias = "soi-congressional-district-2022"

    reviewed = builder._reviewed_exclusions(
        builder.DIRECT_ACTIVE_ALIASES + (acs_cd_alias, soi_cd_alias)
    )

    assert acs_cd_alias not in reviewed
    assert soi_cd_alias not in reviewed
    assert "census-acs-s0101-national-age-2024" in reviewed
    assert "census-acs-s0101-state-age-2024" in reviewed


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
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_amount",
            69_041_649_000.0,
            70_000_000_000.0,
        ),
        diagnostic(
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_returns",
            23_837_149.0,
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


def test_maximum_microsim_batch_size_defaults_and_overrides(monkeypatch) -> None:
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
    assert (
        args.maximum_microsim_batch_size == builder.DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_us_fiscal_refresh_release.py",
            "--ledger-facts",
            "facts.jsonl",
            "--out",
            "release",
            "--maximum-microsim-batch-size",
            "0",
        ],
    )
    args = builder._parse_args()
    assert args.maximum_microsim_batch_size == 0


def test_staging_repo_can_default_from_environment(monkeypatch) -> None:
    builder = _load_builder_module()
    monkeypatch.setenv("POPULACE_STAGING_REPO_ID", "policyengine/populace-us-staging")
    monkeypatch.setenv("POPULACE_STAGING_PREFIX", "candidate-runs")
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

    assert args.staging_repo_id == "policyengine/populace-us-staging"
    assert args.staging_prefix == "candidate-runs"


def test_soi_indicator_rows_flag_positive_component_items() -> None:
    builder = _load_builder_module()

    assert np.array_equal(
        builder._soi_component_row(
            np.array([-5.0, 0.0, 7.0]),
            "capital_gains_gross",
            indicator=True,
        ),
        np.array([0.0, 0.0, 1.0]),
    )
    assert np.array_equal(
        builder._soi_component_row(
            np.array([-5.0, 0.0, 7.0]),
            "capital_gains_losses",
            indicator=True,
        ),
        np.array([0.0, 0.0, 1.0]),
    )
    assert np.array_equal(
        builder._soi_component_row(
            np.array([-5.0, 0.0, 7.0]),
            "business_net_losses",
            indicator=True,
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


def test_unsupported_ledger_filter_metadata_all_value_is_noop() -> None:
    builder = _load_builder_module()
    specs = (
        SimpleNamespace(
            name="all_child_count",
            metadata={"ledger_filter_qualifying_children": "all"},
        ),
        SimpleNamespace(
            name="specific_child_count",
            metadata={"ledger_filter_qualifying_children": "one"},
        ),
    )

    assert builder._unsupported_ledger_filter_metadata(specs) == {
        "specific_child_count": ("ledger_filter_qualifying_children",)
    }


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


def test_base_population_mass_repair_rescales_to_census_benchmark(
    small_frame,
) -> None:
    builder = _load_builder_module()
    benchmark = builder.US_BASE_PERSON_POPULATION_BENCHMARK

    repaired, repair = builder._with_base_population_mass_repair(small_frame)

    assert repair["applied"]
    assert repair["method"] == "rescale_household_weights_to_census_person_population"
    assert repair["initial_population"] == 6000.0
    assert np.isclose(repair["factor"], benchmark / 6000.0)
    assert np.isclose(repair["repaired_population"], benchmark)
    assert np.isclose(float(repaired.resolve_weights("person").values.sum()), benchmark)
    assert repaired.mass_log[-1].entity == "household"
    assert (
        repaired.mass_log[-1].reason == builder.US_BASE_PERSON_POPULATION_REPAIR_REASON
    )

    gate = builder._base_population_scale_gate(repaired, mass_repair=repair)
    assert gate.passed
    assert gate.details["mass_repair"]["initial_population"] == 6000.0
    assert np.isclose(gate.details["mass_repair"]["factor"], benchmark / 6000.0)


def test_social_security_component_value_repair_uses_registry_targets(
    small_frame,
) -> None:
    builder = _load_builder_module()
    person = small_frame.table("person").copy()
    person["social_security_retirement"] = [1.0, 0.0, 2.0, 0.0]
    person["social_security_disability"] = [0.0, 3.0, 0.0, 1.0]
    person["social_security_dependents"] = [2.0, 0.0, 0.0, 1.0]
    person["social_security_survivors"] = [0.0, 1.0, 2.0, 0.0]
    frame = Frame(
        {
            "person": person,
            "household": small_frame.table("household").copy(),
        },
        small_frame.schema,
        {"household": small_frame.weights_for("household")},
    )
    targets = {
        "ssa_retirement_total": 10_000.0,
        "ssa_disability_total": 8_000.0,
        "ssa_dependents_total": 6_000.0,
        "ssa_survivors_total": 12_000.0,
    }
    specs = tuple(
        TargetSpec(
            name=f"ssa.{role}",
            entity="household",
            value=value,
            measure="unused",
            period=builder.PERIOD,
            source="SSA",
            metadata={"target_role": role},
        )
        for role, value in targets.items()
    )

    repaired, repair = builder._with_social_security_component_value_repair(
        frame,
        specs,
    )

    assert repair["applied"]
    weights = pd.Series(repaired.resolve_weights("person").values)
    for role, column in builder.US_SOCIAL_SECURITY_COMPONENT_TARGET_ROLES.items():
        total = float((repaired.table("person")[column] * weights).sum())
        assert np.isclose(total, targets[role])
        assert np.isclose(
            repair["components"][column]["repaired_estimate"],
            targets[role],
        )


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
                measure="national_critical_role",
                value=100.0,
                source="fixture",
                metadata={"target_role": "federal_income_tax_total"},
            ),
            TargetSpec(
                name="state_role_row",
                entity="household",
                measure="state_role_row",
                value=100.0,
                source="fixture",
                metadata={"state_fips": "06", "target_role": "tanf_total"},
            ),
            TargetSpec(
                name="ordinary_distribution_row",
                entity="household",
                measure="ordinary_distribution_row",
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


def test_fiscal_target_loss_weights_hold_concept_budget_when_geography_expands() -> (
    None
):
    builder = _load_builder_module()

    def spec(name: str, value: float, **metadata: str) -> TargetSpec:
        return TargetSpec(
            name=name,
            entity="household",
            measure=metadata.get("variable", "amount"),
            value=value,
            source="fixture",
            metadata={
                "source_measure_id": "amount",
                "source_period": "2024",
                "target_role": "fixture_distribution",
                "measure_mode": "sum",
                **metadata,
            },
        )

    national_income_tax = spec(
        "income_tax_national",
        100.0,
        variable="income_tax",
        ledger_geography_level="country",
        ledger_geography_id="0100000US",
    )
    ctc_national = spec(
        "ctc_national",
        400.0,
        variable="ctc",
        ledger_geography_level="country",
        ledger_geography_id="0100000US",
    )
    ctc_cd_1 = spec(
        "ctc_cd_1",
        100.0,
        variable="ctc",
        ledger_geography_level="congressional_district",
        ledger_geography_id="5001700US0101",
        ledger_geography_name="Alabama Congressional District 1",
        congressional_district_geoid="0101",
        state_fips="01",
    )
    ctc_cd_2 = spec(
        "ctc_cd_2",
        100.0,
        variable="ctc",
        ledger_geography_level="congressional_district",
        ledger_geography_id="5001700US0102",
        ledger_geography_name="Alabama Congressional District 2",
        congressional_district_geoid="0102",
        state_fips="01",
    )

    base_weights = builder._fiscal_target_loss_weights(
        TargetRegistry((national_income_tax, ctc_national), country="us")
    )
    one_child_weights = builder._fiscal_target_loss_weights(
        TargetRegistry(
            (national_income_tax, ctc_national, ctc_cd_1),
            country="us",
        )
    )
    two_child_weights = builder._fiscal_target_loss_weights(
        TargetRegistry(
            (national_income_tax, ctc_national, ctc_cd_1, ctc_cd_2),
            country="us",
        )
    )

    assert np.isclose(base_weights[1] / base_weights.sum(), 2 / 3)
    assert np.isclose(
        one_child_weights[2:].sum() / one_child_weights.sum(),
        two_child_weights[2:].sum() / two_child_weights.sum(),
    )
    assert two_child_weights[1] > two_child_weights[2:].sum()
    assert two_child_weights[2] == two_child_weights[3]


def test_fiscal_target_loss_weights_budget_unparented_cd_rows_by_concept() -> None:
    builder = _load_builder_module()

    def cd_spec(name: str, geoid: str) -> TargetSpec:
        return TargetSpec(
            name=name,
            entity="household",
            measure="tax_filer_individual_count",
            value=100.0,
            source="fixture",
            metadata={
                "source_measure_id": "tax_filer_individual_count",
                "source_period": "2023",
                "target_role": "soi_fiscal_distribution",
                "variable": "tax_filer_individual_count",
                "source_variable": "tax_filer_individual_count",
                "measure_mode": "sum",
                "ledger_geography_level": "congressional_district",
                "ledger_geography_id": f"5001700US{geoid}",
                "ledger_geography_name": f"Congressional District {geoid}",
                "congressional_district_geoid": geoid,
                "state_fips": geoid[:2],
            },
        )

    comparison = TargetSpec(
        name="comparison_amount",
        entity="household",
        measure="adjusted_gross_income",
        value=100.0,
        source="fixture",
        metadata={
            "source_measure_id": "adjusted_gross_income",
            "source_period": "2023",
            "target_role": "soi_fiscal_distribution",
            "variable": "adjusted_gross_income",
            "source_variable": "adjusted_gross_income",
            "measure_mode": "sum",
        },
    )
    one_cd_registry = TargetRegistry(
        (comparison, cd_spec("cd_1", "0101")), country="us"
    )
    many_cd_registry = TargetRegistry(
        (
            comparison,
            cd_spec("cd_1", "0101"),
            cd_spec("cd_2", "0102"),
            cd_spec("cd_3", "0103"),
            cd_spec("cd_4", "0104"),
        ),
        country="us",
    )

    one_cd_weights = builder._fiscal_target_loss_weights(one_cd_registry)
    many_cd_weights = builder._fiscal_target_loss_weights(many_cd_registry)

    assert np.isclose(one_cd_weights[1:].sum() / one_cd_weights.sum(), 0.5)
    assert np.isclose(many_cd_weights[1:].sum() / many_cd_weights.sum(), 0.5)
    assert np.allclose(many_cd_weights[1:], many_cd_weights[1])


def test_fiscal_target_loss_weights_scale_by_sqrt_value_within_basis() -> None:
    builder = _load_builder_module()
    registry = TargetRegistry(
        (
            TargetSpec(
                name="amount_small",
                entity="household",
                measure="amount_small",
                value=100.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="amount_large",
                entity="household",
                measure="amount_large",
                value=300.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="returns_small",
                entity="household",
                measure="returns_small",
                value=10.0,
                source="fixture",
                metadata={
                    "source_measure_id": "income_tax_liability_returns",
                    "measure_mode": "indicator_sum",
                },
            ),
            TargetSpec(
                name="returns_large",
                entity="household",
                measure="returns_large",
                value=30.0,
                source="fixture",
                metadata={
                    "source_measure_id": "ctc_claims",
                    "measure_mode": "indicator_sum",
                },
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
                measure="amount_small",
                value=100.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="amount_large",
                entity="household",
                measure="amount_large",
                value=300.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="returns",
                entity="household",
                measure="returns",
                value=10.0,
                source="fixture",
                metadata={
                    "source_measure_id": "ctc_claims",
                    "measure_mode": "indicator_sum",
                },
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
                measure="zero",
                value=0.0,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="subunit",
                entity="household",
                measure="subunit",
                value=0.25,
                source="fixture",
                metadata={"source_measure_id": "payment_amount"},
            ),
            TargetSpec(
                name="negative",
                entity="household",
                measure="negative",
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
        measure="amount",
        value=100.0,
        source="fixture",
        metadata={"source_measure_id": "payment_amount"},
    )
    return_count = TargetSpec(
        name="return_count",
        entity="household",
        measure="return_count",
        value=100.0,
        source="fixture",
        metadata={
            "source_measure_id": "ctc_claims",
            "measure_mode": "indicator_sum",
        },
    )
    person_count = TargetSpec(
        name="person_count",
        entity="household",
        measure="person_count",
        value=100.0,
        source="fixture",
        metadata={
            "measure_mode": "indicator_sum",
            "source_measure_id": "aptc_recipients",
            "target_role": "aca_ptc_recipients",
            "indicator_map_to": "person",
        },
    )
    bronze_count = TargetSpec(
        name="bronze_count",
        entity="household",
        measure="bronze_count",
        value=100.0,
        source="fixture",
        metadata={
            "measure_mode": "less_than_indicator_sum",
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
        support_value_repairs={"social_security_components": {"applied": True}},
        audit_export_targets=False,
        gate_failures=["ctc failed"],
        timing={
            "target_compilation_seconds": 1.25,
            "calibration_seconds": 2.5,
        },
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
    assert build["support_value_repairs"] == {
        "social_security_components": {"applied": True}
    }
    assert build["timing"] == {
        "target_compilation_seconds": 1.25,
        "calibration_seconds": 2.5,
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
        lambda facts, *, target_period, include_congressional_district_targets=False: (
            registry
        ),
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
    repair_payload = {
        "method": "rescale_household_weights_to_census_person_population",
        "applied": True,
        "factor": 2.0,
    }
    monkeypatch.setattr(
        builder,
        "_with_base_population_mass_repair",
        lambda frame: (frame, repair_payload),
    )
    ss_repair_payload = {
        "method": "rescale_social_security_component_leaves_to_ssa_targets",
        "applied": True,
    }
    monkeypatch.setattr(
        builder,
        "_with_social_security_component_value_repair",
        lambda frame, specs: (frame, ss_repair_payload),
    )
    monkeypatch.setattr(
        builder,
        "_base_population_scale_gate",
        lambda frame, *, mass_repair=None: builder.GateResult(
            name="base_population_scale",
            passed=True,
            details={"checked": True, "mass_repair": mass_repair},
        ),
    )
    monkeypatch.setattr(
        builder,
        "_with_aca_marketplace_source_outputs",
        lambda frame, specs, *, seed, maximum_microsim_batch_size=None: frame,
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

    def fake_materialize_target_frame(frame, specs, **kwargs):
        captured["materialize_kwargs"] = kwargs
        return frame, registry, {"dropped_target_names": []}

    monkeypatch.setattr(
        builder,
        "_materialize_target_frame",
        fake_materialize_target_frame,
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
    assert (
        captured["diagnostics"]["base_population_gate"].details["mass_repair"]
        == repair_payload
    )
    assert captured["diagnostics"]["support_value_repairs"] == {
        "social_security_components": ss_repair_payload
    }
    assert captured["target_loss_cap"] == 1.0
    assert np.array_equal(captured["target_loss_weights"], np.asarray([1.0]))
    assert (
        captured["materialize_kwargs"]["target_materialization_cache_dir"]
        == out / "artifacts" / "target_materialization_cache"
    )


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
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_amount",
            "Earned Income Tax Credit amount",
            69_041_649_000.0,
            83_000_000_000.0,
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
        assert "exceeding 0.15" in failures[0]


def test_critical_gate_allows_eitc_amount_within_credit_tolerance() -> None:
    builder = _load_builder_module()
    name = (
        "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
        f"earned_income_credit.total_earned_income_credit_amount@{builder.PERIOD}"
    )
    target = 69_041_649_000.0
    diagnostics = list(_passing_critical_diagnostics(builder))
    index = next(
        i for i, diagnostic in enumerate(diagnostics) if diagnostic.name == name
    )
    diagnostics[index] = SimpleNamespace(
        name=name,
        target=target,
        initial_estimate=target,
        final_estimate=58_954_970_066.74941,
        relative_error=(58_954_970_066.74941 - target) / target,
    )
    result = SimpleNamespace(
        skipped=(),
        diagnostics=tuple(diagnostics),
        initial_loss=10.0,
        final_loss=5.0,
    )

    assert builder._release_gate_failures(result, {"dropped_target_names": []}) == []


def test_critical_gate_allows_bounded_improvement_over_incumbent() -> None:
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


def test_critical_gate_rejects_improved_miss_past_hard_stop() -> None:
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
        final_estimate=105_000_000_000.0,
        relative_error=(105_000_000_000.0 - target) / target,
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

    failures = builder._release_gate_failures(
        result,
        {"dropped_target_names": []},
        incumbent_diagnostics=incumbent,
    )

    assert len(failures) == 1
    assert "Child Tax Credit amount" in failures[0]
    assert "exceeding 0.15" in failures[0]
    assert "incumbent_relative_error=" in failures[0]
    assert "improvement_hard_stop=0.25" in failures[0]


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
    ratio_diagnostics = gate.details["selected_marketplace_plan_benchmark_ratio"]
    assert ratio_diagnostics["support"] == {"lower": 0.5, "upper": 1.5}
    assert ratio_diagnostics["all_tax_units"] == {
        "count": 3,
        "min": 0.8,
        "max": 1.2,
        "mean": 1.0,
        "neutral_count": 1,
        "below_benchmark_count": 1,
        "above_benchmark_count": 1,
        "below_support_count": 0,
        "above_support_count": 0,
    }
    marketplace_takers = ratio_diagnostics["marketplace_takers"]
    assert marketplace_takers["count"] == 2
    assert abs(marketplace_takers["mean"] - 1.1) < 1e-12
    assert marketplace_takers["below_benchmark_count"] == 0
    assert marketplace_takers["above_benchmark_count"] == 1


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
            metadata={
                "target_role": "aca_ptc_recipients",
                "state_fips": "01",
                "ledger_geography_level": "state",
            },
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


def test_aca_source_tax_unit_table_batches_policyengine_inputs(monkeypatch) -> None:
    builder = _load_builder_module()
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2, 3], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 10, 20, 30], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 100, 200, 300], dtype="int64"),
            "person_family_id": np.asarray([1000, 1000, 2000, 3000], dtype="int64"),
            "person_marital_unit_id": np.asarray(
                [10000, 10000, 20000, 30000], dtype="int64"
            ),
        }
    )
    frame = Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2, 3], dtype="int64"),
                    "state_fips": np.asarray([1, 1, 2]),
                }
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": np.asarray([10, 20, 30], dtype="int64"),
                    "stable_tax_unit_draw": [0.1, 0.2, 0.3],
                }
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": [100, 200, 300]}),
            "family": pd.DataFrame({"family_id": [1000, 2000, 3000]}),
            "marital_unit": pd.DataFrame({"marital_unit_id": [10000, 20000, 30000]}),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([10.0, 20.0, 30.0]),
                kind=WeightKind.DESIGN,
            )
        },
    )
    target_tables = {
        builder.US_ACA_APTC_TARGET_TABLE: pd.DataFrame(
            {
                "state_fips": ["01"],
                "target": [3.0],
            }
        )
    }
    tax_values = {
        10: {
            "is_aca_ptc_eligible": 1.0,
            "aca_ptc": 100.0,
            "health_insurance_premiums_without_medicare_part_b": 400.0,
            "slcsp": 1000.0,
        },
        20: {
            "is_aca_ptc_eligible": 0.0,
            "aca_ptc": 200.0,
            "health_insurance_premiums_without_medicare_part_b": 500.0,
            "slcsp": 1100.0,
        },
        30: {
            "is_aca_ptc_eligible": 1.0,
            "aca_ptc": 300.0,
            "health_insurance_premiums_without_medicare_part_b": 600.0,
            "slcsp": 1200.0,
        },
    }
    person_eligible = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0}
    seen_tax_unit_batches: list[tuple[int, ...]] = []
    formula_owned_assertions: list[int] = []
    dataset_assert_flags: list[bool | None] = []

    class FakeMicrosimulation:
        def __init__(self, *, dataset):
            self.dataset = dataset
            seen_tax_unit_batches.append(
                tuple(dataset.table("tax_unit")["tax_unit_id"].astype(int))
            )

        def _invalidate_all_caches(self):
            pass

    def fake_calculate_array(simulation, variable, *, map_to=None):
        if map_to == "person":
            return np.asarray(
                [
                    person_eligible[int(person_id)]
                    for person_id in simulation.dataset.table("person")["person_id"]
                ],
                dtype=np.float64,
            )
        assert map_to == "tax_unit"
        return np.asarray(
            [
                tax_values[int(tax_unit_id)][variable]
                for tax_unit_id in simulation.dataset.table("tax_unit")["tax_unit_id"]
            ],
            dtype=np.float64,
        )

    def fake_assert_no_formula_owned_columns(frame_arg):
        formula_owned_assertions.append(frame_arg.n("household"))

    def fake_dataset_from_frame(frame_arg, **kwargs):
        dataset_assert_flags.append(kwargs.get("assert_no_formula_owned_columns"))
        return frame_arg

    monkeypatch.setattr(
        builder,
        "_assert_no_formula_owned_columns",
        fake_assert_no_formula_owned_columns,
    )
    monkeypatch.setattr(builder, "_dataset_from_frame", fake_dataset_from_frame)
    monkeypatch.setattr(builder, "_calculate_array", fake_calculate_array)

    tax_unit = builder._aca_source_tax_unit_table_batched(
        frame,
        target_tables,
        microsimulation_cls=FakeMicrosimulation,
        maximum_microsim_batch_size=1,
    ).set_index("tax_unit_id")

    assert seen_tax_unit_batches == [(10,), (20,), (30,)]
    assert formula_owned_assertions == [3]
    assert dataset_assert_flags == [False, False, False]
    assert tax_unit.loc[10, "tax_unit_weight"] == 20.0
    assert tax_unit.loc[20, "tax_unit_weight"] == 20.0
    assert tax_unit.loc[30, "tax_unit_weight"] == 0.0
    assert bool(tax_unit.loc[10, "is_aca_ptc_eligible"]) is True
    assert bool(tax_unit.loc[20, "is_aca_ptc_eligible"]) is True
    assert bool(tax_unit.loc[30, "is_aca_ptc_eligible"]) is False
    assert tax_unit.loc[10, "assigned_aca_ptc"] == 100.0
    assert (
        tax_unit.loc[20, "health_insurance_premiums_without_medicare_part_b"] == 500.0
    )
    assert tax_unit.loc[30, "slcsp"] == 1200.0
    assert tax_unit.loc[10, "aca_take_up_rate"] == 0.075
    assert tax_unit.loc[30, "aca_take_up_rate"] == 0.0


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


def test_aca_source_runtime_uses_bronze_targets_when_available(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    captured: dict[str, object] = {}
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": [10, 20],
            "takes_up_aca_if_eligible": [False, False],
            "selected_marketplace_plan_benchmark_ratio": [1.0, 1.0],
        }
    )

    class FakeFrame:
        entities = ("tax_unit",)
        schema = object()
        weighted_entities = ()
        strata = None

        def table(self, entity):
            assert entity == "tax_unit"
            return tax_unit

    def fake_run_source_stage(
        stage,
        *,
        tables,
        operation_handlers,
        config,
        stop_after,
    ):
        captured["stage"] = stage.stage
        captured["tables"] = tables
        captured["stop_after"] = stop_after
        return pd.DataFrame(
            {
                "tax_unit_id": [10, 20],
                "takes_up_aca_if_eligible": [True, False],
                "selected_marketplace_plan_benchmark_ratio": [0.8, 1.0],
            }
        )

    monkeypatch.setattr(builder, "_aca_source_person_table", lambda frame: object())
    monkeypatch.setattr(
        builder,
        "_aca_source_tax_unit_table",
        lambda frame, target_tables, *, simulation=None, maximum_microsim_batch_size=None: (
            pd.DataFrame({"tax_unit_id": [10, 20], "state_fips": ["06", "06"]})
        ),
    )
    monkeypatch.setattr(builder, "run_source_stage", fake_run_source_stage)
    monkeypatch.setattr(
        builder,
        "Frame",
        lambda tables, schema, weights, strata: SimpleNamespace(tables=tables),
    )

    specs = (
        TargetSpec(
            name="cms_aca.oep2024.state_marketplace.ca.aptc_recipients",
            entity="household",
            measure="takes_up_aca_if_eligible",
            value=1.0,
            source="CMS Marketplace OEP",
            family="cms_aca",
            metadata={
                "target_role": "aca_ptc_recipients",
                "state_fips": "06",
                "ledger_geography_level": "state",
            },
        ),
        TargetSpec(
            name="cms_aca.oep2024.state_metal.ca.bronze_aptc_consumers",
            entity="household",
            measure="selected_marketplace_plan_benchmark_ratio",
            value=1.0,
            source="CMS Marketplace OEP",
            family="cms_aca",
            metadata={
                "target_role": "aca_bronze_aptc_consumers",
                "state_fips": "06",
                "ledger_geography_level": "state",
            },
        ),
    )

    builder._with_aca_marketplace_source_outputs(
        FakeFrame(),
        specs,
        seed=42,
        simulation=object(),
    )

    assert captured["stage"] == builder.US_ACA_MARKETPLACE_STAGE
    assert captured["stop_after"] is None
    target_tables = captured["tables"]
    assert set(target_tables) >= {
        builder.US_ACA_APTC_TARGET_TABLE,
        "cms_aca_bronze_aptc_consumers_by_state",
    }
    bronze_table = target_tables["cms_aca_bronze_aptc_consumers_by_state"]
    assert bronze_table.to_dict("records") == [
        {
            "state_fips": "06",
            "target": 1.0,
            "source_record_id": (
                "cms_aca.oep2024.state_metal.ca.bronze_aptc_consumers"
            ),
        }
    ]


def test_aca_source_target_tables_ignore_congressional_district_targets() -> None:
    builder = _load_builder_module()

    specs = (
        TargetSpec(
            name="irs_soi.ty2022.historic_table_2.state_broad.ca.all."
            "premium_tax_credit_amount",
            entity="household",
            measure="assigned_aca_ptc",
            value=100.0,
            source="SOI",
            family="irs_soi",
            metadata={
                "target_role": "aca_spending",
                "state_fips": "06",
                "ledger_geography_level": "state",
            },
        ),
        TargetSpec(
            name="irs_soi.ty2023.congressional_district_2022.all_returns."
            "ca_01.premium_tax_credit_amount",
            entity="household",
            measure="assigned_aca_ptc",
            value=75.0,
            source="SOI",
            family="irs_soi",
            metadata={
                "target_role": "aca_spending",
                "state_fips": "06",
                "ledger_geography_level": "congressional_district",
                "congressional_district_geoid": "0601",
            },
        ),
        TargetSpec(
            name="irs_soi.ty2023.congressional_district_2022.all_returns."
            "ca_total.premium_tax_credit_amount",
            entity="household",
            measure="assigned_aca_ptc",
            value=125.0,
            source="SOI",
            family="irs_soi",
            metadata={
                "target_role": "aca_spending",
                "state_fips": "06",
                "ledger_geography_level": "state",
                "ledger_layout_groupby_dimension": ("irs_soi.congressional_district"),
                "ledger_layout_groupby_value_id": "ca_total",
            },
        ),
    )

    tables = builder._aca_source_target_tables(specs)

    amount_table = tables["irs_soi_premium_tax_credit_amount_by_state"]
    assert amount_table.to_dict("records") == [
        {
            "state_fips": "06",
            "target": 100.0,
            "source_record_id": (
                "irs_soi.ty2022.historic_table_2.state_broad.ca.all."
                "premium_tax_credit_amount"
            ),
        }
    ]


def test_jct_materialization_collapses_reform_tax_units_and_clears_caches(
    monkeypatch,
    tmp_path,
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
    formula_owned_assertions: list[int] = []

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
            tax_unit_ids = (
                self.dataset["frame"].table("tax_unit")["tax_unit_id"].to_numpy()
            )
            if self.reform is not None:
                assert variable == "income_tax"
                assert kwargs == {}
                reform_income_tax_by_id = {10: 90.0, 20: 25.0, 30: 40.0}
                return np.asarray(
                    [reform_income_tax_by_id[id_] for id_ in tax_unit_ids]
                )
            arrays_by_id = {
                "income_tax": {10: 100.0, 20: 30.0, 30: 70.0},
                "taxable_income": {10: 1000.0, 20: 2000.0, 30: 3000.0},
                "adjusted_gross_income": {10: 1100.0, 20: 2100.0, 30: 3100.0},
                "filing_status": {10: "SINGLE", 20: "SINGLE", 30: "SINGLE"},
                "state_income_tax": {10: 5.0, 20: 6.0, 30: 7.0},
            }
            assert kwargs == {}
            return np.asarray([arrays_by_id[variable][id_] for id_ in tax_unit_ids])

        def _invalidate_all_caches(self):
            self.cache_invalidations += 1

    def fake_dataset_from_frame(
        frame_arg,
        *,
        zero_variables=(),
        system=None,
        assert_no_formula_owned_columns=True,
    ):
        datasets.append(
            (
                frame_arg,
                tuple(zero_variables),
                system,
                assert_no_formula_owned_columns,
            )
        )
        return {"frame": frame_arg, "zero_variables": tuple(zero_variables)}

    def fake_make_zero_variable_reform(system, variable_name):
        assert isinstance(system, FakeSystem)
        assert variable_name == "mock_credit"
        return object()

    def fake_assert_no_formula_owned_columns(frame_arg):
        formula_owned_assertions.append(frame_arg.n("household"))

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=FakeSystem,
            Microsimulation=FakeMicrosimulation,
        ),
    )
    monkeypatch.setattr(
        builder,
        "_assert_no_formula_owned_columns",
        fake_assert_no_formula_owned_columns,
    )
    monkeypatch.setattr(builder, "_dataset_from_frame", fake_dataset_from_frame)
    monkeypatch.setattr(
        builder, "_make_zero_variable_reform", fake_make_zero_variable_reform
    )
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", (reform_spec,))
    monkeypatch.setattr(builder, "SOI_VARIABLE_MAP", {})

    cache_context = {
        "base_dataset_sha256": "test-base-sha",
        "build_commit": "test-commit",
        "policyengine_us_version": "test-policyengine-us",
        "seed": 0,
        "target_period": builder.PERIOD,
        "target_registry_version": "test-target-registry",
    }
    target_frame, registry, dropped = builder._materialize_target_frame(
        frame,
        (target,),
        maximum_microsim_batch_size=1,
        target_materialization_cache_dir=tmp_path,
        target_materialization_cache_context=cache_context,
    )

    household = target_frame.table("household")
    assert np.array_equal(household["income_tax"], np.asarray([130.0, 70.0]))
    assert np.array_equal(
        household["jct_mock_tax_expenditure"], np.asarray([-15.0, -30.0])
    )
    assert len(registry) == 1
    assert dropped["dropped_target_names"] == []
    assert dropped["target_materialization_cache"]["hits"] == 0
    assert dropped["target_materialization_cache"]["misses"] == 1
    assert dropped["target_materialization_cache"]["writes"] == 1
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert len(list(tmp_path.glob("*.npy"))) == 1
    assert [dataset[1] for dataset in datasets] == [
        (),
        ("mock_credit",),
        ("mock_credit",),
    ]
    assert [dataset[0].n("household") for dataset in datasets] == [2, 1, 1]
    assert [dataset[3] for dataset in datasets] == [False, False, False]
    assert formula_owned_assertions == [2, 2]
    assert len(simulations) == 3
    assert [simulation.cache_invalidations for simulation in simulations] == [1, 1, 1]

    target_frame_again, registry_again, dropped_again = (
        builder._materialize_target_frame(
            frame,
            (target,),
            maximum_microsim_batch_size=1,
            target_materialization_cache_dir=tmp_path,
            target_materialization_cache_context=cache_context,
        )
    )

    household_again = target_frame_again.table("household")
    assert np.array_equal(
        household_again["jct_mock_tax_expenditure"], np.asarray([-15.0, -30.0])
    )
    assert len(registry_again) == 1
    assert dropped_again["dropped_target_names"] == []
    assert dropped_again["target_materialization_cache"]["hits"] == 1
    assert dropped_again["target_materialization_cache"]["misses"] == 0
    assert dropped_again["target_materialization_cache"]["writes"] == 0
    assert [dataset[1] for dataset in datasets] == [
        (),
        ("mock_credit",),
        ("mock_credit",),
        (),
    ]
    assert [dataset[0].n("household") for dataset in datasets] == [2, 1, 1, 2]
    assert [dataset[3] for dataset in datasets] == [False, False, False, False]
    assert formula_owned_assertions == [2, 2, 2]
    assert len(simulations) == 4
    assert [simulation.cache_invalidations for simulation in simulations] == [
        1,
        1,
        1,
        1,
    ]


def test_soi_eitc_child_targets_materialize_distinct_child_slices(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
                    "person_household_id": np.asarray([1, 1, 2, 2], dtype="int64"),
                    "person_tax_unit_id": np.asarray([10, 20, 30, 40], dtype="int64"),
                    "person_spm_unit_id": np.asarray(
                        [100, 100, 200, 200], dtype="int64"
                    ),
                    "person_family_id": np.asarray(
                        [1000, 1000, 2000, 2000], dtype="int64"
                    ),
                    "person_marital_unit_id": np.asarray(
                        [10000, 20000, 30000, 40000], dtype="int64"
                    ),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([1, 2], dtype="int64"),
                    "state_fips": np.asarray([6, 6], dtype="int64"),
                    "congressional_district_geoid": np.asarray(
                        [601, 602], dtype="int64"
                    ),
                }
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([10, 20, 30, 40], dtype="int64")}
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
            "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([10000, 20000, 30000, 40000])}
            ),
        },
        builder.US_SCHEMA,
        {
            "household": builder.Weights(
                values=np.asarray([1.0, 1.0]), kind=WeightKind.DESIGN
            )
        },
    )

    def eitc_spec(name, measure, child_filter, *, count=False, variable="eitc"):
        metadata = {
            "variable": variable,
            "agi_lower_bound": "-inf",
            "agi_upper_bound": "inf",
            "filing_status": "All",
            "source_measure_id": "eitc_returns" if count else "eitc_total",
            "ledger_filter_eitc_child_count": child_filter,
            "measure_mode": "indicator_sum" if count else "sum",
        }
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
        eitc_spec(
            "three_plus_return_count",
            "three_plus_return_count",
            "three_or_more_qualifying_children",
            count=True,
        ),
        TargetSpec(
            name="eitc_return_agi",
            entity="household",
            measure="eitc_return_agi",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "adjusted_gross_income",
                "source_variable": "adjusted_gross_income",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "adjusted_gross_income",
                "ledger_domain": (
                    "individual_income_tax_returns_with_earned_income_credit"
                ),
            },
        ),
        TargetSpec(
            name="eitc_return_count",
            entity="household",
            measure="eitc_return_count",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "count",
                "source_variable": "count",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "return_count",
                "ledger_domain": (
                    "individual_income_tax_returns_with_earned_income_credit"
                ),
                "measure_mode": "indicator_sum",
            },
        ),
        TargetSpec(
            name="form_w2_social_security_tips",
            entity="household",
            measure="form_w2_social_security_tips",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "tip_income",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "return_count",
                "measure_mode": "indicator_sum",
            },
        ),
        TargetSpec(
            name="cd_0601_agi",
            entity="household",
            measure="cd_0601_agi",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "adjusted_gross_income",
                "source_variable": "adjusted_gross_income",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "adjusted_gross_income",
                "congressional_district_geoid": "0601",
            },
        ),
        TargetSpec(
            name="cd_0601_tax_filer_individual_count",
            entity="household",
            measure="cd_0601_tax_filer_individual_count",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "tax_filer_individual_count",
                "source_variable": "tax_filer_individual_count",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "tax_filer_individual_count",
                "congressional_district_geoid": "0601",
            },
        ),
        TargetSpec(
            name="medical_dental_expense_amount",
            entity="household",
            measure="medical_dental_expense_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "medical_expense_deduction",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "medical_dental_expense_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="medical_dental_expense_returns",
            entity="household",
            measure="medical_dental_expense_returns",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "medical_expense_deduction",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "medical_dental_expense_returns",
                "measure_mode": "indicator_sum",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="real_estate_taxes_amount",
            entity="household",
            measure="real_estate_taxes_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "real_estate_taxes",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "real_estate_taxes_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="real_estate_taxes_claims",
            entity="household",
            measure="real_estate_taxes_claims",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "real_estate_taxes",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "real_estate_taxes_claims",
                "measure_mode": "indicator_sum",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="limited_state_local_taxes_amount",
            entity="household",
            measure="limited_state_local_taxes_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "salt_deduction",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "limited_state_local_taxes_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="limited_state_local_taxes_returns",
            entity="household",
            measure="limited_state_local_taxes_returns",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "salt_deduction",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "limited_state_local_taxes_returns",
                "measure_mode": "indicator_sum",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="itemized_deductions_amount",
            entity="household",
            measure="itemized_deductions_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "itemized_taxable_income_deductions",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "itemized_deductions_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="total_itemized_deductions_amount",
            entity="household",
            measure="total_itemized_deductions_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "itemized_taxable_income_deductions",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "total_itemized_deductions_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="charitable_amount",
            entity="household",
            measure="charitable_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "charitable_deduction",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "charitable_amount",
                "itemized_only": "true",
            },
        ),
        TargetSpec(
            name="interest_paid_deduction_amount",
            entity="household",
            measure="interest_paid_deduction_amount",
            value=1.0,
            source="fixture",
            family="irs_soi",
            metadata={
                "variable": "interest_deduction",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
                "filing_status": "All",
                "source_measure_id": "interest_paid_deduction_amount",
                "itemized_only": "true",
            },
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
                "eitc",
                "eitc_child_count",
                "itemized_taxable_income_deductions",
                "charitable_deduction",
                "interest_deduction",
                "medical_expense_deduction",
                "real_estate_taxes",
                "salt_deduction",
                "tip_income",
                "tax_unit_size",
                "tax_unit_itemizes",
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
                "income_tax": np.asarray([0.0, 0.0, 0.0, 0.0]),
                "taxable_income": np.asarray([0.0, 0.0, 0.0, 0.0]),
                "adjusted_gross_income": np.asarray(
                    [10_000.0, 20_000.0, 30_000.0, 40_000.0]
                ),
                "filing_status": np.asarray(["SINGLE", "SINGLE", "SINGLE", "SINGLE"]),
                "state_income_tax": np.asarray([0.0, 0.0, 0.0, 0.0]),
                "eitc": np.asarray([100.0, 200.0, 300.0, 0.0]),
                "eitc_child_count": np.asarray([0.0, 2.0, 3.0, 3.0]),
                "itemized_taxable_income_deductions": np.asarray(
                    [1_000.0, 2_000.0, 3_000.0, 4_000.0]
                ),
                "charitable_deduction": np.asarray([10.0, 20.0, 30.0, 40.0]),
                "interest_deduction": np.asarray([1.0, 2.0, 3.0, 4.0]),
                "medical_expense_deduction": np.asarray([100.0, 200.0, 300.0, 400.0]),
                "real_estate_taxes": np.asarray([5_000.0, 6_000.0, 7_000.0, 8_000.0]),
                "salt_deduction": np.asarray([500.0, 600.0, 700.0, 800.0]),
                "tip_income": np.asarray([0.0, 50.0, 0.0, 0.0]),
                "tax_unit_size": np.asarray([1.0, 2.0, 3.0, 4.0]),
                "tax_unit_itemizes": np.asarray([False, True, False, False]),
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
            "adjusted_gross_income": "adjusted_gross_income",
            "eitc": "eitc",
            "itemized_taxable_income_deductions": (
                "itemized_taxable_income_deductions"
            ),
            "charitable_deduction": "charitable_deduction",
            "interest_deduction": "interest_deduction",
            "medical_expense_deduction": "medical_expense_deduction",
            "real_estate_taxes": "real_estate_taxes",
            "salt_deduction": "salt_deduction",
            "tip_income": "tip_income",
            "tax_filer_individual_count": "tax_unit_size",
        },
    )
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", ())

    target_frame, registry, compilation = builder._materialize_target_frame(
        frame, targets
    )

    household = target_frame.table("household")
    assert np.array_equal(household["no_child_amount"], np.asarray([100.0, 0.0]))
    assert np.array_equal(household["two_child_amount"], np.asarray([200.0, 0.0]))
    assert np.array_equal(household["three_plus_amount"], np.asarray([0.0, 300.0]))
    assert np.array_equal(household["two_child_returns"], np.asarray([1.0, 0.0]))
    assert np.array_equal(household["three_plus_return_count"], np.asarray([0.0, 1.0]))
    assert np.array_equal(
        household["eitc_return_agi"], np.asarray([30_000.0, 30_000.0])
    )
    assert np.array_equal(household["eitc_return_count"], np.asarray([2.0, 1.0]))
    assert np.array_equal(
        household["form_w2_social_security_tips"], np.asarray([1.0, 0.0])
    )
    assert np.array_equal(household["cd_0601_agi"], np.asarray([30_000.0, 0.0]))
    assert np.array_equal(
        household["cd_0601_tax_filer_individual_count"], np.asarray([3.0, 0.0])
    )
    assert np.array_equal(
        household["medical_dental_expense_amount"], np.asarray([200.0, 0.0])
    )
    assert np.array_equal(
        household["medical_dental_expense_returns"], np.asarray([1.0, 0.0])
    )
    assert np.array_equal(
        household["real_estate_taxes_amount"], np.asarray([6_000.0, 0.0])
    )
    assert np.array_equal(household["real_estate_taxes_claims"], np.asarray([1.0, 0.0]))
    assert np.array_equal(
        household["limited_state_local_taxes_amount"], np.asarray([600.0, 0.0])
    )
    assert np.array_equal(
        household["limited_state_local_taxes_returns"], np.asarray([1.0, 0.0])
    )
    assert np.array_equal(
        household["itemized_deductions_amount"], np.asarray([2_000.0, 0.0])
    )
    assert np.array_equal(
        household["total_itemized_deductions_amount"], np.asarray([2_000.0, 0.0])
    )
    assert np.array_equal(household["charitable_amount"], np.asarray([20.0, 0.0]))
    assert np.array_equal(
        household["interest_paid_deduction_amount"], np.asarray([2.0, 0.0])
    )
    assert len(registry) == 20
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
            "measure_mode": "indicator_sum" if count else "sum",
        }
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
                    "congressional_district_geoid": np.asarray(
                        ["0601", "1201"], dtype=object
                    ),
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

    def population_age_spec(
        name,
        lower,
        upper,
        *,
        state_fips=None,
        congressional_district_geoid=None,
    ):
        metadata = {
            "materializer": "population_age",
            "measure_mode": "indicator_sum",
            "target_role": "population_age",
            "geography_scope": (
                "congressional_district"
                if congressional_district_geoid
                else "state"
                if state_fips
                else "national"
            ),
            "age_lower_bound": str(lower),
            "age_upper_bound": str(upper),
        }
        if state_fips:
            metadata["state_fips"] = state_fips
        if congressional_district_geoid:
            metadata["congressional_district_geoid"] = congressional_district_geoid
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
        population_age_spec(
            "ca_01_age_0_to_4",
            0,
            5,
            state_fips="06",
            congressional_district_geoid="0601",
        ),
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
    assert np.array_equal(household["ca_01_age_0_to_4"], np.asarray([1.0, 0.0]))
    assert len(registry) == 4
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
                "mass_repair": {
                    "method": "rescale_household_weights_to_census_person_population",
                    "applied": True,
                    "factor": 5.87,
                },
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
        timing={
            "target_compilation_seconds": 3.0,
            "calibration_seconds": 4.0,
            "total_build_seconds": 7.0,
        },
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
    assert (
        build_manifest["gates"]["base_population_scale"]["details"]["mass_repair"][
            "method"
        ]
        == "rescale_household_weights_to_census_person_population"
    )
    assert manifest["data_package"] == {"name": "populace-data", "version": "0.1.0"}
    assert manifest["default_datasets"] == {"national": "populace_us_2024"}
    assert manifest["build"]["built_with_model_package"] == {
        "name": "policyengine-us",
        "version": "1.729.0",
    }
    assert build_manifest["timing"] == {
        "target_compilation_seconds": 3.0,
        "calibration_seconds": 4.0,
        "total_build_seconds": 7.0,
    }
    assert manifest["build"]["timing"] == {
        "target_compilation_seconds": 3.0,
        "calibration_seconds": 4.0,
        "total_build_seconds": 7.0,
    }
    assert (
        manifest["build"]["base_population_scale"]["details"]["mass_repair"]["factor"]
        == 5.87
    )
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


def test_export_frame_rejects_formula_owned_columns(monkeypatch, small_frame) -> None:
    builder = _load_builder_module()

    class FakePolicyEngineUSEngine:
        def _engine_computed_columns(self, tables, *, period):
            assert period == builder.PERIOD
            assert "income" in tables["person"]
            return {"income"}

    monkeypatch.setattr(builder, "PolicyEngineUSEngine", FakePolicyEngineUSEngine)

    with pytest.raises(ValueError, match="Formula-owned.*income"):
        builder._with_calibrated_weights(
            small_frame,
            np.array([1000.0, 2000.0]),
        )


def test_dataset_from_frame_rejects_formula_owned_columns_by_default(
    monkeypatch,
    small_frame,
) -> None:
    builder = _load_builder_module()

    def fake_assert_no_formula_owned_columns(frame):
        assert frame is small_frame
        raise ValueError("formula-owned guard fired")

    monkeypatch.setattr(
        builder,
        "_assert_no_formula_owned_columns",
        fake_assert_no_formula_owned_columns,
    )

    with pytest.raises(ValueError, match="formula-owned guard fired"):
        builder._dataset_from_frame(small_frame)


def test_export_frame_accepts_leaf_only_columns(monkeypatch, small_frame) -> None:
    builder = _load_builder_module()

    class FakePolicyEngineUSEngine:
        def _engine_computed_columns(self, tables, *, period):
            assert period == builder.PERIOD
            assert "income" in tables["person"]
            return set()

    monkeypatch.setattr(builder, "PolicyEngineUSEngine", FakePolicyEngineUSEngine)

    exported = builder._with_calibrated_weights(
        small_frame,
        np.array([1000.0, 2000.0]),
    )

    assert "income" in exported.table("person")
    assert exported.weights_for("household").kind == WeightKind.CALIBRATED


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
        lambda frame, target_specs, **kwargs: (
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
        lambda frame, target_specs, **kwargs: (
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


def test_fiscal_refresh_keeps_unregistered_aca_state_metal_alias_inactive() -> None:
    builder = _load_builder_module()

    assert "cms-aca-oep-state-level" in builder.DIRECT_ACTIVE_ALIASES
    assert "cms-aca-oep-state-metal" not in builder.DIRECT_ACTIVE_ALIASES


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
