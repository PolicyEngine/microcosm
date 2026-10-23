"""Federal Student Aid portfolio targets for federal student loan balances.

The fixture ``us_fsa_portfolio_chronicle_347_facts.jsonl.gz`` holds, byte for
byte, 244 of the 461 rows ``chronicle build-bundle --year 2026`` writes for the
six ``packages/ed_fsa`` source packages of PolicyEngine/chronicle#347 (first
drafted at ``4448a7c``; the exports at the PR head ``81e3f98``, which reads
the raw files from R2, and at its merge commit ``acbedb5`` on Chronicle main
are byte-identical): every
portfolio-summary, location, age, debt-size and loan-type row, and the FY2026
Q2 Direct Loan repayment-plan dollars. The tests pin the plan's reference quarter, its one in-scope control,
each family's reconciled values, every fact's disposition, the release
materializer and the release's restated-filter guard (microcosm#1074).
"""

from __future__ import annotations

import dataclasses
import gzip
import importlib.util
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import compile_us_fiscal_target_registry
from microcosm.build.us_runtime.fsa_student_loan_targets import (
    FSA_MATERIALIZER,
    FSA_PERIOD_ALIGNMENT,
    FSA_TARGET_FAMILY,
    fsa_student_loan_household_values,
    fsa_student_loan_target_receipt,
    plan_fsa_student_loan_targets,
)
from microcosm.build.us_runtime.target_aging import find_period_contract_violations
from microcosm.calibrate import TargetRegistry
from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_POSTAL
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_fiscal_targets import packaged_reference_facts
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_FIXTURE = (
    _TEST_PATHS.tests / "fixtures" / "us_fsa_portfolio_chronicle_347_facts.jsonl.gz"
)
_BILLION = 1e9
# Published FY2026 Q2 (31 March 2026) levels in the fixture, from the FSA
# workbooks: location rows in $ billions and thousands of borrowers.
_STATES_AND_DC_DOLLARS = 1_627.4 * _BILLION
_PUERTO_RICO_AND_OTHER_DOLLARS = (11.7 + 5.0) * _BILLION
_NOT_REPORTED_DOLLARS = 52.2 * _BILLION
_STATES_AND_DC_BORROWERS = 42_375_900.0
_PUERTO_RICO_AND_OTHER_BORROWERS = 336_900.0 + 96_500.0
_NOT_REPORTED_BORROWERS = 1_953_300.0


def _facts() -> list[dict]:
    with gzip.open(_FIXTURE, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _plan(facts: list[dict] | None = None):
    return plan_fsa_student_loan_targets(
        _facts() if facts is None else facts, target_period=2024
    )


def _registry() -> TargetRegistry:
    registry = compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *_facts()],
        target_period=2024,
        allow_unaged_dollar_targets=True,
        student_loan_targets=True,
    )
    return TargetRegistry(
        tuple(spec for spec in registry.specs if spec.family == FSA_TARGET_FAMILY),
        country="us",
    )


def _by_role(registry: TargetRegistry) -> dict[str, list]:
    roles: dict[str, list] = {}
    for spec in registry.specs:
        roles.setdefault(spec.metadata["target_role"], []).append(spec)
    return roles


def _control(dollars: bool) -> float:
    if dollars:
        states, out, not_reported = (
            _STATES_AND_DC_DOLLARS,
            _PUERTO_RICO_AND_OTHER_DOLLARS,
            _NOT_REPORTED_DOLLARS,
        )
    else:
        states, out, not_reported = (
            _STATES_AND_DC_BORROWERS,
            _PUERTO_RICO_AND_OTHER_BORROWERS,
            _NOT_REPORTED_BORROWERS,
        )
    return states * (states + out + not_reported) / (states + out)


def test_plan_binds_the_latest_quarter_every_table_publishes() -> None:
    plan = _plan()

    assert plan.status == "bound"
    assert plan.reference_period == "2026-03"
    assert plan.reference_label == "Data as of Mar. 31, 2026"
    # 51 states x 2 measures, 5 age and 9 debt-size bands x 2, 2 loan types.
    assert len(plan.references) == 51 * 2 + 5 * 2 + 9 * 2 + 2
    for measure, dollars in (("total_balance", True), ("borrowers", False)):
        control = plan.controls[measure]
        assert control["in_scope_control"] == pytest.approx(_control(dollars))
        assert control["published_total"] == pytest.approx(
            (_STATES_AND_DC_DOLLARS + _PUERTO_RICO_AND_OTHER_DOLLARS)
            + _NOT_REPORTED_DOLLARS
            if dollars
            else _STATES_AND_DC_BORROWERS
            + _PUERTO_RICO_AND_OTHER_BORROWERS
            + _NOT_REPORTED_BORROWERS
        )
    assert plan.controls["total_balance"]["in_scope_control"] == pytest.approx(
        1_679.07e9, rel=1e-5
    )
    assert plan.controls["borrowers"]["in_scope_control"] == pytest.approx(
        44_309_425, rel=1e-6
    )


def test_registry_compiles_every_family_onto_one_in_scope_control() -> None:
    registry = _registry()
    roles = _by_role(registry)
    balance = _control(dollars=True)
    borrowers = _control(dollars=False)

    assert len(registry) == 132
    assert {role: len(specs) for role, specs in roles.items()} == {
        "fsa_state_balance": 51,
        "fsa_state_borrowers": 51,
        "fsa_age_band_balance": 5,
        "fsa_age_band_borrowers": 5,
        "fsa_debt_size_band_balance": 9,
        "fsa_debt_size_band_borrowers": 9,
        "fsa_parent_plus_balance": 1,
        "fsa_balance_excluding_parent_plus": 1,
    }
    for role, total in (
        ("fsa_state_balance", balance),
        ("fsa_age_band_balance", balance),
        ("fsa_debt_size_band_balance", balance),
        ("fsa_state_borrowers", borrowers),
        ("fsa_age_band_borrowers", borrowers),
        ("fsa_debt_size_band_borrowers", borrowers),
    ):
        assert math.fsum(spec.value for spec in roles[role]) == pytest.approx(total)
    loan_types = (
        roles["fsa_parent_plus_balance"] + roles["fsa_balance_excluding_parent_plus"]
    )
    assert math.fsum(spec.value for spec in loan_types) == pytest.approx(balance)
    # Parent PLUS is $117.9B of the $1,723.8B the six loan-type rows publish.
    assert roles["fsa_parent_plus_balance"][0].value == pytest.approx(
        balance * 117.9 / 1_723.8, rel=1e-6
    )


def test_state_rows_allocate_not_reported_pro_rata() -> None:
    registry = _registry()
    states = {
        spec.metadata["state_fips"]
        for spec in registry.specs
        if "state_fips" in spec.metadata
    }
    (california,) = (
        spec
        for spec in registry.specs
        if spec.metadata.get("state_fips") == "06"
        and spec.metadata["fsa_measure"] == "total_balance"
    )
    factor = (
        _STATES_AND_DC_DOLLARS + _PUERTO_RICO_AND_OTHER_DOLLARS + _NOT_REPORTED_DOLLARS
    ) / (_STATES_AND_DC_DOLLARS + _PUERTO_RICO_AND_OTHER_DOLLARS)

    assert states == set(US_STATE_FIPS_TO_POSTAL)
    assert "72" not in states
    assert california.value == pytest.approx(156.7e9 * factor)
    assert float(california.metadata["fsa_published_value"]) == 156.7e9
    assert california.metadata["fsa_reconciliation"] == (
        "not_reported_allocated_pro_rata"
    )
    assert float(california.metadata["fsa_reconciliation_factor"]) == pytest.approx(
        factor
    )
    assert california.hierarchy.geography.label == "California"
    assert california.hierarchy.target.label == (
        "Federal student loan balance, California"
    )


def test_bands_bind_as_shares_and_compile_their_edges() -> None:
    roles = _by_role(_registry())
    age = {
        spec.metadata["ledger_layout_groupby_value_id"]: spec
        for spec in roles["fsa_age_band_borrowers"]
    }
    debt = {
        spec.metadata["ledger_layout_groupby_value_id"]: spec
        for spec in roles["fsa_debt_size_band_balance"]
    }

    # Reported age bands publish 44.8M borrowers; "not reported" is 0.1M.
    assert age["age_25_to_34"].value == pytest.approx(
        _control(dollars=False) * 14.0 / 44.8, rel=1e-9
    )
    assert age["age_25_to_34"].metadata["age_lower_bound"] == "25"
    assert age["age_25_to_34"].metadata["age_upper_bound"] == "35"
    assert "age_lower_bound" not in age["age_24_and_younger"].metadata
    assert age["age_62_and_older"].metadata["age_lower_bound"] == "62"
    assert "age_upper_bound" not in age["age_62_and_older"].metadata
    assert age["age_25_to_34"].metadata["age_bound_stamp_source"] == ("constraint_rows")
    assert debt["debt_80k_to_100k"].metadata["fsa_balance_lower_bound"] == "80000"
    assert debt["debt_80k_to_100k"].metadata["fsa_balance_upper_bound"] == "100000"
    assert "fsa_balance_lower_bound" not in debt["debt_less_than_5k"].metadata
    assert "fsa_balance_upper_bound" not in debt["debt_200k_plus"].metadata
    assert debt["debt_200k_plus"].hierarchy.dimensions[0].value_label == "200K+"


def test_specs_sit_at_the_build_period_with_a_declared_alignment() -> None:
    registry = _registry()

    assert {spec.period for spec in registry.specs} == {2024}
    for spec in registry.specs:
        metadata = spec.metadata
        assert metadata["materializer"] == FSA_MATERIALIZER
        assert metadata["source_period"] == "2026-03"
        assert metadata["target_period"] == "2024"
        assert metadata["period_alignment"] == FSA_PERIOD_ALIGNMENT
        assert "period_contract_waiver" not in metadata
        assert "diagnostic_target_label" not in metadata
        assert spec.hierarchy.provider.id == "ed_fsa"
        assert spec.hierarchy.provider.label == "ED · Federal Student Aid"
        assert spec.hierarchy.target.id == spec.name
    assert find_period_contract_violations(registry, target_period=2024) == ()


def test_aged_compile_keeps_fsa_dollars_at_the_reference_date() -> None:
    from microcosm.build.us_runtime.target_aging import age_us_dollar_targets

    registry = age_us_dollar_targets(_registry(), _facts(), target_period=2024)

    for spec in registry.specs:
        if spec.metadata["measure_mode"] == "sum":
            assert spec.metadata["aging_factor_source"] == FSA_PERIOD_ALIGNMENT
            assert float(spec.metadata["aging_factor"]) == 1.0


def test_compile_without_the_flag_ignores_fsa_facts() -> None:
    registry = compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *_facts()],
        target_period=2024,
        allow_unaged_dollar_targets=True,
    )

    assert not [spec for spec in registry.specs if spec.family == FSA_TARGET_FAMILY]


def test_every_fsa_fact_has_one_disposition() -> None:
    facts = _facts()
    receipt = fsa_student_loan_target_receipt(facts, target_period=2024)
    by_id = {item["source_record_id"]: item for item in receipt["dispositions"]}

    assert len(receipt["dispositions"]) == len(facts) == 244
    assert len(by_id) == len(facts)
    assert receipt["disposition_counts"] == {
        "allocated": 4,
        "bound": 136,
        "diagnostic": 34,
        "excluded": 8,
        "other_quarter": 58,
        "out_of_scope": 4,
    }
    location = "ed_fsa.portfolio_by_location.2026_03_31"
    assert by_id[f"{location}.pr.dollars_outstanding"]["disposition"] == (
        "out_of_scope"
    )
    assert by_id[f"{location}.other.borrowers"]["disposition"] == "out_of_scope"
    assert by_id[f"{location}.not_reported.dollars_outstanding"]["disposition"] == (
        "allocated"
    )
    loan_type = "ed_fsa.portfolio_by_loan_type.fy2026_q2"
    assert by_id[f"{loan_type}.recipients.parent_plus.recipients"]["disposition"] == (
        "excluded"
    )
    assert by_id[f"{loan_type}.dollars_outstanding.consolidation.dollars_outstanding"][
        "target"
    ] == (f"{loan_type}.dollars_outstanding.excluding_parent_plus.dollars_outstanding")
    assert {
        item["disposition"]
        for item in receipt["dispositions"]
        if item["table"] in {"portfolio_summary", "dl_by_repayment_plan"}
    } == {"diagnostic"}
    assert {
        item["period"]
        for item in receipt["dispositions"]
        if item["disposition"] == "other_quarter"
    } == {"2025-12", "2026-06"}
    assert len(receipt["targets"]) == 132


def test_a_partial_feed_refuses() -> None:
    facts = _facts()
    without_age = [
        fact
        for fact in facts
        if ".portfolio_by_age." not in fact["layout"]["record_set_id"]
    ]
    with pytest.raises(ValueError, match="portfolio_by_age"):
        _plan(without_age)

    only_q1_age = [
        fact
        for fact in facts
        if not (
            ".portfolio_by_age." in fact["layout"]["record_set_id"]
            and fact["period"]["value"] == "2026-03"
        )
    ]
    with pytest.raises(ValueError, match="share no quarter"):
        _plan(only_q1_age)

    without_texas = [fact for fact in facts if fact["geography"]["id"] != "0400000US48"]
    with pytest.raises(ValueError, match="missing states"):
        _plan(without_texas)


def test_a_gapped_band_partition_refuses() -> None:
    facts = _facts()
    gapped = [
        fact
        for fact in facts
        if not fact["lineage"]["source_record_id"].endswith(
            "fy2026_q2.borrowers.debt_40k_to_60k.borrowers"
        )
    ]
    with pytest.raises(ValueError, match="not contiguous"):
        _plan(gapped)


def test_no_fsa_facts_plans_nothing() -> None:
    plan = _plan(packaged_reference_facts())

    assert plan.status == "no_fsa_facts"
    assert plan.references == ()


def _frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "person_household_id": [10, 10, 20, 20, 30, 30],
            "age": [24.0, 52.0, 30.0, 61.0, 40.0, 70.0],
            "federal_student_loan_balance": [3_000.0, 0.0, 85_000.0, 0.0, 0.0, 0.0],
            "federal_parent_plus_loan_balance": [0.0, 40_000.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    household_ids = np.array([10, 20, 30])
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_spm_unit_id"] = person["person_household_id"] + 2_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(len(person)) + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": household_ids, "state_fips": [6, 48, 6]}
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": household_ids + 1_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 2_000}),
        "family": pd.DataFrame({"family_id": household_ids + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(len(person)) + 4_000}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.full(len(household_ids), 1_000.0), kind=WeightKind.DESIGN
            )
        },
    )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"fsa_measure": "total_balance"}, [43_000.0, 85_000.0, 0.0]),
        ({"fsa_measure": "borrowers"}, [2.0, 1.0, 0.0]),
        ({"fsa_measure": "parent_plus_balance"}, [40_000.0, 0.0, 0.0]),
        ({"fsa_measure": "balance_excluding_parent_plus"}, [3_000.0, 85_000.0, 0.0]),
        (
            {"fsa_measure": "total_balance", "state_fips": "06"},
            [43_000.0, 0.0, 0.0],
        ),
        (
            {"fsa_measure": "borrowers", "age_upper_bound": "25"},
            [1.0, 0.0, 0.0],
        ),
        (
            {"fsa_measure": "total_balance", "age_lower_bound": "50"},
            [40_000.0, 0.0, 0.0],
        ),
        (
            {
                "fsa_measure": "borrowers",
                "fsa_balance_lower_bound": "80000",
                "fsa_balance_upper_bound": "100000",
            },
            [0.0, 1.0, 0.0],
        ),
        (
            {"fsa_measure": "borrowers", "fsa_balance_upper_bound": "5000"},
            [1.0, 0.0, 0.0],
        ),
    ],
)
def test_household_values_sum_the_stored_balances(
    metadata: dict[str, str], expected: list[float]
) -> None:
    np.testing.assert_array_equal(
        fsa_student_loan_household_values(_frame(), metadata), expected
    )


def test_household_values_need_both_balance_columns() -> None:
    frame = _frame()
    person = frame.table("person").drop(columns=["federal_parent_plus_loan_balance"])
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    stripped = Frame(tables, US_SCHEMA, {"household": frame.weights_for("household")})

    assert (
        fsa_student_loan_household_values(stripped, {"fsa_measure": "borrowers"})
        is None
    )
    with pytest.raises(ValueError, match="Unknown FSA measure"):
        fsa_student_loan_household_values(frame, {"fsa_measure": "recipients"})


def test_compiled_fsa_targets_round_trip_through_the_materializer() -> None:
    registry = _registry()
    frame = _frame()
    columns = {
        spec.measure: fsa_student_loan_household_values(frame, spec.metadata)
        for spec in registry.specs
    }
    weights = frame.weights_for("household").values
    total = {
        role: math.fsum(float(np.dot(columns[spec.measure], weights)) for spec in specs)
        for role, specs in _by_role(registry).items()
    }

    # Every family partitions the same people, so each sums to the same level.
    assert total["fsa_state_balance"] == total["fsa_age_band_balance"]
    assert total["fsa_age_band_balance"] == total["fsa_debt_size_band_balance"]
    assert total["fsa_debt_size_band_balance"] == pytest.approx(128_000.0 * 1_000.0)
    assert total["fsa_state_borrowers"] == total["fsa_age_band_borrowers"] == 3_000.0
    assert total["fsa_debt_size_band_borrowers"] == 3_000.0
    assert (
        total["fsa_parent_plus_balance"] + total["fsa_balance_excluding_parent_plus"]
        == total["fsa_state_balance"]
    )


def _load_builder_module():
    path = _TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_filter_guard_accepts_the_compiled_fsa_bands() -> None:
    builder = _load_builder_module()
    registry = _registry()
    restated = Counter(
        key
        for spec in registry.specs
        for key in spec.metadata
        if key.startswith("ledger_filter_")
    )

    assert restated == {
        "ledger_filter_age_lower_bound": 8,
        "ledger_filter_age_upper_bound": 8,
        "ledger_filter_federal_student_loan_balance_lower_bound": 16,
        "ledger_filter_federal_student_loan_balance_upper_bound": 16,
    }
    assert builder._unsupported_ledger_filter_metadata(registry.specs) == {}


def test_release_filter_guard_refuses_a_balance_band_it_would_ignore() -> None:
    builder = _load_builder_module()
    spec = next(
        spec
        for spec in _registry().specs
        if spec.metadata.get("fsa_balance_lower_bound") == "80000"
    )
    other_materializer = dataclasses.replace(
        spec, metadata={**spec.metadata, "materializer": "policyengine_variable"}
    )
    disagreeing = dataclasses.replace(
        spec,
        name=f"{spec.name}.disagreeing",
        hierarchy=None,
        metadata={**spec.metadata, "fsa_balance_lower_bound": "60000"},
    )

    refused = builder._unsupported_ledger_filter_metadata(
        (other_materializer, disagreeing)
    )
    assert any(
        "applies no balance band" in entry for entry in refused[other_materializer.name]
    )
    assert any("disagrees with" in entry for entry in refused[disagreeing.name])


def test_release_parity_contract_reads_the_registry_without_fsa_rows() -> None:
    builder = _load_builder_module()
    registry = compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *_facts()],
        target_period=2024,
        allow_unaged_dollar_targets=True,
        student_loan_targets=True,
    )

    parity = builder._without_opt_in_student_loan_targets(registry)
    assert len(parity) == len(registry) - 132
    assert not [spec for spec in parity.specs if spec.family == FSA_TARGET_FAMILY]
