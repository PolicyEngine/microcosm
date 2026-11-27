"""The pre-R2 county AGI check against R2's SOI AGI measure (microcosm#922).

``tools/check_us_acs_county_agi_support.py`` computes each household's AGI
with R2's projection, reviewed-null fill and dataset builder, in household
chunks. The engine counts only the positive income sources of a tax unit's
head and spouse (``irs_gross_income``), and only an adult can be a head or a
spouse, so income held by a dependent, or by a tax unit without an adult,
adds nothing. This test runs the check's engine pass on six households and
compares it with the production SOI materializer's AGI measure, the value
R2's state AGI row stores (#1660 review).
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_batched_target_materialization import (
    _load_builder_module,
    _soi,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
#: Household AGI: a dependent's wages (1), a head's wages (2), a retiree's
#: interest beside a dependent's wages (3), a spouse's wages (4), a tax unit
#: of one 16-year-old with wages (5), and a capital loss alone (6).
_EXPECTED_AGI = [0.0, 30_000.0, 100.0, 20_000.0, 0.0, -2_000.0]


def _check():
    path = _TEST_PATHS.repository / "tools" / "check_us_acs_county_agi_support.py"
    spec = importlib.util.spec_from_file_location("check_us_county_agi_engine", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame(*, with_nulls: bool) -> Frame:
    household = np.asarray([1, 1, 2, 3, 3, 4, 4, 5, 6, 6], "int64")
    tax_unit = np.asarray([10, 10, 20, 30, 30, 40, 40, 50, 60, 60], "int64")
    interest = [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if with_nulls:
        # A registered null: the fill step gives it the engine default, 0.
        interest[0] = np.nan
    persons = pd.DataFrame(
        {
            "person_id": np.arange(1, 11, dtype="int64"),
            "person_household_id": household,
            "person_tax_unit_id": tax_unit,
            "person_spm_unit_id": tax_unit * 10,
            "person_family_id": tax_unit * 100,
            "person_marital_unit_id": np.asarray(
                [1, 2, 3, 4, 5, 6, 6, 7, 8, 9], "int64"
            )
            * 10_000,
            "age": np.asarray(
                [40.0, 10.0, 40.0, 70.0, 16.0, 40.0, 38.0, 16.0, 50.0, 12.0]
            ),
            "employment_income_before_lsr": np.asarray(
                [0.0, 5_000.0, 30_000.0, 0.0, 8_000.0, 0.0, 20_000.0, 6_000.0, 0.0, 0.0]
            ),
            "taxable_interest_income": np.asarray(interest),
            "short_term_capital_gains": np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2_000.0, 0.0]
            ),
        }
    )
    tables = {"person": persons}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(persons[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    if with_nulls:
        # A runtime-owned take-up flag that R2's stages fill on ACS rows and
        # the AGI does not read: the check holds it back while its nulls are
        # on ACS rows only.
        tables["spm_unit"]["takes_up_snap_if_eligible"] = pd.Series(
            [True, None, True, True, True, True], dtype=object
        )
        tables["spm_unit"]["spm_unit_spine"] = "acs_2024_1yr"
    tables["household"] = pd.DataFrame(
        {
            "household_id": np.arange(1, 7, dtype="int64"),
            "state_fips": np.full(6, 6, dtype="int64"),
        }
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(6), WeightKind.DESIGN)},
    )


def test_no_column_an_r2_stage_fills_is_an_agi_input() -> None:
    """The check holds back a null stage-filled column on ACS rows because
    the AGI does not read it in R2's year; an engine release whose AGI read
    one would make the check refuse a staging file R2 accepts (review of
    #1660). The closure follows reads inside the engine's helper functions:
    the self-employed health insurance deduction reads its two Form 7206
    inputs only in the module-level ``form_7206_line_13`` (microcosm#1675)."""

    from policyengine_us import CountryTaxBenefitSystem

    from microcosm.frame.adapters.policyengine_us import (
        PolicyEngineUSVariableMetadataIndex,
    )

    check = _check()
    release = check._release_tool()
    unread = check.unread_variables(release.PERIOD, system=CountryTaxBenefitSystem())
    inputs = check.agi_engine_inputs(
        PolicyEngineUSVariableMetadataIndex(follow_helpers=True), unread=unread
    )
    gapped = check.agi_engine_inputs(
        PolicyEngineUSVariableMetadataIndex(), unread=unread
    )
    stage_filled = check.reviewed_fills.stage_filled_never_default(release)

    assert "employment_income_before_lsr" in inputs
    form_7206 = {
        "s_corp_shareholder_medicare_wages",
        "self_employed_health_insurance_plan_business",
    }
    assert form_7206 <= inputs - gapped
    assert stage_filled
    assert inputs.isdisjoint(column for _entity, column in stage_filled)


def test_the_closure_stops_at_the_behavioral_responses_a_pass_without_a_reform_skips() -> (
    None
):
    """Through the engine's ``get_behavioral_response_measurements`` helper,
    the behavioral responses read ``household_net_income``, and through it
    the take-up flags R2's stages fill. An engine pass without a reform has no
    baseline, so the responses return 0 before reading anything and the labor
    supply elasticity effects are never computed (microcosm#1675)."""

    from policyengine_us import CountryTaxBenefitSystem, Simulation

    from microcosm.frame.adapters.policyengine_us import (
        PolicyEngineUSVariableMetadataIndex,
    )

    check = _check()
    release = check._release_tool()
    system = CountryTaxBenefitSystem()
    index = PolicyEngineUSVariableMetadataIndex(follow_helpers=True)
    stage_filled = {
        column
        for _entity, column in check.reviewed_fills.stage_filled_never_default(release)
    }
    deductions = check.unread_deductions(release.PERIOD, system=system)
    through_responses = check.agi_engine_inputs(index, unread=deductions)
    assert not through_responses.isdisjoint(stage_filled)
    assert check.unread_variables(release.PERIOD, system=system) == (
        *deductions,
        *check.NO_REFORM_UNREAD,
    )
    stopped = check.agi_engine_inputs(
        index, unread=check.unread_variables(release.PERIOD, system=system)
    )
    assert stopped.isdisjoint(stage_filled)

    y = str(release.PERIOD)
    members = {"members": ["filer"]}
    simulation = Simulation(
        situation={
            "people": {
                "filer": {
                    "age": {y: 40},
                    "employment_income_before_lsr": {y: 50_000},
                    "self_employment_income_before_lsr": {y: 10_000},
                    "long_term_capital_gains_before_response": {y: 20_000},
                    "weekly_hours_worked_before_lsr": {y: 40},
                }
            },
            "tax_units": {"tax_unit": members},
            "spm_units": {"spm_unit": members},
            "families": {"family": members},
            "marital_units": {"marital_unit": members},
            "households": {"household": {**members, "state_code_str": {y: "TX"}}},
        }
    )
    assert simulation.baseline is None
    for name in (
        "labor_supply_behavioral_response",
        "capital_gains_behavioral_response",
        "weekly_hours_worked_behavioural_response",
    ):
        assert float(simulation.calculate(name, release.PERIOD)[0]) == 0.0, name
    assert float(simulation.calculate("adjusted_gross_income", release.PERIOD)[0]) > 0
    for name in ("income_elasticity_lsr", "substitution_elasticity_lsr"):
        assert simulation.get_holder(name).get_known_periods() == [], name


def test_the_closure_stops_only_at_a_deduction_whose_year_does_not_list_it() -> None:
    """policyengine-us 2.145.3's 2020 charitable deduction for non-itemizers
    reads the itemization choice in 2020 alone: 2020 lists it and it lowers
    AGI there, while in R2's year it is not listed and is 0, so the walk may
    stop at it."""

    from policyengine_us import CountryTaxBenefitSystem, Simulation

    from microcosm.frame.adapters.policyengine_us import (
        PolicyEngineUSVariableMetadataIndex,
    )

    check = _check()
    release = check._release_tool()
    system = CountryTaxBenefitSystem()
    assert check.unread_deductions(2020, system=system) == ()
    unread = check.unread_deductions(release.PERIOD, system=system)
    assert unread == ("charitable_contributions_ald",)

    index = PolicyEngineUSVariableMetadataIndex(follow_helpers=True)
    static = check.agi_engine_inputs(index)
    assert check.agi_engine_inputs(index, unread=unread) <= static

    for year, deduction in ((2020, 300.0), (release.PERIOD, 0.0)):
        y = str(year)
        members = {"members": ["filer"]}
        simulation = Simulation(
            situation={
                "people": {
                    "filer": {
                        "age": {y: 40},
                        "employment_income": {y: 50_000},
                        "charitable_cash_donations": {y: 1_000},
                    }
                },
                "tax_units": {"tax_unit": members},
                "spm_units": {"spm_unit": members},
                "families": {"family": members},
                "marital_units": {"marital_unit": members},
                "households": {"household": {**members, "state_code_str": {y: "TX"}}},
            }
        )
        got = simulation.calculate("charitable_contributions_ald", year)
        assert float(got[0]) == deduction, year


def test_household_agi_counts_only_heads_and_spouses_as_the_soi_row_does(
    tmp_path,
) -> None:
    check = _check()
    release = check._release_tool()
    summary = tmp_path / "staging.summary.json"
    summary.write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [
                    {
                        "entity": "person",
                        "column": "taxable_interest_income",
                        "missing_rows": 1,
                        "rows": 10,
                    }
                ]
            }
        )
    )

    agi, receipt = check.household_agi(
        _frame(with_nulls=True), summary, release=release, hh_chunk=2
    )

    np.testing.assert_array_equal(agi, _EXPECTED_AGI)
    assert receipt["chunks"] == 3
    assert receipt["reviewed_fill_columns"] == 1
    assert receipt["null_runtime_columns_held_back"] == {
        "spm_unit": ["takes_up_snap_if_eligible"]
    }

    builder = _load_builder_module()
    target_frame, _registry, compilation = builder._materialize_target_frame(
        _frame(with_nulls=False),
        (_soi("agi_amount", "adjusted_gross_income"),),
        maximum_microsim_batch_size=None,
    )
    assert compilation["dropped_target_names"] == []
    np.testing.assert_array_equal(
        target_frame.table("household")["agi_amount"].to_numpy(dtype=np.float64), agi
    )


def test_r2s_own_agi_pass_stores_the_soi_rows_agi(tmp_path) -> None:
    """R2 computes each household's AGI before minimum support on a county
    SOI surface (microcosm#1661), one chunk at a time; it must be the value
    the state SOI AGI row stores, as the pre-check's is."""

    check = _check()
    release = check._release_tool()
    summary = tmp_path / "staging.summary.json"
    summary.write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [
                    {
                        "entity": "person",
                        "column": "taxable_interest_income",
                        "missing_rows": 1,
                        "rows": 10,
                    }
                ]
            }
        )
    )

    # R2's stages have filled the take-up flags by then, so its pass holds
    # nothing back; the registered interest null is filled chunk by chunk.
    frame = _frame(with_nulls=True)
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["spm_unit"] = tables["spm_unit"].assign(takes_up_snap_if_eligible=True)
    frame = Frame(tables, frame.schema, {"household": frame.weights_for("household")})

    agi = release._household_agi(frame, summary, hh_chunk=2)

    np.testing.assert_array_equal(agi, _EXPECTED_AGI)
