"""JCT reform batches against policyengine-us.

Target compilation scores each JCT tax-expenditure reform in household batches,
each batch engine given the reform's tax-benefit system. These tests compare
it with the construction it replaced, which passed ``reform=`` as well: given
both, policyengine-us 2.2.1 clones the supplied system on every construction
and adds a ``baseline`` branch to the engine. They count system clones, and
exercise the two refusals that stand in for the missing ``baseline`` branch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_batched_target_materialization import (
    _load_builder_module,
    _medicaid_frame,
)

#: (state_fips, persons, tax-unit HSA deduction, SPM-unit childcare). One tax
#: unit, SPM unit and family per household; the first two persons of a
#: household are its head and spouse. Each JCT reform's variable is nonzero
#: for at least one household.
_JCT_HOUSEHOLDS = (
    (
        6,
        (
            {
                "age": 45,
                "employment_income_before_lsr": 300_000.0,
                "real_estate_taxes": 12_000.0,
                "home_mortgage_interest": 20_000.0,
                "charitable_cash_donations": 8_000.0,
                "other_medical_expenses": 40_000.0,
            },
            {"age": 44},
        ),
        0.0,
        0.0,
    ),
    (
        36,
        (
            {
                "age": 50,
                "self_employment_income_before_lsr": 120_000.0,
                # Self-employed premiums are defined only for people marked
                # self-employed.
                "is_self_employed": True,
                "health_insurance_premiums": 9_000.0,
                "self_employed_pension_contributions_desired": 10_000.0,
            },
        ),
        0.0,
        0.0,
    ),
    (
        48,
        (
            {
                "age": 29,
                "employment_income_before_lsr": 60_000.0,
                "student_loan_interest": 2_500.0,
                "traditional_ira_contributions_desired": 6_000.0,
            },
        ),
        3_000.0,
        0.0,
    ),
    (
        6,
        (
            {"age": 38, "employment_income_before_lsr": 70_000.0},
            {"age": 36, "employment_income_before_lsr": 40_000.0},
            {"age": 4},
            {"age": 7},
        ),
        0.0,
        8_000.0,
    ),
    (
        24,
        (
            {
                "age": 61,
                "employment_income_before_lsr": 95_000.0,
                "charitable_cash_donations": 30_000.0,
                "real_estate_taxes": 9_000.0,
                "home_mortgage_interest": 15_000.0,
            },
        ),
        0.0,
        0.0,
    ),
)
#: Boolean person inputs and the value of a person who does not set one.
_BOOLEAN_INPUTS = {"is_self_employed": False}
#: Five households in batches of two: three batch engines per reform.
_BATCH_SIZE = 2
_BATCHES = 3


def _jct_frame() -> Frame:
    input_columns = sorted(
        {
            column
            for _, persons, _, _ in _JCT_HOUSEHOLDS
            for person in persons
            for column in person
        }
        - {"age"}
    )
    rows = []
    for household_id, (_, persons, _, _) in enumerate(_JCT_HOUSEHOLDS, start=1):
        for position, person in enumerate(persons):
            rows.append(
                {
                    "person_id": len(rows) + 1,
                    "person_household_id": household_id,
                    "person_tax_unit_id": household_id * 10,
                    "person_spm_unit_id": household_id * 100,
                    "person_family_id": household_id * 1_000,
                    # Head and spouse share a marital unit.
                    "person_marital_unit_id": household_id * 10_000
                    + max(position - 1, 0),
                    "age": float(person["age"]),
                    **{
                        column: person.get(column, _BOOLEAN_INPUTS.get(column, 0.0))
                        for column in input_columns
                    },
                }
            )
    person = pd.DataFrame(rows)
    for column in input_columns:
        person[column] = person[column].astype(
            bool if column in _BOOLEAN_INPUTS else np.float64
        )
    household_ids = np.arange(1, len(_JCT_HOUSEHOLDS) + 1, dtype="int64")
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": household_ids,
                    "state_fips": np.asarray(
                        [state for state, _, _, _ in _JCT_HOUSEHOLDS], "int64"
                    ),
                }
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": household_ids * 10,
                    "health_savings_account_ald": [
                        hsa for _, _, hsa, _ in _JCT_HOUSEHOLDS
                    ],
                }
            ),
            "spm_unit": pd.DataFrame(
                {
                    "spm_unit_id": household_ids * 100,
                    "spm_unit_pre_subsidy_childcare_expenses": [
                        childcare for _, _, _, childcare in _JCT_HOUSEHOLDS
                    ],
                }
            ),
            "family": pd.DataFrame({"family_id": household_ids * 1_000}),
            "marital_unit": pd.DataFrame(
                {
                    "marital_unit_id": np.unique(
                        person["person_marital_unit_id"].to_numpy()
                    )
                }
            ),
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(_JCT_HOUSEHOLDS), dtype=np.float64), WeightKind.DESIGN
            )
        },
    )


def _record_reform_systems(monkeypatch) -> list[tuple[object, object]]:
    """Record each ``(reform, system)`` that ``Microsimulation`` builds through
    ``default_tax_benefit_system(reform=...)``, as
    ``_reform_household_income_tax`` does once per reform."""

    from policyengine_us import Microsimulation

    built: list[tuple[object, object]] = []
    build_system = Microsimulation.default_tax_benefit_system

    def recording_build(*args, **kwargs):
        reform_system = build_system(*args, **kwargs)
        built.append((kwargs.get("reform"), reform_system))
        return reform_system

    monkeypatch.setattr(
        Microsimulation, "default_tax_benefit_system", staticmethod(recording_build)
    )
    return built


def _income_tax_given_reform_argument(
    builder,
    frame: Frame,
    reform_spec,
    system,
    *,
    reform,
    reform_system,
    batch_size: int,
) -> np.ndarray:
    """The household income tax as each batch engine computed it when given
    ``reform=`` as well as the reform's system, which the reform's batches
    shared (the construction before microcosm 1.6 · 1). Each engine carries a
    ``baseline`` branch; at the installed engine's zero default elasticities
    it scores no labor-supply or capital-gains response, which a batch engine
    without that branch scores as zero too."""

    from policyengine_us import Microsimulation

    assert (
        builder._moved_behavioral_response_parameters(
            reform_system,
            Microsimulation.default_tax_benefit_system_instance,
            builder.PERIOD,
        )
        == []
    ), reform_spec.measure
    n_households = frame.n("household")
    income_tax = np.zeros(n_households, dtype=np.float64)
    for positions in builder._household_position_batches(n_households, batch_size):
        batch_frame = builder._select_households_by_position(frame, positions)
        simulation = Microsimulation(
            tax_benefit_system=reform_system,
            dataset=builder._dataset_from_frame(
                batch_frame,
                zero_variables=(reform_spec.neutralized_variable,),
                system=system,
                assert_no_formula_owned_columns=False,
                period_only=True,
            ),
            reform=reform,
        )
        try:
            assert simulation.baseline is not None
            income_tax[positions] = builder._collapse_tax_unit(
                builder._calculate_array(simulation, "income_tax"),
                builder._tax_unit_to_household_positions(batch_frame),
                batch_frame.n("household"),
            )
            for response in builder.POST_EXPORT_BEHAVIORAL_RESPONSE_PARAMETERS:
                values = builder._calculate_array(simulation, response)
                assert not np.any(values), (reform_spec.measure, response)
        finally:
            builder.release_engine_simulation(simulation)
    return income_tax


def _count_system_clones(monkeypatch) -> list[int]:
    """Count whole tax-benefit-system clones (policyengine-core's
    ``TaxBenefitSystem.clone``, which policyengine-us 2.2.1's
    ``clone_spm_system`` calls)."""

    from policyengine_core.taxbenefitsystems import TaxBenefitSystem

    clones = [0]
    real_clone = TaxBenefitSystem.clone

    def counting_clone(self, *args, **kwargs):
        clones[0] += 1
        return real_clone(self, *args, **kwargs)

    monkeypatch.setattr(TaxBenefitSystem, "clone", counting_clone)
    return clones


def _reform_income_tax_both_ways(
    builder, frame: Frame, reform_spec, system, built, *, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """``_reform_household_income_tax``'s household income tax, then the same
    batches' given ``reform=`` as well, on the one reform system it built."""

    from policyengine_us import Microsimulation

    systems_before = len(built)
    observed = builder._reform_household_income_tax(
        base_frame=frame,
        reform_spec=reform_spec,
        system=system,
        microsimulation_cls=Microsimulation,
        n_households=frame.n("household"),
        batch_size=batch_size,
    )
    assert len(built) == systems_before + 1, reform_spec.measure
    reform, reform_system = built[-1]
    expected = _income_tax_given_reform_argument(
        builder,
        frame,
        reform_spec,
        system,
        reform=reform,
        reform_system=reform_system,
        batch_size=batch_size,
    )
    return observed, expected


def test_jct_reform_batches_match_the_reform_argument_construction(
    monkeypatch,
) -> None:
    """Every shipped JCT reform scores the same household income tax, byte for
    byte, on batch engines given the reform's system alone as on batch
    engines given ``reform=`` as well. Each reform changes income tax for at
    least one fixture household and passes both refusals that stand in for
    the missing ``baseline`` branch."""

    builder = _load_builder_module()
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    frame = _jct_frame()
    n_households = frame.n("household")
    system = CountryTaxBenefitSystem()
    builder._assert_no_formula_owned_columns(frame)
    built = _record_reform_systems(monkeypatch)
    baseline = Microsimulation(
        dataset=builder._dataset_from_frame(
            frame, assert_no_formula_owned_columns=False, period_only=True
        )
    )
    try:
        baseline_income_tax = builder._collapse_tax_unit(
            builder._calculate_array(baseline, "income_tax"),
            builder._tax_unit_to_household_positions(frame),
            n_households,
        )
    finally:
        builder.release_engine_simulation(baseline)

    assert len(builder.US_JCT_TAX_EXPENDITURE_REFORMS) == 11
    for reform_spec in builder.US_JCT_TAX_EXPENDITURE_REFORMS:
        observed, expected = _reform_income_tax_both_ways(
            builder, frame, reform_spec, system, built, batch_size=_BATCH_SIZE
        )
        assert observed.dtype == expected.dtype == np.float64
        assert observed.tobytes() == expected.tobytes(), (
            reform_spec.measure,
            observed,
            expected,
        )
        assert np.any(observed != baseline_income_tax), (
            f"{reform_spec.measure}: the fixture does not reach "
            f"{reform_spec.neutralized_variable}"
        )


def test_jct_reform_batches_clone_no_tax_benefit_system(monkeypatch) -> None:
    """One JCT reform in three batches clones no tax-benefit system. The
    construction with ``reform=`` as well cloned the reform's system once per
    batch engine, which the counter sees."""

    builder = _load_builder_module()
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    frame = _jct_frame()
    system = CountryTaxBenefitSystem()
    reform_spec = builder.US_JCT_TAX_EXPENDITURE_REFORMS[0]
    assert (
        len(
            tuple(
                builder._household_position_batches(frame.n("household"), _BATCH_SIZE)
            )
        )
        == _BATCHES
    )
    built = _record_reform_systems(monkeypatch)
    clones = _count_system_clones(monkeypatch)

    builder._reform_household_income_tax(
        base_frame=frame,
        reform_spec=reform_spec,
        system=system,
        microsimulation_cls=Microsimulation,
        n_households=frame.n("household"),
        batch_size=_BATCH_SIZE,
    )
    assert clones[0] == 0
    assert len(built) == 1

    reform, reform_system = built[0]
    _income_tax_given_reform_argument(
        builder,
        frame,
        reform_spec,
        system,
        reform=reform,
        reform_system=reform_system,
        batch_size=_BATCH_SIZE,
    )
    assert clones[0] == _BATCHES


def test_jct_reform_refuses_moved_behavioral_responses_on_policyengine_us(
    monkeypatch,
) -> None:
    """A reform whose system moves a labor-supply elasticity is refused by
    name, as read from the real reform system's parameters, before any batch
    engine is built."""

    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    builder = _load_builder_module()
    income_elasticity = "gov.simulation.labor_supply_responses.elasticities.income"
    monkeypatch.setattr(
        builder,
        "_make_zero_variable_reform",
        lambda system, variable_name: Reform.from_dict(
            {income_elasticity: {f"{builder.PERIOD}-01-01": -0.05}},
            country_id="us",
        ),
    )
    constructions = [0]

    class CountingMicrosimulation(Microsimulation):
        def __init__(self, *args, **kwargs):
            constructions[0] += 1
            super().__init__(*args, **kwargs)

    frame = _jct_frame()
    reform_spec = builder.US_JCT_TAX_EXPENDITURE_REFORMS[0]
    with pytest.raises(
        ValueError,
        match=(
            rf"{reform_spec.measure}.* off baseline at {builder.PERIOD}: "
            r"gov\.simulation\.labor_supply_responses\.elasticities\.income\."
        ),
    ):
        builder._reform_household_income_tax(
            base_frame=frame,
            reform_spec=reform_spec,
            system=CountryTaxBenefitSystem(),
            microsimulation_cls=CountingMicrosimulation,
            n_households=frame.n("household"),
            batch_size=_BATCH_SIZE,
        )
    assert constructions[0] == 0


def test_jct_reform_batch_refuses_a_baseline_branch_reader_on_policyengine_us(
    monkeypatch,
) -> None:
    """A JCT batch engine that computed a formula reading the ``baseline``
    branch is refused, in one batch, where the population-aggregate refusal is
    off. ``income_tax`` reaches none of those formulas on the installed engine,
    so the engine here also computes ``medicaid_cost``, which reaches the
    Medicaid state denominator: given ``reform=``, that denominator would be
    read from the ``baseline`` branch."""

    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    builder = _load_builder_module()
    real_calculate_array = builder._calculate_array

    def calculate_array_reaching_medicaid(simulation, variable, **kwargs):
        if variable == "income_tax":
            real_calculate_array(simulation, "medicaid_cost")
        return real_calculate_array(simulation, variable, **kwargs)

    monkeypatch.setattr(builder, "_calculate_array", calculate_array_reaching_medicaid)
    frame = _medicaid_frame()
    reform_spec = builder.US_JCT_TAX_EXPENDITURE_REFORMS[0]
    with pytest.raises(
        ValueError,
        match=(
            rf"{reform_spec.measure}.*household batch 1/1 computed "
            rf"medicaid_slcsp_state_denominator@{builder.PERIOD}\."
        ),
    ):
        builder._reform_household_income_tax(
            base_frame=frame,
            reform_spec=reform_spec,
            system=CountryTaxBenefitSystem(),
            microsimulation_cls=Microsimulation,
            n_households=frame.n("household"),
            batch_size=None,
        )
