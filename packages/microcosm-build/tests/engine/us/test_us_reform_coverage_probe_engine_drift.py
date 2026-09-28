"""Reform-coverage probes stay in step with the installed PolicyEngine-US.

The 2026-09-27 Route A export failed its post-export smoke on four probes whose
definitions predated PolicyEngine-US 2.2.1, not on missing inputs:

- Three SNAP source-exclusion probes pinned a person-level unearned-source list
  that still named ``tanf``. 2.2.1 counts TANF once on ``unearned_spm_unit``, and
  a group variable named in a person-level ``adds`` list is projected onto every
  member, so the reform re-added TANF once per SPM-unit member and cut SNAP.
- The alimony-expense probe flipped ``divorce_year_threshold``, which since
  PolicyEngine-US a8e8a2e2ab also gates recipients' ``taxable_alimony_income``,
  so the "abolition" untaxed receipts and lowered income tax.

These tests pin the invariants that would have caught both before a release:
each list-valued probe differs from the engine's own baseline list by exactly
the one item it means to change, and each source/deduction probe moves only
its own leaf on a household that also carries the neighbouring channels.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime.release_input_coverage import (
    us_release_reform_coverage_probes,
)

# A probe whose one-item change is an addition (re-enabling an item the
# baseline list no longer carries) instead of a removal.
_ADDITION_PROBES = {"domestic_production_ald_reactivation": "domestic_production_ald"}

_SNAP_SOURCE_PROBES = {
    "child_support_received_snap_exclusion": "child_support_received",
    "disability_benefits_snap_exclusion": "disability_benefits",
    "workers_compensation_snap_exclusion": "workers_compensation",
}


def _probes():
    return {probe.id: probe for probe in us_release_reform_coverage_probes()}


def _list_valued_changes():
    return [
        (probe.id, path, period, value)
        for probe in us_release_reform_coverage_probes()
        for path, periods in probe.parameter_changes.items()
        for period, value in periods.items()
        if isinstance(value, list)
    ]


@pytest.fixture(scope="module")
def system():
    from policyengine_us import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


@pytest.fixture(scope="module")
def reformed_systems():
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem

    probes = _probes()
    return {
        probe_id: CountryTaxBenefitSystem(
            reform=(
                Reform.from_dict(
                    dict(probes[probe_id].parameter_changes), country_id="us"
                ),
            )
        )
        for probe_id in (*_SNAP_SOURCE_PROBES, "alimony_expense_ald_abolition")
    }


def test_list_valued_probes_are_enumerated() -> None:
    # Guards the parametrization below against silently collecting nothing.
    ids = {probe_id for probe_id, *_ in _list_valued_changes()}
    assert set(_SNAP_SOURCE_PROBES) <= ids
    assert {"alimony_expense_ald_abolition", *_ADDITION_PROBES} <= ids


@pytest.mark.parametrize(
    ("probe_id", "path", "period", "value"),
    _list_valued_changes(),
    ids=[f"{probe_id}-{path}" for probe_id, path, *_ in _list_valued_changes()],
)
def test_list_probe_differs_from_engine_baseline_by_exactly_one_item(
    system, probe_id: str, path: str, period: str, value: list
) -> None:
    baseline = list(system.parameters.get_child(path)(period.split(".")[0]))

    assert len(value) == len(set(value)), f"{probe_id}: duplicate items in {path}"
    removed = set(baseline) - set(value)
    added = set(value) - set(baseline)
    if probe_id in _ADDITION_PROBES:
        assert (removed, added) == (set(), {_ADDITION_PROBES[probe_id]}), (
            f"{probe_id}: {path} must equal the engine baseline plus "
            f"{_ADDITION_PROBES[probe_id]}; removed {sorted(removed)}, "
            f"added {sorted(added)}"
        )
    else:
        assert len(removed) == 1 and not added, (
            f"{probe_id}: {path} must equal the installed engine's baseline "
            f"minus one item; removed {sorted(removed)}, added {sorted(added)}. "
            "Re-derive the list from the engine's current value."
        )


@pytest.mark.parametrize("probe_id", sorted(_SNAP_SOURCE_PROBES))
def test_snap_source_probe_does_not_readd_spm_unit_sources(
    system, probe_id: str
) -> None:
    probe = _probes()[probe_id]
    ((period, sources),) = probe.parameter_changes[
        "gov.usda.snap.income.sources.unearned"
    ].items()
    spm_unit_sources = system.parameters.gov.usda.snap.income.sources.unearned_spm_unit(
        period.split(".")[0]
    )

    assert set(spm_unit_sources), "the engine counts no SPM-unit unearned sources"
    assert not set(sources) & set(spm_unit_sources)


def _snap_household(leaf: str, amount: float, tanf: float) -> dict:
    members = ["adult", "child1", "child2"]
    return {
        "people": {
            "adult": {
                "age": {"2024": 35},
                "employment_income": {"2024": 8_000.0},
                leaf: {"2024": amount},
            },
            "child1": {"age": {"2024": 8}},
            "child2": {"age": {"2024": 5}},
        },
        "tax_units": {
            "tax_unit": {
                "members": members,
                "filing_status": {"2024": "HEAD_OF_HOUSEHOLD"},
            }
        },
        "spm_units": {"spm_unit": {"members": members, "tanf": {"2024": tanf}}},
        "households": {"household": {"members": members, "state_code": {"2024": "TX"}}},
    }


@pytest.mark.parametrize("probe_id", sorted(_SNAP_SOURCE_PROBES))
def test_snap_source_probe_raises_snap_when_the_unit_also_gets_tanf(
    system, reformed_systems, probe_id: str
) -> None:
    from policyengine_us import Simulation

    situation = _snap_household(_SNAP_SOURCE_PROBES[probe_id], 3_000.0, 4_800.0)
    baseline = Simulation(tax_benefit_system=system, situation=situation)
    reformed = Simulation(
        tax_benefit_system=reformed_systems[probe_id], situation=situation
    )

    assert baseline.calculate("snap_unearned_income", 2024)[0] == pytest.approx(7_800.0)
    assert reformed.calculate("snap_unearned_income", 2024)[0] == pytest.approx(4_800.0)
    # SNAP phases out at 30 cents per dollar of net income: excluding $3,000
    # a year raises the benefit by $900 while the unit stays in the phase-out.
    effect = reformed.calculate("snap", 2024)[0] - baseline.calculate("snap", 2024)[0]
    assert effect == pytest.approx(900.0)


@pytest.mark.parametrize("probe_id", sorted(_SNAP_SOURCE_PROBES))
@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    leaf_amount=st.integers(min_value=0, max_value=24_000),
    tanf=st.integers(min_value=0, max_value=12_000),
)
def test_snap_source_probe_removes_exactly_its_leaf(
    system, reformed_systems, probe_id: str, leaf_amount: int, tanf: int
) -> None:
    # Invariant: for any leaf and TANF amount, the reform lowers SNAP countable
    # unearned income by exactly the leaf and never lowers SNAP.
    from policyengine_us import Simulation

    situation = _snap_household(
        _SNAP_SOURCE_PROBES[probe_id], float(leaf_amount), float(tanf)
    )
    baseline = Simulation(tax_benefit_system=system, situation=situation)
    reformed = Simulation(
        tax_benefit_system=reformed_systems[probe_id], situation=situation
    )

    drop = (
        baseline.calculate("snap_unearned_income", 2024)[0]
        - reformed.calculate("snap_unearned_income", 2024)[0]
    )
    assert drop == pytest.approx(float(leaf_amount), abs=0.01)
    assert (
        reformed.calculate("snap", 2024)[0]
        >= baseline.calculate("snap", 2024)[0] - 0.01
    )


def _tax_household(alimony_expense: float, alimony_income: float) -> dict:
    return {
        "people": {
            "adult": {
                "age": {"2024": 45},
                "employment_income": {"2024": 90_000.0},
                "alimony_expense": {"2024": alimony_expense},
                "alimony_income": {"2024": alimony_income},
            }
        },
        "tax_units": {
            "tax_unit": {"members": ["adult"], "filing_status": {"2024": "SINGLE"}}
        },
        "spm_units": {"spm_unit": {"members": ["adult"]}},
        "households": {
            "household": {"members": ["adult"], "state_code": {"2024": "TX"}}
        },
    }


def _income_tax_change(system, reformed_systems, situation: dict) -> float:
    from policyengine_us import Simulation

    baseline = Simulation(tax_benefit_system=system, situation=situation)
    reformed = Simulation(
        tax_benefit_system=reformed_systems["alimony_expense_ald_abolition"],
        situation=situation,
    )
    return float(
        baseline.calculate("income_tax", 2024)[0]
        - reformed.calculate("income_tax", 2024)[0]
    )


def test_alimony_probe_taxes_payers_and_leaves_recipients_alone(
    system, reformed_systems
) -> None:
    payer = _income_tax_change(system, reformed_systems, _tax_household(20_000.0, 0.0))
    recipient = _income_tax_change(
        system, reformed_systems, _tax_household(0.0, 20_000.0)
    )

    # baseline_minus_reform, as the probe scores it: negative for payers.
    assert payer < -1_000.0
    assert recipient == 0.0


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    alimony_expense=st.integers(min_value=0, max_value=40_000),
    alimony_income=st.integers(min_value=0, max_value=40_000),
)
def test_alimony_probe_raises_agi_by_exactly_the_payer_deduction(
    system, reformed_systems, alimony_expense: int, alimony_income: int
) -> None:
    # Invariant: the reform adds back exactly the baseline alimony ALD and
    # leaves the recipient side of AGI untouched.
    from policyengine_us import Simulation

    situation = _tax_household(float(alimony_expense), float(alimony_income))
    baseline = Simulation(tax_benefit_system=system, situation=situation)
    reformed = Simulation(
        tax_benefit_system=reformed_systems["alimony_expense_ald_abolition"],
        situation=situation,
    )

    agi_change = (
        reformed.calculate("adjusted_gross_income", 2024)[0]
        - baseline.calculate("adjusted_gross_income", 2024)[0]
    )
    assert agi_change == pytest.approx(
        baseline.calculate("alimony_expense_ald", 2024)[0], abs=0.01
    )
