"""Tests split from packages/microcosm-build/tests/test_us_disability_benefits.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_disability_benefits import *


def test_policyengine_us_2_2_1_contract_and_positive_annual_behavior() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "2.2.1"
    variable = CountryTaxBenefitSystem().variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0

    situation = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                # SNAP 1.769.0+ applies this person's countable-income share
                # to unearned income; make the graph fixture work-eligible.
                "weekly_hours_worked_before_lsr": {"2024": 40},
                _OUTPUT: {"2024": 6_000.0},
            }
        },
        "tax_units": {
            "tax_unit": {
                "members": ["adult"],
                "filing_status": {"2024": "SINGLE"},
            }
        },
        "spm_units": {"spm_unit": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
    }
    simulation = Simulation(situation=situation)

    assert simulation.calculate(_OUTPUT, 2024)[0] == pytest.approx(6_000.0)
    assert simulation.calculate(_OUTPUT, "2024-01")[0] == pytest.approx(500.0)
    assert simulation.calculate("snap_unearned_income", "2024-01")[0] == pytest.approx(
        500.0
    )


def test_shipped_snap_exclusion_probe_binds_with_positive_sign() -> None:
    from datetime import date, timedelta

    from policyengine_us import CountryTaxBenefitSystem, Simulation

    from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform

    path = "gov.usda.snap.income.sources.unearned"
    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "disability_benefits_snap_exclusion"
    )
    assert probe.parameter_changes == {}
    assert set(probe.list_edits) == {path}
    edit = probe.list_edits[path]
    assert (edit.remove, edit.add) == ((_OUTPUT,), ())
    # The smoke's own reform, its list edit resolved on the installed engine,
    # applied the way the release scorer builds a reform system.
    reform = _build_reform(probe)
    reformed_system = CountryTaxBenefitSystem(reform=(reform,))
    situation = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                "employment_income": {"2024": 12_000.0},
                "weekly_hours_worked_before_lsr": {"2024": 40},
                _OUTPUT: {"2024": 6_000.0},
            }
        },
        "tax_units": {
            "tax_unit": {
                "members": ["adult"],
                "filing_status": {"2024": "SINGLE"},
            }
        },
        "spm_units": {"spm_unit": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
    }
    baseline = Simulation(situation=situation)
    reformed = Simulation(tax_benefit_system=reformed_system, situation=situation)

    # Not a no-op: over the edit period the reformed engine counts every
    # baseline source but disability benefits, and outside it the baseline.
    sources = baseline.tax_benefit_system.parameters.get_child(path)
    reformed_sources = reformed_system.parameters.get_child(path)
    assert _OUTPUT in sources(edit.start)
    expected = [source for source in sources(edit.start) if source != _OUTPUT]
    assert reform.resolved_list_edits == {path: expected}
    for instant in (edit.start, edit.stop):
        assert list(reformed_sources(instant)) == expected
    for instant in (
        date.fromisoformat(edit.start) - timedelta(days=1),
        date.fromisoformat(edit.stop) + timedelta(days=1),
    ):
        assert list(reformed_sources(str(instant))) == list(sources(str(instant)))

    effect = reformed.calculate("snap", 2024)[0] - baseline.calculate("snap", 2024)[0]
    assert effect > 1_000.0
