"""The four H.R.1 immigration probes bind through the engine (microcosm #767).

The release-blocking reform-coverage smoke gate scores the pinned probe
reforms on the full export; these tests prove the same channels move,
nonzero and sign-correct, on synthetic households whose
``immigration_status_str`` carries the humanitarian values the source stage
now imputes. A probe whose channel dies here would score $0 on every
release, so this is the PR-CI early warning for the engine side of the
silent-zero failure — without needing restricted microdata.

Each situation pins the take-up flag its program reads (the same
data-seeded surface the release carries) and, for the ACA channels, an
``slcsp`` override so no rating-area geography is required.
"""

from __future__ import annotations

import pytest

pytest.importorskip("policyengine_us")

from policyengine_us import Simulation  # noqa: E402

from microcosm.build.us_runtime.release_input_coverage import (  # noqa: E402
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime.reform_coverage_smoke import (  # noqa: E402
    _build_reform,
)

_HR1_PROBE_IDS = (
    "hr1_medicaid_humanitarian_eligibility_restoration",
    "hr1_aca_below_fpl_exception_restoration",
    "hr1_aca_lawful_presence_restoration",
    "hr1_snap_humanitarian_eligibility_restoration",
)


@pytest.fixture(scope="module")
def hr1_probes():
    probes = {
        probe.id: probe
        for probe in us_release_reform_coverage_probes()
        if probe.id.startswith("hr1_")
    }
    assert set(probes) == set(_HR1_PROBE_IDS)
    return probes


def _household(
    status: str,
    income: float,
    year: int,
    *,
    person: dict | None = None,
    tax_unit: dict | None = None,
    spm_unit: dict | None = None,
) -> dict:
    people = {
        "parent": {
            "age": {year: 35},
            "employment_income": {year: income},
            "immigration_status_str": {year: status},
            **(person or {}),
        },
        "child": {
            "age": {year: 8},
            "immigration_status_str": {year: status},
            **(person or {}),
        },
    }
    return {
        "people": people,
        "families": {"f": {"members": ["parent", "child"]}},
        "marital_units": {"m": {"members": ["parent"]}},
        "tax_units": {"t": {"members": ["parent", "child"], **(tax_unit or {})}},
        "spm_units": {"s": {"members": ["parent", "child"], **(spm_unit or {})}},
        "households": {
            "h": {"members": ["parent", "child"], "state_name": {year: "TX"}}
        },
    }


def _effect(probe, situation: dict, year: int) -> tuple[float, float]:
    baseline = Simulation(situation=situation)
    reformed = Simulation(situation=situation, reform=_build_reform(probe))
    base = float(baseline.calculate(probe.budget_measure, year).sum())
    reform = float(reformed.calculate(probe.budget_measure, year).sum())
    return base, reform


def test_medicaid_restoration_reenrolls_refugee_family(hr1_probes) -> None:
    # At 2027 law refugees are outside the §71109-narrowed qualified list;
    # restoring the pre-H.R.1 list must re-enroll the anchored taker.
    probe = hr1_probes["hr1_medicaid_humanitarian_eligibility_restoration"]
    situation = _household(
        "REFUGEE",
        25_000.0,
        2027,
        person={"takes_up_medicaid_if_eligible": {2027: True}},
    )
    base, reform = _effect(probe, situation, 2027)
    assert base == 0.0
    assert reform > 0.0


def test_below_fpl_exception_restoration_pays_ptc_to_tps_unit(hr1_probes) -> None:
    # A below-poverty TPS unit is lawfully present for the ACA at 2026 law
    # yet Medicaid-ineligible by status — exactly the §71302 population.
    probe = hr1_probes["hr1_aca_below_fpl_exception_restoration"]
    situation = _household(
        "TPS",
        12_000.0,
        2026,
        tax_unit={
            "takes_up_aca_if_eligible": {2026: True},
            "slcsp": {2026: 8_000.0},
        },
    )
    base, reform = _effect(probe, situation, 2026)
    assert base == 0.0
    assert reform > 0.0


def test_lawful_presence_restoration_requalifies_asylee_unit(hr1_probes) -> None:
    # At 2027 law §71301 adds asylees to the ACA ineligible list; restoring
    # the pre-H.R.1 list (DACA/UNDOCUMENTED only) must re-qualify the unit.
    probe = hr1_probes["hr1_aca_lawful_presence_restoration"]
    situation = _household(
        "ASYLEE",
        35_000.0,
        2027,
        tax_unit={
            "takes_up_aca_if_eligible": {2027: True},
            "slcsp": {2027: 8_000.0},
        },
    )
    base, reform = _effect(probe, situation, 2027)
    assert base == 0.0
    assert reform > 0.0


def test_snap_restoration_reincludes_refugee_members(hr1_probes) -> None:
    # At 2026 law §10108 makes refugees excluded SNAP members; restoring the
    # pre-H.R.1 list must raise the household allotment.
    probe = hr1_probes["hr1_snap_humanitarian_eligibility_restoration"]
    situation = _household(
        "REFUGEE",
        20_000.0,
        2026,
        spm_unit={"takes_up_snap_if_eligible": {2026: True}},
    )
    base, reform = _effect(probe, situation, 2026)
    assert reform > base


def test_probe_magnitude_floors_sit_far_below_plausible_scale(hr1_probes) -> None:
    # The floors are release-scale guards: far above numerical noise, far
    # below the plausible aggregate effect of the imputed stocks (about
    # 1.6M weighted persons across the humanitarian categories).
    floors = {
        "hr1_medicaid_humanitarian_eligibility_restoration": 50_000_000.0,
        "hr1_aca_below_fpl_exception_restoration": 5_000_000.0,
        "hr1_aca_lawful_presence_restoration": 10_000_000.0,
        "hr1_snap_humanitarian_eligibility_restoration": 25_000_000.0,
    }
    for probe_id, floor in floors.items():
        probe = hr1_probes[probe_id]
        assert probe.min_abs_effect == floor
        assert probe.expected_sign == "positive"
        assert probe.binding_inputs == ("immigration_status_str",)
        assert probe.issue == "PolicyEngine/microcosm#767"
