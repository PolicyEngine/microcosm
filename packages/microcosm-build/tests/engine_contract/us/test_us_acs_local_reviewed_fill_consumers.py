"""The reviewed-fill consumer register against the installed policyengine-us
(microcosm#1022, part 6).

Each register entry declares the means-tested programs its column reaches.
This test recomputes them with the engine's static dependency graph
(``PolicyEngineUSVariableMetadataIndex.consumer_receipts``, walked by
:func:`means_tested_consumers`) and fails when an entry misses a program the
walk reaches, or lists one it no longer reaches, so a policyengine-us bump
cannot silently add a consumer the notes never reviewed. It also checks that
the program roots are engine variables covering the engine's own CBO
means-tested transfer list, and that each entry's entity and fill value are
the engine's. For the program couplings register validation enforces
(``REVIEWED_FILL_COUPLINGS``, microcosm#1071 review) it checks that each is an
edge of the engine's graph and that every column reaching one declares both
programs, so the engine-free rule covers it.

The walk tests hold only for the engine the register was reviewed against.
On any other engine the ACS local release gate refuses the release (fail
closed), so they skip until the register is re-reviewed, and one test checks
that the gate refuses the installed engine meanwhile (policyengine-us 2.145.3:
microcosm#1675).
"""

from __future__ import annotations

from enum import Enum
from importlib.metadata import version

import numpy as np
import pytest

from microcosm.build.us_runtime.acs_local_reviewed_fill_consumers import (
    REVIEWED_FILL_COUPLINGS,
    acs_local_reviewed_fill_consumer_gate,
    load_reviewed_fill_consumer_register,
    means_tested_consumers,
)
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSVariableMetadataIndex,
)

_DOCUMENT = load_reviewed_fill_consumer_register()
_ENTRIES = {entry["column"]: entry for entry in _DOCUMENT["entries"]}
_PROGRAMS = {
    program: tuple(spec["roots"]) for program, spec in _DOCUMENT["programs"].items()
}
#: The engine variable through which each coupling's upstream program reaches
#: the downstream one (microcosm#1071 review).
_COUPLING_NODES = {
    "medicaid": "is_medicaid_eligible",
    "chip": "is_chip_eligible",
    "ssi": "ssi",
    "tanf": "tanf",
}
_REACHED: dict[str, frozenset[str]] = {}
_INSTALLED_ENGINE = version("policyengine-us")
_REVIEWED_ENGINE = _DOCUMENT["reviewed_against"]["policyengine_us"]
_ON_THE_REVIEWED_ENGINE = pytest.mark.skipif(
    _INSTALLED_ENGINE != _REVIEWED_ENGINE,
    reason=(
        f"the register was reviewed against policyengine-us {_REVIEWED_ENGINE} "
        f"and {_INSTALLED_ENGINE} is installed; the ACS local release gate "
        "refuses this engine until the re-review (microcosm#1675)"
    ),
)


@pytest.fixture(scope="module")
def index() -> PolicyEngineUSVariableMetadataIndex:
    return PolicyEngineUSVariableMetadataIndex()


@pytest.fixture(scope="module")
def system():
    from policyengine_us import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


def test_the_release_gate_holds_the_register_to_its_reviewed_engine() -> None:
    """A release on an engine the register was not reviewed against fails."""

    gate = acs_local_reviewed_fill_consumer_gate(
        {}, installed_engine_version=_INSTALLED_ENGINE
    )
    matches = _INSTALLED_ENGINE == _REVIEWED_ENGINE
    assert gate.details["engine_matches_review"] is matches
    refusal = (
        f"the release runs policyengine-us {_INSTALLED_ENGINE} but the register "
        f"was reviewed against {_REVIEWED_ENGINE}"
    )
    assert any(refusal in failure for failure in gate.failures) is not matches


def test_the_program_roots_cover_the_engines_means_tested_list(system) -> None:
    roots = {root for program_roots in _PROGRAMS.values() for root in program_roots}
    unknown = sorted(root for root in roots if root not in system.variables)
    assert unknown == [], f"program roots are not engine variables: {unknown}"
    cbo = system.parameters.gov.household.cbo_means_tested_transfers("2024-01-01")
    missing = sorted(set(cbo) - roots)
    assert missing == [], (
        "gov.household.cbo_means_tested_transfers lists means-tested transfers "
        f"the register has no program for: {missing}"
    )
    assert "household_state_benefits" in roots


@_ON_THE_REVIEWED_ENGINE
@pytest.mark.parametrize("column", sorted(_ENTRIES))
def test_each_entry_declares_exactly_the_walked_consumers(index, column) -> None:
    reached = means_tested_consumers(column, index, _PROGRAMS)
    declared = _ENTRIES[column]["consumers"]
    missing = {
        program: " > ".join(path)
        for program, path in reached.items()
        if program not in declared
    }
    stale = sorted(set(declared) - set(reached))
    assert not missing and not stale, (
        f"{column}: the register misses consumer(s) {missing} and lists "
        f"{stale} the walk no longer reaches; add each with a harmless or "
        "known-bias note, or drop it."
    )


@_ON_THE_REVIEWED_ENGINE
@pytest.mark.parametrize("column", sorted(_ENTRIES))
def test_each_entry_matches_its_engine_variable(system, column) -> None:
    entry = _ENTRIES[column]
    variable = system.variables[column]
    assert variable.entity.key == entry["entity"]
    default = variable.default_value
    if isinstance(default, Enum):
        default = default.name
    elif isinstance(default, (bool, np.bool_)):
        default = bool(default)
    # fill_reviewed_nulls records repr() of this value in the fill manifests.
    assert repr(default) == entry["fill_value"]


def _reached(index, column: str) -> frozenset[str]:
    """Every variable the static walk reaches from ``column``."""

    if column not in _REACHED:
        seen = {column}
        queue = [column]
        while queue:
            node = queue.pop()
            try:
                receipts = index.consumer_receipts(node)
            except ValueError:
                continue
            for receipt in receipts:
                if receipt.consumer not in seen:
                    seen.add(receipt.consumer)
                    queue.append(receipt.consumer)
        _REACHED[column] = frozenset(seen)
    return _REACHED[column]


@_ON_THE_REVIEWED_ENGINE
def test_each_program_coupling_is_an_engine_edge(index) -> None:
    """The couplings register validation enforces exist in the engine."""

    assert {upstream for upstream, _, _ in REVIEWED_FILL_COUPLINGS} == set(
        _COUPLING_NODES
    )
    for upstream, downstream, _bars in REVIEWED_FILL_COUPLINGS:
        reached = means_tested_consumers(_COUPLING_NODES[upstream], index, _PROGRAMS)
        assert upstream in reached, (upstream, sorted(reached))
        assert downstream in reached, (upstream, downstream, sorted(reached))

    def consumers(name):
        return {receipt.consumer for receipt in index.consumer_receipts(name)}

    # The bars: Medicaid and CHIP eligibility gate the PTC.
    assert "pays_aca_premium" in consumers("is_medicaid_eligible")
    assert "pays_aca_premium" in consumers("is_chip_eligible")
    assert "is_aca_ptc_eligible" in consumers("pays_aca_premium")
    # SSI and TANF reach SNAP categorical eligibility.
    assert "meets_snap_categorical_eligibility" in consumers("ssi")
    assert "meets_snap_categorical_eligibility" in consumers("tanf")


@_ON_THE_REVIEWED_ENGINE
@pytest.mark.parametrize("column", sorted(_ENTRIES))
def test_each_column_reaching_a_coupling_declares_both_programs(index, column) -> None:
    """So the engine-free coupling rule sees every coupling a column reaches."""

    reached = _reached(index, column)
    declared = set(_ENTRIES[column]["consumers"])
    missing = [
        f"{upstream} -> {downstream}"
        for upstream, downstream, _bars in REVIEWED_FILL_COUPLINGS
        if _COUPLING_NODES[upstream] in reached
        and not {upstream, downstream} <= declared
    ]
    assert missing == [], f"{column} reaches {missing} without declaring both."


@_ON_THE_REVIEWED_ENGINE
def test_the_key_snap_paths(index) -> None:
    """The paths the register's SNAP notes rest on (microcosm#1022 triage)."""

    def reach(column):
        return means_tested_consumers(column, index, _PROGRAMS)

    vehicles = reach("household_vehicles_value")
    assert "meets_tanf_non_cash_asset_test" in vehicles["snap"]
    # SNAP's utility allowance does not read the energy subsidy; the subsidy is
    # the LIHEAP program's own value.
    assert reach("spm_unit_energy_subsidy") == {"liheap": ("spm_unit_energy_subsidy",)}
    # No program reads these at all.
    for column in ("weeks_unemployed", "net_worth", "tenure_type", "ssi_reported"):
        assert reach(column) == {}, column
    # SNAP's medical deduction reads the medical-expense aggregate, which
    # treats a zero direct premium as not supplied.
    premiums = reach("health_insurance_premiums")["snap"]
    assert premiums[:2] == (
        "health_insurance_premiums",
        "medical_expense_health_insurance_premiums",
    )

    # SNAP's shelter deduction reads rent through housing_cost (the shortest
    # path from the rent runs through Arizona TANF, so check the edges).
    def consumers(name):
        return {receipt.consumer for receipt in index.consumer_receipts(name)}

    assert "rent" in consumers("pre_subsidy_rent")
    assert "housing_cost" in consumers("rent")
    assert "snap_excess_shelter_expense_deduction" in consumers("housing_cost")
    assert "snap" in reach("pre_subsidy_rent")
