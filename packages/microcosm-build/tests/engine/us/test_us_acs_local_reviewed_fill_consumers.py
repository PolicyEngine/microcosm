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
the engine's.
"""

from __future__ import annotations

from enum import Enum
from importlib.metadata import version

import numpy as np
import pytest

from microcosm.build.us_runtime.acs_local_reviewed_fill_consumers import (
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


@pytest.fixture(scope="module")
def index() -> PolicyEngineUSVariableMetadataIndex:
    return PolicyEngineUSVariableMetadataIndex()


@pytest.fixture(scope="module")
def system():
    from policyengine_us import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


def test_the_register_was_reviewed_against_the_installed_engine() -> None:
    installed = version("policyengine-us")
    assert _DOCUMENT["reviewed_against"]["policyengine_us"] == installed, (
        "acs_local_reviewed_fill_consumers.yaml was reviewed against "
        f"policyengine-us {_DOCUMENT['reviewed_against']['policyengine_us']} "
        f"but {installed} is installed: re-run the consumer walk, review every "
        "note whose program's rules changed, and update reviewed_against."
    )


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
