"""``UK_WEIGHT_SHARE_FORMULA_INPUTS`` mirrors policyengine-uk's allocation keys.

microcosm#1115 review round 2: the constant names, by hand, the frame columns
each weight-share formula's allocation key reads. This test reads the same
dependencies from the installed engine, so an engine bump that changes a key's
``adds`` list, or drops a formula, fails here instead of letting the
per-block exactness check go stale.
"""

from __future__ import annotations

import pytest

from microcosm.build.uk_runtime.full_measure import UK_WEIGHT_SHARE_FORMULA_INPUTS

#: The allocation-key variable each weight-share formula divides by, as the
#: engine's formula reads it (``x * w / sum(x * w)``).
ALLOCATION_KEYS = {
    "corporate_land_value": "corporate_sector_wealth",
    "shareholding": "corporate_sector_wealth",
    "consumption_shareholding": "consumption",
}


@pytest.fixture(scope="module")
def system():
    from policyengine_uk import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


def test_constant_names_every_weight_share_formula_and_its_engine_inputs(system):
    assert set(UK_WEIGHT_SHARE_FORMULA_INPUTS) == set(ALLOCATION_KEYS)
    for formula, key in ALLOCATION_KEYS.items():
        variable = system.variables[formula]
        assert variable.entity.key == "household", formula
        assert not variable.is_input_variable(), f"{formula} is no longer a formula"
        key_variable = system.variables[key]
        adds = tuple(key_variable.adds)
        assert adds, f"{key} no longer declares its inputs through `adds`"
        assert adds == UK_WEIGHT_SHARE_FORMULA_INPUTS[formula], (
            f"{formula}: the engine's {key} adds {adds}; the constant names "
            f"{UK_WEIGHT_SHARE_FORMULA_INPUTS[formula]}"
        )
        for column in adds:
            leaf = system.variables[column]
            assert leaf.entity.key == "household", column
            assert leaf.is_input_variable(), f"{column} is not an input column"


def test_named_columns_are_release_surface_household_columns():
    """The columns the check reads leave with the release: every one is an
    enhanced-FRS household input (the parity reference) or a reviewed extra
    household column of the export surface (the allow-list)."""
    from microcosm.build.uk_runtime import load_efrs_parity_reference
    from microcosm.build.uk_runtime.terminal_gates import (
        UK_ALLOWED_EXTRA_EXPORT_COLUMNS,
    )

    reference = load_efrs_parity_reference().input_entities
    household = {
        column for column, entity in reference.items() if entity == "household"
    }
    household |= {
        name.split(".", 1)[1]
        for name in UK_ALLOWED_EXTRA_EXPORT_COLUMNS
        if name.startswith("household.")
    }
    for formula, columns in UK_WEIGHT_SHARE_FORMULA_INPUTS.items():
        missing = [c for c in columns if c not in household]
        assert not missing, f"{formula}: {missing} are not release household columns"
