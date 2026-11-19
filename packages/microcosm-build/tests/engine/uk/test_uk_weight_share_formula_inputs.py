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
    """The columns the check reads leave with the release: every one is a
    household column the spine declares and the release boundary keeps.

    Until microcosm#1095 this read the enhanced-FRS reference and the export
    allow-list; the export surface now compares with the current build, so
    the spine's declared surface through the production export path stands
    in for both.
    """
    import numpy as np
    import pandas as pd

    from microcosm.build.uk_runtime.graph import uk_spine_endpoint, uk_spine_graph
    from microcosm.build.uk_runtime.national_frame import (
        uk_national_frame,
        uk_release_export_frame,
    )

    columns = {
        "person": ["person_id", "person_benunit_id", "person_household_id"],
        "benunit": ["benunit_id"],
        "household": ["household_id"],
    }
    for cells in uk_spine_endpoint(uk_spine_graph()).inputs:
        columns[cells.entity] = list(
            dict.fromkeys([*columns[cells.entity], *cells.columns])
        )
    tables = {
        entity: pd.DataFrame({column: np.array([1]) for column in names})
        for entity, names in columns.items()
    }
    frame = uk_national_frame(**tables, time_period="2024", household_weights=[1.0])
    household = set(uk_release_export_frame(frame).table("household").columns)
    for formula, columns_read in UK_WEIGHT_SHARE_FORMULA_INPUTS.items():
        missing = [c for c in columns_read if c not in household]
        assert not missing, f"{formula}: {missing} are not release household columns"
