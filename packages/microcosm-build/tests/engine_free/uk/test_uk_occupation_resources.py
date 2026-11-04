"""Licensed-only: the APS donor cleaning covers the pinned SN 9408 tab.

Skipped where the licensed tab is absent (CI is secrets-free by design). Set
POPULACE_UK_APS_TAB to the pinned apsp_a24m25_eul_pwta22.tab to run it. It
asserts structure only, never a count.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from microcosm.build import load_country_spec
from microcosm.build.uk_runtime.aps_occupation import (
    UK_APS_OCCUPATION_CLEANING,
    UK_APS_PERSON_TAB_ROLE,
    _artifact,
    clean_aps_occupation_table,
)
from microcosm.build.uk_runtime.frs_spine import REGION_MAP, read_pinned_tab
from microcosm.build.uk_runtime.occupation_constants import (
    UK_SOC2020_SUB_MAJOR_GROUPS,
)


def test_the_aps_cleaning_maps_every_code_of_the_licensed_tab() -> None:
    tab = os.environ.get("POPULACE_UK_APS_TAB")
    if not tab or not Path(tab).is_file():
        pytest.skip("licensed APS tab not available (set POPULACE_UK_APS_TAB)")
    stage = next(
        s for s in load_country_spec("uk").sources.stages if s.stage == "aps_occupation"
    )
    raw = read_pinned_tab(
        Path(tab),
        _artifact(stage, UK_APS_PERSON_TAB_ROLE),
        columns=list(UK_APS_OCCUPATION_CLEANING["columns"].values()),
    )
    donors, receipt = clean_aps_occupation_table(raw)
    assert set(donors["soc2020_sub_major_group"]) == set(
        UK_SOC2020_SUB_MAJOR_GROUPS.values()
    )
    assert set(donors["region"]) == set(REGION_MAP.values())
    assert set(donors["gender"]) == {"MALE", "FEMALE"}
    assert receipt["dropped"]["no_region"] == 0
    assert receipt["dropped"]["no_sex"] == 0
    assert (donors["weight"] > 0).all()
