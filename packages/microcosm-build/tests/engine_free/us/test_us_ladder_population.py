"""The US ladder population rows on the shared code give what the tool code gave.

microcosm#1728 moved the state and district population rows and specs of the
ACS local release to the shared ``microcosm.build.ladder_population_targets``,
with the US levels in ``us/local_target_construction.json``. These tests hold
the tool functions, which now call the shared code, to verbatim copies of the
code they replace (``test_support/microcosm_build/us_ladder_population_reference.py``)
on random households and ladders: the same records, matrix rows, dropped
cells and target specs. ``test_us_acs_local_surface_bytes.py`` checks the same
on the pinned feed and ladder.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.congressional_district_vintage import (
    CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
)
from microcosm.build.us_runtime.local_target_construction import (
    us_ladder_population,
)
from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_NAME
from test_support.microcosm_build import us_ladder_population_reference as reference
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
SEEDS = range(10)
GEOGRAPHY_LISTS = (["state"], ["cd"], ["state", "cd"], ["cd", "state"])


@pytest.fixture(scope="module")
def tool():
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location("build_us_acs_local_release", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ladder(seed: int):
    """Households, sizes and ladder populations, with some cells unsupported."""

    rng = np.random.default_rng(seed)
    states = sorted(int(code) for code in US_STATE_FIPS_TO_NAME)
    chosen = sorted(rng.choice(states, size=6, replace=False).tolist())
    districts = sorted(
        {state * 100 + int(rng.integers(0, 8)) for state in chosen for _ in range(3)}
    )
    populations = {
        "state": {state: float(rng.uniform(1e5, 4e7)) for state in chosen},
        "cd": {district: float(rng.uniform(5e5, 9e5)) for district in districts},
    }
    supported = [code for code in districts if rng.uniform() < 0.8]
    rows = []
    for _ in range(int(rng.integers(20, 80))):
        district = supported[int(rng.integers(len(supported)))]
        rows.append(
            {"state_fips": district // 100, "congressional_district_geoid": district}
        )
    households = pd.DataFrame(rows)
    sizes = rng.integers(0, 6, size=len(households)).astype(np.float32)
    return households, sizes, populations


def _matrix_bytes(matrix) -> tuple:
    return (
        matrix.shape,
        matrix.indptr.tolist(),
        matrix.indices.tolist(),
        matrix.data.astype(np.float32).tolist(),
    )


def test_the_declaration_holds_the_tool_values(tool) -> None:
    declaration = us_ladder_population()
    district = declaration.level("cd")
    assert district.geography_id_prefix == CURRENT_CONGRESSIONAL_DISTRICT_PREFIX
    assert {level.geography: level.code_digits for level in declaration.levels} == {
        geography: digits
        for geography, digits in reference.POPULATION_TARGET_CODE_DIGITS.items()
        if geography != "county"
    }
    assert tool.cd_surface.POPULATION_TARGET_CODE_DIGITS == (
        reference.POPULATION_TARGET_CODE_DIGITS
    )
    for geography, code in (("state", 6), ("cd", 601), ("county", 6037)):
        assert tool.cd_surface.population_target_name(geography, code) == (
            reference.population_target_name(geography, code)
        )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("geographies", GEOGRAPHY_LISTS)
def test_population_rows_match(tool, seed: int, geographies: list[str]) -> None:
    households, sizes, populations = _ladder(seed)
    old_records, old_matrix, old_dropped = reference.population_rows(
        households, sizes, populations, geographies
    )
    new_records, new_matrix, new_dropped = tool.cd_surface.population_rows(
        households, sizes, populations, geographies
    )
    assert json.dumps(new_records) == json.dumps(old_records)
    assert _matrix_bytes(new_matrix) == _matrix_bytes(old_matrix)
    assert new_dropped == old_dropped


@pytest.mark.parametrize("seed", SEEDS)
def test_population_specs_match(tool, seed: int) -> None:
    _households, _sizes, populations = _ladder(seed)
    for geographies in GEOGRAPHY_LISTS:
        old = reference.ladder_population_target_specs(populations, geographies)
        new = tool.ladder_population_target_specs(populations, geographies)
        assert new == old
        assert json.dumps([spec.to_dict() for spec in new]) == json.dumps(
            [spec.to_dict() for spec in old]
        )


def test_county_specs_keep_their_place(tool) -> None:
    names = ["pop_state_06", "pop_county_06037", "pop_cd_0601", "pop_county_36061"]
    values = [3.9e7, 9.7e6, 7.6e5, 1.6e6]
    specs = tool.population_target_specs(names, values)
    assert [spec.name for spec in specs] == names
    assert specs[0] == reference.population_target_specs(names[:1], values[:1])[0]
    assert specs[2] == reference.population_target_specs(names[2:3], values[2:3])[0]
    assert specs[1] == tool.county_surface.population_spec(
        names[1], values[1], period=tool.PERIOD
    )
    with pytest.raises(ValueError):
        tool.population_target_specs(names, values[:-1])


def test_a_state_code_with_no_name_refuses(tool) -> None:
    with pytest.raises(ValueError) as old:
        reference.population_target_specs(["pop_state_99"], [1.0])
    with pytest.raises(ValueError) as new:
        tool.population_target_specs(["pop_state_99"], [1.0])
    assert str(new.value) == str(old.value)


class _Frame:
    def __init__(self, households: pd.DataFrame, persons: pd.DataFrame) -> None:
        self._tables = {"household": households, "person": persons}

    def table(self, name: str) -> pd.DataFrame:
        return self._tables[name]


def test_the_tool_picks_the_county_rows_only_for_a_county_run(tool) -> None:
    households, _sizes, populations = _ladder(4)
    households = households.assign(household_id=np.arange(len(households)))
    persons = pd.DataFrame(
        {"person_household_id": np.repeat(households["household_id"], 2)}
    )
    frame = _Frame(households, persons)
    sizes = tool.household_sizes(frame)

    records, matrix, dropped = tool.population_targets(
        frame, populations, ["state", "cd"]
    )
    old_records, old_matrix, old_dropped = reference.population_rows(
        households, sizes, populations, ["state", "cd"]
    )
    assert json.dumps(records) == json.dumps(old_records)
    assert _matrix_bytes(matrix) == _matrix_bytes(old_matrix)
    assert dropped == old_dropped

    with pytest.raises(ValueError, match="county rung"):
        tool.cd_surface.population_rows(
            households, sizes, populations, ["state", "county"]
        )
