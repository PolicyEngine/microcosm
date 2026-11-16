"""Synthetic block ladders for US block-location tests.

``write_location_ladder`` writes a schema-1 block-ladder NPZ (the loader's
required arrays and metadata) plus the additive per-block ``puma`` array, so
tests exercise the real loaders. ``synthetic_blocks`` builds a small nested
geography: states > counties > tracts > blocks, with PUMAs as unions of whole
tracts and congressional districts that cut across tracts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LADDER_LAYERS = {
    "congressional_district": {"vintage": "119th_congress", "source": "test CD119"},
    "sldu": {"vintage": "2020_baf", "source": "test SLDU"},
    "sldl": {"vintage": "2020_baf", "source": "test SLDL"},
    "place": {"vintage": "2020_census", "source": "test place"},
    "cbsa": {"vintage": "omb_2023_delineations", "source": "test CBSA"},
    "puma": {"vintage": "2020_puma", "source": "test tract-to-PUMA"},
}


def ladder_metadata(*, with_puma: bool = True) -> dict:
    layers = {
        name: dict(spec)
        for name, spec in LADDER_LAYERS.items()
        if with_puma or name != "puma"
    }
    return {
        "schema_version": 1,
        "kind": "us_block_ladder",
        "block_vintage": "2020_tabulation_blocks",
        "sampling_basis": "population",
        "layers": layers,
    }


def synthetic_blocks(seed: int = 0, *, states=(1, 2, 36)) -> dict[str, np.ndarray]:
    """A small nested ladder: 2-3 counties per state, 1-3 tracts per county,
    1-4 blocks per tract, 1-2 PUMAs per state (unions of whole tracts), 1-3
    districts per state assigned per block (districts cut across tracts)."""

    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int, int, int, str, str, int, int]] = []
    for state in states:
        n_puma = int(rng.integers(1, 3))
        n_cd = int(rng.integers(1, 4))
        for county_index in range(int(rng.integers(2, 4))):
            county = 1 + 2 * county_index
            cbsa = int(rng.choice([0, 10000 + state * 10 + county_index]))
            for tract_index in range(int(rng.integers(1, 4))):
                # Nonzero last digit, so a tract derived from a wrong prefix
                # length cannot coincide with the true one.
                tract = 101 * (tract_index + 1)
                puma = state * 10**5 + 100 + int(rng.integers(0, n_puma))
                for block_index in range(int(rng.integers(1, 5))):
                    block = int(
                        f"{state:02d}{county:03d}{tract:06d}{1000 + block_index:04d}"
                    )
                    population = int(rng.choice([1, 7, 40, 250, 3000]))
                    district = int(rng.integers(0, n_cd)) + (1 if n_cd > 1 else 0)
                    rows.append(
                        (
                            block,
                            population,
                            state * 100 + district,
                            puma,
                            str(rng.choice(["001", "002", "00A"])),
                            str(rng.choice(["", "010", "011"])),
                            int(rng.choice([0, 12345, 67890])),
                            cbsa,
                        )
                    )
    rows.sort()
    return {
        "block_geoid": np.asarray([row[0] for row in rows], dtype=np.int64),
        "population": np.asarray([row[1] for row in rows], dtype=np.int64),
        "congressional_district_geoid": np.asarray(
            [row[2] for row in rows], dtype=np.int64
        ),
        "puma": np.asarray([row[3] for row in rows], dtype=np.int64),
        "sldu": np.asarray([row[4] for row in rows], dtype="U3"),
        "sldl": np.asarray([row[5] for row in rows], dtype="U3"),
        "place_fips": np.asarray([row[6] for row in rows], dtype=np.int32),
        "cbsa_code": np.asarray([row[7] for row in rows], dtype=np.int32),
    }


#: The states of :func:`local_jurisdiction_blocks`: Colorado, Delaware, New
#: York and Pennsylvania.
LOCAL_JURISDICTION_STATES = (8, 10, 36, 42)


def local_jurisdiction_blocks(seed: int = 0) -> dict[str, np.ndarray]:
    """``synthetic_blocks`` over CO, DE, NY and PA with every geography the
    local-jurisdiction registry names, and a decoy for each place code.

    The codes are written here, not read from the registry, so a test's
    expectation does not come from the code under test:

    - Colorado's county 001 becomes 031 (Denver) and Pennsylvania's becomes
      101 (Philadelphia): the two county-rule flags;
    - New York's county 003 is all place 84000 (Yonkers) and Delaware's
      county 003 is all place 77580 (Wilmington): the two place-rule flags;
    - place codes repeat across states, so Delaware's county 001 is all place
      84000 and New York's county 001 is all place 77580. Neither is Yonkers
      or Wilmington. That New York county is renamed 047 (Kings), a New York
      City county, which the PUMA-ladder gate's NYC checks need.

    Every other block keeps its synthetic place. Each state keeps at least a
    county 003, so every jurisdiction has blocks at every seed.
    """

    arrays = synthetic_blocks(seed, states=LOCAL_JURISDICTION_STATES)
    block = arrays["block_geoid"].astype(np.int64)
    state = block // 10**13
    county = block // 10**10 % 1000
    renamed = block.copy()
    for state_fips, old, new in ((8, 1, 31), (42, 1, 101), (36, 1, 47)):
        renamed[(state == state_fips) & (county == old)] += (new - old) * 10**10
    place = arrays["place_fips"].copy()
    for state_fips, county_code, place_code in (
        (36, 3, 84000),
        (10, 3, 77580),
        (10, 1, 84000),
        (36, 1, 77580),
    ):
        place[(state == state_fips) & (county == county_code)] = place_code
    order = np.argsort(renamed, kind="stable")
    located = {name: values[order] for name, values in arrays.items()}
    located["block_geoid"] = renamed[order]
    located["place_fips"] = place[order]
    return located


def expected_local_jurisdiction_flags(household) -> dict[str, np.ndarray]:
    """Each registry flag for each household row, from the row's own
    ``county_fips`` and ``place_fips``.

    The four rules are written out here (Denver is county 08031; Philadelphia
    county 42101; Yonkers New York place 84000; Wilmington Delaware place
    77580), so a test compares the build's flags with an expectation the
    registry and its module had no part in. The person flag
    ``pa_philadelphia_wage_tax_resident`` is given per household; a test maps
    it to persons itself.
    """

    county = household["county_fips"].astype(str).to_numpy()
    place = household["place_fips"].astype(str).to_numpy()
    state = np.array([value[:2] for value in county.tolist()])
    return {
        "in_denver": county == "08031",
        "pa_philadelphia_wage_tax_resident": county == "42101",
        "in_yonkers": (state == "36") & (place == "84000"),
        "in_wilmington": (state == "10") & (place == "77580"),
    }


def assert_local_jurisdiction_flags_follow_geography(household, person) -> None:
    """Every flag column equals :func:`expected_local_jurisdiction_flags`.

    All four must be present as boolean columns: the three household flags on
    ``household`` and the Philadelphia flag on ``person``, each person taking
    its household's value.
    """

    expected = expected_local_jurisdiction_flags(household)
    for flag in ("in_denver", "in_yonkers", "in_wilmington"):
        assert flag in household.columns, flag
        assert household[flag].dtype == bool, (flag, household[flag].dtype)
        np.testing.assert_array_equal(
            household[flag].to_numpy(), expected[flag], err_msg=flag
        )
    flag = "pa_philadelphia_wage_tax_resident"
    assert flag in person.columns, flag
    assert person[flag].dtype == bool, (flag, person[flag].dtype)
    by_household = dict(
        zip(household["household_id"].tolist(), expected[flag].tolist(), strict=True)
    )
    np.testing.assert_array_equal(
        person[flag].to_numpy(),
        np.array([by_household[key] for key in person["person_household_id"].tolist()]),
        err_msg=flag,
    )


def write_location_ladder(
    path: Path,
    arrays: dict[str, np.ndarray] | None = None,
    *,
    with_puma: bool = True,
    metadata: dict | None = None,
) -> Path:
    """Write a block-ladder NPZ; ``arrays`` defaults to ``synthetic_blocks()``."""

    payload = dict(arrays if arrays is not None else synthetic_blocks())
    if not with_puma:
        payload.pop("puma", None)
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata or ladder_metadata(with_puma=with_puma), sort_keys=True)
    )
    np.savez_compressed(path, **payload)
    return path


__all__ = [name for name in globals() if not name.startswith("__")]
