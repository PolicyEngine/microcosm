"""Invented Census-shaped mappings through the shared atomic-area contract."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from microcosm.build import atomic_geography as geo
from microcosm.build.us_runtime import atomic_block_support as us


def inputs():
    blocks = [int(v) for v in ("010010201001000", "010010201001001", "020130001001000")]
    return {
        "block_population": dict(zip(blocks, (2, 6, 1), strict=True)),
        "cd_by_block": dict(zip(blocks, (101, 102, 200), strict=True)),
        "puma_by_tract": {blocks[0] // 10000: 100001, blocks[2] // 10000: 200002},
        "source_ids": {k: "invented-" + k for k in ("population", "district", "puma")},
    }


def test_source_rows_and_leading_zeroes_are_preserved():
    data = inputs()
    before = deepcopy(data)
    payload = us.assemble_atomic_block_support(**data)
    support = geo.decode_atomic_support(payload)
    assert data == before
    assert support.arrays["area"].tolist() == [
        "010010201001000",
        "010010201001001",
        "020130001001000",
    ]
    assert support.arrays["state"].tolist() == ["01", "01", "02"]
    assert support.arrays["county"].tolist() == ["01001", "01001", "02013"]
    assert support.arrays["tract"].tolist() == [
        "01001020100",
        "01001020100",
        "02013000100",
    ]
    assert support.arrays["puma"].tolist() == ["0100001", "0100001", "0200002"]
    assert support.arrays["district"].tolist() == ["0101", "0102", "0200"]
    assert np.array_equal(support.arrays["population"], [2, 6, 1])
    assert support.metadata["columns"]["district"]["relation"] == "official_tabulation"
    assert support.metadata["columns"]["population"]["basis"] == "2020_census_persons"
    for key in ("block_population", "cd_by_block", "puma_by_tract"):
        data[key] = dict(reversed(list(data[key].items())))
    assert us.assemble_atomic_block_support(**data) == payload


def test_country_declaration_uses_shared_assignment_and_derivation():
    payload = us.assemble_atomic_block_support(**inputs())
    support = geo.decode_atomic_support(payload)
    spec = us.assignment_definition(
        identity=("source", "source_id"),
        state_column="observed_state",
        puma_column="observed_puma",
        source_ids=inputs()["source_ids"],
        seed=17,
    )
    households = pd.DataFrame(
        {
            "source": ["acs", "asec", "acs"],
            "source_id": [1, 1, 2],
            "observed_state": ["01", "01", "02"],
            "observed_puma": ["0100001", None, "0200002"],
        }
    )
    supports = {us.SYSTEM: support}
    assigned = pd.concat(
        [households, geo.assign_atomic(households, spec, supports)], axis=1
    )
    result = pd.concat(
        [assigned, geo.derive_geography(assigned, spec, supports)], axis=1
    )
    geo.validate_geography(result, spec, supports)
    assert result.assigned_state_fips.tolist() == households.observed_state.tolist()
    assert result.congressional_district_geoid.iloc[2] == "0200"
    assert len(result) == len(households)
    pd.testing.assert_frame_equal(result[households.columns], households)
    subset = result.iloc[[2, 0]]
    geo.validate_geography(subset, spec, supports)
    pd.testing.assert_frame_equal(
        geo.assign_atomic(households.iloc[[2, 0]], spec, supports),
        assigned.drop(columns=households.columns).iloc[[2, 0]],
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_puma",
        "missing_district",
        "puma_wrong_state",
        "district_wrong_state",
        "negative_population",
        "boolean_population",
        "fractional_population",
        "invalid_block",
        "boolean_block",
        "missing_source",
        "empty_source",
    ],
)
def test_malformed_or_incomplete_source_mappings_refuse(change):
    data = inputs()
    block = next(iter(data["block_population"]))
    if change == "missing_puma":
        data["puma_by_tract"].pop(block // 10000)
    elif change == "missing_district":
        data["cd_by_block"].pop(block)
    elif change == "puma_wrong_state":
        data["puma_by_tract"][block // 10000] = 200001
    elif change == "district_wrong_state":
        data["cd_by_block"][block] = 201
    elif change == "negative_population":
        data["block_population"][block] = -1
    elif change == "boolean_population":
        data["block_population"][block] = True
    elif change == "fractional_population":
        data["block_population"][block] = 1.5
    elif change == "invalid_block":
        data["block_population"][10**15] = 1
    elif change == "boolean_block":
        data["block_population"][True] = 1
    elif change == "missing_source":
        data["source_ids"].pop("district")
    elif change == "empty_source":
        data["source_ids"]["population"] = " "
    with pytest.raises(ValueError, match="US atomic support:"):
        us.assemble_atomic_block_support(**data)
