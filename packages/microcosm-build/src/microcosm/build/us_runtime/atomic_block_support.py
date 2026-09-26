"""Normalize US block mappings for shared geography operators; no new kernel.

Publisher parsers supply block POP100, the CD block-equivalency mapping and the
2020 tract-to-PUMA mapping. Source authentication remains upstream. This module
retains block rows and supplies a country declaration; assignment and geographic
derivation are executed by the shared country-neutral operators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

from microcosm.build.atomic_geography import (
    encode_atomic_support,
    validate_assignment_spec,
)

SYSTEM = "us_census_block_2020"
SOURCE = "us_atomic_block_support"
_STATES = frozenset(
    {
        1,
        2,
        4,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        53,
        54,
        55,
        56,
    }
)


def _require(condition, message):
    if not condition:
        raise ValueError("US atomic support: " + message)


def _integer(value, *, minimum=0, maximum=2**53):
    return (
        isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        and minimum <= value <= maximum
    )


def _sources(source_ids):
    _require(
        isinstance(source_ids, Mapping)
        and set(source_ids) == {"population", "district", "puma"}
        and all(
            isinstance(v, str) and v and v.strip() == v for v in source_ids.values()
        ),
        "three explicit nonempty source identities required",
    )
    return dict(source_ids)


def assemble_atomic_block_support(
    *,
    block_population: Mapping,
    cd_by_block: Mapping,
    puma_by_tract: Mapping,
    source_ids: Mapping,
) -> bytes:
    """Keep each supplied block, with string codes and source-labeled mappings.

    Population is the declared 2020-person sampling proxy. This does not imply
    household counts or support for subsequent construction in unpopulated
    2020 blocks. Existing PL parsers supply populated blocks only. Extra mapping
    entries may cover unselected areas; every supplied support row must map.
    """
    sources = _sources(source_ids)
    _require(
        all(
            isinstance(m, Mapping)
            for m in (block_population, cd_by_block, puma_by_tract)
        )
        and bool(block_population),
        "nonempty block support and mappings required",
    )
    for block, population in block_population.items():
        _require(
            _integer(block, minimum=10**13, maximum=10**15 - 1)
            and int(block) // 10**13 in _STATES,
            "invalid block GEOID",
        )
        _require(_integer(population), "invalid population weight")
    blocks = sorted(int(block) for block in block_population)
    pumas, districts = [], []
    for block in blocks:
        state, tract = block // 10**13, block // 10000
        puma, district = puma_by_tract.get(tract), cd_by_block.get(block)
        _require(
            _integer(puma, minimum=1, maximum=9999999)
            and int(puma) // 100000 == state
            and int(puma) % 100000 > 0,
            "missing or inconsistent tract-to-PUMA mapping",
        )
        _require(
            _integer(district, minimum=100, maximum=5699)
            and int(district) // 100 == state
            and int(district) % 100 <= 53,
            "missing or inconsistent block-to-district mapping",
        )
        pumas.append(f"{int(puma):07d}")
        districts.append(f"{int(district):04d}")
    arrays = {
        "area": np.asarray([f"{v:015d}" for v in blocks]),
        "state": np.asarray([f"{v // 10**13:02d}" for v in blocks]),
        "county": np.asarray([f"{v // 10**10:05d}" for v in blocks]),
        "tract": np.asarray([f"{v // 10000:011d}" for v in blocks]),
        "puma": np.asarray(pumas),
        "district": np.asarray(districts),
        "population": np.asarray(
            [int(block_population[v]) for v in blocks], dtype=np.int64
        ),
    }
    columns = {
        name: {
            "kind": "code",
            "source": sources["population"],
            "vintage": "2020",
            "relation": "exact",
        }
        for name in ("area", "state", "county", "tract")
    }
    columns.update(
        {
            "puma": {
                "kind": "code",
                "source": sources["puma"],
                "vintage": "2020",
                "relation": "exact",
            },
            "district": {
                "kind": "code",
                "source": sources["district"],
                "vintage": "119th_congress",
                "relation": "official_tabulation",
            },
            "population": {
                "kind": "weight",
                "source": sources["population"],
                "basis": "2020_census_persons",
            },
        }
    )
    return encode_atomic_support(
        {
            "version": 1,
            "system": SYSTEM,
            "level": "block",
            "code_system": "census_geoid",
            "vintage": "2020",
            "columns": columns,
        },
        arrays,
    )


def assignment_definition(
    *,
    identity: Sequence[str],
    state_column: str,
    puma_column: str | None,
    source_ids: Mapping,
    seed: int,
) -> dict:
    """Declare one block draw constrained by normalized observed state/PUMA.

    Input codes must already be strings from the survey's source projection.
    Callers declare stable source plus completed initial-clone identity columns.
    Assignment follows those clones; subsequent views retain the block and all
    functionally derived geographies.
    """
    sources = _sources(source_ids)
    _require(type(seed) is int and 0 <= seed < 2**63, "invalid seed")
    _require(not isinstance(identity, (str, bytes)), "identity must be columns")
    constraints = [{"input": state_column, "support": "state", "required": True}]
    if puma_column is not None:
        constraints.append({"input": puma_column, "support": "puma", "required": False})
    layers = [
        {
            "input": name,
            "output": output,
            "vintage": vintage,
            "relation": relation,
            "source": sources[source],
        }
        for name, output, vintage, relation, source in (
            ("state", "assigned_state_fips", "2020", "exact", "population"),
            ("county", "county_fips", "2020", "exact", "population"),
            ("tract", "census_tract_geoid", "2020", "exact", "population"),
            ("puma", "assigned_puma_geoid", "2020", "exact", "puma"),
            (
                "district",
                "congressional_district_geoid",
                "119th_congress",
                "official_tabulation",
                "district",
            ),
        )
    ]
    return validate_assignment_spec(
        {
            "version": 1,
            "identity": list(identity),
            "stream": ["sha256-u53-v1", "us-atomic-geography", 0, seed],
            "outputs": {
                "area": "census_block_geoid",
                "system": "geography_system",
                "basis": "geography_assignment_basis",
            },
            "systems": [
                {
                    "id": SYSTEM,
                    "level": "block",
                    "code_system": "census_geoid",
                    "vintage": "2020",
                    "source": SOURCE,
                    "selector": {},
                    "constraints": constraints,
                    "observed_area": None,
                    "stages": [{"level": "area", "weight": "population"}],
                    "layers": layers,
                }
            ],
        }
    )
