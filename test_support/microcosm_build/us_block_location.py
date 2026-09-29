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
                tract = 100 * (tract_index + 1)
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
