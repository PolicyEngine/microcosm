"""The bytes the ACS local materialize stage writes for its targets (microcosm#1728).

The shared-code work of microcosm#1728 moves the code that builds the local
target surface out of the tools. Each move must leave every target, role and
receipt as it was. ``TargetRegistry.version`` sorts metadata keys, so it does
not see a change of key order; the stage writes ``target_registry.json`` and
``target_roles.json`` with the keys in insertion order, and their sha256 is in
the run identity. These tests pin those bytes.

For each non-county ``--soi-mode``, the test follows the target part of
``do_materialize``: the admin surface, the ladder populations, the role
records, the district pro-rata populations (``state_cd``), the ladder
population rows, and the district and county hold-outs. The household frame
and the engine pass are not needed for that: a release keeps every compiled
target (a lost one stops the run) and every ladder cell (a cell with no
household stops the run without ``--allow-partial-geography``). So the test
gives the population rows a small household table that supports every ladder
cell, and pins the rows' matrix on that table too.

The pins were measured at ``3316038cc``, before any code moved. They run only
when ``MICROCOSM_US_CHRONICLE_FACTS`` names the pinned feed and
``MICROCOSM_US_PUMA_LADDER_ARTIFACT`` names the pinned PUMA ladder. The
county mode (``state_county``) needs the pinned county facts and a household
frame for its support moves; it is not pinned here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.calibrate import TargetRegistry
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

#: The pinned PUMA ladder (as in ``test_us_target_sets.py``).
PINNED_LADDER_SHA256 = (
    "39a2ab2abeab07a88362af7ab2940e0e1d50a297c919e4bbc6fb65bab51147d8"
)

#: The tool's defaults: ``--families`` and ``--geographies``.
FAMILIES = ("snap", "medicaid", "soi")
GEOGRAPHIES = ("state", "cd")

#: For each mode on the pinned feed and ladder: the number of targets, and the
#: sha256 of ``target_registry.json``, of ``target_roles.json``, of the target
#: receipts (SOI surface, district hold-out, county hold-out) and of the
#: population rows' matrix on the supporting household table.
PINNED_BYTES: dict[str, dict[str, object]] = {
    "full": {
        "targets": 30_464,
        "target_registry_sha256": (
            "d568f29827f8c24a07021628c747c4437b212984be9e82ff93cbebc1ec4ecb69"
        ),
        "target_roles_sha256": (
            "a55d88c7c2b747e7321de6cdae4671dfcb29b3ee5c97587d7983ac888652743a"
        ),
        "receipts_sha256": (
            "f102782cd2fcfdd896a91bbb163ab1a5120cf06e008aade657cb2b54b4b1eddb"
        ),
        "population_matrix_sha256": (
            "ae7785b655538f802e9ef7203c3ae6160ad250ed9e22c06c39475ae4614994f4"
        ),
    },
    "state": {
        "targets": 4_867,
        "target_registry_sha256": (
            "0157f2b653e51147b8e1b1fc364b0616d1198665718426384349c37cea8a0e4c"
        ),
        "target_roles_sha256": (
            "929b1b27b62f57c492829b51dd9c0e72a21b470b32345f0475a545f7d76de28d"
        ),
        "receipts_sha256": (
            "c5f148b6b2f72745dd1c3d470b88743f5acf9c5cc07c10fedeef6c6831f31afc"
        ),
        "population_matrix_sha256": (
            "ae7785b655538f802e9ef7203c3ae6160ad250ed9e22c06c39475ae4614994f4"
        ),
    },
    "state_cd": {
        "targets": 26_912,
        "target_registry_sha256": (
            "fad3ddf61915407fe187d9c739bd379d17c804347dc446e85f3e0303e8518bf3"
        ),
        "target_roles_sha256": (
            "ffa9c4c964ea84385f0bfe5ef3c1053937897bfcecff941315d0a61f0ab0cca4"
        ),
        "receipts_sha256": (
            "9c67654cd7c8ab53b68969bf7ea30134871a1407af37d87ae3b2067294efed7f"
        ),
        "population_matrix_sha256": (
            "ae7785b655538f802e9ef7203c3ae6160ad250ed9e22c06c39475ae4614994f4"
        ),
    },
    "totals": {
        "targets": 742,
        "target_registry_sha256": (
            "07a69bf412892b2e5e73868a6d49886c0077c875b07cce293bc742745d858d16"
        ),
        "target_roles_sha256": (
            "c1d5eca2710f97a073c111273b78d2d9e0f58777314b1bd74eb8dd95410a686e"
        ),
        "receipts_sha256": (
            "7195a0a24ce81a4f63d680781d9fbdc2b3f4a50c9d2b3fe1e16e9f4eff66e3e9"
        ),
        "population_matrix_sha256": (
            "ae7785b655538f802e9ef7203c3ae6160ad250ed9e22c06c39475ae4614994f4"
        ),
    },
}


def _load_tool(name: str):
    path = _TEST_PATHS.repository / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _pinned_inputs() -> tuple[str, Path]:
    feed = os.environ.get("MICROCOSM_US_CHRONICLE_FACTS", "")
    if not feed or not Path(feed).exists():
        pytest.skip("set MICROCOSM_US_CHRONICLE_FACTS to the pinned 41b2fb2 feed")
    ladder = os.environ.get("MICROCOSM_US_PUMA_LADDER_ARTIFACT", "")
    if not ladder or not Path(ladder).exists():
        pytest.skip("set MICROCOSM_US_PUMA_LADDER_ARTIFACT to the PUMA ladder")
    assert hashlib.sha256(Path(ladder).read_bytes()).hexdigest() == PINNED_LADDER_SHA256
    return feed, Path(ladder)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _supporting_households(populations) -> tuple[pd.DataFrame, np.ndarray]:
    """One household in each ladder district, and in each state with none."""

    districts = sorted(int(code) for code in populations["cd"])
    states = {code // 100 for code in districts}
    rows = [
        {"state_fips": code // 100, "congressional_district_geoid": code}
        for code in districts
    ]
    rows += [
        {"state_fips": int(code), "congressional_district_geoid": 0}
        for code in sorted(populations["state"])
        if int(code) not in states
    ]
    households = pd.DataFrame(rows)
    return households, np.ones(len(households), dtype=np.float32)


def materialize_target_part(local_tool, feed: str, ladder: Path, soi_mode: str):
    """The target part of ``do_materialize`` for one mode, without a frame.

    Returns ``(specs, roles, receipts, population_matrix)`` in the order the
    stage writes them.
    """

    cd_surface = local_tool.cd_surface
    county_surface = local_tool.county_surface
    args = SimpleNamespace(
        soi_mode=soi_mode,
        cd_holdout_fraction=None,
        county_column="county_fips",
        county_holdout_fraction=county_surface.DEFAULT_COUNTY_HOLDOUT_FRACTION,
        county_holdout_salt=county_surface.COUNTY_HOLDOUT_SALT,
    )
    holdout_fraction = local_tool._cd_holdout_fraction(args)
    surface = local_tool.state_admin_surface(feed, list(FAMILIES), soi_mode=soi_mode)
    registry = surface.registry
    populations = local_tool.ladder_population(
        ladder,
        sorted(set(GEOGRAPHIES) | {"cd"}),
        county_column="county_fips",
    )
    local_tool.require_one_label_per_geography(registry, populations, list(GEOGRAPHIES))
    compiled_specs = list(registry.specs)
    admin_roles = [county_surface.target_record(spec) for spec in compiled_specs]
    if soi_mode == local_tool.SOI_MODE_STATE_CD:
        local_tool._attach_pro_rata_populations(admin_roles, populations["cd"])
        local_tool._require_pro_rata_populations(admin_roles)
    households, sizes = _supporting_households(populations)
    pop_roles, pop_matrix, pop_dropped = cd_surface.population_rows(
        households,
        sizes,
        populations,
        list(GEOGRAPHIES),
        county_column="county_fips",
    )
    assert pop_dropped == []
    pop_specs = local_tool.population_target_specs(
        [record["name"] for record in pop_roles],
        [record["value"] for record in pop_roles],
    )
    roles = admin_roles + pop_roles
    holdout = cd_surface.assign_target_roles(roles, fraction=holdout_fraction)
    county_holdout = county_surface.assign_roles(roles, args)
    receipts = {
        "soi_surface": surface.soi_receipt,
        "cd_holdout": holdout,
        "county_holdout": county_holdout,
    }
    return (*compiled_specs, *pop_specs), roles, receipts, pop_matrix


def surface_bytes(local_tool, feed: str, ladder: Path, soi_mode: str, tmp_path: Path):
    """The pinned measurements of one mode (see ``PINNED_BYTES``)."""

    specs, roles, receipts, pop_matrix = materialize_target_part(
        local_tool, feed, ladder, soi_mode
    )
    registry_path = tmp_path / f"target_registry_{soi_mode}.json"
    TargetRegistry(specs, country="us").to_json(registry_path)
    matrix = (
        str(pop_matrix.shape).encode()
        + pop_matrix.indptr.astype(np.int64).tobytes()
        + pop_matrix.indices.astype(np.int64).tobytes()
        + pop_matrix.data.astype(np.float32).tobytes()
    )
    return {
        "targets": len(specs),
        "target_registry_sha256": _sha256(registry_path.read_bytes()),
        "target_roles_sha256": _sha256(json.dumps(roles, indent=1).encode()),
        "receipts_sha256": _sha256(json.dumps(receipts, indent=1).encode()),
        "population_matrix_sha256": _sha256(matrix),
    }


@pytest.fixture(scope="module")
def local_tool():
    return _load_tool("build_us_acs_local_release")


@pytest.mark.parametrize("soi_mode", sorted(PINNED_BYTES))
def test_pinned_feed_local_surface_bytes_are_unchanged(
    local_tool, soi_mode: str, tmp_path: Path
) -> None:
    feed, ladder = _pinned_inputs()
    assert (
        surface_bytes(local_tool, feed, ladder, soi_mode, tmp_path)
        == (PINNED_BYTES[soi_mode])
    )


def test_every_non_county_mode_is_pinned(local_tool) -> None:
    assert set(PINNED_BYTES) == set(local_tool.SOI_MODES) - {
        local_tool.county_surface.SOI_MODE
    }


def test_the_supporting_households_cover_every_cell() -> None:
    populations = {"state": {6: 1.0, 11: 1.0, 72: 1.0}, "cd": {601: 1.0, 1198: 1.0}}
    households, sizes = _supporting_households(populations)
    assert households.to_dict("records") == [
        {"state_fips": 6, "congressional_district_geoid": 601},
        {"state_fips": 11, "congressional_district_geoid": 1198},
        {"state_fips": 72, "congressional_district_geoid": 0},
    ]
    assert sizes.tolist() == [1.0, 1.0, 1.0]
