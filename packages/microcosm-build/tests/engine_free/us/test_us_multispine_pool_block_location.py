"""The stacked multispine pool's ``--location-rule block_v1`` slice (microcosm#696).

The legacy geography contract, operator order and parser surface are pinned by
the existing pool-tool and h5_io tests; these tests cover the opt-in rule: its
contract and operator order, the CLI refusals, the draw on a small synthetic
stacked frame, and the round trip through h5_io's consumer validators.

The ``--cd-plan-registry`` section at the bottom covers the optional pinned
block -> CD-plan registry.  Its invariants, each executed below:

1. Legacy is byte-identical: the legacy contract, run config and root attrs
   are unchanged, and the flag is refused under the legacy rule.
2. block_v1 without the flag is byte-identical: contract, configured pins
   digest and run config equal their pre-registry goldens, and the receipt,
   household_location record and root attrs gain no key.
3. Only the pinned registry loads, and a wrong file is refused at input
   verification, before any source is loaded.
4. Every plan column equals the registry's district of the drawn block and
   lies in the household's state (property-tested).
5. Attaching never moves a drawn block or any other column (property-tested).
6. The registry digest recorded in the contract, the household_location
   record, the verified input pins and the H5 root attr is the loaded file's.
7. A no-registry checkpoint cannot resume a registry run, and vice versa:
   the configured namespace and the checkpoint identity both differ.
8. The tool's contract equals h5_io's consumer copy with and without the
   registry (differential, property-tested).
"""

from __future__ import annotations

# Test functions take the imported ``pool_tool`` fixture as a parameter.
# ruff: noqa: F811
import copy
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import h5_io
from microcosm.build.us_runtime.block_location import (
    POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR,
    US_BLOCK_LOCATION_RULE_ID,
    US_CD_PLAN_COLUMN_PREFIX,
    attach_pinned_cd_plan_registry,
    load_us_location_ladder,
)
from microcosm.build.us_runtime.cd_plan_registry import (
    PRIMARY_CD_PLAN,
    load_us_cd_plan_registry,
)
from microcosm.build.us_runtime.congressional_district_vintage import (
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
    CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
)
from microcosm.build.us_runtime.cps_source_geography import cps_source_geography
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_block_location import (
    synthetic_blocks,
    write_location_ladder,
)
from test_support.microcosm_build.us_cd_plan_registry import (
    file_sha256,
    pin_cd_plan_registry,
    registry_ready_blocks,
    write_cd_plan_registry,
)
from test_support.microcosm_build.us_multispine_pool_tool import (  # noqa: F401
    _stacked_main_argv,
    pool_tool,
)

# The canonical 117th -> 119th crosswalk pin (rebuilt from the CD plan
# registry in #1043); the tool refuses any other.
_CROSSWALK_SHA256 = "a347303fff3fea145f758488c43cb94355df5a8cb5e512552b2ee3791d2aa233"
_CPS_SHA256 = "a" * 64
_INCOME_YEAR = 2023
# Seed 2 of the synthetic ladder carries the Bronx (36005), which the NYC
# checks of the PUMA-ladder gate need, plus Baldwin, AL (01003).
_LADDER_SEED = 2
# The pinned registry's plans other than the ladder's primary (119th).
_NON_PRIMARY_PLANS = ("117th_congress", "118th_congress", "120th_congress")
_PLAN_COLUMNS = tuple(
    f"{US_CD_PLAN_COLUMN_PREFIX}{plan}" for plan in _NON_PRIMARY_PLANS
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ladder_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("block-ladder") / "ladder.npz"
    return write_location_ladder(path, synthetic_blocks(_LADDER_SEED))


@pytest.fixture(scope="module")
def ladder(ladder_path: Path):
    return load_us_location_ladder(ladder_path)


def _crosswalk(ladder) -> pd.DataFrame:
    districts = sorted(set(ladder.blocks.congressional_district_geoid.tolist()))
    return pd.DataFrame(
        {
            "target_geography_id": [
                f"{CURRENT_CONGRESSIONAL_DISTRICT_PREFIX}{district:04d}"
                for district in districts
            ]
        }
    )


# (household channel, state, observed ACS PUMA or None, CPS GTCO or None, weight)
_HOUSEHOLDS = (
    ("acs", 1, "0100100", None, 15.0),
    ("acs", 1, "0100101", None, 15.0),
    ("acs", 2, "0200100", None, 15.0),
    ("acs", 1, "0100100", None, 15.0),
    ("acs", 1, "0100101", None, 15.0),
    ("acs", 2, "0200100", None, 15.0),
    ("asec", 36, None, 5, 4.0),  # identified Bronx: the NYC anchor
    ("asec", 36, None, 0, 6.0),
    ("asec", 1, None, 3, 5.0),  # identified Baldwin
    ("asec", 1, None, 0, 5.0),
    ("asec", 2, None, 0, 5.0),
)


def _stacked_frame(
    *, asec_source_year: int = _INCOME_YEAR, id_offset: int = 0
) -> Frame:
    n = len(_HOUSEHOLDS)
    rows = np.arange(1, n + 1, dtype=np.int64)
    ids = rows + id_offset
    channel = np.asarray([row[0] for row in _HOUSEHOLDS], dtype=object)
    is_asec = channel == "asec"
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "source_year": np.where(is_asec, float(asec_source_year), np.nan),
            # The CPS source key is the source row's, whatever the pool id.
            "source_household_id": np.where(is_asec, 900.0 + rows, np.nan),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": ids,
            "state_fips": np.asarray([row[1] for row in _HOUSEHOLDS], dtype=np.int64),
            "puma": pd.array([row[2] for row in _HOUSEHOLDS], dtype="string"),
            support_channel_column("household"): channel,
        }
    )
    tables = {
        "person": person,
        "household": household,
        **{
            entity: pd.DataFrame({f"{entity}_id": ids})
            for entity in US_SCHEMA.group_entities
            if entity != "household"
        },
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([row[4] for row in _HOUSEHOLDS], dtype=np.float64),
                WeightKind.IMPORTANCE,
            )
        },
        pd.Series(["fixture"] * n, dtype=object),
    )


def _cps_household_geography() -> pd.DataFrame:
    rows = [
        (900 + index + 1, state, gtco)
        for index, (channel, state, _puma, gtco, _weight) in enumerate(_HOUSEHOLDS)
        if channel == "asec"
    ]
    return pd.DataFrame(
        {
            "source_year": _INCOME_YEAR,
            "source_household_id": np.asarray([row[0] for row in rows], np.int64),
            "state_fips": np.asarray([row[1] for row in rows], np.int64),
            "gtco": np.asarray([row[2] for row in rows], np.int64),
            "gtcbsa": np.zeros(len(rows), dtype=np.int64),
            "person_weight": np.full(len(rows), 1_000.0),
        }
    )


def _contract(pool_tool: ModuleType, ladder, *, seed: int = 11) -> dict:
    return pool_tool._stacked_block_location_contract(
        seed,
        1,
        ladder.sha256,
        cps_asec_sha256={_INCOME_YEAR: _CPS_SHA256},
    )


def _assign(pool_tool: ModuleType, ladder, *, seed: int = 11, frame=None):
    return pool_tool._assign_stacked_household_block_location(
        _stacked_frame() if frame is None else frame,
        ladder=ladder,
        crosswalk=_crosswalk(ladder),
        cps_household_geography=_cps_household_geography(),
        contract=_contract(pool_tool, ladder, seed=seed),
    )


def _manifest(receipt: dict, ladder) -> dict:
    """A JSON round-tripped stacked manifest carrying a block_v1 receipt."""

    stored = json.loads(json.dumps(receipt))
    authorities = stored["contract"]["authorities"]
    return {
        "pipeline": "us-stacked-pool",
        "random_seed": 0,
        "operator_order": list(h5_io.US_STACKED_POOL_BLOCK_LOCATION_OPERATOR_ORDER),
        "geography_assignment": stored,
        "stage_receipts": {"geography_assignment": copy.deepcopy(stored)},
        "provenance_pins": {
            role: {
                "expected_sha256": authority["sha256"],
                "actual_sha256": authority["sha256"],
            }
            for role, authority in authorities.items()
        },
        "household_location": copy.deepcopy(stored["household_location"]),
    }


def _with_lineage(household: pd.DataFrame, *, clone_of: int | None = None):
    """Add PUF-support lineage columns, optionally one clone of a household."""

    table = household.copy()
    table[support_source_id_column("household")] = table["household_id"]
    table[support_clone_index_column("household")] = 0
    if clone_of is not None:
        clone = table.loc[table["household_id"] == clone_of].copy()
        clone["household_id"] = int(table["household_id"].max()) + 1
        clone[support_clone_index_column("household")] = 1
        table = pd.concat([table, clone], ignore_index=True)
    return table


def _root_attributes(receipt: dict, ladder) -> dict[str, str]:
    seed = receipt["contract"]["seed"]["value"]
    attributes = {
        CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR: _CROSSWALK_SHA256,
        CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR: "119th_congress",
        h5_io.POPULACE_LOCATION_RULE_ATTR: "block_v1",
        h5_io.POPULACE_LOCATION_SEED_ATTR: str(seed),
        h5_io.POPULACE_LOCATION_CLONES_ATTR: "1",
        h5_io.POPULACE_BLOCK_LADDER_SHA256_ATTR: ladder.sha256,
        h5_io.POPULACE_BLOCK_LADDER_VINTAGES_ATTR: json.dumps(
            receipt["household_location"]["block_ladder"]["layer_vintages"],
            sort_keys=True,
        ),
    }
    registry = receipt["household_location"].get("cd_plan_registry")
    if registry is not None:
        attributes[POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR] = registry["sha256"]
    return attributes


def _block_args(**overrides) -> SimpleNamespace:
    values = {
        "location_rule": "block_v1",
        "location_seed": 3,
        "location_clones": 1,
        "block_ladder": Path("ladder.npz"),
        "block_ladder_sha256": "b" * 64,
        "cps_asec_h5": [f"{_INCOME_YEAR}=census_cps_{_INCOME_YEAR}.h5"],
        "cps_asec_h5_sha256": [f"{_INCOME_YEAR}={_CPS_SHA256}"],
        "puma_ladder": None,
        "puma_ladder_sha256": None,
        "congressional_district_vintage_crosswalk": Path("crosswalk.csv"),
        "congressional_district_vintage_crosswalk_sha256": _CROSSWALK_SHA256,
        "legacy_two_spine": False,
        "config_authority": "constants",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# --------------------------------------------------------------------------
# Contract, operator order and the default path
# --------------------------------------------------------------------------


def test_default_rule_keeps_the_legacy_contract_and_operator_order(
    pool_tool: ModuleType,
) -> None:
    legacy = pool_tool._stacked_geography_assignment_contract()
    assert pool_tool._stacked_location_contract_for_args(SimpleNamespace()) == legacy
    assert (
        pool_tool._stacked_operator_order_for_contract(legacy)
        == h5_io.US_STACKED_POOL_OPERATOR_ORDER
    )
    assert h5_io.us_stacked_pool_operator_order({}) == (
        h5_io.US_STACKED_POOL_OPERATOR_ORDER
    )
    assert pool_tool._with_location_run_config(
        {"config_authority": "constants"}, SimpleNamespace()
    ) == {"config_authority": "constants"}
    assert pool_tool._stacked_location_root_attributes(None, {}) == {}
    assert pool_tool._stacked_location_root_attributes(legacy, {}) == {}


def test_block_operator_order_replaces_only_the_geography_slot() -> None:
    legacy = h5_io.US_STACKED_POOL_OPERATOR_ORDER
    block = h5_io.US_STACKED_POOL_BLOCK_LOCATION_OPERATOR_ORDER
    assert len(block) == len(legacy)
    differing = [
        index for index, (a, b) in enumerate(zip(legacy, block, strict=True)) if a != b
    ]
    assert differing == [legacy.index("assign_us_puma_ladder")]
    assert block[differing[0]] == "assign_us_block_location"


@pytest.mark.parametrize("registry_sha256", [None, "c" * 64])
@pytest.mark.parametrize("seed", [0, 1, 578, 2**31])
def test_tool_contract_equals_the_consumer_copy(
    pool_tool: ModuleType, seed: int, registry_sha256: str | None
) -> None:
    cps = {2022: "1" * 64, 2024: "2" * 64}
    tool = pool_tool._stacked_block_location_contract(
        seed,
        1,
        "f" * 64,
        cps_asec_sha256=cps,
        cd_plan_registry_sha256=registry_sha256,
    )
    consumer = h5_io.expected_us_stacked_block_location_contract(
        seed=seed,
        clones=1,
        block_ladder_sha256="f" * 64,
        crosswalk_sha256=_CROSSWALK_SHA256,
        cps_asec_sha256=cps,
        cd_plan_registry_sha256=registry_sha256,
    )
    assert json.loads(json.dumps(tool)) == consumer
    base = pool_tool._stacked_block_location_contract(
        seed, 1, "f" * 64, cps_asec_sha256=cps
    )
    if registry_sha256 is None:
        assert tool == base
        assert "cd_plan_registry" not in tool["authorities"]
    else:
        # The registry adds exactly one authority and three derived columns.
        assert tool["authorities"]["cd_plan_registry"] == {
            "input_role": "cd_plan_registry",
            "source_ref": "source:us_cd_plan_registry_2020",
            "sha256": registry_sha256,
        }
        assert tool["declaration"]["derive"] == [
            *base["declaration"]["derive"],
            *(f"{US_CD_PLAN_COLUMN_PREFIX}{plan}" for plan in _NON_PRIMARY_PLANS),
        ]
        stripped = copy.deepcopy(tool)
        del stripped["authorities"]["cd_plan_registry"]
        stripped["declaration"]["derive"] = base["declaration"]["derive"]
        assert stripped == base
    assert tool["algorithm"]["id"] == US_BLOCK_LOCATION_RULE_ID
    assert h5_io.US_STACKED_BLOCK_LOCATION_ALGORITHM_ID == US_BLOCK_LOCATION_RULE_ID
    assert tool["seed"]["value"] == seed
    # The block rule's seed site/stream stay outside the legacy-v1 protocol.
    assert tool["seed"]["site"] != "legacy_puma_ladder"
    assert tool["seed"]["stream"] != "geography_legacy"
    assert (
        pool_tool._stacked_operator_order_for_contract(tool)
        == h5_io.US_STACKED_POOL_BLOCK_LOCATION_OPERATOR_ORDER
    )


def test_block_identity_binds_its_contract_and_order(
    pool_tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The engine-free lane has no policyengine-us metadata index.
    monkeypatch.setattr(
        pool_tool,
        "pool_remaining_stage_input_manifest_receipt",
        lambda: {"fixture": "remaining-stage manifest"},
    )
    kwargs = {
        "stack_receipt": {"sample_fraction": 1.0, "sample_seed": 578},
        "sample_fraction": 1.0,
        "sample_seed": 578,
        "clone_attachment_fraction": 1.0,
        "clone_attachment_seed": 578,
        "policyengine_us_version": "fixture",
    }
    legacy = pool_tool._stacked_checkpoint_base_identity({}, **kwargs)
    contract = pool_tool._stacked_block_location_contract(
        4, 1, "f" * 64, cps_asec_sha256={_INCOME_YEAR: _CPS_SHA256}
    )
    block = pool_tool._stacked_checkpoint_base_identity(
        {}, geography_contract=contract, **kwargs
    )
    assert legacy["geography_assignment"] == (
        pool_tool._stacked_geography_assignment_contract()
    )
    assert legacy["pool_code"]["operator_order"] == list(
        h5_io.US_STACKED_POOL_OPERATOR_ORDER
    )
    assert block["geography_assignment"] == contract
    assert block["pool_code"]["operator_order"] == list(
        h5_io.US_STACKED_POOL_BLOCK_LOCATION_OPERATOR_ORDER
    )
    stripped = {
        key: value
        for key, value in block.items()
        if key not in {"geography_assignment", "pool_code"}
    }
    assert stripped == {
        key: value
        for key, value in legacy.items()
        if key not in {"geography_assignment", "pool_code"}
    }


# --------------------------------------------------------------------------
# CLI surface and refusals
# --------------------------------------------------------------------------


def test_location_options_parse_beside_the_pinned_parser(
    pool_tool: ModuleType, tmp_path: Path
) -> None:
    legacy = pool_tool._parse_arguments(_stacked_main_argv(tmp_path))
    assert (legacy.location_rule, legacy.location_seed, legacy.location_clones) == (
        "legacy",
        0,
        1,
    )
    pool_tool._require_location_rule_arguments(legacy)
    argv = [
        argument
        for argument in _stacked_main_argv(tmp_path)
        if argument
        not in {
            "--puma-ladder",
            str(tmp_path / "puma-ladder"),
            "--puma-ladder-sha256",
            "39a2ab2abeab07a88362af7ab2940e0e1d50a297c919e4bbc6fb65bab51147d8",
        }
    ]
    block = pool_tool._parse_arguments(
        [
            *argv,
            "--location-rule",
            "block_v1",
            "--location-seed",
            "7",
            "--block-ladder",
            str(tmp_path / "ladder.npz"),
            "--block-ladder-sha256",
            "b" * 64,
            "--cps-asec-h5",
            f"2023={tmp_path / 'census_cps_2023.h5'}",
            "--cps-asec-h5-sha256",
            f"2023={_CPS_SHA256}",
        ]
    )
    assert (block.location_rule, block.location_seed) == ("block_v1", 7)
    pool_tool._require_location_rule_arguments(block)
    contract = pool_tool._stacked_location_contract_for_args(block)
    assert contract["seed"]["value"] == 7
    assert contract["authorities"]["block_ladder"]["sha256"] == "b" * 64
    assert set(contract["authorities"]) == {
        "block_ladder",
        "congressional_district_vintage_crosswalk",
        "cps_asec_household_geography_2023",
    }
    assert pool_tool._with_location_run_config(
        {"config_authority": "constants"}, block
    ) == {
        "config_authority": "constants",
        "location_rule": "block_v1",
        "location_seed": 7,
        "location_clones": 1,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"location_rule": "legacy", "location_seed": 5},
            "apply only to --location-rule block_v1",
        ),
        (
            {"location_rule": "legacy", "location_seed": 0, "location_clones": 3},
            "apply only to --location-rule block_v1",
        ),
        (
            {
                "location_rule": "legacy",
                "location_seed": 0,
                "cps_asec_h5": None,
                "cps_asec_h5_sha256": None,
                "block_ladder_sha256": None,
            },
            "--block-ladder apply only",
        ),
        ({"legacy_two_spine": True}, "--legacy-two-spine assigns no"),
        ({"config_authority": "constants_adapter"}, "constants_adapter"),
        (
            {"location_clones": 2},
            "validate_assembly_provenance.*validate_stacked_spine_frame",
        ),
        ({"location_seed": -1}, "non-negative"),
        ({"block_ladder": None}, "missing --block-ladder"),
        ({"block_ladder_sha256": None}, "missing --block-ladder-sha256"),
        ({"puma_ladder": Path("puma.npz")}, "legacy-rule inputs"),
        ({"cps_asec_h5": None}, "requires --cps-asec-h5"),
        (
            {"cps_asec_h5_sha256": [f"2024={_CPS_SHA256}"]},
            "must name the same income years",
        ),
        ({"cps_asec_h5": ["2023"]}, "INCOME_YEAR=VALUE"),
        (
            {"congressional_district_vintage_crosswalk_sha256": "0" * 64},
            "canonical US authority pin",
        ),
        (
            {
                "location_rule": "legacy",
                "location_seed": 0,
                "block_ladder": None,
                "block_ladder_sha256": None,
                "cps_asec_h5": None,
                "cps_asec_h5_sha256": None,
                "cd_plan_registry": Path("registry.npz"),
            },
            "^--cd-plan-registry apply only to --location-rule block_v1",
        ),
    ],
)
def test_location_arguments_are_refused(
    pool_tool: ModuleType, overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pool_tool._require_location_rule_arguments(_block_args(**overrides))


def test_main_refuses_block_v1_with_the_legacy_two_spine_route(
    pool_tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pool_tool, "_main_legacy", lambda _a: calls.append("l"))
    monkeypatch.setattr(pool_tool, "_main_stacked", lambda _a: calls.append("s"))
    with pytest.raises(ValueError, match="--legacy-two-spine"):
        pool_tool.main(
            [
                *_stacked_main_argv(tmp_path),
                "--legacy-two-spine",
                "--location-rule",
                "block_v1",
            ]
        )
    with pytest.raises(ValueError, match="apply only to --location-rule block_v1"):
        pool_tool.main([*_stacked_main_argv(tmp_path), "--location-seed", "9"])
    with pytest.raises(ValueError, match="constants_adapter"):
        pool_tool.main(
            [
                *_stacked_main_argv(tmp_path),
                "--config-authority",
                "constants_adapter",
                "--location-rule",
                "block_v1",
            ]
        )
    assert calls == []


def test_block_root_attributes_record_rule_seed_and_ladder(
    pool_tool: ModuleType, ladder
) -> None:
    _assigned, receipt, _districts = _assign(pool_tool, ladder, seed=21)
    verified = {
        "block_ladder": SimpleNamespace(actual_sha256=ladder.sha256),
    }
    attributes = pool_tool._stacked_location_root_attributes(
        receipt["contract"],
        verified,
        pool_tool._stacked_household_location_record({"geography_assignment": receipt}),
    )
    assert attributes == {
        key: value
        for key, value in _root_attributes(receipt, ladder).items()
        if key in h5_io.US_STACKED_BLOCK_LOCATION_ROOT_ATTRIBUTES
    }
    assert attributes[h5_io.POPULACE_LOCATION_SEED_ATTR] == "21"
    # The registry digest attr is the one conditional entry: absent here.
    assert set(attributes) == set(h5_io.US_STACKED_BLOCK_LOCATION_ROOT_ATTRIBUTES) - {
        POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR
    }
    with pytest.raises(ValueError, match="layer vintages"):
        pool_tool._stacked_location_root_attributes(receipt["contract"], verified)


# --------------------------------------------------------------------------
# The draw on a synthetic stacked frame
# --------------------------------------------------------------------------


def test_block_draw_respects_each_households_source_geography(
    pool_tool: ModuleType, ladder
) -> None:
    frame = _stacked_frame()
    assigned, receipt, districts = _assign(pool_tool, ladder, frame=frame)
    household = assigned.table("household")
    source = frame.table("household")
    assert list(household["household_id"]) == list(source["household_id"])

    block = household["block_geoid"].astype(str)
    assert block.str.fullmatch(r"[0-9]{15}").all()
    block_index = np.searchsorted(ladder.block_geoid, block.astype(np.int64))
    assert (ladder.block_geoid[block_index] == block.astype(np.int64)).all()
    # Every geography is the block's lookup; the state never moves.
    assert (household["tract_geoid"] == block.str[:11]).all()
    assert (household["county_fips"] == block.str[:5]).all()
    assert (block.str[:2].astype(int) == source["state_fips"]).all()
    assert (
        household["congressional_district_geoid"].to_numpy(np.int64)
        == ladder.blocks.congressional_district_geoid[block_index]
    ).all()
    assert (
        household["puma"].astype(str).astype(np.int64).to_numpy()
        == ladder.puma[block_index]
    ).all()

    # ACS: the drawn block lies inside the observed PUMA.
    is_acs = source[support_channel_column("household")].eq("acs").to_numpy()
    assert (
        household.loc[is_acs, "puma"].astype(str).to_numpy()
        == source.loc[is_acs, "puma"].astype(str).to_numpy()
    ).all()

    # ASEC: identified counties are kept; GTCO = 0 avoids every county that is
    # both coded in the file and on Census's official identified list.
    cps = _cps_household_geography()
    reference = cps_source_geography(
        ladder,
        cps,
        source_year=cps["source_year"].to_numpy(),
        source_household_id=cps["source_household_id"].to_numpy(),
        state_fips=cps["state_fips"].to_numpy(),
    )
    excluded = reference.identified_counties[_INCOME_YEAR]
    asec_rows = np.flatnonzero(~is_acs)
    for row, gtco in zip(asec_rows, cps["gtco"], strict=True):
        state = int(source.loc[row, "state_fips"])
        county = int(household.loc[row, "county_fips"])
        if gtco:
            assert county == state * 1000 + int(gtco)
        else:
            assert county not in excluded

    record = receipt["household_location"]
    assert record["rule"] == US_BLOCK_LOCATION_RULE_ID
    assert record["location_rule"] == "block_v1"
    assert record["seed"] == 11
    assert record["clones"] == 1
    assert record["block_ladder"]["sha256"] == ladder.sha256
    assert record["source_geography_households"]["puma"] == int(is_acs.sum())
    assert record["candidate_set_rule"]["cps_asec"]["cbsa_narrowing"]["requested"]
    assert receipt["gate"]["passed"] is True
    assert set(receipt["gate"]["gates"]) == {"us_block_location", "us_puma_ladder"}
    assert districts == tuple(
        sorted(set(ladder.blocks.congressional_district_geoid.tolist()))
    )


def test_block_draw_is_deterministic_and_seeded(pool_tool: ModuleType, ladder) -> None:
    _frame_a, first, _ = _assign(pool_tool, ladder, seed=11)
    _frame_b, again, _ = _assign(pool_tool, ladder, seed=11)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)
    moved = {
        _assign(pool_tool, ladder, seed=seed)[1]["assigned_household_geography"][
            "sha256"
        ]
        for seed in range(12, 20)
    }
    assert len(moved | {first["assigned_household_geography"]["sha256"]}) > 1


def test_asec_rows_without_a_pinned_source_year_fail_closed(
    pool_tool: ModuleType, ladder
) -> None:
    with pytest.raises(ValueError, match="pinned CPS ASEC household H5s"):
        _assign(pool_tool, ladder, frame=_stacked_frame(asec_source_year=2022))


def test_rows_without_an_observed_puma_or_cps_key_fail_closed(
    pool_tool: ModuleType, ladder
) -> None:
    frame = _stacked_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"].loc[0, "puma"] = pd.NA
    broken = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )
    # Spine-blind: without an observed PUMA a household must resolve through
    # its CPS ASEC source key, which an ACS record does not have.
    with pytest.raises(ValueError, match="source_household_id|ASEC source row"):
        _assign(pool_tool, ladder, frame=broken)


def test_block_draw_refuses_a_ladder_with_other_bytes(
    pool_tool: ModuleType, ladder
) -> None:
    contract = pool_tool._stacked_block_location_contract(
        11, 1, "0" * 64, cps_asec_sha256={_INCOME_YEAR: _CPS_SHA256}
    )
    with pytest.raises(ValueError, match="differ from the contract's pinned"):
        pool_tool._assign_stacked_household_block_location(
            _stacked_frame(),
            ladder=ladder,
            crosswalk=_crosswalk(ladder),
            cps_household_geography=_cps_household_geography(),
            contract=contract,
        )


def test_receipt_validator_selects_the_contract_by_algorithm_id(
    pool_tool: ModuleType, ladder
) -> None:
    assigned, receipt, districts = _assign(pool_tool, ladder)
    contract = receipt["contract"]
    validate = pool_tool._validate_stacked_geography_assignment_receipt
    validate(
        assigned,
        receipt,
        target_districts=districts,
        boundary="fixture",
        require_exact_assembled_rows=True,
        configured_contract=contract,
        location_ladder=ladder,
    )
    # A block receipt is never accepted by a legacy-configured run ...
    for configured in (None, pool_tool._stacked_geography_assignment_contract()):
        with pytest.raises(ValueError, match="configured location rule"):
            validate(
                assigned,
                receipt,
                target_districts=districts,
                boundary="fixture",
                require_exact_assembled_rows=True,
                configured_contract=configured,
            )
    # ... nor under another seed, and a mismatched algorithm id is refused.
    other_seed = _contract(pool_tool, ladder, seed=12)
    with pytest.raises(ValueError, match="configured location rule"):
        validate(
            assigned,
            receipt,
            target_districts=districts,
            boundary="fixture",
            require_exact_assembled_rows=True,
            configured_contract=other_seed,
        )
    renamed = copy.deepcopy(receipt)
    renamed["contract"]["algorithm"]["id"] = "us_block_location.population_draw.v2"
    with pytest.raises(ValueError, match="contract"):
        validate(
            assigned,
            renamed,
            target_districts=districts,
            boundary="fixture",
            require_exact_assembled_rows=True,
            configured_contract=contract,
        )
    tampered = copy.deepcopy(receipt)
    tampered["household_location"]["seed"] = 99
    with pytest.raises(ValueError, match="household_location record"):
        validate(
            assigned,
            tampered,
            target_districts=districts,
            boundary="fixture",
            require_exact_assembled_rows=True,
            configured_contract=contract,
        )


def _cloned_frame(assigned: Frame, household: pd.DataFrame) -> Frame:
    """A frame whose extra household rows are PUF-support copies with members."""

    ids = household["household_id"].to_numpy(np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
        }
    )
    tables = {
        "person": person,
        "household": household.reset_index(drop=True),
        **{
            entity: pd.DataFrame({f"{entity}_id": ids})
            for entity in US_SCHEMA.group_entities
            if entity != "household"
        },
    }
    return Frame(
        tables,
        assigned.schema,
        {"household": Weights(np.ones(len(ids)), WeightKind.IMPORTANCE)},
        pd.Series(["fixture"] * len(ids), dtype=object),
    )


def test_support_clones_inherit_the_located_block(
    pool_tool: ModuleType, ladder
) -> None:
    assigned, receipt, districts = _assign(pool_tool, ladder)
    household = _with_lineage(assigned.table("household"), clone_of=7)
    validate = pool_tool._validate_stacked_geography_assignment_receipt
    validate(
        _cloned_frame(assigned, household),
        receipt,
        target_districts=districts,
        boundary="fixture",
        require_exact_assembled_rows=False,
        configured_contract=receipt["contract"],
    )
    moved = household.copy()
    moved.loc[moved.index[-1], "sldu"] = "zzz"
    with pytest.raises(
        ValueError, match="disagree on assigned geography column 'sldu'"
    ):
        validate(
            _cloned_frame(assigned, moved),
            receipt,
            target_districts=districts,
            boundary="fixture",
            require_exact_assembled_rows=False,
            configured_contract=receipt["contract"],
        )


# --------------------------------------------------------------------------
# h5_io consumer validators
# --------------------------------------------------------------------------


def test_h5_io_accepts_a_block_manifest_round_trip(
    pool_tool: ModuleType, ladder
) -> None:
    assigned, receipt, _districts = _assign(pool_tool, ladder)
    manifest = _manifest(receipt, ladder)
    path = Path("fixture.manifest.json")
    assert h5_io.us_stacked_pool_operator_order(manifest) == (
        h5_io.US_STACKED_POOL_BLOCK_LOCATION_OPERATOR_ORDER
    )
    h5_io._validate_stacked_geography_assignment_manifest_binding(
        manifest, manifest_path=path
    )
    household = _with_lineage(assigned.table("household"), clone_of=3)
    h5_io._validate_stacked_geography_h5_binding(
        manifest,
        household,
        _root_attributes(receipt, ladder),
        manifest_path=path,
        pool_path=Path("fixture.h5"),
    )
    # The late-DAG binding accepts the block order and then asks for the DAG.
    with pytest.raises(ValueError, match="late-producer DAG receipt"):
        h5_io._validate_stacked_late_dag_manifest_binding(manifest, manifest_path=path)
    legacy_order = dict(
        manifest, operator_order=list(h5_io.US_STACKED_POOL_OPERATOR_ORDER)
    )
    with pytest.raises(ValueError, match="canonical late-DAG operator order"):
        h5_io._validate_stacked_late_dag_manifest_binding(
            legacy_order, manifest_path=path
        )


def _mutated(manifest: dict, mutate) -> dict:
    changed = copy.deepcopy(manifest)
    mutate(changed)
    changed["stage_receipts"]["geography_assignment"] = copy.deepcopy(
        changed["geography_assignment"]
    )
    return changed


def _set(path: tuple, value):
    def mutate(manifest: dict) -> None:
        target = manifest
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            _set(("geography_assignment", "contract", "algorithm", "id"), "other.v1"),
            "geography declaration changed",
        ),
        (
            _set(("geography_assignment", "contract", "algorithm", "clones"), 2),
            "location clones",
        ),
        (
            _set(("geography_assignment", "contract", "declaration", "anchor"), "puma"),
            "block-location contract changed",
        ),
        (
            _set(("geography_assignment", "contract", "seed", "stream"), "x"),
            "block-location contract changed",
        ),
        (_set(("household_location", "seed"), 12345), "household_location record"),
        (
            _set(("household_location", "block_ladder", "layer_vintages"), {}),
            "household_location record",
        ),
        (_set(("random_seed",), 1), "random_seed"),
        (
            _set(("provenance_pins", "block_ladder", "actual_sha256"), "0" * 64),
            "'block_ladder' differs from its authenticated input pin",
        ),
        (
            _set(
                (
                    "geography_assignment",
                    "gate",
                    "gates",
                    "us_block_location",
                    "passed",
                ),
                False,
            ),
            "passed block-location and PUMA-ladder gates",
        ),
        (
            _set(
                ("geography_assignment", "assigned_household_geography", "codec"), "x"
            ),
            "block-location receipt",
        ),
    ],
)
def test_h5_io_refuses_a_changed_block_manifest(
    pool_tool: ModuleType, ladder, mutate, message: str
) -> None:
    _assigned, receipt, _districts = _assign(pool_tool, ladder)
    manifest = _mutated(_manifest(receipt, ladder), mutate)
    with pytest.raises(ValueError, match=message):
        h5_io._validate_stacked_geography_assignment_manifest_binding(
            manifest, manifest_path=Path("fixture.manifest.json")
        )


def test_h5_io_refuses_block_h5_drift(pool_tool: ModuleType, ladder) -> None:
    assigned, receipt, _districts = _assign(pool_tool, ladder)
    manifest = _manifest(receipt, ladder)
    household = _with_lineage(assigned.table("household"), clone_of=3)
    attributes = _root_attributes(receipt, ladder)

    def check(table: pd.DataFrame, attrs: dict) -> None:
        h5_io._validate_stacked_geography_h5_binding(
            manifest,
            table,
            attrs,
            manifest_path=Path("fixture.manifest.json"),
            pool_path=Path("fixture.h5"),
        )

    with pytest.raises(ValueError, match="location root attributes"):
        check(household, {**attributes, h5_io.POPULACE_LOCATION_SEED_ATTR: "12"})
    with pytest.raises(ValueError, match="location root attributes"):
        check(
            household,
            {**attributes, h5_io.POPULACE_BLOCK_LADDER_VINTAGES_ATTR: "{}"},
        )
    with pytest.raises(ValueError, match="location root attributes"):
        check(
            household,
            {
                key: value
                for key, value in attributes.items()
                if key != h5_io.POPULACE_LOCATION_RULE_ATTR
            },
        )
    moved = household.copy()
    first_block = str(moved.loc[0, "block_geoid"])
    other = next(
        f"{value:015d}"
        for value in ladder.block_geoid.tolist()
        if f"{value:015d}"[:5] == first_block[:5] and f"{value:015d}" != first_block
    )
    moved.loc[0, "block_geoid"] = other
    with pytest.raises(ValueError, match="block location differs"):
        check(moved, attributes)
    incoherent = household.copy()
    incoherent.loc[incoherent.index[-1], "place_fips"] = "99999"
    with pytest.raises(ValueError, match="disagree on assigned geography column"):
        check(incoherent, attributes)
    prefix = household.copy()
    prefix["tract_geoid"] = prefix["tract_geoid"].astype(str).str[:10] + "9"
    with pytest.raises(ValueError, match="prefix of its household's located block"):
        check(prefix, attributes)


def test_cps_pins_must_be_the_raw_stage_sources(pool_tool: ModuleType) -> None:
    raw_stage = {
        "source_receipt": {
            "sources": [
                {"year": 2023, "sha256": "1" * 64},
                {"year": 2024, "sha256": "2" * 64},
            ]
        }
    }
    bind = pool_tool._require_cps_asec_pins_match_raw_stage
    bind(raw_stage, {2023: "1" * 64, 2024: "2" * 64})
    with pytest.raises(ValueError, match="differ from the ASEC raw-stage sources"):
        bind(raw_stage, {2023: "1" * 64})
    with pytest.raises(ValueError, match="differ from the ASEC raw-stage sources"):
        bind(raw_stage, {2023: "1" * 64, 2024: "3" * 64})
    with pytest.raises(ValueError, match="records no source_receipt.sources"):
        bind({}, {2023: "1" * 64})


def test_an_acs_household_that_lost_its_puma_is_refused_not_joined_to_cps(
    pool_tool: ModuleType, ladder
) -> None:
    """Production ACS rows carry the ``puma_geoid`` alias and raw ``PUMA`` and
    share the CPS 2024 source-key space, so one without a valid ``puma`` must
    be refused rather than routed through a CPS source-key join."""

    frame = _stacked_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    household = tables["household"]
    has_puma = household["puma"].astype("string").str.len().fillna(0) > 0
    household["puma_geoid"] = household["puma"].where(has_puma, pd.NA)
    first_acs = int(np.flatnonzero(has_puma.to_numpy())[0])
    household.loc[first_acs, "puma"] = pd.NA
    broken = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )
    with pytest.raises(ValueError, match="carry an ACS 'puma_geoid' but no valid"):
        _assign(pool_tool, ladder, frame=broken)

    household.loc[first_acs, "puma"] = household.loc[first_acs, "puma_geoid"]
    household.loc[first_acs, "puma_geoid"] = "0100999"
    mismatched = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )
    with pytest.raises(ValueError, match="disagrees with their puma_geoid"):
        _assign(pool_tool, ladder, frame=mismatched)


# --------------------------------------------------------------------------
# --cd-plan-registry: the pinned block -> CD-plan registry
# --------------------------------------------------------------------------
#
# ``synthetic_blocks`` ladders are not apportionment-valid, so the registry
# tests use their own ladder: the module ladder's blocks, PUMAs and layers
# with only its (119th) districts rewritten by ``registry_ready_blocks``.  The
# households above keep their states and candidate sets on it.

# Pre-registry goldens, computed from the tool and h5_io as committed before
# --cd-plan-registry existed (and after #1043's crosswalk pin): the block_v1
# contract of _block_args(), its configured pins digest and configured
# stacked identity with the stacked pins of _configured_args(), and the
# legacy geography contract.
_NO_REGISTRY_CONTRACT_SHA256 = (
    "9f788c81805228feb2d551200d26c410694d16d6d5e735c24603a8aadc8cf78e"
)
_NO_REGISTRY_PINS_DIGEST = (
    "1d502538e42572c4162da3c9a49d74db8a4e9fcdeca38f9b7cdf3e2906b95602"
)
_NO_REGISTRY_CONFIGURED_IDENTITY_SHA256 = (
    "3f30f0db53445473717aa6433e590118fffc7759b04ea334d9ad2459fc70c1d7"
)
_LEGACY_CONTRACT_SHA256 = (
    "44e7f5f6cd7ceeef85d532c3d46306d03de4f4ee811376e6d29920742746b65c"
)


@pytest.fixture(scope="module")
def registry(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """A registry-ready ladder, its synthetic registry, and the attachment.

    The pin is active only while attaching; tests that call the tool's pin
    readers pin again with their own ``monkeypatch``.
    """

    root = tmp_path_factory.mktemp("cd-plan-registry")
    arrays = registry_ready_blocks(synthetic_blocks(_LADDER_SEED))
    ladder_path = write_location_ladder(root / "ladder.npz", arrays)
    registry_path = write_cd_plan_registry(root / "registry.npz", arrays)
    plain = load_us_location_ladder(ladder_path)
    with pytest.MonkeyPatch.context() as patch:
        sha256 = pin_cd_plan_registry(patch, registry_path)
        attached, record = attach_pinned_cd_plan_registry(plain, registry_path)
    return SimpleNamespace(
        path=registry_path,
        ladder_path=ladder_path,
        sha256=sha256,
        plain=plain,
        attached=attached,
        record=record,
        lookup=load_us_cd_plan_registry(registry_path),
    )


def _registry_contract(
    pool_tool: ModuleType, registry: SimpleNamespace, *, seed: int = 11
) -> dict:
    return pool_tool._stacked_block_location_contract(
        seed,
        1,
        registry.plain.sha256,
        cps_asec_sha256={_INCOME_YEAR: _CPS_SHA256},
        cd_plan_registry_sha256=registry.sha256,
    )


def _assign_plain(
    pool_tool: ModuleType, registry: SimpleNamespace, *, seed: int = 11, frame=None
):
    """The registry-free draw on the registry-ready ladder."""

    return pool_tool._assign_stacked_household_block_location(
        _stacked_frame() if frame is None else frame,
        ladder=registry.plain,
        crosswalk=_crosswalk(registry.plain),
        cps_household_geography=_cps_household_geography(),
        contract=_contract(pool_tool, registry.plain, seed=seed),
    )


def _assign_registry(
    pool_tool: ModuleType,
    registry: SimpleNamespace,
    *,
    seed: int = 11,
    frame=None,
    ladder=None,
    record=...,
    contract=None,
):
    return pool_tool._assign_stacked_household_block_location(
        _stacked_frame() if frame is None else frame,
        ladder=registry.attached if ladder is None else ladder,
        crosswalk=_crosswalk(registry.plain),
        cps_household_geography=_cps_household_geography(),
        contract=(
            _registry_contract(pool_tool, registry, seed=seed)
            if contract is None
            else contract
        ),
        cd_plan_registry=registry.record if record is ... else record,
    )


def _configured_args(**overrides) -> SimpleNamespace:
    """_block_args plus the stacked pins and scale controls identity reads."""

    values = {
        "acs_household_zip_sha256": "1" * 64,
        "acs_person_zip_sha256": "2" * 64,
        "acs_rent_h5_sha256": "3" * 64,
        "asec_raw_stage_h5_sha256": "4" * 64,
        "puf_h5_sha256": "5" * 64,
        "puf_source_year_csv_sha256": "6" * 64,
        "sample_fraction": 0.01,
        "sample_seed": 578,
        "clone_attachment_fraction": 1.0,
        "clone_attachment_seed": 579,
    }
    values.update(overrides)
    return _block_args(**values)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _verified(pool_tool: ModuleType, contract: dict) -> dict:
    """A verified input for every contract authority, at its pinned digest."""

    return {
        role: pool_tool._VerifiedInput(
            role=role,
            path=Path(f"/fixture/{role}"),
            expected_sha256=authority["sha256"],
            actual_sha256=authority["sha256"],
            size_bytes=1,
        )
        for role, authority in contract["authorities"].items()
    }


def _lookup(registry: SimpleNamespace, household: pd.DataFrame, plan: str):
    blocks = household["block_geoid"].astype(str).astype(np.int64).to_numpy()
    index = np.searchsorted(registry.lookup.block_geoid, blocks)
    assert (registry.lookup.block_geoid[index] == blocks).all()
    return registry.lookup.plans[plan][index]


# -- 1-2. Byte identity without the flag -------------------------------------


def test_without_the_flag_every_surface_equals_its_pre_registry_golden(
    pool_tool: ModuleType,
) -> None:
    legacy = pool_tool._stacked_geography_assignment_contract()
    assert _canonical_sha256(legacy) == _LEGACY_CONTRACT_SHA256
    assert pool_tool._stacked_location_contract_for_args(SimpleNamespace()) == legacy

    block = _configured_args()
    assert not hasattr(block, "cd_plan_registry")
    explicit = _configured_args(cd_plan_registry=None)
    for args in (block, explicit):
        contract = pool_tool._stacked_location_contract_for_args(args)
        assert _canonical_sha256(contract) == _NO_REGISTRY_CONTRACT_SHA256
        assert pool_tool._configured_input_pins_digest(args) == (
            _NO_REGISTRY_PINS_DIGEST
        )
        assert _canonical_sha256(pool_tool._configured_stacked_identity(args)) == (
            _NO_REGISTRY_CONFIGURED_IDENTITY_SHA256
        )
        assert pool_tool._with_location_run_config(
            {"config_authority": "constants"}, args
        ) == {
            "config_authority": "constants",
            "location_rule": "block_v1",
            "location_seed": 3,
            "location_clones": 1,
        }
        assert pool_tool._block_location_source_paths(args) == {
            Path("ladder.npz").resolve(),
            Path("crosswalk.csv").resolve(),
            Path(f"census_cps_{_INCOME_YEAR}.h5").resolve(),
        }
    assert _canonical_sha256(
        h5_io.expected_us_stacked_block_location_contract(
            seed=3,
            clones=1,
            block_ladder_sha256="b" * 64,
            crosswalk_sha256=_CROSSWALK_SHA256,
            cps_asec_sha256={_INCOME_YEAR: _CPS_SHA256},
        )
    ) == (_NO_REGISTRY_CONTRACT_SHA256)


def test_without_the_flag_the_receipt_gains_no_registry_key(
    pool_tool: ModuleType, ladder
) -> None:
    _assigned, receipt, _districts = _assign(pool_tool, ladder)
    assert set(receipt) == {
        "artifact_kind",
        "schema_version",
        "contract",
        "pre_assignment_household_order",
        "assigned_household_geography",
        "target_universe",
        "output",
        "summary",
        "household_location",
        "gate",
    }
    record = receipt["household_location"]
    assert "cd_plan_registry" not in record
    assert not any(
        column.startswith(US_CD_PLAN_COLUMN_PREFIX)
        for column in record["geography_columns"]
    )
    assert set(record["block_ladder"]["congressional_district_plans"]) == {
        PRIMARY_CD_PLAN
    }
    assert "cd_plan_registry" not in receipt["contract"]["authorities"]


# -- 3. The flag: parsing, refusals, pinning --------------------------------


def test_registry_option_parses_beside_the_pinned_parser(
    pool_tool: ModuleType, tmp_path: Path
) -> None:
    assert (
        pool_tool._parse_arguments(_stacked_main_argv(tmp_path)).cd_plan_registry
        is None
    )
    parsed = pool_tool._parse_arguments(
        [*_stacked_main_argv(tmp_path), "--cd-plan-registry", "registry.npz"]
    )
    assert parsed.cd_plan_registry == Path("registry.npz")
    # Only the location parser owns it; the pinned stacked parser does not.
    assert "cd_plan_registry" not in vars(
        pool_tool._parser().parse_args(_stacked_main_argv(tmp_path))
    )


def test_main_refuses_the_registry_under_the_legacy_rule(
    pool_tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pool_tool, "_main_legacy", lambda _a: calls.append("l"))
    monkeypatch.setattr(pool_tool, "_main_stacked", lambda _a: calls.append("s"))
    for extra in ([], ["--legacy-two-spine"]):
        with pytest.raises(
            ValueError, match="^--cd-plan-registry apply only to --location-rule"
        ):
            pool_tool.main(
                [
                    *_stacked_main_argv(tmp_path),
                    *extra,
                    "--cd-plan-registry",
                    str(tmp_path / "registry.npz"),
                ]
            )
    assert calls == []


def test_registry_configuration_is_argv_determined_and_conditional(
    pool_tool: ModuleType,
    registry: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin = pin_cd_plan_registry(monkeypatch, registry.path)
    plain = _configured_args()
    given = _configured_args(cd_plan_registry=registry.path)
    # The configured pin is the packaged digest, read before the file: a
    # path that does not exist configures the same identity.
    absent = _configured_args(cd_plan_registry=tmp_path / "not-there.npz")

    contract = pool_tool._stacked_location_contract_for_args(given)
    assert contract["authorities"]["cd_plan_registry"]["sha256"] == pin
    assert contract == pool_tool._stacked_block_location_contract(
        3,
        1,
        "b" * 64,
        cps_asec_sha256={_INCOME_YEAR: _CPS_SHA256},
        cd_plan_registry_sha256=pin,
    )
    assert pool_tool._configured_input_pins_digest(given) != (
        pool_tool._configured_input_pins_digest(plain)
    )
    assert pool_tool._with_location_run_config({}, given) == {
        **pool_tool._with_location_run_config({}, plain),
        "cd_plan_registry_sha256": pin,
    }
    assert pool_tool._block_location_source_paths(given) == {
        *pool_tool._block_location_source_paths(plain),
        registry.path.resolve(),
    }

    configured = pool_tool._configured_stacked_identity(given)
    assert configured == pool_tool._configured_stacked_identity(absent)
    unconfigured = pool_tool._configured_stacked_identity(plain)
    assert {key for key in configured if configured[key] != unconfigured[key]} == {
        "expected_input_pins_digest",
        "geography_assignment",
    }
    outputs = pool_tool._stacked_output_paths(tmp_path / "pool.h5")
    assert pool_tool._stacked_checkpoint_root(outputs, configured) != (
        pool_tool._stacked_checkpoint_root(outputs, unconfigured)
    )


def test_an_unpinned_or_missing_registry_is_refused_at_verification(
    pool_tool: ModuleType,
    registry: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {}
    for name in ("ladder.npz", "crosswalk.csv", "census_cps.h5"):
        files[name] = tmp_path / name
        files[name].write_bytes(name.encode())
    args = _block_args(
        block_ladder=files["ladder.npz"],
        block_ladder_sha256=file_sha256(files["ladder.npz"]),
        congressional_district_vintage_crosswalk=files["crosswalk.csv"],
        congressional_district_vintage_crosswalk_sha256=file_sha256(
            files["crosswalk.csv"]
        ),
        cps_asec_h5=[f"{_INCOME_YEAR}={files['census_cps.h5']}"],
        cps_asec_h5_sha256=[f"{_INCOME_YEAR}={file_sha256(files['census_cps.h5'])}"],
    )
    assert "cd_plan_registry" not in pool_tool._verify_block_location_inputs(args)

    registry_args = _block_args(**{**vars(args), "cd_plan_registry": registry.path})
    # The packaged pin names the real artifact, so the synthetic file fails.
    with pytest.raises(ValueError, match="US CD plan registry SHA-256 mismatch"):
        pool_tool._verify_block_location_inputs(registry_args)
    missing = _block_args(
        **{**vars(args), "cd_plan_registry": tmp_path / "missing.npz"}
    )
    with pytest.raises(FileNotFoundError, match="US CD plan registry"):
        pool_tool._verify_block_location_inputs(missing)

    pin = pin_cd_plan_registry(monkeypatch, registry.path)
    verified = pool_tool._verify_block_location_inputs(registry_args)
    assert verified["cd_plan_registry"].expected_sha256 == pin
    assert verified["cd_plan_registry"].actual_sha256 == file_sha256(registry.path)
    other = write_cd_plan_registry(
        tmp_path / "other.npz",
        registry_ready_blocks(synthetic_blocks(_LADDER_SEED)),
        known_deviations=False,
    )
    with pytest.raises(ValueError, match="US CD plan registry SHA-256 mismatch"):
        pool_tool._verify_block_location_inputs(
            _block_args(**{**vars(args), "cd_plan_registry": other})
        )


def test_main_refuses_an_unpinned_registry_before_loading_any_source(
    pool_tool: ModuleType,
    registry: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)
    real_verify = pool_tool._verify_file

    def verify(role: str, path: Path, expected: str):
        if role == "US CD plan registry":
            return real_verify(role, path, expected)
        return pool_tool._VerifiedInput(
            role=role,
            path=Path(path),
            expected_sha256=expected,
            actual_sha256=expected,
            size_bytes=1,
        )

    monkeypatch.setattr(pool_tool, "_verify_file", verify)
    monkeypatch.setattr(
        pool_tool,
        "_verify_acs_file",
        lambda role, path, expected, _packaged: verify(role, path, expected),
    )
    monkeypatch.setattr(pool_tool, "ACS_2022_RENT_ARTIFACT_SHA256", "1" * 64)
    monkeypatch.setattr(pool_tool, "_git_code_pin", lambda: "a" * 40)
    expensive: list[str] = []

    def load_inputs(*_args, **_kwargs):
        expensive.append("sources")
        raise RuntimeError("fixture stops after input verification")

    monkeypatch.setattr(pool_tool, "_load_inputs", load_inputs)
    monkeypatch.setattr(
        pool_tool,
        "load_congressional_district_vintage_crosswalk",
        lambda _path: _crosswalk(registry.plain),
    )
    argv = [
        argument
        for argument in _stacked_main_argv(tmp_path)
        if argument
        not in {
            "--puma-ladder",
            str(tmp_path / "puma-ladder"),
            "--puma-ladder-sha256",
            "39a2ab2abeab07a88362af7ab2940e0e1d50a297c919e4bbc6fb65bab51147d8",
        }
    ] + [
        "--location-rule",
        "block_v1",
        "--block-ladder",
        str(tmp_path / "ladder.npz"),
        "--block-ladder-sha256",
        "b" * 64,
        "--cps-asec-h5",
        f"{_INCOME_YEAR}={tmp_path / 'census_cps.h5'}",
        "--cps-asec-h5-sha256",
        f"{_INCOME_YEAR}={_CPS_SHA256}",
        "--cd-plan-registry",
        str(registry.path),
    ]

    def phases() -> list[list[str]]:
        return [
            json.loads(path.read_text())["phases_reached"]
            for path in sorted((tmp_path / "logbook-receipts").glob("*/error.json"))
        ]

    with pytest.raises(ValueError, match="US CD plan registry SHA-256 mismatch"):
        pool_tool.main(argv)
    assert expensive == []
    # The error receipt lists the phases reached before the failure.
    assert phases() == [["attempt_started", "configured"]]

    # Pinned, the same file verifies and the run reaches the sources.
    pin_cd_plan_registry(monkeypatch, registry.path)
    with pytest.raises(RuntimeError, match="fixture stops"):
        pool_tool.main(argv)
    assert expensive == ["sources"]
    assert ["attempt_started", "configured", "inputs_verified"] in phases()


# -- 7. Checkpoint identity: resumes bind the registry ------------------------


def test_checkpoints_never_cross_between_registry_and_plain_runs(
    pool_tool: ModuleType,
    registry: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin_cd_plan_registry(monkeypatch, registry.path)
    monkeypatch.setattr(
        pool_tool,
        "pool_remaining_stage_input_manifest_receipt",
        lambda: {"fixture": "remaining-stage manifest"},
    )
    scale = {
        "sample_fraction": 0.01,
        "sample_seed": 578,
        "clone_attachment_fraction": 1.0,
        "clone_attachment_seed": 579,
    }
    stack = {"sample_fraction": 0.01, "sample_seed": 578}
    plain_contract = _contract(pool_tool, registry.plain)
    registry_contract = _registry_contract(pool_tool, registry)
    runs = {
        "plain": (_verified(pool_tool, plain_contract), plain_contract),
        "registry": (_verified(pool_tool, registry_contract), registry_contract),
    }
    identities = {
        name: pool_tool._stacked_checkpoint_base_identity(
            verified, stack_receipt=stack, geography_contract=contract, **scale
        )
        for name, (verified, contract) in runs.items()
    }
    changed = {
        key
        for key in identities["plain"]
        if identities["plain"][key] != identities["registry"][key]
    }
    assert changed == {"inputs", "geography_assignment"}
    assert identities["registry"]["inputs"]["cd_plan_registry"]["sha256"] == (
        registry.sha256
    )

    # A resume is discovered only by the run whose identity wrote it.
    for writer in runs:
        root = tmp_path / writer
        root.mkdir()
        stage = "assembled"
        manifest_path = (
            root / pool_tool._POOL_STAGE_CHECKPOINT_FILENAMES[stage]
        ).with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(
                {
                    "artifact_kind": (
                        pool_tool._POOL_STAGE_CHECKPOINT_MANIFEST_ARTIFACT_KIND
                    ),
                    "schema_version": pool_tool.POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
                    "materializer_version": (
                        pool_tool.POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION
                    ),
                    "stage": stage,
                    "identity": pool_tool._pool_checkpoint_stage_identity(
                        identities[writer], stage
                    ),
                }
            )
        )
        for reader, (reader_verified, reader_contract) in runs.items():
            found = pool_tool._discover_stacked_checkpoint_identity(
                root,
                verified_inputs=reader_verified,
                geography_contract=reader_contract,
                **scale,
            )
            assert (found is not None) == (reader == writer), (writer, reader)

    # Verified inputs and contract must agree on the registry either way, and
    # the verified registry must be the packaged pin.
    with pytest.raises(ValueError, match="exactly when its contract names"):
        pool_tool._stacked_checkpoint_base_identity(
            runs["registry"][0],
            stack_receipt=stack,
            geography_contract=plain_contract,
            **scale,
        )
    with pytest.raises(ValueError, match="requires verified input role"):
        pool_tool._stacked_checkpoint_base_identity(
            runs["plain"][0],
            stack_receipt=stack,
            geography_contract=registry_contract,
            **scale,
        )
    unpinned = pool_tool._stacked_block_location_contract(
        11,
        1,
        registry.plain.sha256,
        cps_asec_sha256={_INCOME_YEAR: _CPS_SHA256},
        cd_plan_registry_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="differs from its packaged pin"):
        pool_tool._stacked_checkpoint_base_identity(
            _verified(pool_tool, unpinned),
            stack_receipt=stack,
            geography_contract=unpinned,
            **scale,
        )


# -- 4-6. The draw with the registry attached --------------------------------


def test_registry_draw_writes_each_plan_as_the_registry_lookup(
    pool_tool: ModuleType, registry: SimpleNamespace
) -> None:
    assigned, receipt, districts = _assign_registry(pool_tool, registry)
    plain_frame, plain_receipt, plain_districts = _assign_plain(pool_tool, registry)
    household = assigned.table("household")
    plain = plain_frame.table("household")

    assert list(household.columns) == [*plain.columns, *_PLAN_COLUMNS]
    pd.testing.assert_frame_equal(household[list(plain.columns)], plain)
    assert districts == plain_districts
    for plan, column in registry.record["columns"].items():
        values = household[column].to_numpy(np.int64)
        assert (values == _lookup(registry, household, plan)).all(), plan
        assert (values // 100 == household["state_fips"].to_numpy()).all(), plan
    assert registry.record["columns"][PRIMARY_CD_PLAN] == (
        "congressional_district_geoid"
    )
    # At least one household sits in a district the 119th plan does not use.
    assert any(
        (household[column] != household["congressional_district_geoid"]).any()
        for column in _PLAN_COLUMNS
    )

    record = receipt["household_location"]
    assert record["cd_plan_registry"] == json.loads(json.dumps(registry.record))
    assert record["cd_plan_registry"]["sha256"] == file_sha256(registry.path)
    assert receipt["contract"]["authorities"]["cd_plan_registry"]["sha256"] == (
        registry.sha256
    )
    assert record["geography_columns"][-3:] == list(_PLAN_COLUMNS)
    assert receipt["gate"]["passed"] is True
    assert receipt["gate"]["gates"]["us_block_location"]["details"]["columns"][
        -3:
    ] == list(_PLAN_COLUMNS)
    # Everything the plan columns do not touch equals the plain receipt.
    for key in (
        "pre_assignment_household_order",
        "assigned_household_geography",
        "target_universe",
        "output",
        "summary",
    ):
        assert receipt[key] == plain_receipt[key], key
    for key in set(plain_receipt["household_location"]) - {
        "block_ladder",
        "geography_columns",
    }:
        assert record[key] == plain_receipt["household_location"][key], key


def test_registry_draw_refuses_a_mismatched_attachment(
    pool_tool: ModuleType, registry: SimpleNamespace
) -> None:
    other_sha = dict(registry.record, sha256="d" * 64)
    with pytest.raises(ValueError, match="registry bytes differ from the contract"):
        _assign_registry(pool_tool, registry, record=other_sha)
    with pytest.raises(ValueError, match="registry attached: False"):
        _assign_registry(pool_tool, registry, record=None)
    with pytest.raises(ValueError, match="contract binds it: False"):
        _assign_registry(
            pool_tool,
            registry,
            contract=_contract(pool_tool, registry.plain),
        )
    # The record alone is not an attachment: the ladder must carry the plans.
    with pytest.raises(ValueError, match="CD plan columns differ"):
        _assign_registry(pool_tool, registry, ladder=registry.plain)
    # ... and a ladder carrying plans needs a contract that derives them.
    with pytest.raises(ValueError, match="CD plan columns differ"):
        pool_tool._assign_stacked_household_block_location(
            _stacked_frame(),
            ladder=registry.attached,
            crosswalk=_crosswalk(registry.plain),
            cps_household_geography=_cps_household_geography(),
            contract=_contract(pool_tool, registry.plain),
        )


def test_registry_receipt_validator_binds_the_record_and_plan_columns(
    pool_tool: ModuleType, registry: SimpleNamespace
) -> None:
    assigned, receipt, districts = _assign_registry(pool_tool, registry)
    contract = receipt["contract"]
    validate = pool_tool._validate_stacked_geography_assignment_receipt

    def check(frame, changed_receipt=None, *, exact=True, configured=None):
        validate(
            frame,
            receipt if changed_receipt is None else changed_receipt,
            target_districts=districts,
            boundary="fixture",
            require_exact_assembled_rows=exact,
            configured_contract=contract if configured is None else configured,
            location_ladder=registry.attached if exact else None,
        )

    check(assigned)
    # A registry receipt never binds a registry-free run, nor the reverse.
    with pytest.raises(ValueError, match="configured location rule"):
        check(assigned, configured=_contract(pool_tool, registry.plain))
    plain_frame, plain_receipt, _ = _assign_plain(pool_tool, registry)
    with pytest.raises(ValueError, match="configured location rule"):
        check(plain_frame, plain_receipt)

    tampered = copy.deepcopy(receipt)
    tampered["household_location"]["cd_plan_registry"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="CD plan registry record differs"):
        check(assigned, tampered)
    dropped = copy.deepcopy(receipt)
    del dropped["household_location"]["cd_plan_registry"]
    with pytest.raises(ValueError, match="CD plan registry record differs"):
        check(assigned, dropped)
    added = copy.deepcopy(plain_receipt)
    added["household_location"]["cd_plan_registry"] = registry.record
    with pytest.raises(ValueError, match="does not bind"):
        check(plain_frame, added, configured=plain_receipt["contract"])
    underived = copy.deepcopy(receipt)
    underived["contract"]["declaration"]["derive"] = underived["contract"][
        "declaration"
    ]["derive"][:-1]
    with pytest.raises(ValueError, match="contract changed"):
        check(assigned, underived, configured=underived["contract"])

    column = f"{US_CD_PLAN_COLUMN_PREFIX}118th_congress"
    household = assigned.table("household")
    with pytest.raises(ValueError, match=f"lost CD plan column '{column}'"):
        check(_cloned_frame(assigned, household.drop(columns=column)), exact=False)
    elsewhere = household.copy()
    elsewhere.loc[0, column] = 3601  # a household in Alabama (state 01)
    with pytest.raises(ValueError, match="in every household's state"):
        check(_cloned_frame(assigned, elsewhere), exact=False)
    # A same-state move passes the state check but not the live gate.
    new_york = int(np.flatnonzero(household["state_fips"].to_numpy() == 36)[0])
    moved = household.copy()
    current = int(moved.loc[new_york, column])
    moved.loc[new_york, column] = 3601 if current != 3601 else 3602
    moved_frame = Frame(
        {entity: assigned.table(entity) for entity in assigned.entities}
        | {"household": moved},
        assigned.schema,
        {"household": assigned.weights_for("household")},
        assigned.strata,
    )
    with pytest.raises(ValueError, match="block-location gate receipt changed"):
        check(moved_frame)

    # Support clones inherit every plan column.
    cloned = _with_lineage(household, clone_of=7)
    check(_cloned_frame(assigned, cloned), exact=False)
    drift = cloned.copy()
    current = int(drift.loc[drift.index[-1], column])
    drift.loc[drift.index[-1], column] = 3601 if current != 3601 else 3602
    with pytest.raises(
        ValueError, match=f"disagree on assigned geography column '{column}'"
    ):
        check(_cloned_frame(assigned, drift), exact=False)


def test_registry_root_attribute_is_the_recorded_and_verified_digest(
    pool_tool: ModuleType, registry: SimpleNamespace
) -> None:
    _assigned, receipt, _districts = _assign_registry(pool_tool, registry, seed=21)
    record = pool_tool._stacked_household_location_record(
        {"geography_assignment": receipt}
    )
    verified = {
        "block_ladder": SimpleNamespace(actual_sha256=registry.plain.sha256),
        "cd_plan_registry": SimpleNamespace(actual_sha256=registry.sha256),
    }
    attributes = pool_tool._stacked_location_root_attributes(
        receipt["contract"], verified, record
    )
    assert attributes[POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR] == registry.sha256
    assert set(attributes) == set(h5_io.US_STACKED_BLOCK_LOCATION_ROOT_ATTRIBUTES)
    assert attributes == {
        key: value
        for key, value in _root_attributes(receipt, registry.plain).items()
        if key in h5_io.US_STACKED_BLOCK_LOCATION_ROOT_ATTRIBUTES
    }
    for changed in (
        {**verified, "cd_plan_registry": SimpleNamespace(actual_sha256="d" * 64)},
        {"block_ladder": verified["block_ladder"]},
    ):
        with pytest.raises(ValueError, match="CD plan registry digest"):
            pool_tool._stacked_location_root_attributes(
                receipt["contract"], changed, record
            )
    with pytest.raises(ValueError, match="CD plan registry digest"):
        pool_tool._stacked_location_root_attributes(
            _contract(pool_tool, registry.plain, seed=21), verified, record
        )


# -- 6. h5_io consumer validators on a registry pool -------------------------


def test_h5_io_accepts_a_registry_manifest_and_h5_round_trip(
    pool_tool: ModuleType, registry: SimpleNamespace
) -> None:
    assigned, receipt, _districts = _assign_registry(pool_tool, registry)
    manifest = _manifest(receipt, registry.plain)
    path = Path("fixture.manifest.json")
    assert manifest["provenance_pins"]["cd_plan_registry"]["actual_sha256"] == (
        registry.sha256
    )
    h5_io._validate_stacked_geography_assignment_manifest_binding(
        manifest, manifest_path=path
    )
    household = _with_lineage(assigned.table("household"), clone_of=7)
    h5_io._validate_stacked_geography_h5_binding(
        manifest,
        household,
        _root_attributes(receipt, registry.plain),
        manifest_path=path,
        pool_path=Path("fixture.h5"),
    )


def _without_registry_authority(manifest: dict) -> None:
    contract = manifest["geography_assignment"]["contract"]
    del contract["authorities"]["cd_plan_registry"]
    contract["declaration"]["derive"] = contract["declaration"]["derive"][:-3]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            _set(("provenance_pins", "cd_plan_registry", "actual_sha256"), "d" * 64),
            "'cd_plan_registry' differs from its authenticated input pin",
        ),
        (
            lambda manifest: manifest["provenance_pins"].pop("cd_plan_registry"),
            "'cd_plan_registry' differs from its authenticated input pin",
        ),
        (
            _set(
                (
                    "geography_assignment",
                    "contract",
                    "authorities",
                    "cd_plan_registry",
                    "sha256",
                ),
                "d" * 64,
            ),
            "'cd_plan_registry' differs from its authenticated input pin",
        ),
        (
            lambda manifest: manifest["geography_assignment"]["contract"][
                "declaration"
            ]["derive"].pop(),
            "block-location contract changed",
        ),
        (_without_registry_authority, "does not bind"),
    ],
)
def test_h5_io_refuses_a_changed_registry_manifest(
    pool_tool: ModuleType, registry: SimpleNamespace, mutate, message: str
) -> None:
    _assigned, receipt, _districts = _assign_registry(pool_tool, registry)
    manifest = _mutated(_manifest(receipt, registry.plain), mutate)
    with pytest.raises(ValueError, match=message):
        h5_io._validate_stacked_geography_assignment_manifest_binding(
            manifest, manifest_path=Path("fixture.manifest.json")
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda location: location["cd_plan_registry"].update({"sha256": "d" * 64}),
        lambda location: location.pop("cd_plan_registry"),
        lambda location: location["cd_plan_registry"]["columns"].pop("118th_congress"),
    ],
)
def test_h5_io_refuses_a_changed_registry_record(
    pool_tool: ModuleType, registry: SimpleNamespace, mutate
) -> None:
    _assigned, receipt, _districts = _assign_registry(pool_tool, registry)
    manifest = _manifest(receipt, registry.plain)
    for location in (
        manifest["household_location"],
        manifest["geography_assignment"]["household_location"],
        manifest["stage_receipts"]["geography_assignment"]["household_location"],
    ):
        mutate(location)
    with pytest.raises(ValueError, match="CD-plan registry record does not match"):
        h5_io._validate_stacked_geography_assignment_manifest_binding(
            manifest, manifest_path=Path("fixture.manifest.json")
        )


def test_h5_io_refuses_registry_h5_drift(
    pool_tool: ModuleType, registry: SimpleNamespace
) -> None:
    assigned, receipt, _districts = _assign_registry(pool_tool, registry)
    manifest = _manifest(receipt, registry.plain)
    household = _with_lineage(assigned.table("household"), clone_of=7)
    attributes = _root_attributes(receipt, registry.plain)
    column = f"{US_CD_PLAN_COLUMN_PREFIX}120th_congress"

    def check(table: pd.DataFrame, attrs: dict, bound=manifest) -> None:
        h5_io._validate_stacked_geography_h5_binding(
            bound,
            table,
            attrs,
            manifest_path=Path("fixture.manifest.json"),
            pool_path=Path("fixture.h5"),
        )

    with pytest.raises(ValueError, match="location root attributes"):
        check(
            household, {**attributes, POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR: "d" * 64}
        )
    with pytest.raises(ValueError, match="location root attributes"):
        check(
            household,
            {
                key: value
                for key, value in attributes.items()
                if key != POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR
            },
        )
    with pytest.raises(ValueError, match="missing block-location column"):
        check(household.drop(columns=column), attributes)
    elsewhere = household.copy()
    elsewhere.loc[0, column] = 3601
    with pytest.raises(ValueError, match="not an integral congressional district"):
        check(elsewhere, attributes)
    drift = household.copy()
    current = int(drift.loc[drift.index[-1], column])
    drift.loc[drift.index[-1], column] = 3601 if current != 3601 else 3602
    with pytest.raises(
        ValueError, match=f"disagree on assigned geography column '{column}'"
    ):
        check(drift, attributes)

    # A registry-free block_v1 pool refuses a stray plan column or digest attr.
    plain_frame, plain_receipt, _ = _assign_plain(pool_tool, registry)
    plain_manifest = _manifest(plain_receipt, registry.plain)
    plain_household = _with_lineage(plain_frame.table("household"), clone_of=7)
    plain_attributes = _root_attributes(plain_receipt, registry.plain)
    check(plain_household, plain_attributes, plain_manifest)
    stray = plain_household.copy()
    stray[column] = household[column].to_numpy()
    with pytest.raises(ValueError, match="carries CD-plan column"):
        check(stray, plain_attributes, plain_manifest)
    with pytest.raises(ValueError, match="location root attributes"):
        check(
            plain_household,
            {
                **plain_attributes,
                POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR: registry.sha256,
            },
            plain_manifest,
        )


def _with_household(frame: Frame, household: pd.DataFrame) -> Frame:
    return Frame(
        {entity: frame.table(entity) for entity in frame.entities}
        | {"household": household},
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


def test_a_carried_plan_column_the_draw_does_not_write_is_refused(
    pool_tool: ModuleType, ladder, registry: SimpleNamespace
) -> None:
    """A plan column the draw does not rewrite would keep stale values beside
    the drawn block: refused at the draw and at every receipt rebind."""

    source = _stacked_frame()
    carried = source.table("household").copy()
    carried[_PLAN_COLUMNS[0]] = 101
    with pytest.raises(ValueError, match=r"carries CD plan column.*does not write"):
        _assign(pool_tool, ladder, frame=_with_household(source, carried))
    # With the registry attached, a plan outside the registry is still stale.
    unknown = source.table("household").copy()
    unknown[f"{US_CD_PLAN_COLUMN_PREFIX}116th_congress"] = 101
    with pytest.raises(ValueError, match=r"116th_congress.*does not write"):
        _assign_registry(pool_tool, registry, frame=_with_household(source, unknown))

    # A registry-free receipt refuses a plan column appearing after the draw.
    assigned, receipt, districts = _assign(pool_tool, ladder)
    household = assigned.table("household").copy()
    household[_PLAN_COLUMNS[2]] = household["congressional_district_geoid"]
    with pytest.raises(ValueError, match="its block-location contract does not derive"):
        pool_tool._validate_stacked_geography_assignment_receipt(
            _with_household(assigned, household),
            receipt,
            target_districts=districts,
            boundary="fixture",
            require_exact_assembled_rows=True,
            configured_contract=receipt["contract"],
        )


def test_every_copy_names_the_packaged_registry_plans(pool_tool: ModuleType) -> None:
    """Differential: the tool's and h5_io's plan-column lists are the packaged
    registry's non-primary plans, so an attachment from the pinned registry is
    exactly what the argv-determined contract derives."""

    from microcosm.build.us_runtime import cd_plan_registry

    provenance = cd_plan_registry.packaged_us_cd_plan_registry_provenance()
    assert provenance["primary_plan"] == PRIMARY_CD_PLAN
    expected = tuple(
        f"{US_CD_PLAN_COLUMN_PREFIX}{plan}"
        for plan in sorted(provenance["plans"])
        if plan != PRIMARY_CD_PLAN
    )
    assert expected == _PLAN_COLUMNS
    assert pool_tool._STACKED_CD_PLAN_REGISTRY_COLUMNS == expected
    assert h5_io._STACKED_CD_PLAN_REGISTRY_COLUMNS == expected
    assert pool_tool._STACKED_CD_PLAN_REGISTRY_SOURCE_REF == (
        h5_io._STACKED_CD_PLAN_REGISTRY_SOURCE_REF
    )


# -- Properties ---------------------------------------------------------------


def test_attaching_the_registry_never_moves_a_drawn_block(
    pool_tool: ModuleType, registry: SimpleNamespace
) -> None:
    """For every seed and household ids: the registry draw is the plain draw
    plus exactly the plan columns, each the registry's district of the drawn
    block in the household's state, and its receipt validates and records the
    loaded file's digest."""

    pytest.importorskip("hypothesis")
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    @settings(
        max_examples=25,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(seed=st.integers(0, 2**32 - 1), id_offset=st.integers(0, 10**6))
    def check(seed: int, id_offset: int) -> None:
        frame = _stacked_frame(id_offset=id_offset)
        plain_frame, plain_receipt, _ = _assign_plain(
            pool_tool, registry, seed=seed, frame=frame
        )
        assigned, receipt, _ = _assign_registry(
            pool_tool, registry, seed=seed, frame=frame
        )
        household = assigned.table("household")
        plain = plain_frame.table("household")
        assert list(household.columns) == [*plain.columns, *_PLAN_COLUMNS]
        pd.testing.assert_frame_equal(household[list(plain.columns)], plain)
        for entity in assigned.entities:
            if entity != "household":
                pd.testing.assert_frame_equal(
                    assigned.table(entity), plain_frame.table(entity)
                )
        assert (
            receipt["assigned_household_geography"]
            == (plain_receipt["assigned_household_geography"])
        )
        state = household["state_fips"].to_numpy()
        for plan, column in registry.record["columns"].items():
            values = household[column].to_numpy(np.int64)
            assert (values == _lookup(registry, household, plan)).all()
            assert (values // 100 == state).all()
        assert (
            receipt["household_location"]["cd_plan_registry"]["sha256"]
            == receipt["contract"]["authorities"]["cd_plan_registry"]["sha256"]
            == file_sha256(registry.path)
        )

    check()


def test_tool_and_consumer_contracts_agree_for_every_input(
    pool_tool: ModuleType,
) -> None:
    """Differential: the producer contract and h5_io's copy agree on every
    seed, digest set and registry choice, and the registry changes exactly
    one authority and the derived plan columns."""

    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    digest = st.text("0123456789abcdef", min_size=64, max_size=64)

    @settings(max_examples=200, deadline=None)
    @given(
        seed=st.integers(0, 2**63 - 1),
        ladder_sha256=digest,
        crosswalk_sha256=digest,
        cps=st.dictionaries(st.integers(2000, 2030), digest, min_size=1, max_size=4),
        registry_sha256=st.none() | digest,
    )
    def check(seed, ladder_sha256, crosswalk_sha256, cps, registry_sha256) -> None:
        tool = pool_tool._stacked_block_location_contract(
            seed,
            1,
            ladder_sha256,
            cps_asec_sha256=cps,
            crosswalk_sha256=crosswalk_sha256,
            cd_plan_registry_sha256=registry_sha256,
        )
        consumer = h5_io.expected_us_stacked_block_location_contract(
            seed=seed,
            clones=1,
            block_ladder_sha256=ladder_sha256,
            crosswalk_sha256=crosswalk_sha256,
            cps_asec_sha256=cps,
            cd_plan_registry_sha256=registry_sha256,
        )
        assert json.loads(json.dumps(tool)) == consumer
        # The receipt round trip rebuilds the same contract.
        assert (
            pool_tool._expected_block_location_contract_from(
                json.loads(json.dumps(tool))
            )
            == tool
        )
        assert pool_tool._contract_cd_plan_registry_sha256(tool) == registry_sha256
        assert pool_tool._contract_cd_plan_columns(tool) == (
            _PLAN_COLUMNS if registry_sha256 is not None else ()
        )

    check()
