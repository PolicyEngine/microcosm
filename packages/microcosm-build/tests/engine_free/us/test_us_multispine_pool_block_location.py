"""The stacked multispine pool's ``--location-rule block_v1`` slice (microcosm#696).

The legacy geography contract, operator order and parser surface are pinned by
the existing pool-tool and h5_io tests; these tests cover the opt-in rule: its
contract and operator order, the CLI refusals, the draw on a small synthetic
stacked frame, and the round trip through h5_io's consumer validators.
"""

from __future__ import annotations

# Test functions take the imported ``pool_tool`` fixture as a parameter.
# ruff: noqa: F811
import copy
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import h5_io
from microcosm.build.us_runtime.block_location import (
    US_BLOCK_LOCATION_RULE_ID,
    load_us_location_ladder,
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
from test_support.microcosm_build.us_multispine_pool_tool import (  # noqa: F401
    _stacked_main_argv,
    pool_tool,
)

_CROSSWALK_SHA256 = "c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec"
_CPS_SHA256 = "a" * 64
_INCOME_YEAR = 2023
# Seed 2 of the synthetic ladder carries the Bronx (36005), which the NYC
# checks of the PUMA-ladder gate need, plus Baldwin, AL (01003).
_LADDER_SEED = 2


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


def _stacked_frame(*, asec_source_year: int = _INCOME_YEAR) -> Frame:
    n = len(_HOUSEHOLDS)
    ids = np.arange(1, n + 1, dtype=np.int64)
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
            "source_household_id": np.where(is_asec, 900.0 + ids, np.nan),
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
    return {
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


@pytest.mark.parametrize("seed", [0, 1, 578, 2**31])
def test_tool_contract_equals_the_consumer_copy(
    pool_tool: ModuleType, seed: int
) -> None:
    cps = {2022: "1" * 64, 2024: "2" * 64}
    tool = pool_tool._stacked_block_location_contract(
        seed, 1, "f" * 64, cps_asec_sha256=cps
    )
    consumer = h5_io.expected_us_stacked_block_location_contract(
        seed=seed,
        clones=1,
        block_ladder_sha256="f" * 64,
        crosswalk_sha256=_CROSSWALK_SHA256,
        cps_asec_sha256=cps,
    )
    assert json.loads(json.dumps(tool)) == consumer
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
    assert set(attributes) == set(h5_io.US_STACKED_BLOCK_LOCATION_ROOT_ATTRIBUTES)
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
