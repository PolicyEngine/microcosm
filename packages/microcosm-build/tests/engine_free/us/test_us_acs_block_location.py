"""The ACS local-release line under ``--location-rule block_v1`` (microcosm#696).

The pooled ASEC-by-PUF + ACS staging frame locates every household on one
2020 census block:

- ACS rows draw a block within their observed 2020 PUMA;
- donor rows that already carry a ladder block keep it (never redrawn) and
  re-derive every geography from it;
- donor rows without a block draw within their state, counted under an
  explicit reason;

and every geography column then equals the ladder's lookup of the block, so
the block-location release gate passes. The legacy default is untouched
(the existing ACS multispine tests pin it); these tests cover the new
branch, its refusals and its records.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build import FitWeightRecord, GateResult
from microcosm.build.us_runtime import acs_multispine
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.base_pool import (
    ACS_POOL_BLOCK_LOCATION_METADATA_KEY,
    ACS_POOL_LOCATION_CLONES_REFUSAL,
    DONOR_BLOCK_PRESERVED,
    DONOR_BLOCK_STATE_DRAWN,
    preflight_pooled_ladder_geography,
    spine_column,
    with_optional_acs_spine,
)
from microcosm.build.us_runtime.block_location import (
    SOURCE_GEOGRAPHY_PUMA,
    SOURCE_GEOGRAPHY_STATE,
    US_BLOCK_LOCATION_RULE_ID,
    US_LOCATION_RULE_BLOCK_V1,
    UsLocationLadder,
    derive_us_block_geography,
    load_us_location_ladder,
    location_geography_columns,
    us_block_location_gate,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_block_location import (
    synthetic_blocks,
    write_location_ladder,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
requires_pytables = pytest.mark.skipif(
    importlib.util.find_spec("tables") is None,
    reason="requires pytables (the build environment)",
)

STATES = (1, 2, 36)


@pytest.fixture(scope="module")
def ladder(tmp_path_factory: pytest.TempPathFactory) -> UsLocationLadder:
    path = tmp_path_factory.mktemp("ladder") / "blocks.npz"
    write_location_ladder(path, synthetic_blocks(3, states=STATES))
    return load_us_location_ladder(path)


@pytest.fixture(scope="module")
def ladder_without_puma(tmp_path_factory: pytest.TempPathFactory) -> UsLocationLadder:
    path = tmp_path_factory.mktemp("ladder") / "blocks_no_puma.npz"
    write_location_ladder(path, synthetic_blocks(3, states=STATES), with_puma=False)
    return load_us_location_ladder(path)


def _frame(
    household: pd.DataFrame, *, weights: np.ndarray, spine: str | None = None
) -> Frame:
    ids = household["household_id"].to_numpy(dtype=np.int64)
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
        "household": household,
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    if spine is not None:
        for entity, table in tables.items():
            table[spine_column(entity)] = spine
    return Frame(tables, US_SCHEMA, {"household": Weights(weights, WeightKind.DESIGN)})


def _ladder_block(ladder: UsLocationLadder, state: int, which: int = 0) -> str:
    in_state = ladder.block_geoid[ladder.block_geoid // 10**13 == state]
    return f"{int(in_state[which % len(in_state)]):015d}"


def _donor_base(
    ladder: UsLocationLadder, *, blocks: str = "all", legacy_columns: bool = True
) -> Frame:
    """Six donor households; ``blocks`` is ``all``, ``some`` or ``none``."""

    states = [1, 1, 2, 36, 36, 2]
    household = pd.DataFrame(
        {"household_id": np.arange(1, 7, dtype=np.int64), "state_fips": states}
    )
    if blocks != "none":
        geoids = [_ladder_block(ladder, state, i) for i, state in enumerate(states)]
        if blocks == "some":
            geoids[1] = ""
            geoids[4] = None
        household["block_geoid"] = geoids
        if legacy_columns:
            # What a legacy base-h5 donor carries next to its block.
            index = np.searchsorted(
                ladder.block_geoid,
                [int(g) if g else ladder.block_geoid[0] for g in geoids],
            )
            derived = derive_us_block_geography(ladder, index)
            for column in ("tract_geoid", "county_fips"):
                household[column] = derived[column]
            household["congressional_district_geoid"] = derived[
                "congressional_district_geoid"
            ]
    household["puma"] = pd.Series([None] * len(household), dtype=object)
    return _frame(household, weights=np.full(6, 50.0), spine="asec_puf")


def _acs(ladder: UsLocationLadder, *, n: int = 9) -> Frame:
    assert ladder.puma is not None
    pumas = np.unique(ladder.puma)
    chosen = [int(pumas[i % len(pumas)]) for i in range(n)]
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1, dtype=np.int64),
            "state_fips": [puma // 10**5 for puma in chosen],
            "puma": [f"{puma:07d}" for puma in chosen],
        }
    )
    return _frame(household, weights=np.full(n, 10.0), spine=ACS_2024_1YR_SPINE)


def _pool(ladder: UsLocationLadder, base: Frame, acs: Frame, **options) -> Frame:
    return with_optional_acs_spine(
        base,
        acs,
        location_rule=US_LOCATION_RULE_BLOCK_V1,
        block_ladder=ladder,
        max_peak_bytes=None,
        **options,
    )


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _record(pooled: Frame) -> dict:
    return _thaw(pooled.metadata[ACS_POOL_BLOCK_LOCATION_METADATA_KEY])


def _block_index(ladder: UsLocationLadder, blocks: pd.Series) -> np.ndarray:
    return np.searchsorted(ladder.block_geoid, blocks.astype(np.int64).to_numpy())


def test__block_v1__acs_blocks_lie_in_observed_puma(ladder: UsLocationLadder) -> None:
    acs = _acs(ladder)
    pooled = _pool(ladder, _donor_base(ladder), acs)
    household = pooled.table("household")
    acs_rows = household[spine_column("household")].eq(ACS_2024_1YR_SPINE)
    observed = acs.table("household")["puma"].astype(np.int64).to_numpy()
    index = _block_index(ladder, household.loc[acs_rows, "block_geoid"])
    assert ladder.puma is not None
    np.testing.assert_array_equal(ladder.puma[index], observed)
    np.testing.assert_array_equal(
        household.loc[acs_rows, "puma"].astype(np.int64).to_numpy(), observed
    )


def test__block_v1__donor_blocks_are_preserved_not_redrawn(
    ladder: UsLocationLadder,
) -> None:
    base = _donor_base(ladder)
    pooled = _pool(ladder, base, _acs(ladder))
    household = pooled.table("household")
    donors = household[spine_column("household")].eq("asec_puf")
    assert (
        household.loc[donors, "block_geoid"].tolist()
        == base.table("household")["block_geoid"].tolist()
    )
    record = _record(pooled)
    assert record["donor_blocks"]["mode"] == DONOR_BLOCK_PRESERVED
    assert record["donor_blocks"][DONOR_BLOCK_PRESERVED] == 6
    assert record["donor_blocks"][DONOR_BLOCK_STATE_DRAWN] == 0
    assert record["donor_blocks"]["state_drawn_reason"] is None
    # The fixture's legacy columns came from this ladder: nothing changed.
    assert record["donor_blocks"]["rederived_columns_changed"] == {}


def test__block_v1__every_geography_derives_from_the_block_and_gate_passes(
    ladder: UsLocationLadder,
) -> None:
    pooled = _pool(ladder, _donor_base(ladder, blocks="some"), _acs(ladder))
    household = pooled.table("household")
    gate = us_block_location_gate(household, ladder)
    assert gate.passed, gate.failures
    for column in location_geography_columns(ladder):
        assert household[column].notna().all(), column
    assert household["county_fips"].map(len).eq(5).all()
    assert household["puma"].map(len).eq(7).all()
    assert (
        household["block_geoid"].str[:2].astype(int).to_numpy()
        == household["state_fips"].to_numpy()
    ).all()


def test__block_v1__donors_without_block_draw_within_state_and_are_counted(
    ladder: UsLocationLadder,
) -> None:
    base = _donor_base(ladder, blocks="some")
    pooled = _pool(ladder, base, _acs(ladder))
    record = _record(pooled)
    donor_blocks = record["donor_blocks"]
    assert donor_blocks["mode"] == f"{DONOR_BLOCK_PRESERVED}_with_state_draws"
    assert donor_blocks[DONOR_BLOCK_PRESERVED] == 4
    assert donor_blocks[DONOR_BLOCK_STATE_DRAWN] == 2
    assert "no block_geoid" in donor_blocks["state_drawn_reason"]
    assert record["source_geography_households"][SOURCE_GEOGRAPHY_STATE] == 2
    assert record["source_geography_households"][SOURCE_GEOGRAPHY_PUMA] == 9
    household = pooled.table("household")
    donors = household.loc[household[spine_column("household")].eq("asec_puf")]
    kept = base.table("household")["block_geoid"]
    for position in (0, 2, 3, 5):
        assert donors["block_geoid"].iloc[position] == kept.iloc[position]


def test__block_v1__donor_without_any_block_column_draws_within_state(
    ladder: UsLocationLadder,
) -> None:
    pooled = _pool(ladder, _donor_base(ladder, blocks="none"), _acs(ladder))
    record = _record(pooled)
    assert record["donor_blocks"]["mode"] == DONOR_BLOCK_STATE_DRAWN
    assert record["donor_blocks"][DONOR_BLOCK_STATE_DRAWN] == 6
    assert us_block_location_gate(pooled.table("household"), ladder).passed


def test__block_v1__record_carries_rule_seed_and_ladder(
    ladder: UsLocationLadder,
) -> None:
    pooled = _pool(ladder, _donor_base(ladder), _acs(ladder), location_seed=17)
    record = _record(pooled)
    assert record["rule"] == US_BLOCK_LOCATION_RULE_ID
    assert record["location_rule"] == US_LOCATION_RULE_BLOCK_V1
    assert record["seed"] == 17
    assert record["clones"] == 1
    assert record["block_ladder"]["sha256"] == ladder.sha256
    assert record["block_ladder"]["carries_puma"] is True
    assert record["households"] == 15
    assert record["spine_households"] == {ACS_2024_1YR_SPINE: 9, "asec_puf": 6}
    assert set(record["candidate_set_rule"]) == {ACS_2024_1YR_SPINE, "asec_puf"}


def test__block_v1__is_deterministic_and_seed_sensitive(
    ladder: UsLocationLadder,
) -> None:
    base = _donor_base(ladder, blocks="none")
    acs = _acs(ladder, n=40)
    first = _pool(ladder, base, acs, location_seed=3).table("household")
    again = _pool(ladder, base, acs, location_seed=3).table("household")
    other = _pool(ladder, base, acs, location_seed=4).table("household")
    pd.testing.assert_frame_equal(first, again)
    assert not first["block_geoid"].equals(other["block_geoid"])


def test__block_v1__conserves_household_mass(ladder: UsLocationLadder) -> None:
    base = _donor_base(ladder)
    pooled = _pool(ladder, base, _acs(ladder))
    assert pooled.weights_for("household").total == pytest.approx(
        base.weights_for("household").total, rel=1e-12
    )


def test__block_v1__refuses_more_than_one_location_clone(
    ladder: UsLocationLadder,
) -> None:
    with pytest.raises(ValueError) as caught:
        _pool(ladder, _donor_base(ladder), _acs(ladder), location_clones=2)
    message = str(caught.value)
    assert message == ACS_POOL_LOCATION_CLONES_REFUSAL
    assert "_require_stored_inputs" in message
    assert "US_STORED_NON_VARIABLE_COLUMNS" in message
    assert "_assign_pooled_block_location" in message


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"location_clones": 0}, "location_clones must be >= 1"),
        ({"location_seed": -1}, "location_seed must be non-negative"),
        ({"geography_seed": 5}, "geography_seed seeds only the legacy"),
        ({"location_seed": 1.5}, "location_seed must be an integer"),
    ],
)
def test__block_v1__refuses_invalid_options(
    ladder: UsLocationLadder, options: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _pool(ladder, _donor_base(ladder), _acs(ladder), **options)


def test__block_v1__refuses_a_puma_ladder_it_would_ignore(
    ladder: UsLocationLadder,
) -> None:
    with pytest.raises(ValueError, match="Refusing puma_ladder"):
        _pool(
            ladder,
            _donor_base(ladder),
            _acs(ladder),
            puma_ladder=_legacy_puma_ladder(),
        )


def test__block_v1__refuses_missing_or_puma_less_ladder(
    ladder: UsLocationLadder, ladder_without_puma: UsLocationLadder
) -> None:
    base, acs = _donor_base(ladder), _acs(ladder)
    with pytest.raises(ValueError, match="requires a block_ladder"):
        with_optional_acs_spine(base, acs, location_rule=US_LOCATION_RULE_BLOCK_V1)
    with pytest.raises(ValueError, match="per-block 'puma'"):
        _pool(ladder_without_puma, base, acs)


@pytest.mark.parametrize(
    "options",
    [
        {"location_seed": 1},
        {"location_clones": 2},
        {"location_rule": "nope"},
    ],
)
def test__legacy__refuses_block_only_options(
    ladder: UsLocationLadder, options: dict
) -> None:
    with pytest.raises(ValueError):
        with_optional_acs_spine(_donor_base(ladder), _acs(ladder), **options)


def test__legacy__refuses_a_block_ladder(ladder: UsLocationLadder) -> None:
    with pytest.raises(ValueError, match="only used by location_rule='block_v1'"):
        with_optional_acs_spine(_donor_base(ladder), _acs(ladder), block_ladder=ladder)


def test__legacy__no_acs_identity_is_unchanged(ladder: UsLocationLadder) -> None:
    base = _donor_base(ladder)
    assert with_optional_acs_spine(base, None) is base


def test__block_v1__refuses_unknown_donor_block(ladder: UsLocationLadder) -> None:
    base = _donor_base(ladder, legacy_columns=False)
    household = base.table("household").copy()
    household.loc[0, "block_geoid"] = "010019999999999"
    bad = _frame(household, weights=np.full(6, 50.0), spine="asec_puf")
    with pytest.raises(ValueError, match="absent from the block ladder"):
        _pool(ladder, bad, _acs(ladder))


def test__block_v1__refuses_donor_block_in_another_state(
    ladder: UsLocationLadder,
) -> None:
    base = _donor_base(ladder, legacy_columns=False)
    household = base.table("household").copy()
    household.loc[0, "block_geoid"] = _ladder_block(ladder, 36)
    bad = _frame(household, weights=np.full(6, 50.0), spine="asec_puf")
    with pytest.raises(ValueError, match="state prefixes disagree"):
        _pool(ladder, bad, _acs(ladder))


def test__block_v1__refuses_acs_puma_outside_the_ladder(
    ladder: UsLocationLadder,
) -> None:
    acs = _acs(ladder)
    household = acs.table("household").copy()
    household.loc[0, "puma"] = "0199999"
    household = household.drop(columns=[spine_column("household")])
    bad = _frame(
        household, weights=np.full(len(household), 10.0), spine=ACS_2024_1YR_SPINE
    )
    with pytest.raises(ValueError, match="absent from the block ladder's PUMA layer"):
        _pool(ladder, _donor_base(ladder), bad)


def test__block_v1__integer_donor_blocks_keep_their_leading_zero(
    ladder: UsLocationLadder,
) -> None:
    base = _donor_base(ladder, legacy_columns=False)
    household = base.table("household").copy()
    household["block_geoid"] = household["block_geoid"].astype(np.int64)
    as_int = _frame(household, weights=np.full(6, 50.0), spine="asec_puf")
    pooled = _pool(ladder, as_int, _acs(ladder))
    donors = pooled.table("household").iloc[:6]
    assert (
        donors["block_geoid"].tolist()
        == base.table("household")["block_geoid"].tolist()
    )


def test__block_v1__records_donor_columns_the_ladder_rederives(
    ladder: UsLocationLadder,
) -> None:
    base = _donor_base(ladder)
    household = base.table("household").copy()
    household.loc[0, "county_fips"] = "01999"
    stale = _frame(household, weights=np.full(6, 50.0), spine="asec_puf")
    pooled = _pool(ladder, stale, _acs(ladder))
    record = _record(pooled)
    assert record["donor_blocks"]["rederived_columns_changed"] == {"county_fips": 1}
    assert us_block_location_gate(pooled.table("household"), ladder).passed


def test__preflight__block_v1_reports_donor_mode(ladder: UsLocationLadder) -> None:
    acs = _acs(ladder).table("household")
    for blocks, expected in (
        ("all", DONOR_BLOCK_PRESERVED),
        ("none", DONOR_BLOCK_STATE_DRAWN),
        ("some", f"{DONOR_BLOCK_PRESERVED}_with_state_draws"),
    ):
        base = _donor_base(ladder, blocks=blocks).table("household")
        assert (
            preflight_pooled_ladder_geography(
                base,
                acs,
                None,
                location_rule=US_LOCATION_RULE_BLOCK_V1,
                block_ladder=ladder,
            )
            == expected
        )


def test__preflight__block_v1_fails_before_transfer_on_bad_inputs(
    ladder: UsLocationLadder,
) -> None:
    acs = _acs(ladder).table("household").copy()
    base = _donor_base(ladder).table("household")
    acs.loc[0, "puma"] = "0199999"
    with pytest.raises(ValueError, match="PUMA layer"):
        preflight_pooled_ladder_geography(
            base,
            acs,
            None,
            location_rule=US_LOCATION_RULE_BLOCK_V1,
            block_ladder=ladder,
        )
    base = base.copy()
    base.loc[0, "block_geoid"] = "not-a-block"
    with pytest.raises(ValueError, match="15-digit"):
        preflight_pooled_ladder_geography(
            base,
            _acs(ladder).table("household"),
            None,
            location_rule=US_LOCATION_RULE_BLOCK_V1,
            block_ladder=ladder,
        )


def test__preflight__legacy_refuses_block_ladder(ladder: UsLocationLadder) -> None:
    with pytest.raises(ValueError, match="only used by location_rule='block_v1'"):
        preflight_pooled_ladder_geography(
            _donor_base(ladder).table("household"),
            _acs(ladder).table("household"),
            None,
            block_ladder=ladder,
        )


# ---------------------------------------------------------------------------
# build_optional_acs_multispine threading
# ---------------------------------------------------------------------------


def test__multispine__block_v1_threads_through_and_records_location(
    monkeypatch: pytest.MonkeyPatch, ladder: UsLocationLadder
) -> None:
    base = _donor_base(ladder, blocks="some")
    acs = _acs(ladder)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        acs_multispine,
        "build_acs_pums_unit_frame",
        lambda source, *, chunksize: (acs, {"spine": ACS_2024_1YR_SPINE}),
    )
    monkeypatch.setattr(
        acs_multispine,
        "map_acs_native_inputs",
        lambda raw: type("Mapped", (), {"frame": raw, "native_inputs": {}})(),
    )

    def fake_transfer(recipient, donor, **kwargs):
        captured["transfer"] = kwargs
        return type(
            "Transfer",
            (),
            {
                "frame": recipient,
                "imputed_inputs": (),
                "fit_records": (),
                "deferred_inputs": (
                    "block_geoid",
                    "congressional_district_geoid",
                    "county_fips",
                    "tract_geoid",
                ),
                "resolved_donor_channel": "asec",
            },
        )()

    monkeypatch.setattr(acs_multispine, "transfer_acs_inputs", fake_transfer)
    result = acs_multispine.build_optional_acs_multispine(
        base,
        object(),  # type: ignore[arg-type]
        location_rule=US_LOCATION_RULE_BLOCK_V1,
        block_ladder=ladder,
        location_seed=9,
    )
    provenance = result.provenance
    json.dumps(provenance, allow_nan=False)
    assert provenance["deferred_inputs"] == []
    location = provenance["household_location"]
    assert location["rule"] == US_BLOCK_LOCATION_RULE_ID
    assert location["seed"] == 9
    assert location["donor_blocks"][DONOR_BLOCK_STATE_DRAWN] == 2
    geography = provenance["geography_ladder"]
    assert geography["applied"] is True
    assert geography["location_rule"] == US_LOCATION_RULE_BLOCK_V1
    assert geography["seed"] == 9
    assert geography["unresolved_sub_puma_inputs"] == []
    assert geography["block_ladder_sha256"] == ladder.sha256
    assert geography["resolved_model_inputs"] == list(
        location_geography_columns(ladder)
    )
    assert us_block_location_gate(result.frame.table("household"), ladder).passed


def test__multispine__refuses_invalid_location_options_before_loading(
    monkeypatch: pytest.MonkeyPatch, ladder: UsLocationLadder
) -> None:
    def must_not_load(*_args, **_kwargs):
        raise AssertionError("location options must be validated first")

    monkeypatch.setattr(acs_multispine, "build_acs_pums_unit_frame", must_not_load)
    for options in (
        {"location_seed": 3},
        {"location_rule": US_LOCATION_RULE_BLOCK_V1},
        {
            "location_rule": US_LOCATION_RULE_BLOCK_V1,
            "block_ladder": ladder,
            "location_clones": 2,
        },
    ):
        with pytest.raises(ValueError):
            acs_multispine.build_optional_acs_multispine(
                _donor_base(ladder),
                object(),  # type: ignore[arg-type]
                **options,
            )


# ---------------------------------------------------------------------------
# tools/_legacy/build_us_acs_multispine_base.py
# ---------------------------------------------------------------------------


def _load_legacy_builder():
    path = (
        _TEST_PATHS.repository
        / "tools"
        / "_legacy"
        / ("build_us_acs_multispine_base.py")
    )
    spec = importlib.util.spec_from_file_location(
        "legacy_build_us_acs_multispine_base_block_v1", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_release_tool():
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release_block_v1", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test__legacy_tool__parser_defaults_are_legacy() -> None:
    builder = _load_legacy_builder()
    args = builder._parse_args(["--base-h5", "a.h5", "--out-h5", "b.h5"])
    assert args.location_rule == "legacy"
    assert args.location_seed == 0
    assert args.location_clones == 1
    assert args.block_ladder is None
    builder._validate_location_arguments(args)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--location-seed", "1"], "apply only to --location-rule block_v1"),
        (["--location-clones", "2"], "apply only to --location-rule block_v1"),
        (["--block-ladder", "x.npz"], "only used by --location-rule block_v1"),
        (["--location-rule", "block_v1"], "requires --block-ladder"),
        (
            ["--location-rule", "block_v1", "--block-ladder", "x.npz"]
            + ["--location-clones", "2"],
            "_require_stored_inputs",
        ),
        (
            ["--location-rule", "block_v1", "--block-ladder", "x.npz"]
            + ["--geography-seed", "3"],
            "--geography-seed seeds only the legacy",
        ),
    ],
)
def test__legacy_tool__refuses_inconsistent_location_flags(
    extra: list[str], match: str
) -> None:
    builder = _load_legacy_builder()
    args = builder._parse_args(["--base-h5", "a.h5", "--out-h5", "b.h5", *extra])
    with pytest.raises(SystemExit, match=match):
        builder._validate_location_arguments(args)


def test__legacy_tool__refuses_zero_location_clones_at_parse() -> None:
    builder = _load_legacy_builder()
    with pytest.raises(SystemExit):
        builder._parse_args(
            ["--base-h5", "a.h5", "--out-h5", "b.h5", "--location-clones", "0"]
        )


def _with_person_inputs(frame: Frame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["takes_up_snap_if_eligible"] = True
    return Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


def _puma_ladder_stub(builder, ladder: UsLocationLadder):
    """The PUMA ladder of ``ladder``'s own blocks (the same 2020 geography)."""

    assert ladder.puma is not None
    frame = pd.DataFrame(
        {
            "puma": ladder.puma,
            "cd": ladder.congressional_district_plans[
                ladder.primary_congressional_district_plan
            ],
            "county": ladder.block_geoid // 10**10,
            "tract": ladder.block_geoid // 10**4,
            "population": ladder.population.astype(np.float64),
        }
    )

    def overlap(column: str):
        grouped = frame.groupby(["puma", column], sort=True)["population"].sum()
        return (
            grouped.index.get_level_values(0).to_numpy(np.int64),
            grouped.index.get_level_values(1).to_numpy(np.int64),
            grouped.to_numpy(np.float64),
        )

    anchor = frame.groupby("puma", sort=True)["population"].sum()
    cd_puma, cd, cd_population = overlap("cd")
    county_puma, county, county_population = overlap("county")
    tract_puma, tract, tract_population = overlap("tract")
    return builder.UsPumaLadder(
        puma=anchor.index.to_numpy(np.int64),
        puma_population=anchor.to_numpy(np.float64),
        cd_overlap_puma=cd_puma,
        cd_overlap_cd=cd,
        cd_overlap_population=cd_population,
        county_overlap_puma=county_puma,
        county_overlap_county=county.astype(np.int32),
        county_overlap_population=county_population,
        tract_overlap_puma=tract_puma,
        tract_overlap_tract=tract,
        tract_overlap_population=tract_population,
        metadata={
            "schema_version": 1,
            "kind": "us_puma_ladder",
            "puma_vintage": "2020_puma",
            "sampling_basis": "population",
            "layers": {
                "congressional_district": {
                    "vintage": ladder.primary_congressional_district_plan
                },
                "county": {"vintage": "2020_census"},
                "tract": {"vintage": "2020_census"},
            },
        },
    )


def _run_block_v1_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    tamper: bool = False,
    seed: int = 5,
    puma_ladder=None,
):
    builder = _load_legacy_builder()
    ladder_path = write_location_ladder(
        tmp_path / "blocks.npz", synthetic_blocks(3, states=STATES)
    )
    ladder = load_us_location_ladder(ladder_path)
    base = _with_person_inputs(_donor_base(ladder, blocks="some"))
    acs_frame = _acs(ladder)
    acs_tables = {e: acs_frame.table(e).copy() for e in acs_frame.entities}
    acs_tables["person"]["takes_up_snap_if_eligible"] = False
    acs_tables["household"]["TYPEHUGQ"] = 1.0
    acs_frame = Frame(
        acs_tables, US_SCHEMA, {"household": acs_frame.weights_for("household")}
    )
    combined = with_optional_acs_spine(
        base,
        acs_frame,
        location_rule=US_LOCATION_RULE_BLOCK_V1,
        block_ladder=ladder,
        location_seed=seed,
        max_peak_bytes=None,
    )
    if tamper:
        tables = {e: combined.table(e).copy() for e in combined.entities}
        tables["household"].loc[0, "county_fips"] = "99999"
        combined = Frame(
            tables,
            US_SCHEMA,
            {"household": combined.weights_for("household")},
            combined.strata,
            metadata=combined.metadata,
        )
    if puma_ladder is None:
        puma_ladder = _puma_ladder_stub(builder, ladder)
    donor_mode = _record(combined)["donor_blocks"]["mode"]
    provenance = {
        "enabled": True,
        "deferred_inputs": [],
        "household_location": _record(combined),
        "geography_ladder": acs_multispine._block_geography_provenance(
            combined,
            ladder,
            puma_ladder,
            location_seed=seed,
            donor_geography=donor_mode,
            expected_congressional_district_vintage="119th_congress",
        ),
        "imputed_inputs": [
            {
                "column": "takes_up_snap_if_eligible",
                "family": "benefit_participation",
                "unmodeled_recipient_rows": 0,
            }
        ],
    }
    base_h5 = tmp_path / "dense.h5"
    base_h5.write_bytes(b"dense-base")
    puma_ladder_path = tmp_path / "us_puma_ladder_2020.npz"
    puma_ladder_path.write_bytes(b"puma-ladder")
    manifest_path = tmp_path / "acs_sources.json"
    manifest_path.write_text("{}", encoding="utf-8")
    output_h5 = tmp_path / "combined.h5"
    summary_path = tmp_path / "combined.summary.json"
    captured: dict[str, object] = {}
    transfer_plan = {
        "person": {"benefit_participation": ("takes_up_snap_if_eligible",)}
    }

    def fake_build(actual_base, actual_source, **kwargs):
        captured["build"] = kwargs
        return builder.AcsMultispineResult(
            frame=combined,
            fit_records=(FitWeightRecord("acs_transfer:person:benefits", "design"),),
            provenance=provenance,
        )

    def fake_write(frame, path, **kwargs):
        captured["write"] = kwargs
        path.write_bytes(b"combined-output")

    manifest = type("Manifest", (), {})()
    monkeypatch.setattr(builder, "_load_base_frame", lambda path: base)
    monkeypatch.setattr(
        builder,
        "_base_location_rule",
        lambda path: {
            "location_rule": "block_v1",
            "recorded": True,
            "attributes": {"populace_location_rule": "block_v1"},
        },
    )
    monkeypatch.setattr(builder, "_require_dense_donor_coverage", lambda *a, **k: None)
    monkeypatch.setattr(
        builder, "acs_local_transfer_target_families", lambda: transfer_plan
    )
    monkeypatch.setattr(
        builder, "declared_acs_transfer_target_families", lambda: transfer_plan
    )
    monkeypatch.setattr(
        builder.acs_sources, "load_acs_source_manifest", lambda p: manifest
    )
    monkeypatch.setattr(
        builder.acs_sources,
        "fetch_acs_pums_sources",
        lambda cache, *, manifest: builder.AcsPumsSource(
            cache / "h.zip", cache / "p.zip"
        ),
    )
    monkeypatch.setattr(builder, "_source_provenance", lambda *a, **k: {})
    monkeypatch.setattr(builder, "build_optional_acs_multispine", fake_build)
    monkeypatch.setattr(builder, "load_us_puma_ladder", lambda path: puma_ladder)
    monkeypatch.setattr(builder, "_engine_input_null_audit", lambda frame: [])
    monkeypatch.setattr(builder, "_preflight_staging_export", lambda frame: 1)
    monkeypatch.setattr(builder, "_write_dataset", fake_write)
    monkeypatch.setattr(
        builder,
        "acs_local_hours_signal_gate",
        lambda frame, *, source_null_audit: GateResult(
            name="acs_local_hours_signal", passed=True, failures=(), details={}
        ),
    )
    arguments = [
        "--base-h5",
        str(base_h5),
        "--out-h5",
        str(output_h5),
        "--summary",
        str(summary_path),
        "--source-manifest",
        str(manifest_path),
        "--inputs-dir",
        str(tmp_path / "inputs"),
        "--puma-ladder",
        str(puma_ladder_path),
        "--location-rule",
        "block_v1",
        "--location-seed",
        str(seed),
        "--block-ladder",
        str(ladder_path),
    ]
    exit_code = builder.main(arguments)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return builder, ladder, exit_code, captured, summary


def test__legacy_tool__block_v1_main_records_rule_seed_gate_and_attributes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    builder, ladder, exit_code, captured, summary = _run_block_v1_main(
        monkeypatch, tmp_path
    )
    assert exit_code == 0
    build_options = dict(captured["build"])
    block_ladder = build_options.pop("block_ladder")
    assert block_ladder.sha256 == ladder.sha256
    build_options.pop("puma_ladder")
    build_options.pop("hours_donor_factory")
    build_options.pop("target_families")
    assert build_options == {
        "chunksize": builder.DEFAULT_CHUNKSIZE,
        "acs_share": 0.5,
        "hours_under15_policy": None,
        "donor_channel": builder.ACS_DONOR_CHANNEL_AUTO,
        "seed": 0,
        "n_estimators": 32,
        "max_targets_per_fit": 8,
        "geography_seed": 0,
        "location_rule": "block_v1",
        "location_seed": 5,
        "location_clones": 1,
    }

    location = summary["household_location"]
    assert location["rule"] == US_BLOCK_LOCATION_RULE_ID
    assert location["location_rule"] == "block_v1"
    assert location["seed"] == 5
    assert location["clones"] == 1
    assert location["block_ladder"]["sha256"] == ladder.sha256
    assert location["block_ladder_path"] == str((tmp_path / "blocks.npz").resolve())
    assert (
        location["puma_ladder"]["sha256"] == hashlib.sha256(b"puma-ladder").hexdigest()
    )
    assert location["donor_base_location"]["location_rule"] == "block_v1"
    assert location["gate"]["passed"] is True
    agreement = location["ladder_agreement"]
    assert agreement["congressional_district_plan"] == "119th_congress"
    for layer in ("pumas", "states", "congressional_districts"):
        assert agreement[layer]["only_in_block_ladder"] == []
        assert agreement[layer]["only_in_puma_ladder"] == []
        assert agreement[layer]["max_abs_population_difference"] == 0.0
    assert location["donor_blocks"][DONOR_BLOCK_STATE_DRAWN] == 2

    orchestration = summary["orchestration"]
    assert orchestration["location_rule"] == "block_v1"
    assert orchestration["location_seed"] == 5
    assert orchestration["location_clones"] == 1
    assert orchestration["block_ladder_sha256"] == ladder.sha256
    assert orchestration["geography_seed"] == 0

    geography = summary["geography_ladder"]
    assert geography["sha256"] == hashlib.sha256(b"puma-ladder").hexdigest()
    assert geography["block_ladder"]["sha256"] == ladder.sha256
    assert geography["seed"] == 5
    assert geography["location_rule"] == "block_v1"

    limitation = summary["reviewed_limitations"][2]
    assert limitation["id"] == "sub_puma_geographic_precision"
    assert limitation["location_rule"] == "block_v1"
    assert limitation["unavailable_exact_geography"] == []
    assert limitation["assignment_seed"] == 5
    assert "Do not synthesize" not in limitation["treatment"]
    assert "block/tract/place/SLD/CBSA" in limitation["treatment"]

    attributes = captured["write"]["root_attributes"]
    assert attributes["populace_location_rule"] == "block_v1"
    assert attributes["populace_location_seed"] == "5"
    assert attributes["populace_location_clones"] == "1"
    assert attributes["populace_block_ladder_sha256"] == ladder.sha256
    assert json.loads(attributes["populace_block_ladder_vintages"]) == (
        ladder.layer_vintages
    )
    assert attributes["populace_puma_ladder_artifact_sha256"] == (
        hashlib.sha256(b"puma-ladder").hexdigest()
    )
    assert json.loads(attributes["populace_puma_ladder_vintages"]) == (
        _puma_ladder_stub(builder, ladder).layer_vintages
    )


def test__legacy_tool__block_v1_gate_failure_is_hard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="Block location gate failed.*county_fips"):
        _run_block_v1_main(monkeypatch, tmp_path, tamper=True)
    assert not (tmp_path / "combined.h5").exists()


def test__legacy_tool__refuses_a_puma_ladder_of_another_geography(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="not the same 2020 geography"):
        _run_block_v1_main(monkeypatch, tmp_path, puma_ladder=_legacy_puma_ladder())
    assert not (tmp_path / "combined.h5").exists()


def test__legacy_tool__refuses_a_puma_ladder_of_another_district_vintage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ladder: UsLocationLadder
) -> None:
    builder = _load_legacy_builder()
    stub = _puma_ladder_stub(builder, ladder)
    layers = dict(stub.metadata["layers"])
    layers["congressional_district"] = {"vintage": "118th_congress"}
    other = builder.UsPumaLadder(
        **{
            **{name: getattr(stub, name) for name in stub.__dataclass_fields__},
            "metadata": {**stub.metadata, "layers": layers},
        }
    )
    with pytest.raises(SystemExit, match="'118th_congress'"):
        _run_block_v1_main(monkeypatch, tmp_path, puma_ladder=other)


@requires_pytables
def test__legacy_tool__staging_root_attributes_round_trip(
    tmp_path: Path, ladder: UsLocationLadder
) -> None:
    builder = _load_legacy_builder()
    pooled = _pool(ladder, _donor_base(ladder, blocks="some"), _acs(ladder))
    attributes = builder.location_root_attributes(
        location_rule="block_v1",
        location_seed=4,
        location_clones=1,
        block_ladder_sha256=str(ladder.sha256),
        block_ladder_vintages=ladder.layer_vintages,
        puma_ladder_sha256="0" * 64,
        puma_ladder_vintages={"congressional_district": "119th_congress"},
    )
    path = tmp_path / "staging.h5"
    builder._write_dataset(pooled, path, period=2024, root_attributes=attributes)
    with pd.HDFStore(path, mode="r") as store:
        stored = store.get_node("/")._v_attrs
        assert {
            key: builder._attribute_text(stored[key]) for key in attributes
        } == attributes
    assert builder._base_location_rule(path) == {
        "location_rule": "block_v1",
        "recorded": True,
        "attributes": {
            "populace_location_rule": "block_v1",
            "populace_location_seed": "4",
            "populace_location_clones": "1",
        },
    }
    legacy = tmp_path / "legacy.h5"
    builder._write_dataset(pooled, legacy, period=2024)
    assert builder._base_location_rule(legacy) == {
        "location_rule": "legacy",
        "recorded": False,
        "attributes": {},
    }


# ---------------------------------------------------------------------------
# tools/build_us_acs_local_release.py
# ---------------------------------------------------------------------------


def _block_v1_staging_summary(ladder_path: Path, sha256: str) -> dict:
    return {
        "orchestration": {
            "max_households": None,
            "seed": 0,
            "geography_seed": 0,
            "location_rule": "block_v1",
            "location_seed": 7,
            "location_clones": 1,
            "block_ladder_sha256": sha256,
        },
        "geography_ladder": {"sha256": "p" * 64},
        "household_location": {
            "rule": US_BLOCK_LOCATION_RULE_ID,
            "seed": 7,
            "clones": 1,
            "block_ladder": {
                "sha256": sha256,
                "layer_vintages": {"congressional_district": "119th_congress"},
            },
            "block_ladder_path": str(ladder_path),
            "donor_blocks": {DONOR_BLOCK_PRESERVED: 3, DONOR_BLOCK_STATE_DRAWN: 1},
        },
    }


def test__release__legacy_staging_orchestration_projection_is_unchanged() -> None:
    release = _load_release_tool()
    projected = release._require_uncapped_staging(
        {"orchestration": {"max_households": None, "seed": 1}}
    )
    assert list(projected) == list(release._STAGING_ORCHESTRATION_KEYS)
    assert "household_location" not in projected
    assert release.staging_household_location({"orchestration": {}}) is None
    assert release.staging_refresh_recipe({}) == release.LEGACY_STAGING_REFRESH_RECIPE
    assert release.calibrated_location_root_attributes({}) is None


def test__release__block_v1_staging_orchestration_adds_location_keys(
    tmp_path: Path,
) -> None:
    release = _load_release_tool()
    summary = _block_v1_staging_summary(tmp_path / "blocks.npz", "a" * 64)
    projected = release._require_uncapped_staging(summary)
    assert projected["location_rule"] == "block_v1"
    assert projected["location_seed"] == 7
    assert projected["location_clones"] == 1
    assert projected["block_ladder_sha256"] == "a" * 64
    assert projected["household_location"] == summary["household_location"]
    assert list(projected)[: len(release._STAGING_ORCHESTRATION_KEYS)] == list(
        release._STAGING_ORCHESTRATION_KEYS
    )


def test__release__block_v1_summary_without_record_is_refused() -> None:
    release = _load_release_tool()
    with pytest.raises(SystemExit, match="no household_location"):
        release.staging_household_location(
            {"orchestration": {"location_rule": "block_v1"}}
        )
    with pytest.raises(SystemExit, match="unknown location_rule"):
        release.staging_household_location(
            {"orchestration": {"location_rule": "block_v9"}}
        )


def test__release__block_v1_limitations_state_the_rule(tmp_path: Path) -> None:
    release = _load_release_tool()
    legacy = {
        entry["id"]: entry
        for entry in release.finalize_reviewed_limitations(
            {"reviewed_limitations": []}, {}, {}
        )
    }
    block = {
        entry["id"]: entry
        for entry in release.finalize_reviewed_limitations(
            _block_v1_staging_summary(tmp_path / "b.npz", "a" * 64), {}, {}
        )
    }
    assert list(legacy) == list(block)
    assert "location_rule" not in legacy["mixed_sub_puma_column_coverage"]
    assert "donor-spine-only" in legacy["mixed_sub_puma_column_coverage"]["reason"]
    mixed = block["mixed_sub_puma_column_coverage"]
    assert mixed["location_rule"] == "block_v1"
    assert "Every household on both spines" in mixed["reason"]
    assert "donor-spine-only" not in mixed["reason"]
    assert mixed["donor_blocks"][DONOR_BLOCK_STATE_DRAWN] == 1
    cd = block["cd_population_marginal_vintage_2020"]
    assert cd["household_district_plan"] == "119th_congress"
    assert "one 2020 census block" in cd["reason"]
    assert "checked that the two ladders" not in cd["reason"]
    summary = _block_v1_staging_summary(tmp_path / "b.npz", "a" * 64)
    summary["household_location"]["ladder_agreement"] = {
        "congressional_districts": {"max_abs_population_difference": 0.0}
    }
    checked = {
        entry["id"]: entry
        for entry in release.finalize_reviewed_limitations(summary, {}, {})
    }["cd_population_marginal_vintage_2020"]
    assert "same PUMAs, states and districts" in checked["reason"]
    assert "difference between them was 0.0." in checked["reason"]
    for key in legacy:
        if key not in {
            "mixed_sub_puma_column_coverage",
            "cd_population_marginal_vintage_2020",
        }:
            assert legacy[key] == block[key]


def test__release__block_v1_refresh_recipe_parses_as_block_v1(tmp_path: Path) -> None:
    release = _load_release_tool()
    builder = _load_legacy_builder()
    recipe = shlex.split(
        release.staging_refresh_recipe(
            _block_v1_staging_summary(tmp_path / "b.npz", "a" * 64)
        )
    )
    assert recipe[:3] == ["uv", "run", "tools/build_us_acs_multispine_base.py"]
    args = builder._parse_args(recipe[3:])
    assert args.location_rule == "block_v1"
    assert args.location_seed == 7
    assert args.block_ladder is not None
    builder._validate_location_arguments(args)


def test__release__block_ladder_is_sha_verified(
    tmp_path: Path, ladder: UsLocationLadder
) -> None:
    release = _load_release_tool()
    path = write_location_ladder(
        tmp_path / "blocks.npz", synthetic_blocks(3, states=STATES)
    )
    good = _block_v1_staging_summary(path, release._sha256(path))
    loaded = release.load_staging_block_ladder(good)
    assert loaded.sha256 == ladder.sha256
    bad = _block_v1_staging_summary(path, "b" * 64)
    with pytest.raises(SystemExit, match="not the"):
        release.load_staging_block_ladder(bad)
    with pytest.raises(SystemExit, match="missing"):
        release.load_staging_block_ladder(
            _block_v1_staging_summary(tmp_path / "gone.npz", "a" * 64)
        )
    with pytest.raises(SystemExit, match="applies only to a block_v1"):
        release.load_staging_block_ladder({}, path)
    assert release.load_staging_block_ladder({}) is None


def test__release__calibrated_root_attributes_match_the_staging_writer(
    tmp_path: Path,
) -> None:
    release = _load_release_tool()
    attributes = release.calibrated_location_root_attributes(
        _block_v1_staging_summary(tmp_path / "b.npz", "a" * 64)
    )
    assert attributes == {
        "populace_location_rule": "block_v1",
        "populace_location_seed": "7",
        "populace_location_clones": "1",
        "populace_block_ladder_sha256": "a" * 64,
        "populace_block_ladder_vintages": json.dumps(
            {"congressional_district": "119th_congress"}, sort_keys=True
        ),
        "populace_puma_ladder_artifact_sha256": "p" * 64,
        "populace_puma_ladder_vintages": json.dumps({}, sort_keys=True),
    }


# ---------------------------------------------------------------------------
# Invariants, for every input (property-based), and the legacy differential
# ---------------------------------------------------------------------------

try:
    import hypothesis
    from hypothesis import strategies as st
except ModuleNotFoundError:  # pragma: no cover - the workspace env installs it
    hypothesis = None


def _pool_scenario(draw):
    """Donors in any ladder state, each with or without a block; ACS in any PUMA."""

    n_donors = draw(st.integers(1, 10))
    donors = draw(
        st.lists(
            st.tuples(
                st.sampled_from(STATES),
                st.sampled_from(("block", "", None)),
                st.integers(0, 10_000),
            ),
            min_size=n_donors,
            max_size=n_donors,
        )
    )
    acs_picks = draw(st.lists(st.integers(0, 10_000), min_size=1, max_size=12))
    seed = draw(st.integers(0, 2**32))
    return donors, acs_picks, seed


def _scenario_frames(
    ladder: UsLocationLadder, donors, acs_picks, *, strip_blocks: bool = False
) -> tuple[Frame, Frame]:
    states = [state for state, _, _ in donors]
    blocks = [
        _ladder_block(ladder, state, which)
        if kind == "block" and not strip_blocks
        else (None if kind is None else "")
        for state, kind, which in donors
    ]
    base = _frame(
        pd.DataFrame(
            {
                "household_id": np.arange(1, len(donors) + 1, dtype=np.int64),
                "state_fips": states,
                "block_geoid": pd.Series(blocks, dtype=object),
                "puma": pd.Series([None] * len(donors), dtype=object),
            }
        ),
        weights=np.linspace(10.0, 60.0, len(donors)),
        spine="asec_puf",
    )
    assert ladder.puma is not None
    pumas = np.unique(ladder.puma)
    chosen = [int(pumas[pick % len(pumas)]) for pick in acs_picks]
    acs = _frame(
        pd.DataFrame(
            {
                "household_id": np.arange(1, len(chosen) + 1, dtype=np.int64),
                "state_fips": [puma // 10**5 for puma in chosen],
                "puma": [f"{puma:07d}" for puma in chosen],
            }
        ),
        weights=np.full(len(chosen), 7.0),
        spine=ACS_2024_1YR_SPINE,
    )
    return base, acs


def _pool_invariants(ladder: UsLocationLadder, scenario) -> None:
    donors, acs_picks, seed = scenario
    base, acs = _scenario_frames(ladder, donors, acs_picks)
    pooled = _pool(ladder, base, acs, location_seed=seed)
    household = pooled.table("household")
    is_acs = household[spine_column("household")].eq(ACS_2024_1YR_SPINE).to_numpy()

    # Consistency: every geography is its block's ladder lookup (the gate).
    gate = us_block_location_gate(household, ladder)
    assert gate.passed, gate.failures
    # Conservation: rows and the base's household mass.
    assert len(household) == len(donors) + len(acs_picks)
    assert pooled.weights_for("household").total == pytest.approx(
        base.weights_for("household").total, rel=1e-12
    )
    # state_fips is never rewritten.
    assert household.loc[~is_acs, "state_fips"].tolist() == [s for s, _, _ in donors]
    assert (
        household.loc[is_acs, "state_fips"].tolist()
        == acs.table("household")["state_fips"].tolist()
    )
    # ACS rows: the block lies in the observed PUMA.
    assert ladder.puma is not None
    observed = acs.table("household")["puma"].astype(np.int64).to_numpy()
    index = _block_index(ladder, household.loc[is_acs, "block_geoid"])
    np.testing.assert_array_equal(ladder.puma[index], observed)
    # Donor rows: a carried block is kept verbatim, never redrawn.
    kept = base.table("household")["block_geoid"].tolist()
    donor_blocks = household.loc[~is_acs, "block_geoid"].tolist()
    for (_, kind, _), before, after in zip(donors, kept, donor_blocks, strict=True):
        if kind == "block":
            assert after == before
    # Accounting: preserved + state-drawn = donors; drawn rows are sourced.
    record = _record(pooled)
    n_kept = sum(kind == "block" for _, kind, _ in donors)
    assert record["donor_blocks"][DONOR_BLOCK_PRESERVED] == n_kept
    assert record["donor_blocks"][DONOR_BLOCK_STATE_DRAWN] == len(donors) - n_kept
    assert record["source_geography_households"][SOURCE_GEOGRAPHY_PUMA] == len(
        acs_picks
    )
    assert record["source_geography_households"][SOURCE_GEOGRAPHY_STATE] == (
        len(donors) - n_kept
    )
    assert record["seed"] == seed
    # Determinism.
    again = _pool(ladder, base, acs, location_seed=seed).table("household")
    pd.testing.assert_frame_equal(household, again)
    # Keyed draws: an ACS household's block does not depend on which donors
    # carry a block (the legacy rng.choice path couples them; this one must
    # not).
    stripped_base, _ = _scenario_frames(ladder, donors, acs_picks, strip_blocks=True)
    stripped = _pool(ladder, stripped_base, acs, location_seed=seed)
    assert (
        stripped.table("household").loc[is_acs, "block_geoid"].tolist()
        == household.loc[is_acs, "block_geoid"].tolist()
    )


if hypothesis is not None:

    @hypothesis.settings(
        max_examples=60,
        deadline=None,
        suppress_health_check=[hypothesis.HealthCheck.too_slow],
    )
    @hypothesis.given(scenario=st.composite(_pool_scenario)())
    def test__block_v1__pool_invariants_hold_for_every_donor_and_acs_mix(
        ladder: UsLocationLadder, scenario
    ) -> None:
        _pool_invariants(ladder, scenario)

else:  # pragma: no cover

    @pytest.mark.skip(reason="requires hypothesis")
    def test__block_v1__pool_invariants_hold_for_every_donor_and_acs_mix() -> None:
        pass


def _legacy_puma_ladder():
    from microcosm.build.us_runtime.puma_ladder import UsPumaLadder

    puma = np.asarray([100_101, 100_202], dtype=np.int64)
    population = np.asarray([40.0, 60.0])
    return UsPumaLadder(
        puma=puma,
        puma_population=population,
        cd_overlap_puma=puma.copy(),
        cd_overlap_cd=np.asarray([101, 102], dtype=np.int64),
        cd_overlap_population=population.copy(),
        county_overlap_puma=puma.copy(),
        county_overlap_county=np.asarray([1_001, 1_003], dtype=np.int32),
        county_overlap_population=population.copy(),
        tract_overlap_puma=puma.copy(),
        tract_overlap_tract=np.asarray([1_001_000_100, 1_003_000_100], dtype=np.int64),
        tract_overlap_population=population.copy(),
        metadata={
            "schema_version": 1,
            "kind": "us_puma_ladder",
            "puma_vintage": "2020_puma",
            "sampling_basis": "population",
            "layers": {
                "congressional_district": {"vintage": "119th_congress"},
                "county": {"vintage": "2020_census"},
                "tract": {"vintage": "2020_census"},
            },
        },
    )


def test__legacy__explicit_default_location_options_reproduce_the_puma_ladder_pool() -> (
    None
):
    """Differential: the new kwargs at their defaults are the old call, exactly."""

    base = _frame(
        pd.DataFrame(
            {"household_id": np.arange(1, 6, dtype=np.int64), "state_fips": [1] * 5}
        ),
        weights=np.linspace(10.0, 50.0, 5),
        spine="asec_puf",
    )
    acs = _frame(
        pd.DataFrame(
            {
                "household_id": np.arange(1, 5, dtype=np.int64),
                "state_fips": [1] * 4,
                "puma": ["0100101", "0100202", "0100101", "0100202"],
            }
        ),
        weights=np.full(4, 3.0),
        spine=ACS_2024_1YR_SPINE,
    )
    options = {
        "puma_ladder": _legacy_puma_ladder(),
        "geography_seed": 11,
        "expected_congressional_district_vintage": "119th_congress",
        "max_peak_bytes": None,
    }
    implicit = with_optional_acs_spine(base, acs, **options)
    explicit = with_optional_acs_spine(
        base,
        acs,
        location_rule="legacy",
        block_ladder=None,
        location_seed=0,
        location_clones=1,
        **options,
    )
    for entity in implicit.entities:
        pd.testing.assert_frame_equal(implicit.table(entity), explicit.table(entity))
    np.testing.assert_array_equal(
        implicit.weights_for("household").values,
        explicit.weights_for("household").values,
    )
    assert ACS_POOL_BLOCK_LOCATION_METADATA_KEY not in explicit.metadata
    assert "block_geoid" not in explicit.table("household")


@requires_pytables
def test__legacy_tool__reads_a_base_rule_written_by_the_base_h5_line(
    tmp_path: Path, ladder: UsLocationLadder
) -> None:
    """The base-H5 line writes its rule with h5py; this line reads it with PyTables."""

    h5py = pytest.importorskip("h5py")
    builder = _load_legacy_builder()
    path = tmp_path / "base.h5"
    builder._write_dataset(_donor_base(ladder), path, period=2024)
    with h5py.File(path, "a") as h5:
        # As tools/build_us_puf_support_base.py:_write_block_v1_location_attrs.
        h5.attrs["populace_location_rule"] = "block_v1"
        h5.attrs["populace_location_seed"] = str(11)
        h5.attrs["populace_location_clones"] = str(1)
    assert builder._base_location_rule(path) == {
        "location_rule": "block_v1",
        "recorded": True,
        "attributes": {
            "populace_location_rule": "block_v1",
            "populace_location_seed": "11",
            "populace_location_clones": "1",
        },
    }


@requires_pytables
def test__legacy_tool__block_v1_staging_bytes_reload_through_the_release_loader(
    tmp_path: Path, ladder: UsLocationLadder
) -> None:
    """Staging writes with the legacy writer; the release reads it back and gates it."""

    builder = _load_legacy_builder()
    release = _load_release_tool()
    pooled = _pool(ladder, _donor_base(ladder, blocks="some"), _acs(ladder, n=30))
    written = pooled.table("household")
    assert (written["place_fips"] == "").any() or (written["sldl"] == "").any()
    path = tmp_path / "staging.h5"
    builder._write_dataset(pooled, path, period=2024)
    reloaded = release._load_staging_frame(path).table("household")
    gate = us_block_location_gate(reloaded, ladder)
    assert gate.passed, gate.failures
    assert reloaded["block_geoid"].tolist() == written["block_geoid"].tolist()


@pytest.mark.parametrize(
    ("donor_geography", "kept", "drawn"),
    [
        ("preserved_donor_block", True, False),
        ("state_drawn_missing_donor_block", False, True),
        ("preserved_donor_block_with_state_draws", True, True),
    ],
)
def test__block_v1__limitation_states_what_happened_to_donors(
    donor_geography: str, kept: bool, drawn: bool
) -> None:
    """The recorded reason matches the donor mode: never a false claim that
    base blocks were kept when donors were drawn within their state."""

    builder = _load_legacy_builder()
    record = builder._block_v1_sub_puma_limitation(
        {"donor_geography": donor_geography, "resolved_model_inputs": []}
    )
    reason = record["reason"]
    assert ("keep" in reason) is kept
    assert ("within their state" in reason or "within that state" in reason) is drawn
    observed = record["observed_geography"]["asec_puf"]
    if donor_geography == "state_drawn_missing_donor_block":
        assert observed == ["state_fips"]
    elif donor_geography == "preserved_donor_block":
        assert observed == ["state_fips", "block_geoid"]
    else:
        assert observed == {
            "with_base_block": ["state_fips", "block_geoid"],
            "without_base_block": ["state_fips"],
        }
