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

``--cd-plan-registry`` (the pinned block -> CD-plan registry, attached to the
block ladder) is covered below its own heading. Its invariants, each executed
by the tests named in that section:

1. Legacy is byte-identical and refuses the registry (flag, pool option).
2. ``block_v1`` without the registry is byte-identical: no new build kwarg,
   pool kwarg, manifest key, summary key, orchestration key or root attribute.
3. Only the pinned registry loads, refused before the donor base is read.
4. Every plan column is the registry's district of the drawn block, lies in
   the household's state, and the hard location gate covers it.
5. Attaching never moves a block or changes any other column (property).
6. The registry sha256 recorded (manifest, summary, orchestration, root
   attribute, calibrated attribute) is the loaded file's.
7. A donor base's plan columns the run would not rewrite are refused (the
   probe-verified half-null bug), and so is a base located with another
   registry.
8. The local release binds the registry across its checkpointed stages: the
   run identity pins staging bytes that carry the registry, and finalize
   refuses plan columns its re-attached registry does not verify (and a
   recorded registry whose columns are missing).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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
    refuse_unwritten_cd_plan_columns,
    spine_column,
    unwritten_cd_plan_columns,
    validate_pool_location_options,
    with_optional_acs_spine,
)
from microcosm.build.us_runtime.block_location import (
    POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR,
    SOURCE_GEOGRAPHY_PUMA,
    SOURCE_GEOGRAPHY_STATE,
    US_BLOCK_LOCATION_RULE_ID,
    US_CD_PLAN_COLUMN_PREFIX,
    US_LOCATION_RULE_BLOCK_V1,
    UsLocationLadder,
    attach_pinned_cd_plan_registry,
    derive_us_block_geography,
    load_us_location_ladder,
    location_geography_columns,
    stale_cd_plan_columns,
    us_block_location_gate,
)
from microcosm.build.us_runtime.cd_plan_registry import (
    PRIMARY_CD_PLAN,
    UsCdPlanRegistry,
    load_us_cd_plan_registry,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_block_location import (
    synthetic_blocks,
    write_location_ladder,
)
from test_support.microcosm_build.us_cd_plan_registry import (
    REGISTRY_PLAN_APPORTIONMENT,
    file_sha256,
    pin_cd_plan_registry,
    registry_ready_blocks,
    write_cd_plan_registry,
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


#: The registry's non-current plans: the household columns an attach adds.
NON_CURRENT_PLANS = tuple(
    plan for plan in sorted(REGISTRY_PLAN_APPORTIONMENT) if plan != PRIMARY_CD_PLAN
)
PLAN_COLUMNS = tuple(f"{US_CD_PLAN_COLUMN_PREFIX}{plan}" for plan in NON_CURRENT_PLANS)


@dataclass(frozen=True)
class RegistryLadders:
    """A registry-ready ladder, its synthetic registry, and both attach results.

    The raw ``synthetic_blocks`` ladder is not registry-compatible (its
    districts break apportionment), so the ladder and the registry are built
    from the same :func:`registry_ready_blocks` arrays.
    """

    arrays: dict
    ladder_path: Path
    registry_path: Path
    digest: str
    plain: UsLocationLadder
    attached: UsLocationLadder
    record: dict
    registry: UsCdPlanRegistry


def _registry_ladders(directory: Path, patch) -> RegistryLadders:
    arrays = registry_ready_blocks(synthetic_blocks(3, states=STATES))
    ladder_path = write_location_ladder(directory / "blocks.npz", arrays)
    registry_path = write_cd_plan_registry(directory / "registry.npz", arrays)
    digest = pin_cd_plan_registry(patch, registry_path)
    plain = load_us_location_ladder(ladder_path)
    attached, record = attach_pinned_cd_plan_registry(plain, registry_path)
    return RegistryLadders(
        arrays=arrays,
        ladder_path=ladder_path,
        registry_path=registry_path,
        digest=digest,
        plain=plain,
        attached=attached,
        record=record,
        registry=load_us_cd_plan_registry(registry_path),
    )


@pytest.fixture(scope="module")
def registry_ladders(tmp_path_factory: pytest.TempPathFactory) -> RegistryLadders:
    # The pin is needed only while the pinned loader runs.
    with pytest.MonkeyPatch.context() as patch:
        return _registry_ladders(tmp_path_factory.mktemp("registry"), patch)


def _registry_districts(
    registry: UsCdPlanRegistry, blocks: pd.Series
) -> dict[str, np.ndarray]:
    """Each plan's registry district for ``blocks`` (15-digit text geoids)."""

    values = blocks.astype(np.int64).to_numpy()
    index = np.searchsorted(registry.block_geoid, values)
    assert (registry.block_geoid[index] == values).all()
    return {plan: registry.plans[plan][index] for plan in registry.plans}


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
    assert args.cd_plan_registry is None
    builder._validate_location_arguments(args)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--location-seed", "1"], "apply only to --location-rule block_v1"),
        (["--location-clones", "2"], "apply only to --location-rule block_v1"),
        (["--block-ladder", "x.npz"], "only used by --location-rule block_v1"),
        (
            ["--cd-plan-registry", "x.npz"],
            "--cd-plan-registry is only used by --location-rule block_v1",
        ),
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
    registry: bool = False,
    pool_with_registry: bool | None = None,
    pass_registry_flag: bool | None = None,
    registry_flag_path: Path | None = None,
    repin: str | None = None,
    base_attributes=None,
    tool_base_columns: dict | None = None,
    pool_columns: dict | None = None,
    provenance_registry: dict | None = None,
    captured: dict | None = None,
):
    """Run the tool's ``block_v1`` main with the heavy stages stubbed.

    ``registry`` builds the ladder and a synthetic registry from
    :func:`registry_ready_blocks` arrays, pins the registry, and (unless
    ``pool_with_registry`` is False) pools the stubbed build's frame from the
    attached ladder with the record; ``pass_registry_flag`` (default
    ``registry``) passes ``--cd-plan-registry`` (``registry_flag_path``
    overrides its path). ``repin`` re-pins another digest before main runs.
    ``base_attributes`` is the donor base's recorded root attributes (or a
    callable of the registry setup returning them), ``tool_base_columns``
    extra household columns of the base the tool loads (the stubbed build
    ignores them), ``pool_columns`` extra household columns of the frame the
    stubbed build returns, and ``provenance_registry`` replaces the recorded
    ``cd_plan_registry`` record. ``captured`` (filled even when main exits)
    receives the registry setup under ``"registry"`` and counts the tool's
    block-ladder loads, base-rule reads, base loads and ACS fetches.
    """

    builder = _load_legacy_builder()
    captured = {} if captured is None else captured
    captured.update(
        {"ladder_loads": 0, "base_rule_reads": 0, "base_loads": 0, "fetches": 0}
    )
    pool_options: dict[str, object] = {}
    if registry:
        setup = _registry_ladders(tmp_path, monkeypatch)
        ladder_path = setup.ladder_path
        ladder = setup.plain
        pool_ladder = setup.attached
        if pool_with_registry is not False:
            pool_options["cd_plan_registry"] = setup.record
        else:
            pool_ladder = setup.plain
        captured["registry"] = setup
    else:
        ladder_path = write_location_ladder(
            tmp_path / "blocks.npz", synthetic_blocks(3, states=STATES)
        )
        ladder = load_us_location_ladder(ladder_path)
        pool_ladder = ladder
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
        block_ladder=pool_ladder,
        location_seed=seed,
        max_peak_bytes=None,
        **pool_options,
    )
    if tool_base_columns:
        tables = {e: base.table(e).copy() for e in base.entities}
        tables["household"] = tables["household"].assign(**tool_base_columns)
        tool_base = Frame(
            tables, US_SCHEMA, {"household": base.weights_for("household")}, base.strata
        )
    else:
        tool_base = base
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
    if pool_columns:
        tables = {e: combined.table(e).copy() for e in combined.entities}
        tables["household"] = tables["household"].assign(**pool_columns)
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
    household_location = _record(combined)
    if provenance_registry is not None:
        household_location["cd_plan_registry"] = provenance_registry
    provenance = {
        "enabled": True,
        "deferred_inputs": [],
        "household_location": household_location,
        "geography_ladder": acs_multispine._block_geography_provenance(
            combined,
            pool_ladder,
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
        captured["written_frame"] = frame
        path.write_bytes(b"combined-output")

    def load_ladder(path):
        captured["ladder_loads"] += 1
        return load_us_location_ladder(path)

    def load_base(path):
        captured["base_loads"] += 1
        return tool_base

    def base_rule(path):
        captured["base_rule_reads"] += 1
        attributes = base_attributes
        if callable(attributes):
            attributes = attributes(captured.get("registry"))
        return {
            "location_rule": "block_v1",
            "recorded": True,
            "attributes": dict(attributes or {"populace_location_rule": "block_v1"}),
        }

    def fetch(cache, *, manifest):
        captured["fetches"] += 1
        return builder.AcsPumsSource(cache / "h.zip", cache / "p.zip")

    manifest = type("Manifest", (), {})()
    monkeypatch.setattr(builder, "load_us_location_ladder", load_ladder)
    monkeypatch.setattr(builder, "_load_base_frame", load_base)
    monkeypatch.setattr(builder, "_base_location_rule", base_rule)
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
    monkeypatch.setattr(builder.acs_sources, "fetch_acs_pums_sources", fetch)
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
    pass_flag = registry if pass_registry_flag is None else pass_registry_flag
    if pass_flag:
        flag_path = registry_flag_path or tmp_path / "registry.npz"
        arguments += ["--cd-plan-registry", str(flag_path)]
    if repin is not None:
        from microcosm.build.us_runtime import cd_plan_registry

        monkeypatch.setattr(
            cd_plan_registry,
            "packaged_us_cd_plan_registry_provenance",
            lambda: {"output_sha256": repin},
        )
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


# ---------------------------------------------------------------------------
# --cd-plan-registry: the pinned block -> CD-plan registry on this line
# ---------------------------------------------------------------------------

#: The seven root attributes a block_v1 staging H5 carries without a registry.
NO_REGISTRY_ROOT_ATTRIBUTES = {
    "populace_location_rule",
    "populace_location_seed",
    "populace_location_clones",
    "populace_block_ladder_sha256",
    "populace_block_ladder_vintages",
    "populace_puma_ladder_artifact_sha256",
    "populace_puma_ladder_vintages",
}
#: The block_v1 treatment text of a run without the registry (byte-pinned).
NO_REGISTRY_TREATMENT = (
    "Give each ACS household one 2020 census block drawn with "
    "probability proportional to 2020 block population within its "
    "observed PUMA, keep each donor's already-assigned block (a donor "
    "without one draws within its state), and derive block, tract, "
    "county, place, SLDU/SLDL, CBSA, PUMA and congressional district "
    "from the block through the block ladder. ACS rows therefore now "
    "carry block/tract/place/SLD/CBSA, and no household's geographies "
    "contradict each other."
)
#: Pool kwargs build_optional_acs_multispine passed under block_v1 before the
#: registry existed; without a record they must be exactly these.
BLOCK_V1_POOL_KWARGS = {
    "acs_share",
    "geography_seed",
    "expected_congressional_district_vintage",
    "location_rule",
    "block_ladder",
    "location_seed",
    "location_clones",
}


def _registry_pool(
    registry_ladders: RegistryLadders, base: Frame, acs: Frame, **options
) -> Frame:
    return _pool(
        registry_ladders.attached,
        base,
        acs,
        cd_plan_registry=registry_ladders.record,
        **options,
    )


def _donor_base_with_plans(
    registry_ladders: RegistryLadders, plans: tuple[str, ...] = NON_CURRENT_PLANS
) -> Frame:
    """A donor base as a registry-attached base-H5 build writes it: every
    donor carries its block and each plan's district of that block."""

    base = _donor_base(registry_ladders.plain)
    household = base.table("household").copy()
    index = _block_index(registry_ladders.attached, household["block_geoid"])
    derived = derive_us_block_geography(registry_ladders.attached, index)
    for plan in plans:
        column = f"{US_CD_PLAN_COLUMN_PREFIX}{plan}"
        household[column] = derived[column]
    return _frame(household, weights=np.full(6, 50.0), spine="asec_puf")


def _assert_plan_columns_are_registry_lookups(
    registry_ladders: RegistryLadders, household: pd.DataFrame, record: Mapping
) -> None:
    districts = _registry_districts(registry_ladders.registry, household["block_geoid"])
    states = pd.to_numeric(household["state_fips"]).to_numpy(np.int64)
    for plan, column in record["columns"].items():
        values = household[column].to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(values, districts[plan], err_msg=plan)
        np.testing.assert_array_equal(values // 100, states, err_msg=plan)


# Pool ----------------------------------------------------------------------


def test__registry__plan_columns_are_the_registry_lookup_and_the_gate_covers_them(
    registry_ladders: RegistryLadders,
) -> None:
    """Invariant 4: every plan column is the registry's district of the drawn
    (or preserved) block, in the household's state, and the gate checks it."""

    pooled = _registry_pool(
        registry_ladders,
        _donor_base(registry_ladders.plain, blocks="some"),
        _acs(registry_ladders.plain, n=30),
    )
    household = pooled.table("household")
    record = _record(pooled)
    assert record["cd_plan_registry"] == registry_ladders.record
    assert record["cd_plan_registry"]["sha256"] == file_sha256(
        registry_ladders.registry_path
    )
    assert record["geography_columns"][-3:] == list(PLAN_COLUMNS)
    _assert_plan_columns_are_registry_lookups(
        registry_ladders, household, record["cd_plan_registry"]
    )
    for column in PLAN_COLUMNS:
        assert household[column].dtype == np.int64
    gate = us_block_location_gate(household, registry_ladders.attached)
    assert gate.passed, gate.failures
    assert set(PLAN_COLUMNS) <= set(gate.details["columns"])
    tampered = household.copy()
    tampered.loc[0, PLAN_COLUMNS[0]] = tampered.loc[0, PLAN_COLUMNS[0]] + 1
    failed = us_block_location_gate(tampered, registry_ladders.attached)
    assert not failed.passed
    assert any(line.startswith(PLAN_COLUMNS[0]) for line in failed.failures)


def test__registry__the_pool_record_carries_the_record_only_when_given(
    registry_ladders: RegistryLadders,
) -> None:
    base = _donor_base(registry_ladders.plain, blocks="some")
    acs = _acs(registry_ladders.plain)
    plain = _pool(registry_ladders.plain, base, acs)
    attached = _registry_pool(registry_ladders, base, acs)
    assert "cd_plan_registry" not in _record(plain)
    assert not set(PLAN_COLUMNS) & set(plain.table("household").columns)
    assert _record(attached)["cd_plan_registry"] == registry_ladders.record


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"sha256": "not-a-digest"}, "no valid sha256"),
        ({"plans": []}, "no plans"),
        ({"plans": "117th_congress"}, "no plans"),
        ({"plans": ["116th_congress"]}, "does not attach"),
        ({"columns": {}}, "not the columns"),
    ],
)
def test__registry__a_malformed_record_is_refused(
    registry_ladders: RegistryLadders, change: dict, match: str
) -> None:
    record = {**registry_ladders.record, **change}
    with pytest.raises(ValueError, match=match):
        _pool(
            registry_ladders.attached,
            _donor_base(registry_ladders.plain),
            _acs(registry_ladders.plain),
            cd_plan_registry=record,
        )


def test__registry__the_record_is_refused_under_legacy_and_on_an_unattached_ladder(
    registry_ladders: RegistryLadders,
) -> None:
    """Invariant 1: legacy refuses the registry option; a record whose plans
    the ladder does not attach is refused rather than recorded falsely."""

    with pytest.raises(ValueError, match="cd_plan_registry is only used"):
        validate_pool_location_options(
            "legacy",
            block_ladder=None,
            location_seed=0,
            location_clones=1,
            geography_seed=0,
            cd_plan_registry=registry_ladders.record,
        )
    with pytest.raises(ValueError, match="cd_plan_registry is only used"):
        with_optional_acs_spine(
            _donor_base(registry_ladders.plain),
            _acs(registry_ladders.plain),
            cd_plan_registry=registry_ladders.record,
        )
    with pytest.raises(ValueError, match="does not attach"):
        _pool(
            registry_ladders.plain,
            _donor_base(registry_ladders.plain),
            _acs(registry_ladders.plain),
            cd_plan_registry=registry_ladders.record,
        )


def test__stale_plans__the_probe_scenario_is_refused_before_it_ships_half_null(
    registry_ladders: RegistryLadders,
) -> None:
    """Invariant 7, the probe-verified bug: a base carrying plan columns,
    pooled without the registry, used to ship them NaN on every ACS row with
    every gate passing. The pool and its pre-transfer preflight refuse it."""

    base = _donor_base_with_plans(registry_ladders)
    acs = _acs(registry_ladders.plain)
    with pytest.raises(ValueError, match="would not rewrite") as caught:
        _pool(registry_ladders.plain, base, acs)
    for column in PLAN_COLUMNS:
        assert column in str(caught.value)
    with pytest.raises(ValueError, match="would not rewrite"):
        preflight_pooled_ladder_geography(
            base.table("household"),
            acs.table("household"),
            None,
            location_rule=US_LOCATION_RULE_BLOCK_V1,
            block_ladder=registry_ladders.plain,
        )
    # With the registry the same base pools: every row's plan columns are
    # re-derived from its block, none is null, and the gate covers them.
    pooled = _registry_pool(registry_ladders, base, acs)
    household = pooled.table("household")
    assert household[list(PLAN_COLUMNS)].notna().all().all()
    assert us_block_location_gate(household, registry_ladders.attached).passed
    assert _record(pooled)["donor_blocks"]["rederived_columns_changed"] == {}


def test__stale_plans__a_donor_plan_the_registry_contradicts_is_counted(
    registry_ladders: RegistryLadders,
) -> None:
    base = _donor_base_with_plans(registry_ladders)
    household = base.table("household").copy()
    column = PLAN_COLUMNS[0]
    household.loc[0, column] = int(household.loc[0, column]) + 1
    changed = _frame(household, weights=np.full(6, 50.0), spine="asec_puf")
    pooled = _registry_pool(registry_ladders, changed, _acs(registry_ladders.plain))
    record = _record(pooled)
    assert record["donor_blocks"]["rederived_columns_changed"] == {column: 1}
    assert us_block_location_gate(
        pooled.table("household"), registry_ladders.attached
    ).passed


def test__stale_plans__a_plan_the_registry_does_not_attach_is_refused_with_it(
    registry_ladders: RegistryLadders,
) -> None:
    base = _donor_base(registry_ladders.plain)
    household = base.table("household").copy()
    household[f"{US_CD_PLAN_COLUMN_PREFIX}116th_congress"] = 101
    stale = _frame(household, weights=np.full(6, 50.0), spine="asec_puf")
    with pytest.raises(ValueError, match="116th_congress"):
        _registry_pool(registry_ladders, stale, _acs(registry_ladders.plain))


def test__stale_plans__an_acs_side_plan_column_is_refused_too(
    registry_ladders: RegistryLadders,
) -> None:
    acs = _acs(registry_ladders.plain)
    household = acs.table("household").drop(columns=[spine_column("household")])
    household[PLAN_COLUMNS[0]] = 101
    bad = _frame(household, weights=np.full(9, 10.0), spine=ACS_2024_1YR_SPINE)
    with pytest.raises(ValueError, match="'ACS'"):
        _pool(registry_ladders.plain, _donor_base(registry_ladders.plain), bad)


def _legacy_pool_frames(*, plan_column: bool) -> tuple[Frame, Frame]:
    household = pd.DataFrame(
        {"household_id": np.arange(1, 6, dtype=np.int64), "state_fips": [1] * 5}
    )
    if plan_column:
        household[PLAN_COLUMNS[0]] = 101
    base = _frame(household, weights=np.linspace(10.0, 50.0, 5), spine="asec_puf")
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
    return base, acs


def test__stale_plans__the_legacy_puma_ladder_pool_refuses_plan_columns() -> None:
    """The legacy PUMA-ladder assignment writes no plan column, so a base
    carrying one would ship it half-null there too; its pool and preflight
    refuse it, and a base without one pools exactly as before."""

    base, acs = _legacy_pool_frames(plan_column=True)
    options = {
        "puma_ladder": _legacy_puma_ladder(),
        "geography_seed": 11,
        "expected_congressional_district_vintage": "119th_congress",
        "max_peak_bytes": None,
    }
    with pytest.raises(ValueError, match="legacy PUMA-ladder assignment writes"):
        with_optional_acs_spine(base, acs, **options)
    with pytest.raises(ValueError, match="would not rewrite"):
        preflight_pooled_ladder_geography(
            base.table("household"), acs.table("household"), _legacy_puma_ladder()
        )
    clean, acs = _legacy_pool_frames(plan_column=False)
    with_optional_acs_spine(clean, acs, **options)
    assert (
        preflight_pooled_ladder_geography(
            clean.table("household"), acs.table("household"), _legacy_puma_ladder()
        )
        == "ladder_drawn"
    )


def test__stale_plans__the_plain_pool_without_a_location_step_is_unchanged() -> None:
    """A pool that assigns no geography (no ladder) keeps its contract: every
    base-only column, geography included, is aligned with missing ACS values;
    it is not a location step, so it is not refused."""

    base, acs = _legacy_pool_frames(plan_column=True)
    pooled = with_optional_acs_spine(base, acs, max_peak_bytes=None)
    column = pooled.table("household")[PLAN_COLUMNS[0]]
    assert column.iloc[:5].tolist() == [101] * 5
    assert column.iloc[5:].isna().all()


# Multispine threading --------------------------------------------------------


def _multispine_run(monkeypatch, base: Frame, acs: Frame, **options):
    """build_optional_acs_multispine with its loaders stubbed; the real pool
    runs through a spy that records the kwargs it was called with."""

    calls: list[dict] = []

    def spy(actual_base, actual_acs, **kwargs):
        calls.append(kwargs)
        return with_optional_acs_spine(
            actual_base, actual_acs, max_peak_bytes=None, **kwargs
        )

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
    monkeypatch.setattr(
        acs_multispine,
        "transfer_acs_inputs",
        lambda recipient, donor, **kwargs: type(
            "Transfer",
            (),
            {
                "frame": recipient,
                "imputed_inputs": (),
                "fit_records": (),
                "deferred_inputs": ("block_geoid", "congressional_district_geoid"),
                "resolved_donor_channel": "asec",
            },
        )(),
    )
    monkeypatch.setattr(acs_multispine, "with_optional_acs_spine", spy)
    result = acs_multispine.build_optional_acs_multispine(
        base,
        object(),  # type: ignore[arg-type]
        location_rule=US_LOCATION_RULE_BLOCK_V1,
        location_seed=9,
        **options,
    )
    return result, calls


def test__multispine__threads_the_registry_record_only_when_given(
    monkeypatch: pytest.MonkeyPatch, registry_ladders: RegistryLadders
) -> None:
    """Invariants 2 and 6: without a record the pool call is the old one,
    key for key; with it, the record reaches the pool (no post-hoc merge)
    and provenance['household_location'] carries it."""

    base = _donor_base(registry_ladders.plain, blocks="some")
    acs = _acs(registry_ladders.plain)
    plain, plain_calls = _multispine_run(
        monkeypatch, base, acs, block_ladder=registry_ladders.plain
    )
    assert set(plain_calls[0]) == BLOCK_V1_POOL_KWARGS
    assert "cd_plan_registry" not in plain.provenance["household_location"]

    attached, calls = _multispine_run(
        monkeypatch,
        base,
        acs,
        block_ladder=registry_ladders.attached,
        cd_plan_registry=registry_ladders.record,
    )
    assert set(calls[0]) == BLOCK_V1_POOL_KWARGS | {"cd_plan_registry"}
    assert calls[0]["cd_plan_registry"] is registry_ladders.record
    provenance = attached.provenance
    json.dumps(provenance, allow_nan=False)
    assert provenance["household_location"]["cd_plan_registry"] == (
        registry_ladders.record
    )
    assert provenance["geography_ladder"]["resolved_model_inputs"][-3:] == list(
        PLAN_COLUMNS
    )
    assert provenance["deferred_inputs"] == []
    household = attached.frame.table("household")
    _assert_plan_columns_are_registry_lookups(
        registry_ladders, household, registry_ladders.record
    )
    # Invariant 5 through the whole seam: nothing but the plan columns moved.
    plain_household = plain.frame.table("household")
    pd.testing.assert_frame_equal(
        household[list(plain_household.columns)], plain_household
    )


def test__multispine__refuses_a_registry_record_before_loading(
    monkeypatch: pytest.MonkeyPatch, registry_ladders: RegistryLadders
) -> None:
    def must_not_load(*_args, **_kwargs):
        raise AssertionError("location options must be validated first")

    monkeypatch.setattr(acs_multispine, "build_acs_pums_unit_frame", must_not_load)
    for options in (
        {"cd_plan_registry": registry_ladders.record},
        {
            "location_rule": US_LOCATION_RULE_BLOCK_V1,
            "block_ladder": registry_ladders.plain,
            "cd_plan_registry": registry_ladders.record,
        },
    ):
        with pytest.raises(ValueError):
            acs_multispine.build_optional_acs_multispine(
                _donor_base(registry_ladders.plain),
                object(),  # type: ignore[arg-type]
                **options,
            )


# Properties -------------------------------------------------------------------


def _attached_pool_invariants(registry_ladders: RegistryLadders, scenario) -> None:
    """Invariant 5 (differential): the attached pool equals the plain pool on
    every non-plan column, weight and record entry; plan columns are the
    registry's lookup of each block, in its state; the gate covers them."""

    donors, acs_picks, seed = scenario
    base, acs = _scenario_frames(registry_ladders.plain, donors, acs_picks)
    plain = _pool(registry_ladders.plain, base, acs, location_seed=seed)
    attached = _registry_pool(registry_ladders, base, acs, location_seed=seed)
    household = attached.table("household")
    plain_household = plain.table("household")
    assert [c for c in household.columns if c not in PLAN_COLUMNS] == list(
        plain_household.columns
    )
    assert set(household.columns) - set(plain_household.columns) == set(PLAN_COLUMNS)
    pd.testing.assert_frame_equal(
        household[list(plain_household.columns)], plain_household
    )
    for entity in plain.entities:
        if entity != "household":
            pd.testing.assert_frame_equal(attached.table(entity), plain.table(entity))
    np.testing.assert_array_equal(
        attached.weights_for("household").values, plain.weights_for("household").values
    )
    _assert_plan_columns_are_registry_lookups(
        registry_ladders, household, registry_ladders.record
    )
    gate = us_block_location_gate(household, registry_ladders.attached)
    assert gate.passed, gate.failures
    assert set(PLAN_COLUMNS) <= set(gate.details["columns"])
    record, plain_record = _record(attached), _record(plain)
    assert record["cd_plan_registry"] == registry_ladders.record
    assert "cd_plan_registry" not in plain_record
    for key in set(record) - {"block_ladder", "geography_columns", "cd_plan_registry"}:
        assert record[key] == plain_record[key], key


_COLUMN_NAMES = (
    "household_id",
    "state_fips",
    "congressional_district_geoid",
    "congressional_district_geoid_",
    US_CD_PLAN_COLUMN_PREFIX,
    *PLAN_COLUMNS,
    f"{US_CD_PLAN_COLUMN_PREFIX}116th_congress",
    f"{US_CD_PLAN_COLUMN_PREFIX}119th_congress",
    "xcongressional_district_geoid__117th_congress",
)


def _unwritten_properties(registry_ladders: RegistryLadders, columns) -> None:
    """The legacy copy of the stale-plan rule (no ladder: every plan column)
    agrees with the shared helper on a ladder that attaches no other plan;
    with the registry exactly the attached plans stop being stale."""

    frame = pd.DataFrame(columns=columns)
    legacy = unwritten_cd_plan_columns(columns)
    assert legacy == stale_cd_plan_columns(columns, registry_ladders.plain)
    assert legacy == sorted(
        c for c in columns if c.startswith(US_CD_PLAN_COLUMN_PREFIX)
    )
    attached = unwritten_cd_plan_columns(columns, registry_ladders.attached)
    assert attached == [c for c in legacy if c not in PLAN_COLUMNS]
    for ladder, expected in ((None, legacy), (registry_ladders.attached, attached)):
        if expected:
            with pytest.raises(ValueError, match="would not rewrite"):
                refuse_unwritten_cd_plan_columns(frame, block_ladder=ladder)
        else:
            refuse_unwritten_cd_plan_columns(frame, block_ladder=ladder)


if hypothesis is not None:

    @hypothesis.settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[
            hypothesis.HealthCheck.too_slow,
            hypothesis.HealthCheck.function_scoped_fixture,
        ],
    )
    @hypothesis.given(scenario=st.composite(_pool_scenario)())
    def test__registry__attaching_changes_only_plan_columns_for_every_mix(
        registry_ladders: RegistryLadders, scenario
    ) -> None:
        _attached_pool_invariants(registry_ladders, scenario)
        # The shared pool invariants hold on the attached ladder as well.
        _pool_invariants(registry_ladders.attached, scenario)

    @hypothesis.settings(max_examples=200, deadline=None)
    @hypothesis.given(
        columns=st.lists(st.sampled_from(_COLUMN_NAMES), max_size=12, unique=True)
    )
    def test__stale_plans__the_legacy_rule_agrees_with_the_shared_helper(
        registry_ladders: RegistryLadders, columns
    ) -> None:
        _unwritten_properties(registry_ladders, columns)

else:  # pragma: no cover

    @pytest.mark.skip(reason="requires hypothesis")
    def test__registry__attaching_changes_only_plan_columns_for_every_mix() -> None:
        pass

    @pytest.mark.skip(reason="requires hypothesis")
    def test__stale_plans__the_legacy_rule_agrees_with_the_shared_helper() -> None:
        pass


# The staging tool ---------------------------------------------------------------


def test__legacy_tool__cd_plan_registry_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invariants 3, 4 and 6 through main: the pinned registry is attached,
    threaded into the pool, gated, and its digest recorded everywhere as the
    loaded file's."""

    builder, _ladder, exit_code, captured, summary = _run_block_v1_main(
        monkeypatch, tmp_path, registry=True
    )
    setup = captured["registry"]
    digest = file_sha256(setup.registry_path)
    assert exit_code == 0
    assert setup.digest == digest
    assert captured["ladder_loads"] == 1

    build_options = dict(captured["build"])
    block_ladder = build_options.pop("block_ladder")
    assert set(block_ladder.congressional_district_plans) == set(
        REGISTRY_PLAN_APPORTIONMENT
    )
    record = build_options.pop("cd_plan_registry")
    assert record == setup.record
    assert record["sha256"] == digest
    for key in ("puma_ladder", "hours_donor_factory", "target_families"):
        build_options.pop(key)
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
    provenance_location = summary["orchestration"]["provenance"]["household_location"]
    assert location["cd_plan_registry"] == setup.record
    assert location["cd_plan_registry"] == provenance_location["cd_plan_registry"]
    assert location["cd_plan_registry_path"] == str(setup.registry_path.resolve())
    assert summary["orchestration"]["cd_plan_registry_sha256"] == digest
    attributes = captured["write"]["root_attributes"]
    assert set(attributes) == NO_REGISTRY_ROOT_ATTRIBUTES | {
        POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR
    }
    assert attributes[POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR] == digest

    gate = location["gate"]
    assert gate["passed"] is True
    assert set(PLAN_COLUMNS) <= set(gate["details"]["columns"])
    plans = summary["geography_ladder"]["block_ladder"]["congressional_district_plans"]
    assert set(plans) == set(REGISTRY_PLAN_APPORTIONMENT)
    written = captured["written_frame"].table("household")
    _assert_plan_columns_are_registry_lookups(setup, written, record)
    limitation = summary["reviewed_limitations"][2]
    assert limitation["assigned_geography"][-3:] == list(PLAN_COLUMNS)
    assert limitation["treatment"].startswith(NO_REGISTRY_TREATMENT)
    assert "congressional-district plan registry" in limitation["treatment"]


def test__legacy_tool__block_v1_without_the_registry_records_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invariant 2: no flag, no new kwarg, key, attribute or wording."""

    _builder, _ladder, exit_code, captured, summary = _run_block_v1_main(
        monkeypatch, tmp_path
    )
    assert exit_code == 0
    assert "cd_plan_registry" not in captured["build"]
    assert set(captured["write"]["root_attributes"]) == NO_REGISTRY_ROOT_ATTRIBUTES
    location = summary["household_location"]
    assert "cd_plan_registry" not in location
    assert "cd_plan_registry_path" not in location
    assert "cd_plan_registry_sha256" not in summary["orchestration"]
    assert summary["reviewed_limitations"][2]["treatment"] == NO_REGISTRY_TREATMENT
    assert not any(
        str(column).startswith(US_CD_PLAN_COLUMN_PREFIX)
        for column in captured["written_frame"].table("household").columns
    )


def test__legacy_tool__the_registry_changes_only_its_own_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Differential through main on the same registry-ready ladder: the run
    with the flag writes the same frame plus the plan columns, and its
    summary differs only by the registry's own keys."""

    runs = {}
    for flag in (False, True):
        directory = tmp_path / ("with" if flag else "without")
        directory.mkdir()
        runs[flag] = _run_block_v1_main(
            monkeypatch,
            directory,
            registry=True,
            pool_with_registry=flag,
            pass_registry_flag=flag,
        )
    _, _, _, plain_captured, plain_summary = runs[False]
    _, _, _, captured, summary = runs[True]
    plain_household = plain_captured["written_frame"].table("household")
    household = captured["written_frame"].table("household")
    pd.testing.assert_frame_equal(
        household[list(plain_household.columns)], plain_household
    )
    assert set(household.columns) - set(plain_household.columns) == set(PLAN_COLUMNS)
    assert set(summary["household_location"]) - set(
        plain_summary["household_location"]
    ) == {"cd_plan_registry", "cd_plan_registry_path"}
    assert set(summary["orchestration"]) - set(plain_summary["orchestration"]) == {
        "cd_plan_registry_sha256"
    }
    assert set(captured["write"]["root_attributes"]) - set(
        plain_captured["write"]["root_attributes"]
    ) == {POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR}
    for key in ("rows", "spine_totals", "household_weight_totals"):
        assert summary[key] == plain_summary[key], key
    for key in ("donor_blocks", "source_geography_households", "households", "seed"):
        assert (
            summary["household_location"][key]
            == plain_summary["household_location"][key]
        ), key


def test__legacy_tool__a_registry_flag_with_an_unattached_pool_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The hard gate uses the attached ladder, so a pool without the plan
    columns cannot pass as a registry run."""

    with pytest.raises(SystemExit, match="omitted household column"):
        _run_block_v1_main(
            monkeypatch, tmp_path, registry=True, pool_with_registry=False
        )
    assert not (tmp_path / "combined.h5").exists()


@pytest.mark.parametrize(
    ("registry", "pass_flag", "pool_columns"),
    [
        (False, False, {PLAN_COLUMNS[0]: 101}),
        (True, True, {f"{US_CD_PLAN_COLUMN_PREFIX}116th_congress": 101}),
        (True, False, None),
    ],
    ids=["no-registry", "unattached-plan", "flag-dropped"],
)
def test__legacy_tool__refuses_pool_plan_columns_the_ladder_does_not_attach(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registry: bool,
    pass_flag: bool,
    pool_columns: dict | None,
) -> None:
    """Invariant 5 at the tool's hard gate: the gate verifies exactly the
    columns the run's ladder attaches, so a plan column the pool returns
    beyond them (a plan column without the flag, a plan the registry does not
    attach, or a registry pool whose run dropped the flag) is refused by name
    before the recorded-registry check, and nothing is written."""

    captured: dict = {}
    with pytest.raises(SystemExit, match="no gate can verify them") as caught:
        _run_block_v1_main(
            monkeypatch,
            tmp_path,
            registry=registry,
            pass_registry_flag=pass_flag,
            pool_columns=pool_columns,
            captured=captured,
        )
    expected = list(pool_columns) if pool_columns else list(PLAN_COLUMNS)
    assert all(column in str(caught.value) for column in expected)
    assert "write" not in captured
    assert not (tmp_path / "combined.h5").exists()
    assert not (tmp_path / "combined.summary.json").exists()


def test__legacy_tool__a_recorded_registry_other_than_the_attached_one_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="records CD-plan registry 'fff"):
        _run_block_v1_main(
            monkeypatch,
            tmp_path,
            registry=True,
            provenance_registry={"sha256": "f" * 64},
        )
    assert not (tmp_path / "combined.h5").exists()


@pytest.mark.parametrize("case", ["unpinned", "missing"])
def test__legacy_tool__refuses_a_wrong_registry_before_reading_the_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    """Invariant 3: only the pinned file loads, and the refusal comes before
    the block ladder or the donor base's attributes or tables are read."""

    options = (
        {"repin": "0" * 64}
        if case == "unpinned"
        else {"registry_flag_path": tmp_path / "no-such-registry.npz"}
    )
    captured: dict = {}
    with pytest.raises(SystemExit, match="--cd-plan-registry .* was refused") as caught:
        _run_block_v1_main(
            monkeypatch, tmp_path, registry=True, captured=captured, **options
        )
    assert ("SHA-256" if case == "unpinned" else "not found") in str(caught.value)
    assert captured["ladder_loads"] == 0
    assert captured["base_rule_reads"] == 0
    assert captured["base_loads"] == 0
    assert not (tmp_path / "combined.h5").exists()


def test__legacy_tool__refuses_a_pinned_registry_that_contradicts_the_ladder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pinned registry whose current (119th) plan disagrees with the block
    ladder on one block cannot attach (the primary plan is a differential
    check, never overridden): refused before the donor base is read."""

    arrays = registry_ready_blocks(synthetic_blocks(3, states=STATES))
    blocks = np.asarray(arrays["block_geoid"], dtype=np.int64)
    districts = np.asarray(arrays["congressional_district_geoid"], dtype=np.int64)
    new_york = int(np.flatnonzero(blocks // 10**13 == 36)[0])
    moved = 3600 + int(districts[new_york]) % 100 % 26 + 1
    assert moved != districts[new_york]
    directory = tmp_path / "contradicting"
    directory.mkdir()
    path = write_cd_plan_registry(
        directory / "registry.npz",
        arrays,
        overrides={PRIMARY_CD_PLAN: {int(blocks[new_york]): moved}},
    )
    captured: dict = {}
    with pytest.raises(SystemExit, match="was refused: .*disagrees") as caught:
        _run_block_v1_main(
            monkeypatch,
            tmp_path,
            registry=True,
            registry_flag_path=path,
            repin=file_sha256(path),
            captured=captured,
        )
    assert PRIMARY_CD_PLAN in str(caught.value)
    assert captured["ladder_loads"] == 1
    assert captured["base_rule_reads"] == 0
    assert captured["base_loads"] == 0
    assert not (tmp_path / "combined.h5").exists()


@pytest.mark.parametrize("with_registry", [True, False])
def test__legacy_tool__refuses_a_base_located_with_another_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_registry: bool
) -> None:
    """A base whose location attached another registry (or one, when this run
    attaches none) would have its donors' plan columns silently replaced or
    dropped; it is refused before the base is loaded."""

    captured: dict = {}
    with pytest.raises(SystemExit, match="located with CD-plan registry eee") as caught:
        _run_block_v1_main(
            monkeypatch,
            tmp_path,
            registry=with_registry,
            base_attributes={
                "populace_location_rule": "block_v1",
                POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR: "e" * 64,
            },
            captured=captured,
        )
    assert ("attaches none" in str(caught.value)) is not with_registry
    assert captured["base_loads"] == 0
    assert not (tmp_path / "combined.h5").exists()


def test__legacy_tool__accepts_a_base_located_with_the_same_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, exit_code, _captured, summary = _run_block_v1_main(
        monkeypatch,
        tmp_path,
        registry=True,
        base_attributes=lambda setup: {
            "populace_location_rule": "block_v1",
            POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR: setup.digest,
        },
    )
    assert exit_code == 0
    recorded = summary["household_location"]["donor_base_location"]["attributes"]
    assert (
        recorded[POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR]
        == (summary["orchestration"]["cd_plan_registry_sha256"])
    )


@pytest.mark.parametrize("registry", [False, True])
def test__legacy_tool__refuses_base_plan_columns_the_run_would_not_rewrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, registry: bool
) -> None:
    """Invariant 7 at the tool: the probe scenario (plan columns, no registry)
    and a plan the registry does not attach are refused right after the base
    loads, before the ACS archives are fetched."""

    column = (
        f"{US_CD_PLAN_COLUMN_PREFIX}116th_congress" if registry else PLAN_COLUMNS[0]
    )
    captured: dict = {}
    with pytest.raises(SystemExit, match=f"--base-h5: .*{column}"):
        _run_block_v1_main(
            monkeypatch,
            tmp_path,
            registry=registry,
            tool_base_columns={column: 101},
            captured=captured,
        )
    assert captured["base_loads"] == 1
    assert captured["fetches"] == 0


def test__legacy_tool__accepts_base_plan_columns_the_registry_rewrites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, exit_code, _captured, _summary = _run_block_v1_main(
        monkeypatch,
        tmp_path,
        registry=True,
        tool_base_columns={column: 101 for column in PLAN_COLUMNS},
    )
    assert exit_code == 0


def test__legacy_tool__refuses_a_registry_path_that_is_an_output(
    tmp_path: Path,
) -> None:
    builder = _load_legacy_builder()
    for target in ("out.h5", "out.summary.json"):
        args = builder._parse_args(
            [
                "--base-h5",
                str(tmp_path / "base.h5"),
                "--out-h5",
                str(tmp_path / "out.h5"),
                "--location-rule",
                "block_v1",
                "--block-ladder",
                str(tmp_path / "blocks.npz"),
                "--cd-plan-registry",
                str(tmp_path / target),
            ]
        )
        with pytest.raises(SystemExit, match="--cd-plan-registry must differ"):
            builder._validate_artifact_paths(
                args, summary_path=tmp_path / "out.summary.json"
            )


def test__legacy_tool__root_attributes_add_the_registry_only_when_given() -> None:
    builder = _load_legacy_builder()
    options = {
        "location_rule": "block_v1",
        "location_seed": 4,
        "location_clones": 1,
        "block_ladder_sha256": "a" * 64,
        "block_ladder_vintages": {"block": "2020_census"},
        "puma_ladder_sha256": "b" * 64,
        "puma_ladder_vintages": {"puma": "2020_puma"},
    }
    plain = builder.location_root_attributes(**options)
    assert set(plain) == NO_REGISTRY_ROOT_ATTRIBUTES
    assert builder.location_root_attributes(
        **options, cd_plan_registry_sha256=None
    ) == (plain)
    attached = builder.location_root_attributes(
        **options, cd_plan_registry_sha256="c" * 64
    )
    assert attached == {**plain, POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR: "c" * 64}


@requires_pytables
def test__legacy_tool__the_registry_attribute_round_trips_and_is_read_back(
    tmp_path: Path, registry_ladders: RegistryLadders
) -> None:
    """The staging writer stores the registry attribute; a later run reading
    that H5 as its base (or a base-H5-line H5 written with h5py) records it."""

    builder = _load_legacy_builder()
    pooled = _registry_pool(
        registry_ladders,
        _donor_base(registry_ladders.plain, blocks="some"),
        _acs(registry_ladders.plain),
    )
    attributes = builder.location_root_attributes(
        location_rule="block_v1",
        location_seed=4,
        location_clones=1,
        block_ladder_sha256=str(registry_ladders.attached.sha256),
        block_ladder_vintages=registry_ladders.attached.layer_vintages,
        puma_ladder_sha256="0" * 64,
        puma_ladder_vintages={"congressional_district": "119th_congress"},
        cd_plan_registry_sha256=registry_ladders.digest,
    )
    path = tmp_path / "staging.h5"
    builder._write_dataset(pooled, path, period=2024, root_attributes=attributes)
    assert builder._base_location_rule(path) == {
        "location_rule": "block_v1",
        "recorded": True,
        "attributes": {
            "populace_location_rule": "block_v1",
            "populace_location_seed": "4",
            "populace_location_clones": "1",
            POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR: registry_ladders.digest,
        },
    }
    h5py = pytest.importorskip("h5py")
    base = tmp_path / "base.h5"
    builder._write_dataset(_donor_base(registry_ladders.plain), base, period=2024)
    with h5py.File(base, "a") as h5:
        h5.attrs["populace_location_rule"] = "block_v1"
        h5.attrs[POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR] = registry_ladders.digest
    assert builder._base_location_rule(base)["attributes"] == {
        "populace_location_rule": "block_v1",
        POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR: registry_ladders.digest,
    }


# The ACS local release ------------------------------------------------------


def _registry_staging_summary(ladder_path: Path, sha256: str, digest: str) -> dict:
    summary = _block_v1_staging_summary(ladder_path, sha256)
    summary["orchestration"]["cd_plan_registry_sha256"] = digest
    summary["household_location"]["cd_plan_registry"] = {
        "sha256": digest,
        "plans": sorted(REGISTRY_PLAN_APPORTIONMENT),
    }
    summary["household_location"]["cd_plan_registry_path"] = str(
        ladder_path.with_name("registry.npz")
    )
    return summary


def test__release__staging_registry_is_read_only_when_recorded(tmp_path: Path) -> None:
    release = _load_release_tool()
    assert release.staging_cd_plan_registry({}) is None
    no_registry = _block_v1_staging_summary(tmp_path / "b.npz", "a" * 64)
    assert release.staging_cd_plan_registry(no_registry) is None
    summary = _registry_staging_summary(tmp_path / "b.npz", "a" * 64, "d" * 64)
    assert release.staging_cd_plan_registry(summary)["sha256"] == "d" * 64


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda s: s["orchestration"].pop("cd_plan_registry_sha256"),
        lambda s: s["household_location"].pop("cd_plan_registry"),
        lambda s: s["household_location"].pop("cd_plan_registry_path"),
        lambda s: s["orchestration"].update(cd_plan_registry_sha256="e" * 64),
        lambda s: s["household_location"]["cd_plan_registry"].pop("sha256"),
    ],
    ids=["no-orchestration-sha", "no-record", "no-path", "shas-differ", "no-sha"],
)
def test__release__an_inconsistent_registry_record_is_refused(
    tmp_path: Path, corrupt
) -> None:
    release = _load_release_tool()
    summary = _registry_staging_summary(tmp_path / "b.npz", "a" * 64, "d" * 64)
    corrupt(summary)
    with pytest.raises(SystemExit, match="records a CD-plan registry inconsistently"):
        release.staging_cd_plan_registry(summary)
    with pytest.raises(SystemExit, match="inconsistently"):
        release.staging_refresh_recipe(summary)


def test__release__a_legacy_summary_naming_a_registry_is_refused() -> None:
    release = _load_release_tool()
    with pytest.raises(SystemExit, match="inconsistently"):
        release.staging_cd_plan_registry(
            {"orchestration": {"cd_plan_registry_sha256": "d" * 64}}
        )


def test__release__recipes_name_the_registry_only_for_a_registry_run(
    tmp_path: Path,
) -> None:
    release = _load_release_tool()
    builder = _load_legacy_builder()
    plain = release.staging_refresh_recipe(
        _block_v1_staging_summary(tmp_path / "b.npz", "a" * 64)
    )
    assert plain == (
        f"{release.LEGACY_STAGING_REFRESH_RECIPE} --location-rule block_v1 "
        "--location-seed 7 --block-ladder <us_block_ladder_2020_puma.npz>"
    )
    assert release.staging_refresh_recipe({}) == release.LEGACY_STAGING_REFRESH_RECIPE
    recipe = release.staging_refresh_recipe(
        _registry_staging_summary(tmp_path / "b.npz", "a" * 64, "d" * 64)
    )
    assert recipe == f"{plain} --cd-plan-registry <us_cd_plan_registry_2020.npz>"
    tokens = shlex.split(recipe)
    args = builder._parse_args(tokens[3:])
    assert args.location_rule == "block_v1"
    assert args.cd_plan_registry is not None
    builder._validate_location_arguments(args)


def test__release__the_orchestration_projection_carries_the_registry_sha(
    tmp_path: Path,
) -> None:
    release = _load_release_tool()
    plain = release._require_uncapped_staging(
        _block_v1_staging_summary(tmp_path / "b.npz", "a" * 64)
    )
    assert "cd_plan_registry_sha256" not in plain
    projected = release._require_uncapped_staging(
        _registry_staging_summary(tmp_path / "b.npz", "a" * 64, "d" * 64)
    )
    assert projected["cd_plan_registry_sha256"] == "d" * 64
    assert projected["household_location"]["cd_plan_registry"]["sha256"] == "d" * 64
    assert list(projected)[: len(release._STAGING_ORCHESTRATION_KEYS)] == list(
        release._STAGING_ORCHESTRATION_KEYS
    )


@pytest.mark.parametrize("registry", [False, True])
def test__release__calibrated_attributes_equal_the_staging_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, registry: bool
) -> None:
    """Differential: the release derives the calibrated artifact's location
    attributes from the staging summary; they must equal what the staging
    tool wrote on its H5, with and without the registry."""

    release = _load_release_tool()
    _, _, exit_code, captured, summary = _run_block_v1_main(
        monkeypatch, tmp_path, registry=registry
    )
    assert exit_code == 0
    written = captured["write"]["root_attributes"]
    derived = release.calibrated_location_root_attributes(
        summary, targets_ladder_sha256=summary["geography_ladder"]["sha256"]
    )
    assert derived == written
    assert (POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR in derived) is registry
    projected = release._require_uncapped_staging(summary)
    assert ("cd_plan_registry_sha256" in projected) is registry


def test__release__the_block_ladder_reload_reattaches_the_staged_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finalize's ladder is the one households were located with: the
    recorded registry is re-attached (sha-checked and pinned), so the
    finalize gate covers the plan columns."""

    release = _load_release_tool()
    _, _, _, captured, summary = _run_block_v1_main(
        monkeypatch, tmp_path, registry=True
    )
    setup = captured["registry"]
    ladder = release.load_staging_block_ladder(summary)
    assert set(ladder.congressional_district_plans) == set(REGISTRY_PLAN_APPORTIONMENT)
    for plan, values in setup.attached.congressional_district_plans.items():
        np.testing.assert_array_equal(ladder.congressional_district_plans[plan], values)
    written = captured["written_frame"].table("household")
    gate = us_block_location_gate(written, ladder)
    assert gate.passed and set(PLAN_COLUMNS) <= set(gate.details["columns"])
    plain = release.load_staging_block_ladder(
        _block_v1_staging_summary(setup.ladder_path, file_sha256(setup.ladder_path))
    )
    assert not us_block_location_gate(
        written.drop(columns=list(PLAN_COLUMNS)), ladder
    ).passed
    assert us_block_location_gate(written, plain).passed  # blind to plan columns

    # Relocated bytes are accepted through the override.
    relocated = tmp_path / "moved" / "registry.npz"
    relocated.parent.mkdir()
    relocated.write_bytes(setup.registry_path.read_bytes())
    assert set(
        release.load_staging_block_ladder(
            summary, cd_plan_registry=relocated
        ).congressional_district_plans
    ) == set(REGISTRY_PLAN_APPORTIONMENT)


def test__release__the_registry_reload_refuses_other_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = _load_release_tool()
    _, _, _, captured, summary = _run_block_v1_main(
        monkeypatch, tmp_path, registry=True
    )
    setup = captured["registry"]
    with pytest.raises(SystemExit, match="missing; pass --cd-plan-registry"):
        release.load_staging_block_ladder(
            summary, cd_plan_registry=tmp_path / "gone.npz"
        )
    other = tmp_path / "other.npz"
    write_cd_plan_registry(other, setup.arrays, known_deviations=False)
    with pytest.raises(SystemExit, match="the block_v1 staging run attached"):
        release.load_staging_block_ladder(summary, cd_plan_registry=other)
    # The recorded bytes, but no longer the package pin.
    from microcosm.build.us_runtime import cd_plan_registry

    monkeypatch.setattr(
        cd_plan_registry,
        "packaged_us_cd_plan_registry_provenance",
        lambda: {"output_sha256": "0" * 64},
    )
    with pytest.raises(SystemExit, match="was refused: .*SHA-256"):
        release.load_staging_block_ladder(summary)
    pin_cd_plan_registry(monkeypatch, setup.registry_path)
    edited = json.loads(json.dumps(summary))
    edited["household_location"]["cd_plan_registry"]["plans"] = ["117th_congress"]
    with pytest.raises(SystemExit, match="re-attaches plans"):
        release.load_staging_block_ladder(edited)


def test__release__a_registry_override_needs_a_registry_run(tmp_path: Path) -> None:
    release = _load_release_tool()
    path = write_location_ladder(
        tmp_path / "blocks.npz", synthetic_blocks(3, states=STATES)
    )
    plain = _block_v1_staging_summary(path, release._sha256(path))
    with pytest.raises(SystemExit, match="records none"):
        release.load_staging_block_ladder(plain, cd_plan_registry=tmp_path / "r.npz")
    with pytest.raises(SystemExit, match="legacy location rule"):
        release.load_staging_block_ladder({}, cd_plan_registry=tmp_path / "r.npz")
    args = release._parse_args(
        [
            "--stage",
            "finalize",
            "--staging-h5",
            str(tmp_path / "s.h5"),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--out-h5",
            str(tmp_path / "o.h5"),
        ]
    )
    assert args.cd_plan_registry is None


#: The finite space of what finalize can meet: the staging summary's
#: location record, the recorded registry file, the package pin and the
#: release's --cd-plan-registry override.
_REGISTRY_SUMMARY_KINDS = ("legacy", "plain", "registry", "inconsistent")
_REGISTRY_RECORDED_FILES = ("present", "missing", "other_bytes")
_REGISTRY_PINS = ("current", "moved")
_REGISTRY_OVERRIDES = (None, "relocated", "missing")
#: Preflight stage selections: ``all`` and ``finalize`` need the file,
#: ``calibrate`` only a consistent record, the rest nothing.
_PREFLIGHT_STAGES = {
    "all": ["materialize", "calibrate", "qa", "finalize", "package"],
    "finalize": ["finalize"],
    "calibrate": ["calibrate"],
    "materialize": ["materialize"],
    "qa": ["qa"],
    "package": ["package"],
}


def _registry_file_case(
    setup: RegistryLadders,
    directory: Path,
    patch: pytest.MonkeyPatch,
    *,
    kind: str,
    recorded: str,
    pin: str,
    override: str | None,
) -> tuple[dict, Path | None, str | None]:
    """One staging summary, files and pin; returns ``(summary, override path,
    expected)`` with ``expected`` the refusal pattern finalize must raise, or
    ``None`` when it must re-attach (the specification, written out)."""

    from microcosm.build.us_runtime import cd_plan_registry

    ladder_path = directory / "blocks.npz"
    ladder_path.write_bytes(setup.ladder_path.read_bytes())
    registry_path = directory / "registry.npz"
    if recorded == "present":
        registry_path.write_bytes(setup.registry_path.read_bytes())
    elif recorded == "other_bytes":
        write_cd_plan_registry(registry_path, setup.arrays, known_deviations=False)
    override_path = None
    if override == "relocated":
        override_path = directory / "moved" / "registry.npz"
        override_path.parent.mkdir()
        override_path.write_bytes(setup.registry_path.read_bytes())
    elif override == "missing":
        override_path = directory / "gone" / "registry.npz"
    if pin == "current":
        pin_cd_plan_registry(patch, setup.registry_path)
    else:
        patch.setattr(
            cd_plan_registry,
            "packaged_us_cd_plan_registry_provenance",
            lambda: {"output_sha256": "0" * 64},
        )
    if kind == "legacy":
        summary: dict = {"orchestration": {"max_households": None}}
    else:
        summary = _block_v1_staging_summary(ladder_path, file_sha256(ladder_path))
    if kind in {"registry", "inconsistent"}:
        summary["orchestration"]["cd_plan_registry_sha256"] = setup.digest
        location = summary["household_location"]
        location["cd_plan_registry"] = json.loads(json.dumps(setup.record))
        location["cd_plan_registry_path"] = str(registry_path)
    if kind == "inconsistent":
        del summary["orchestration"]["cd_plan_registry_sha256"]

    if kind == "legacy":
        expected = None if override is None else "legacy location rule"
    elif kind == "plain":
        expected = None if override is None else "records none"
    elif kind == "inconsistent":
        expected = "inconsistently"
    else:
        state = recorded if override is None else {
            "relocated": "present",
            "missing": "missing",
        }[override]
        expected = {
            "missing": "missing; pass --cd-plan-registry",
            "other_bytes": "the block_v1 staging run attached",
            "present": None if pin == "current" else "was refused: .*SHA-256",
        }[state]
    return summary, override_path, expected


@pytest.mark.parametrize("override", _REGISTRY_OVERRIDES)
@pytest.mark.parametrize("pin", _REGISTRY_PINS)
@pytest.mark.parametrize("recorded", _REGISTRY_RECORDED_FILES)
@pytest.mark.parametrize("kind", _REGISTRY_SUMMARY_KINDS)
def test__release__preflight_and_finalize_agree_on_every_registry_case(
    registry_ladders: RegistryLadders,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    recorded: str,
    pin: str,
    override: str | None,
) -> None:
    """Invariant 3 on the release, exhaustively over the finite case space
    (summary kind x recorded file x pin x override): finalize's reload
    (``load_staging_block_ladder``) re-attaches exactly when the written-out
    specification says so and otherwise refuses with its reason before the
    block ladder is loaded; ``main``'s preflight agrees with it whenever
    finalize is among the stages, refuses only an inconsistent record for
    calibrate alone and nothing otherwise, never loads a ladder, and hashes
    nothing for a staging run without a registry."""

    from microcosm.build.us_runtime import block_location as block_location_runtime

    release = _load_release_tool()
    summary, override_path, expected = _registry_file_case(
        registry_ladders,
        tmp_path,
        monkeypatch,
        kind=kind,
        recorded=recorded,
        pin=pin,
        override=override,
    )
    loads: list[Path] = []
    hashes: list[Path] = []
    load_ladder = block_location_runtime.load_us_location_ladder
    sha256 = release._sha256

    def counted_load(path):
        loads.append(Path(path))
        return load_ladder(path)

    def counted_sha256(path):
        hashes.append(Path(path))
        return sha256(path)

    monkeypatch.setattr(block_location_runtime, "load_us_location_ladder", counted_load)
    monkeypatch.setattr(release, "_sha256", counted_sha256)

    if expected is None:
        ladder = release.load_staging_block_ladder(
            summary, cd_plan_registry=override_path
        )
        if kind == "legacy":
            assert ladder is None
        else:
            plans = set(ladder.congressional_district_plans)
            assert plans == (
                set(REGISTRY_PLAN_APPORTIONMENT)
                if kind == "registry"
                else {PRIMARY_CD_PLAN}
            )
    else:
        with pytest.raises(SystemExit, match=expected):
            release.load_staging_block_ladder(summary, cd_plan_registry=override_path)
        assert loads == []

    summary_path = tmp_path / "staging.summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    for label, stages in _PREFLIGHT_STAGES.items():
        loads.clear()
        hashes.clear()
        args = SimpleNamespace(
            stages=list(stages),
            staging_summary=summary_path,
            staging_h5=tmp_path / "staging.h5",
            cd_plan_registry=override_path,
        )
        if "finalize" in stages:
            wanted = expected
        elif "calibrate" in stages and kind == "inconsistent":
            wanted = "inconsistently"
        else:
            wanted = None
        if wanted is None:
            release.preflight_staging_cd_plan_registry(args)
        else:
            with pytest.raises(SystemExit, match=wanted):
                release.preflight_staging_cd_plan_registry(args)
        assert loads == [], label
        if kind in {"legacy", "plain"}:
            assert hashes == [], label
        assert len(hashes) <= 1, label


@requires_pytables
def test__release__the_checkpoint_identity_binds_the_staging_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invariant 8: the local release's run identity pins the staging H5's
    sha256, and a registry run's staging bytes carry the registry (its plan
    columns and root attribute), so a checkpoint materialized from a run
    without the registry cannot resume one with it, and vice versa."""

    release = _load_release_tool()
    # The harness stubs the run's writer and base reader; write and read the
    # real staging bytes with an unpatched copy of the tool.
    writer = _load_legacy_builder()
    staged: dict[bool, Path] = {}
    for flag in (False, True):
        directory = tmp_path / ("with" if flag else "without")
        directory.mkdir()
        _builder, _, exit_code, captured, summary = _run_block_v1_main(
            monkeypatch,
            directory,
            registry=True,
            pool_with_registry=flag,
            pass_registry_flag=flag,
        )
        assert exit_code == 0
        path = directory / "staging.h5"
        writer._write_dataset(
            captured["written_frame"],
            path,
            period=2024,
            root_attributes=captured["write"]["root_attributes"],
        )
        attributes = writer._base_location_rule(path)["attributes"]
        recorded = attributes.get(POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR)
        assert recorded == summary["orchestration"].get("cd_plan_registry_sha256")
        assert (recorded is not None) is flag
        columns = set(release._load_staging_frame(path).table("household").columns)
        assert (columns & set(PLAN_COLUMNS)) == (set(PLAN_COLUMNS) if flag else set())
        staged[flag] = path
    for materialized in (False, True):
        checkpoint = tmp_path / f"checkpoint-{materialized}"
        checkpoint.mkdir()
        (checkpoint / "run_identity.json").write_text(
            json.dumps({"staging_sha256": release._sha256(staged[materialized])})
        )
        resumed = SimpleNamespace(
            checkpoint_dir=checkpoint, staging_h5=staged[materialized]
        )
        assert release._verify_run_identity(resumed)["staging_sha256"]
        resumed.staging_h5 = staged[not materialized]
        with pytest.raises(SystemExit, match="does not match the materialized"):
            release._verify_run_identity(resumed)


@pytest.fixture(scope="module")
def staging_tools():
    """The staging tool and the local release, loaded once for properties."""

    return _load_legacy_builder(), _load_release_tool()


def _staging_projection_properties(
    staging_tools, seed: int, digest: str | None
) -> None:
    """The release's views of a staging summary follow its registry record:
    the refresh recipe, the calibrated root attributes and the build-manifest
    projection name the registry exactly when the run attached one, and the
    calibrated attributes equal what the staging writer's own function
    produces from the same inputs (producer/consumer differential)."""

    builder, release = staging_tools
    ladder_path = Path("/ladders/blocks.npz")
    summary = (
        _block_v1_staging_summary(ladder_path, "a" * 64)
        if digest is None
        else _registry_staging_summary(ladder_path, "a" * 64, digest)
    )
    summary["orchestration"]["location_seed"] = seed
    summary["household_location"]["seed"] = seed
    registry = release.staging_cd_plan_registry(summary)
    assert (registry is None) is (digest is None)
    if registry is not None:
        assert registry["sha256"] == digest

    recipe = release.staging_refresh_recipe(summary)
    plain = (
        f"{release.LEGACY_STAGING_REFRESH_RECIPE} --location-rule block_v1 "
        f"--location-seed {seed} --block-ladder <us_block_ladder_2020_puma.npz>"
    )
    suffix = " --cd-plan-registry <us_cd_plan_registry_2020.npz>"
    assert recipe == (plain if digest is None else plain + suffix)
    args = builder._parse_args(shlex.split(recipe)[3:])
    builder._validate_location_arguments(args)
    assert args.location_seed == seed
    assert (args.cd_plan_registry is None) is (digest is None)

    attributes = release.calibrated_location_root_attributes(
        summary, targets_ladder_sha256="p" * 64
    )
    assert attributes == builder.location_root_attributes(
        location_rule="block_v1",
        location_seed=seed,
        location_clones=1,
        block_ladder_sha256="a" * 64,
        block_ladder_vintages={"congressional_district": "119th_congress"},
        puma_ladder_sha256="p" * 64,
        puma_ladder_vintages={},
        cd_plan_registry_sha256=digest,
    )
    expected = set(NO_REGISTRY_ROOT_ATTRIBUTES)
    if digest is not None:
        expected.add(POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR)
        assert attributes[POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR] == digest
    assert set(attributes) == expected

    projected = release._require_uncapped_staging(summary)
    assert ("cd_plan_registry_sha256" in projected) is (digest is not None)
    assert projected.get("cd_plan_registry_sha256") == digest


if hypothesis is not None:

    @hypothesis.settings(max_examples=80, deadline=None)
    @hypothesis.given(
        seed=st.integers(0, 2**31 - 1),
        digest=st.one_of(
            st.none(),
            st.text(alphabet="0123456789abcdef", min_size=64, max_size=64),
        ),
    )
    def test__release__staging_projections_follow_the_registry_record(
        staging_tools, seed: int, digest: str | None
    ) -> None:
        _staging_projection_properties(staging_tools, seed, digest)

else:  # pragma: no cover

    @pytest.mark.skip(reason="requires hypothesis")
    def test__release__staging_projections_follow_the_registry_record() -> None:
        pass
