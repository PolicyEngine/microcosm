"""The UK's conditioned area draw: definition, inputs and ladder (microcosm#931).

The spine's area draw (G2) conditions each household's area on its IMD decile
and its observed council-tax band, falling back along a declared ladder where
a cell has no mass. These tests hold the definition to canonical conditioned
supports, the ten ladder layers and the spine's stream, Northern Ireland to
households; the draw inputs to O14 iv and v, SPI-channel rows to missing
inputs; each household's level to a per-row reading of the toy supports; and
today's region-only definition to its bytes. Toy supports and invented
households only; no refusal prints a key, a code of a household or a count.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from microcosm.build.atomic_geography import (
    assign_atomic,
    decode_atomic_support,
    encode_atomic_support,
)
from microcosm.build.source_lineage import copy_paths
from microcosm.build.uk_runtime import atomic_area_conditioning as conditioning
from microcosm.build.uk_runtime.atomic_area_conditioning import (
    BAND_INPUT_COLUMN,
    DECILE_INPUT_COLUMN,
    DEFAULT_WEIGHT_COLUMN,
    DRAW_INPUT_COLUMNS,
    IDENTITY_COLUMN,
    REGION_COLUMN,
    UK_ATOMIC_FALLBACK_LEVELS,
    UK_DRAW_CONSTRAINTS,
    UK_FALLBACK_STATUSES,
    UK_SPINE_ATOMIC_AREA_SEED,
    UK_SPINE_ATOMIC_AREA_STREAM,
    UK_SPINE_LADDER_COLUMNS,
    uk_band_weight_map,
    uk_conditioned_assignment_definition,
    uk_conditioned_supports,
    uk_draw_frame,
    uk_draw_inputs,
    uk_draw_levels,
    uk_fallback_reasons,
    uk_support_channel_copies,
    uk_system_fallback_levels,
)
from microcosm.build.uk_runtime.atomic_area_support import (
    ASSIGNMENT_COLUMNS,
    BAND_WEIGHT_COLUMNS,
    CONDITIONED_SOURCES,
    IMD_DECILE_COLUMN,
    IMD_DECILES,
    UK_NATIVE_ALIAS_COLUMNS,
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.atomic_area_support import (
    IDENTITY_COLUMN as SUPPORT_IDENTITY_COLUMN,
)
from microcosm.build.uk_runtime.council_tax_constants import UK_COUNCIL_TAX_BANDS
from microcosm.build.uk_runtime.household_lineage import UK_COPY_LINEAGE
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.uk_atomic_area_support import payloads
from test_support.microcosm_build.uk_atomic_support_fixtures import toy_support_payloads
from test_support.microcosm_build.uk_conditioned_support_fixtures import (
    EW,
    NI,
    SCOT,
    TOY_HOUSEHOLDS,
    ToyHousehold,
    toy_conditioned_support_payloads,
    toy_conditioned_supports,
    toy_household_table,
)

_STRING = pd.StringDtype(storage="python")
#: Today's region-only definition on the invented supports, seed 42: its
#: canonical bytes' SHA-256 at the parent of the conditioned draw. The national
#: problem, the dense measures, the inspection export and the geography
#: identity hash atomic_area_support, so the conditioned draw lives in a
#: module of its own and leaves this definition byte for byte (C-28).
_V1_DEFINITION_SHA256 = (
    "5947915e81ccf8a8fb03f7b856b4309b7d59adac4b86322b07e0d9b2020cdf0d"
)


@pytest.fixture(scope="module")
def definition() -> dict:
    return uk_conditioned_assignment_definition(
        toy_conditioned_support_payloads(), seed=UK_SPINE_ATOMIC_AREA_SEED
    )


@pytest.fixture(scope="module")
def supports() -> dict:
    return toy_conditioned_supports()


def _households(cases=TOY_HOUSEHOLDS, **lineage) -> pd.DataFrame:
    """The cases' raw inputs with invented keys and sources, on an index that
    is not a range, so the adapter must keep the table's own."""

    household = toy_household_table(cases)
    household.index = pd.Index(np.arange(len(cases)) * 7 + 3, name="row")
    household[IDENTITY_COLUMN] = pd.Series(
        [f"invented-key-{position}" for position in range(len(cases))],
        index=household.index,
        dtype=_STRING,
    )
    household["source_household_id"] = np.arange(1, len(cases) + 1, dtype=np.int64)
    for column, values in lineage.items():
        household[column] = values
    return household


def _system_rows(definition: dict, system: str) -> dict:
    (row,) = [entry for entry in definition["systems"] if entry["id"] == system]
    return row


# --- The definition ----------------------------------------------------------------


def test_the_definition_declares_the_conditioned_draw(definition) -> None:
    assert definition["identity"] == [IDENTITY_COLUMN]
    assert IDENTITY_COLUMN == SUPPORT_IDENTITY_COLUMN
    assert definition["stream"] == [
        "sha256-u53-v1",
        UK_SPINE_ATOMIC_AREA_STREAM,
        0,
        UK_SPINE_ATOMIC_AREA_SEED,
    ]
    assert definition["outputs"] == dict(
        zip(("area", "system", "basis"), ASSIGNMENT_COLUMNS, strict=True)
    )
    for system, sources in CONDITIONED_SOURCES.items():
        entry = _system_rows(definition, system)
        assert entry["source"] == sources
        assert entry["observed_area"] is None
        assert entry["constraints"] == [
            {"input": REGION_COLUMN, "support": "frs_region", "required": True},
            {
                "input": DECILE_INPUT_COLUMN,
                "support": IMD_DECILE_COLUMN,
                "required": False,
            },
        ]


def test_great_britain_samples_the_bands_weight_and_northern_ireland_households(
    definition,
) -> None:
    for system in (EW, SCOT):
        (stage,) = _system_rows(definition, system)["stages"]
        assert stage == {
            "level": "area",
            "weight_by": {
                "input": BAND_INPUT_COLUMN,
                "map": uk_band_weight_map(system),
                "default": DEFAULT_WEIGHT_COLUMN,
            },
        }
        assert sorted(uk_band_weight_map(system).values()) == sorted(
            BAND_WEIGHT_COLUMNS[system]
        )
    assert sorted(uk_band_weight_map(EW)) == list("ABCDEFGHI")
    assert sorted(uk_band_weight_map(SCOT)) == list("ABCDEFGH")
    # Domestic rates: no band weights, so no weight_by in Northern Ireland.
    assert _system_rows(definition, NI)["stages"] == [
        {"level": "area", "weight": DEFAULT_WEIGHT_COLUMN}
    ]
    assert uk_band_weight_map(NI) == {}
    assert [level.name for level in uk_system_fallback_levels(NI)] == ["L1", "L3"]
    assert uk_system_fallback_levels(EW) == UK_ATOMIC_FALLBACK_LEVELS


def test_the_layers_are_the_ten_ladder_columns_and_carry_the_supports_metadata(
    definition, supports
) -> None:
    for entry in definition["systems"]:
        outputs = [layer["output"] for layer in entry["layers"]]
        assert outputs == list(UK_SPINE_LADDER_COLUMNS)
        assert not set(outputs) & set(UK_NATIVE_ALIAS_COLUMNS)
        columns = supports[entry["id"]].metadata["columns"]
        for layer in entry["layers"]:
            assert layer["input"] == (
                "area" if layer["output"] == "oa_code" else layer["output"]
            )
            assert {key: layer[key] for key in ("source", "vintage", "relation")} == {
                key: columns[layer["input"]][key]
                for key in ("source", "vintage", "relation")
            }


def test_the_definition_takes_canonical_conditioned_bytes_only() -> None:
    conditioned = toy_conditioned_support_payloads()
    # Today's supports, without the conditioning columns.
    with pytest.raises(ValueError, match="conditioned support columns differ"):
        uk_conditioned_assignment_definition(toy_support_payloads(), seed=0)
    # Self-consistent bytes the adapter would not write: one area's FRS
    # region disagrees with its region code.
    support = decode_atomic_support(conditioned[EW])
    metadata = {
        **dict(support.metadata),
        "columns": {
            name: dict(column) for name, column in support.metadata["columns"].items()
        },
    }
    arrays = {name: np.asarray(array).copy() for name, array in support.arrays.items()}
    arrays["frs_region"][0] = (
        "WALES" if arrays["frs_region"][0] != "WALES" else "LONDON"
    )
    forged = encode_atomic_support(metadata, arrays)
    with pytest.raises(ValueError, match="not the canonical conditioned output"):
        uk_conditioned_assignment_definition({**conditioned, EW: forged}, seed=0)
    with pytest.raises(ValueError, match="three systems"):
        uk_conditioned_assignment_definition(
            {EW: conditioned[EW], SCOT: conditioned[SCOT]}, seed=0
        )
    with pytest.raises(ValueError, match="immutable support bytes"):
        uk_conditioned_assignment_definition(
            {**conditioned, NI: bytearray(conditioned[NI])}, seed=0
        )
    for seed in (-1, 2**32, True, 1.0):
        with pytest.raises(ValueError, match="seed must be a bounded integer"):
            uk_conditioned_assignment_definition(conditioned, seed=seed)


def test_the_conditioned_supports_decode_only_canonical_bytes() -> None:
    decoded = uk_conditioned_supports(toy_conditioned_support_payloads())
    assert list(decoded) == [EW, SCOT, NI]
    assert all(
        support.metadata["system"] == system for system, support in decoded.items()
    )
    with pytest.raises(ValueError, match="conditioned support columns differ"):
        uk_conditioned_supports(toy_support_payloads())


def test_the_region_only_definition_keeps_its_bytes() -> None:
    """Today's v1 definition, which the conditioned one leaves as it is."""

    v1 = uk_atomic_assignment_definition(payloads(), seed=42)
    assert hashlib.sha256(canonical_json(v1)).hexdigest() == _V1_DEFINITION_SHA256
    conditioned = uk_conditioned_assignment_definition(
        toy_conditioned_support_payloads(), seed=42
    )
    assert v1["stream"][1] != conditioned["stream"][1]
    for old, new in zip(v1["systems"], conditioned["systems"], strict=True):
        assert {key: old[key] for key in ("id", "level", "vintage", "selector")} == {
            key: new[key] for key in ("id", "level", "vintage", "selector")
        }
        assert old["constraints"] == new["constraints"][:1]
        assert old["stages"] == [{"level": "area", "weight": "households"}]
        assert {layer["output"] for layer in new["layers"]} < {
            layer["output"] for layer in old["layers"]
        }


def test_the_spellings_follow_their_owners() -> None:
    from microcosm.build.uk_runtime.frs_spine import (
        FRS_COUNCIL_TAX_BAND_OBSERVED_COLUMN,
        FRS_IMD_DECILE_COLUMN,
        FRS_IMD_DECILE_UNOBSERVED,
    )
    from microcosm.build.uk_runtime.geography_ladder import UK_GEOGRAPHY_LADDER_COLUMNS
    from microcosm.build.uk_runtime.local_authority_input import (
        UK_LOCAL_AUTHORITY_INPUT_COLUMN,
    )

    assert conditioning.COUNCIL_TAX_BAND_OBSERVED_COLUMN == (
        FRS_COUNCIL_TAX_BAND_OBSERVED_COLUMN
    )
    assert conditioning.UNOBSERVED_IMD_DECILE == FRS_IMD_DECILE_UNOBSERVED
    assert IMD_DECILE_COLUMN == FRS_IMD_DECILE_COLUMN
    assert UK_SPINE_LADDER_COLUMNS == tuple(
        column
        for column in UK_GEOGRAPHY_LADDER_COLUMNS
        if column != UK_LOCAL_AUTHORITY_INPUT_COLUMN
    )
    assert set(uk_band_weight_map(EW)) <= set(UK_COUNCIL_TAX_BANDS)
    branches = {branch.name for branch in UK_COPY_LINEAGE.branches}
    assert conditioning.SUPPORT_CHANNEL_BRANCH in branches
    assert dict(UK_DRAW_CONSTRAINTS) == {
        REGION_COLUMN: "frs_region",
        DECILE_INPUT_COLUMN: IMD_DECILE_COLUMN,
    }


# --- The draw inputs (O14 iv and v) ------------------------------------------------


def test_the_inputs_read_the_decile_as_text_and_the_band_where_observed() -> None:
    household = _households()
    inputs = uk_draw_inputs(household)
    assert list(inputs.columns) == list(DRAW_INPUT_COLUMNS)
    assert inputs.index.equals(household.index)
    assert all(inputs[column].dtype == _STRING for column in inputs.columns)
    for (_, row), case in zip(inputs.iterrows(), TOY_HOUSEHOLDS, strict=True):
        assert row[REGION_COLUMN] == case.region
        if case.imd_decile:
            assert row[DECILE_INPUT_COLUMN] == str(case.imd_decile)
        else:
            assert pd.isna(row[DECILE_INPUT_COLUMN])
        if case.council_tax_band_observed:
            assert row[BAND_INPUT_COLUMN] == case.council_tax_band
        else:
            assert pd.isna(row[BAND_INPUT_COLUMN])


def test_every_decile_reads_as_its_own_code_ten_included() -> None:
    """The decile reads as the supports' codes "1" to "10", the two-digit one
    included, and 0 as missing."""

    cases = [ToyHousehold("LONDON", decile, "B", True, "L0") for decile in range(0, 11)]
    inputs = uk_draw_inputs(_households(cases=cases))
    assert inputs[DECILE_INPUT_COLUMN].tolist()[1:] == list(IMD_DECILES)
    assert inputs[DECILE_INPUT_COLUMN].tolist()[-1] == "10"
    assert pd.isna(inputs[DECILE_INPUT_COLUMN].iloc[0])


def test_support_channel_copies_draw_without_inputs(definition, supports) -> None:
    copies = np.asarray([position % 2 for position in range(len(TOY_HOUSEHOLDS))])
    household = _households(
        household_support_channel=np.where(copies == 1, "spi", "frs").astype(object),
        household_support_clone_index=copies.astype(np.int64),
    )
    assert uk_support_channel_copies(household).tolist() == (copies == 1).tolist()
    inputs = uk_draw_inputs(household)
    assert (
        inputs.loc[copies == 1, [DECILE_INPUT_COLUMN, BAND_INPUT_COLUMN]]
        .isna()
        .all()
        .all()
    )
    levels = uk_draw_levels(inputs, definition, supports)
    assert set(levels[copies == 1]) == {"L3"}
    assert levels[copies == 0].tolist() == [
        case.level
        for case, copy in zip(TOY_HOUSEHOLDS, copies, strict=True)
        if not copy
    ]
    # A table without the channel's columns has no copy.
    assert not uk_support_channel_copies(_households()).any()


def test_a_later_copy_of_an_spi_copy_is_an_spi_channel_row() -> None:
    household = _households(
        household_support_channel=np.asarray(
            ["spi"] + ["frs"] * (len(TOY_HOUSEHOLDS) - 1), dtype=object
        ),
        household_support_clone_index=np.asarray(
            [1] + [0] * (len(TOY_HOUSEHOLDS) - 1), dtype=np.int64
        ),
        household_is_capital_gains_clone=np.asarray(
            [True, True] + [False] * (len(TOY_HOUSEHOLDS) - 2)
        ),
    )
    _, paths = copy_paths(household, UK_COPY_LINEAGE)
    assert paths[0] == (("spi_support_channel", 1), ("cgt_incidence_clone", 1))
    assert uk_support_channel_copies(household).tolist()[:3] == [True, False, False]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda h: h.drop(columns=["council_tax_band_observed"]), "lacks"),
        (lambda h: h.assign(**{IDENTITY_COLUMN: pd.NA}), IDENTITY_COLUMN),
        (lambda h: h.assign(region=" LONDON"), "region must hold"),
        (lambda h: h.assign(imd_decile=11), "outside 0 to 10"),
        (lambda h: h.assign(imd_decile=-1), "outside 0 to 10"),
        (lambda h: h.assign(imd_decile=h["imd_decile"].astype(float)), "integer"),
        (lambda h: h.assign(imd_decile=h["imd_decile"] > 0), "integer"),
        (
            lambda h: h.assign(
                council_tax_band_observed=h["council_tax_band_observed"]
                .astype("boolean")
                .where(h.index != h.index[0], pd.NA)
            ),
            "complete boolean",
        ),
        (lambda h: h.assign(council_tax_band_observed=1), "complete boolean"),
        (
            lambda h: h.assign(council_tax_band="J", council_tax_band_observed=True),
            "not a band A to I",
        ),
    ],
)
def test_malformed_inputs_are_refused_by_column(mutate, message) -> None:
    household = mutate(_households())
    with pytest.raises(ValueError, match=message) as refused:
        uk_draw_inputs(household)
    # Columns and declared constants only: no key and no household's region.
    text = str(refused.value)
    assert "invented-key" not in text
    assert not any(case.region in text for case in TOY_HOUSEHOLDS)


# --- The levels ---------------------------------------------------------------------


def _first_level_with_mass(support, case) -> str:
    """The first level whose inputs hold and whose cell has mass, row by row."""

    decile = str(case.imd_decile) if case.imd_decile else None
    band = case.council_tax_band if case.council_tax_band_observed else None
    regions = np.asarray(support.arrays["frs_region"])
    deciles = np.asarray(support.arrays[IMD_DECILE_COLUMN])
    weights = uk_band_weight_map(support.metadata["system"])
    for level in uk_system_fallback_levels(support.metadata["system"]):
        keeps = DECILE_INPUT_COLUMN in level.constraints
        if keeps and decile is None:
            continue
        if level.weight_by and band not in weights:
            continue
        rows = regions == case.region
        if keeps:
            rows &= deciles == decile
        column = weights[band] if level.weight_by else DEFAULT_WEIGHT_COLUMN
        if int(np.asarray(support.arrays[column], dtype=np.int64)[rows].sum()) > 0:
            return level.name
    raise AssertionError("an invented household has no level")


def test_each_household_takes_the_first_level_whose_cell_has_mass(
    definition, supports
) -> None:
    levels = uk_draw_levels(uk_draw_inputs(_households()), definition, supports)
    system_of = {"NORTHERN_IRELAND": NI, "SCOTLAND": SCOT}
    expected = [
        _first_level_with_mass(supports[system_of.get(case.region, EW)], case)
        for case in TOY_HOUSEHOLDS
    ]
    assert levels.tolist() == expected == [case.level for case in TOY_HOUSEHOLDS]
    # Every level is exercised, and Northern Ireland takes L1 or L3 only.
    assert set(levels) == {"L0", "L1", "L2", "L3"}
    northern_ireland = [c.region == "NORTHERN_IRELAND" for c in TOY_HOUSEHOLDS]
    assert set(levels[northern_ireland]) == {"L1", "L3"}


def test_the_chooser_reads_its_rule_and_default_from_the_definition(
    definition,
) -> None:
    # The draw samples a missing band by the stage's default; the chooser's
    # default is that same column, and its map the stage's.
    for entry in definition["systems"]:
        rule, default = conditioning._system_rules(entry)
        assert default == DEFAULT_WEIGHT_COLUMN
        if entry["id"] == NI:
            assert rule is None
        else:
            assert rule == {
                "input": BAND_INPUT_COLUMN,
                "map": uk_band_weight_map(entry["id"]),
            }


def test_the_chooser_is_called_with_the_definitions_own_arguments(
    monkeypatch, definition, supports
) -> None:
    """Every toy cell with population has households too, so the levels alone
    cannot tell the default weight from another: the chooser's arguments are
    recorded and held to each system's stage."""

    calls: list[dict] = []
    real = conditioning.choose_fallback_levels

    def recording(support, households, **arguments):
        calls.append({"system": support.metadata["system"], **arguments})
        return real(support, households, **arguments)

    monkeypatch.setattr(conditioning, "choose_fallback_levels", recording)
    uk_draw_levels(uk_draw_inputs(_households()), definition, supports)
    assert sorted(call["system"] for call in calls) == sorted((EW, SCOT, NI))
    for call in calls:
        (entry,) = [e for e in definition["systems"] if e["id"] == call["system"]]
        rule, default = conditioning._system_rules(entry)
        assert call["default_weight"] == default == DEFAULT_WEIGHT_COLUMN
        assert call["weight_by"] == rule
        assert call["constraint_columns"] == dict(UK_DRAW_CONSTRAINTS)
        assert [level.name for level in call["levels"]] == [
            level.name for level in uk_system_fallback_levels(call["system"])
        ]


def test_a_region_no_system_selects_is_refused(definition, supports) -> None:
    household = _households().assign(region="UNKNOWN")
    with pytest.raises(ValueError, match="exactly one area system"):
        uk_draw_levels(uk_draw_inputs(household), definition, supports)
    with pytest.raises(ValueError, match="decoded support for every system"):
        uk_draw_levels(uk_draw_inputs(_households()), definition, {EW: supports[EW]})


def test_the_draw_frame_drops_what_each_level_does_not_keep(
    definition, supports
) -> None:
    inputs = uk_draw_inputs(_households())
    levels = uk_draw_levels(inputs, definition, supports)
    frame = uk_draw_frame(inputs, levels)
    assert all(frame[column].dtype == _STRING for column in frame.columns)
    keeps_decile = np.isin(levels, ["L0", "L1"])
    keeps_band = np.isin(levels, ["L0", "L2"])
    assert frame[DECILE_INPUT_COLUMN][~keeps_decile].isna().all()
    assert frame[BAND_INPUT_COLUMN][~keeps_band].isna().all()
    assert frame[DECILE_INPUT_COLUMN][keeps_decile].equals(
        inputs[DECILE_INPUT_COLUMN][keeps_decile]
    )
    assert frame[BAND_INPUT_COLUMN][keeps_band].equals(
        inputs[BAND_INPUT_COLUMN][keeps_band]
    )
    assert frame[[IDENTITY_COLUMN, REGION_COLUMN]].equals(
        inputs[[IDENTITY_COLUMN, REGION_COLUMN]]
    )
    with pytest.raises(ValueError, match="outside the UK fallback ladder"):
        uk_draw_frame(inputs, ["L4"] * len(inputs))
    with pytest.raises(ValueError, match="one level per household"):
        uk_draw_frame(inputs, levels[:-1])


def test_one_definition_draws_every_level_inside_its_cell(definition, supports) -> None:
    """A level's draw keeps its own inputs alone, and lands where it has mass."""

    inputs = uk_draw_inputs(_households())
    levels = uk_draw_levels(inputs, definition, supports)
    draw = uk_draw_frame(inputs, levels)
    assigned = assign_atomic(draw, definition, supports)
    for (_, row), level, case in zip(
        assigned.iterrows(), levels, TOY_HOUSEHOLDS, strict=True
    ):
        support = supports[row["atomic_area_system"]]
        position = int(
            np.flatnonzero(support.arrays["area"] == row["atomic_area_code"])[0]
        )
        assert support.arrays["frs_region"][position] == case.region
        if level in {"L0", "L1"}:
            assert support.arrays[IMD_DECILE_COLUMN][position] == str(case.imd_decile)
        weight = (
            uk_band_weight_map(row["atomic_area_system"])[case.council_tax_band]
            if level in {"L0", "L2"}
            else DEFAULT_WEIGHT_COLUMN
        )
        assert int(support.arrays[weight][position]) > 0


def test_the_fallback_reasons_name_each_inputs_status(definition, supports) -> None:
    inputs = uk_draw_inputs(_households())
    levels = uk_draw_levels(inputs, definition, supports)
    reasons = uk_fallback_reasons(inputs, levels, definition)
    assert reasons.index.equals(inputs.index)
    for (_, row), case, level in zip(
        reasons.iterrows(), TOY_HOUSEHOLDS, levels, strict=True
    ):
        assert row["decile"] in UK_FALLBACK_STATUSES["decile"]
        assert row["band"] in UK_FALLBACK_STATUSES["band"]
        if level == "L0":
            assert (row["decile"], row["band"]) == ("kept", "kept")
        if not case.imd_decile:
            assert row["decile"] == "unobserved"
        if case.region == "NORTHERN_IRELAND":
            assert row["band"] == "none"
    # London's band-H household in decile 1 keeps its decile and drops the band;
    # its decile-9 household drops the decile and keeps the band.
    assert reasons.iloc[1].tolist() == ["kept", "dropped"]
    assert reasons.iloc[4].tolist() == ["dropped", "kept"]
