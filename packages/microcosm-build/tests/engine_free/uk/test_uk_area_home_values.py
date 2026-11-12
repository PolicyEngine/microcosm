"""The WAS chain's home values off each owner's constituency, G8 (microcosm#931).

Invented households, toy knots and a toy regional reference only. The tests
hold form F: an England and Wales owner's value is its constituency's knots at
its weighted mid-rank within its FRS region, divided by the region's regional
uprating factor on the draw, so the uprating, dividing by the kept draw,
lands it back on the knots; every other row keeps its draw bit for bit. They
also hold the evidence to rounded clone-0 source counts and the declaration
to its reviewed form.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    _reject_executable_parameter_keys,
    _reject_incumbent_dependencies,
)
from microcosm.build.uk_runtime import area_home_values as ahv
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.property_localisation import (
    RANK_BASIS_COLUMN,
    localise_main_residence_values,
    owner_uniforms,
)
from microcosm.build.uk_runtime.regional_home_values import (
    REGIONAL_HOME_VALUES_STREAM,
)
from microcosm.build.uk_runtime.regional_uprating import (
    UK_REGIONAL_LAND_VALUES_RESOURCE,
    regional_reference_factors,
    uprate_household_property_by_region,
)

_GRID = [0.0, 0.5, 1.0]
_CAP = 1_200_000.0
#: Invented constituencies: two London seats, one North East seat, one Welsh
#: seat. The first London seat reads the cap from the median up.
_SEATS = {
    "E14000001": [200_000.0, _CAP, _CAP],
    "E14000002": [150_000.0, 300_000.0, 600_000.0],
    "E14000003": [100_000.0, 200_000.0, 400_000.0],
    "W07000001": [90_000.0, 180_000.0, 360_000.0],
}
_REGIONAL = {
    "values": [
        {"region": "LONDON", "avg_house_price": 500_000.0},
        {"region": "NORTH_EAST", "avg_house_price": 150_000.0},
        {"region": "WALES", "avg_house_price": 200_000.0},
        {"region": "SCOTLAND", "avg_house_price": 180_000.0},
    ]
}


def _knots_payload(seats=None) -> dict:
    return {
        "schema_version": 1,
        "kind": "uk_high_value_home_value_knots",
        "grid": _GRID,
        "cap": _CAP,
        "basis_year": 2024,
        "inputs_sha256": "toy",
        "method": "toy",
        "levels": {
            "constituency": {
                code: {"quantiles": quantiles}
                for code, quantiles in (seats or _SEATS).items()
            }
        },
    }


def _resources(seats=None) -> ahv.UKAreaHomeValuesResources:
    return ahv.load_area_home_values_resources(
        knots=json.dumps(_knots_payload(seats)).encode(),
        regional_reference=json.dumps(_REGIONAL).encode(),
    )


#: id, region, tenure, constituency, support channel, segment-2 draw, weight.
_ROWS = (
    (1, "LONDON", "OWNED_OUTRIGHT", "E14000001", "frs", 450_000.0, 2.0),
    (2, "LONDON", "OWNED_WITH_MORTGAGE", "E14000002", "frs", 300_000.0, 1.0),
    (3, "LONDON", "RENT_PRIVATELY", "E14000001", "frs", 0.0, 1.0),
    # An SPI copy: it ranks beside the FRS owners but is left out of the mean.
    (4, "LONDON", "OWNED_OUTRIGHT", "E14000002", "spi", 900_000.0, 0.5),
    (5, "NORTH_EAST", "OWNED_OUTRIGHT", "E14000003", "frs", 120_000.0, 3.0),
    (6, "NORTH_EAST", "OWNED_WITH_MORTGAGE", "E14000003", "frs", 160_000.0, 1.0),
    (7, "WALES", "OWNED_OUTRIGHT", "W07000001", "frs", 200_000.0, 2.0),
    # An owner the chain gave no main residence.
    (8, "WALES", "OWNED_OUTRIGHT", "W07000001", "frs", 0.0, 1.0),
    (9, "SCOTLAND", "OWNED_OUTRIGHT", "S14000001", "frs", 250_000.0, 2.0),
    (10, "NORTHERN_IRELAND", "OWNED_OUTRIGHT", "N05000001", "frs", 150_000.0, 1.0),
    (11, "SCOTLAND", "RENT_FROM_COUNCIL", "S14000001", "frs", 0.0, 1.0),
)
_ENGLAND_WALES_OWNERS = (1, 2, 4, 5, 6, 7)


def _household(rows=_ROWS) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    household = pd.DataFrame(
        {
            "household_id": np.array([row[0] for row in rows], dtype=np.int64),
            "region": [row[1] for row in rows],
            "tenure_type": [row[2] for row in rows],
            "constituency_code": pd.array([row[3] for row in rows], dtype="string"),
            "household_support_channel": [row[4] for row in rows],
            "source_household_id": np.array([row[0] for row in rows], dtype=np.int64),
        }
    )
    drawn = np.array([row[5] for row in rows], dtype=float)
    weights = np.array([row[6] for row in rows], dtype=float)
    return household, drawn, weights


def _expected(household, drawn, weights, resources) -> tuple[np.ndarray, np.ndarray]:
    """Form F from the public helpers: the knot value at the rank, then
    divided by the region's factor on the draw."""

    rank, _ = owner_uniforms(
        spine_key=household["household_id"].astype(str).to_numpy(dtype=object),
        region=household["region"].to_numpy(),
        tenure_type=household["tenure_type"].to_numpy(),
        value=drawn,
        weight=weights,
        stream=ahv.AREA_HOME_VALUES_STREAM,
    )
    knot, rewritten = localise_main_residence_values(
        u=rank,
        constituency_code=household["constituency_code"].to_numpy(dtype=object),
        tenure_type=household["tenure_type"].to_numpy(),
        value=drawn,
        knots=resources.knots,
    )
    factors = regional_reference_factors(
        household.loc[:, ["region", "household_support_channel"]].assign(
            main_residence_value=drawn
        ),
        _REGIONAL,
    )
    expected = knot.copy()
    for region, factor in factors.items():
        rows = rewritten & (household["region"].to_numpy() == region)
        expected[rows] = knot[rows] / factor
    return expected, knot


def _bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.int64)


def _no_data_digit(message: str) -> bool:
    """No digit beyond the issue reference, the design item and the
    declaration's key names."""

    stripped = re.sub(r"microcosm#\d+|C-\d+|sha256", "", message)
    return re.search(r"\d", stripped) is None


# --- The values ------------------------------------------------------------------


def test_england_and_wales_owners_read_their_seat_divided_by_the_factor() -> None:
    household, drawn, weights = _household()
    resources = _resources()

    result = ahv.localise_area_home_values(
        household, weights, drawn, resources=resources
    )

    expected, knot = _expected(household, drawn, weights, resources)
    assert _bits(result.value).tolist() == _bits(expected).tolist()
    ids = household["household_id"].to_numpy()
    assert sorted(ids[result.rewritten].tolist()) == list(_ENGLAND_WALES_OWNERS)
    # By hand for the North East: two owners at weights 3 and 1 take the
    # mid-ranks 0.375 and 0.875, read 175,000 and 350,000 off their seat, and
    # the region's factor on the draw is 150,000 over their mean of 140,000.
    north_east = household["region"].to_numpy() == "NORTH_EAST"
    assert knot[north_east].tolist() == pytest.approx([175_000.0, 350_000.0])
    assert result.value[north_east].tolist() == pytest.approx(
        [175_000.0 * 140_000.0 / 150_000.0, 350_000.0 * 140_000.0 / 150_000.0]
    )
    # Scotland, Northern Ireland, renters and the owner without a positive
    # draw keep the draw bit for bit; the draw is the rank basis on every row.
    kept = ~result.rewritten
    assert _bits(result.value[kept]).tolist() == _bits(drawn[kept]).tolist()
    assert _bits(result.rank_basis).tolist() == _bits(drawn).tolist()


def test_the_regional_uprating_lands_localised_homes_on_the_knots() -> None:
    """Declaring the kept draw its denominator, the uprating computes each
    region's factor from the same rows and values, bit for bit, so England
    and Wales main residences return to their knot value to one rounding
    (form F, C-15)."""

    household, drawn, weights = _household()
    resources = _resources()
    result = ahv.localise_area_home_values(
        household, weights, drawn, resources=resources
    )
    _, knot = _expected(household, drawn, weights, resources)
    uprating_input = household.assign(
        main_residence_value=result.value,
        property_wealth=result.value + 1_000.0,
        **{RANK_BASIS_COLUMN: result.rank_basis},
    )

    on_draw = regional_reference_factors(
        household.loc[:, ["region", "household_support_channel"]].assign(
            main_residence_value=drawn
        ),
        _REGIONAL,
    )
    on_basis = regional_reference_factors(
        uprating_input, _REGIONAL, denominator_column=RANK_BASIS_COLUMN
    )
    assert list(on_basis) == list(on_draw)
    assert (
        _bits(list(on_basis.values())).tolist()
        == _bits(list(on_draw.values())).tolist()
    )
    uprated = uprate_household_property_by_region(
        uprating_input, _REGIONAL, denominator_column=RANK_BASIS_COLUMN
    )
    rewritten = result.rewritten
    landed = uprated["main_residence_value"].to_numpy()[rewritten]
    assert np.all(np.abs(landed - knot[rewritten]) <= np.spacing(knot[rewritten]))
    # Scotland is uprated by the factor on its draw, as before.
    scotland = (household["region"] == "SCOTLAND").to_numpy() & (drawn > 0)
    assert uprated["main_residence_value"].to_numpy()[scotland].tolist() == (
        pytest.approx((drawn[scotland] * on_draw["SCOTLAND"]).tolist())
    )


def _random_households(n: int = 240, seed: int = 931):
    """Invented owners and renters over the toy seats, in two clones and two
    channels, with non-integer draws and weights, so that a sum taken in row
    order would differ in its last bits; every fifth London owner ties on its
    draw with the one before it."""

    rng = np.random.default_rng(seed)
    seats = {
        "LONDON": ["E14000001", "E14000002"],
        "NORTH_EAST": ["E14000003"],
        "WALES": ["W07000001"],
        "SCOTLAND": ["S14000001"],
        "NORTHERN_IRELAND": ["N05000001"],
    }
    regions = rng.choice(list(seats), n)
    clone = (np.arange(n) % 2).astype(np.int64)
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1, dtype=np.int64) + clone * 10**9,
            "region": regions,
            "tenure_type": rng.choice(
                ["OWNED_OUTRIGHT", "OWNED_WITH_MORTGAGE", "RENT_PRIVATELY"],
                n,
                p=[0.4, 0.3, 0.3],
            ),
            "constituency_code": pd.array(
                [rng.choice(seats[region]) for region in regions], dtype="string"
            ),
            "household_support_channel": rng.choice(["frs", "spi"], n, p=[0.85, 0.15]),
            "source_household_id": np.arange(1, n + 1, dtype=np.int64) // 2,
            "household_clone_index": clone,
        }
    )
    owner = household["tenure_type"].str.startswith("OWNED").to_numpy()
    drawn = np.where(
        owner & (rng.uniform(size=n) > 0.1), rng.lognormal(12.3, 0.6, n), 0.0
    )
    london = np.flatnonzero(owner & (drawn > 0) & (regions == "LONDON"))
    drawn[london[1::5]] = drawn[london[0::5][: len(london[1::5])]]
    return household, drawn, rng.uniform(0.2, 3.0, n)


def test_a_row_reads_its_value_whatever_the_table_order() -> None:
    household, drawn, weights = _random_households()
    resources = _resources()
    result = ahv.localise_area_home_values(
        household, weights, drawn, resources=resources
    )
    assert result.rewritten.any() and not result.rewritten.all()
    rng = np.random.default_rng(1112)
    for _ in range(5):
        order = rng.permutation(len(household))
        shuffled = ahv.localise_area_home_values(
            household.iloc[order].reset_index(drop=True),
            weights[order],
            drawn[order],
            resources=resources,
        )

        assert _bits(shuffled.value).tolist() == _bits(result.value[order]).tolist()
        assert shuffled.rewritten.tolist() == result.rewritten[order].tolist()
        assert shuffled.evidence() == result.evidence()


def test_ties_break_on_the_declared_stream() -> None:
    """Twelve London owners tie on their draw in one seat whose knots rise
    strictly: their order, and so their values, is the declared stream's."""

    rows = tuple(
        (100 + index, "LONDON", "OWNED_OUTRIGHT", "E14000002", "frs", 333_000.0, 1.0)
        for index in range(12)
    )
    household, drawn, weights = _household(rows)
    resources = _resources()

    result = ahv.localise_area_home_values(
        household, weights, drawn, resources=resources
    )

    expected, _ = _expected(household, drawn, weights, resources)
    assert _bits(result.value).tolist() == _bits(expected).tolist()
    assert len(set(result.value.tolist())) == 12


def test_a_copy_sharing_its_parents_source_ranks_on_its_own_key() -> None:
    """Real support copies keep their FRS parent's true source: the rank still
    keys each row on its own household id, and the evidence counts the copy
    and its parent once."""

    household, drawn, weights = _household()
    # Row 4 is the SPI support copy; here it keeps household 1's source.
    household.loc[household["household_id"] == 4, "source_household_id"] = 1
    resources = _resources()

    result = ahv.localise_area_home_values(
        household, weights, drawn, resources=resources
    )

    expected, _ = _expected(household, drawn, weights, resources)
    assert _bits(result.value).tolist() == _bits(expected).tolist()
    # London's rewritten rows rest on two true sources, not three.
    assert result.evidence()["sources_rewritten_by_region"]["E12000007"] == "<3"


def test_the_rank_is_its_own_stream_with_no_seed_of_its_own() -> None:
    """Seed source NONE with a fixed keyed stream, as ruled for
    regional_home_values on 9 Nov (C-43); the stream is the step's own."""

    assert ahv.AREA_HOME_VALUES_STREAM == (
        "sha256-u53-v1",
        "uk-spine-area-home-values-v1",
        0,
        0,
    )
    assert ahv.AREA_HOME_VALUES_STREAM != REGIONAL_HOME_VALUES_STREAM
    parameters = ahv.area_home_values_operation_parameters()
    assert parameters["stream"] == list(ahv.AREA_HOME_VALUES_STREAM)
    assert "seed" not in parameters


# --- The evidence ------------------------------------------------------------------


def test_the_evidence_counts_rounded_sources_and_carries_no_value() -> None:
    household, drawn, weights = _household()
    result = ahv.localise_area_home_values(
        household, weights, drawn, resources=_resources()
    )

    evidence = result.evidence()
    assert evidence == {
        "clone_count": 1,
        "counted_among": "clone_0",
        # Three London sources, two in the North East and one in Wales, as
        # receipt_counts rounds them.
        "sources_rewritten_by_region": {
            "E12000001": "<3",
            "E12000007": "3-9",
            "W92000004": "<3",
        },
        # The first London owner reads its seat's cap.
        "sources_at_cap": "<3",
        "knots_basis_year": 2024,
        "knots_sha256": _resources().knots_sha256,
        "knots_level": "constituency",
        "regional_reference_sha256": _resources().regional_reference_sha256,
        "ranking": {"rank_basis": "weighted_midpoint_within_frs_region"},
        "rank_basis_column": RANK_BASIS_COLUMN,
        "stream": list(ahv.AREA_HOME_VALUES_STREAM),
        "method": ahv.METHOD_VERSION,
    }
    printed = repr(result)
    assert "450000" not in printed and "rank_basis=array" not in printed


def _cloned(k: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    household, drawn, weights = _household()
    tables = []
    for clone in range(k):
        table = household.copy()
        table["household_id"] = household["household_id"] + clone * 10**9
        table["household_clone_index"] = np.int64(clone)
        tables.append(table)
    return (
        pd.concat(tables, ignore_index=True),
        np.tile(drawn, k),
        np.tile(weights / k, k),
    )


@pytest.mark.parametrize("k", [2, 3])
def test_the_evidence_counts_clone_zero_and_is_the_same_at_every_k(k) -> None:
    """The regions' rewritten sources do not grow with K. The sources at the
    cap are clone 0's own: the clones of one household tie on their draw, so
    which of them a tie-break ranks higher is the stream's, and clone 0's
    count is recounted here from its own rows."""

    from microcosm.build.uk_runtime.receipt_counts import (
        distinct_count,
        support_band,
    )

    household, drawn, weights = _household()
    resources = _resources()
    single = ahv.localise_area_home_values(
        household, weights, drawn, resources=resources
    )
    cloned_household, cloned_drawn, cloned_weights = _cloned(k)
    cloned = ahv.localise_area_home_values(
        cloned_household, cloned_weights, cloned_drawn, resources=resources
    )

    single_evidence = single.evidence()
    cloned_evidence = cloned.evidence()
    assert cloned_evidence["clone_count"] == k
    assert (
        cloned_evidence["sources_rewritten_by_region"]
        == single_evidence["sources_rewritten_by_region"]
    )
    assert {
        key: value
        for key, value in cloned_evidence.items()
        if key not in {"clone_count", "sources_at_cap"}
    } == {
        key: value
        for key, value in single_evidence.items()
        if key not in {"clone_count", "sources_at_cap"}
    }
    _, knot = _expected(cloned_household, cloned_drawn, cloned_weights, resources)
    clone_zero = cloned_household["household_clone_index"].to_numpy() == 0
    assert cloned_evidence["sources_at_cap"] == support_band(
        distinct_count(
            cloned_household["source_household_id"].to_numpy(),
            cloned.rewritten & clone_zero & (knot >= _CAP),
            cloned_weights,
        )
    )
    # Every clone of an England and Wales owner is rewritten; the others keep
    # their draw.
    assert int(cloned.rewritten.sum()) == k * int(single.rewritten.sum())


def _area_rows() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """London owners on one seat, ids minted through the copy-id table. Clone
    0: households 1 and 2, an SPI copy of household 1 (its true source 1)
    and household 3, which drew no home. Clone 1: the same rows, with
    household 3 drawn."""

    from microcosm.build.uk_runtime import copy_id_table

    base = (
        (1, 1, "frs", 450_000.0),
        (2, 2, "frs", 300_000.0),
        (1 + copy_id_table.radix("spi_support_channel"), 1, "spi", 400_000.0),
        (3, 3, "frs", 0.0),
    )
    rows = [(*row, 0) for row in base]
    for row_id, source, channel, value in base:
        clone_id = int(
            copy_id_table.mint(np.array([row_id]), "geographic_support", 1)[0]
        )
        rows.append((clone_id, source, channel, 350_000.0 if source == 3 else value, 1))
    household = pd.DataFrame(
        {
            "household_id": np.array([row[0] for row in rows], dtype=np.int64),
            "region": ["LONDON"] * len(rows),
            "tenure_type": ["OWNED_OUTRIGHT"] * len(rows),
            "constituency_code": pd.array(["E14000002"] * len(rows), dtype="string"),
            "household_support_channel": [row[2] for row in rows],
            "source_household_id": np.array([row[1] for row in rows], dtype=np.int64),
            "household_clone_index": np.array([row[4] for row in rows], dtype=np.int64),
        }
    )
    drawn = np.array([row[3] for row in rows], dtype=float)
    return household, drawn, np.ones(len(rows), dtype=float)


def test_the_evidence_counts_distinct_sources_in_clone_zero() -> None:
    """Clone 0 rewrites households 1 and 2 and the copy of household 1: two
    true sources ("<3"). Counted over rows that is three, and over every
    clone household 3 joins: either reads "3-9" (C-44)."""

    household, drawn, weights = _area_rows()

    result = ahv.localise_area_home_values(
        household, weights, drawn, resources=_resources()
    )

    assert result.evidence()["sources_rewritten_by_region"] == {"E12000007": "<3"}


def test_the_evidence_counts_the_sources_at_the_cap_in_clone_zero() -> None:
    """Every rewritten row reads the cap on this seat: clone 0 holds two true
    sources there ("<3"); clone 0's rows are three and every clone's sources
    three ("3-9")."""

    household, drawn, weights = _area_rows()
    household["constituency_code"] = pd.array(
        ["E14000009"] * len(household), dtype="string"
    )
    seats = {**_SEATS, "E14000009": [_CAP, _CAP, _CAP]}

    result = ahv.localise_area_home_values(
        household, weights, drawn, resources=_resources(seats)
    )

    assert result.evidence()["sources_at_cap"] == "<3"


def test_a_row_of_zero_weight_carries_no_source_into_the_evidence() -> None:
    """A rewritten clone-0 row of zero weight carries no localised value into
    a weighted figure, so its source does not count (receipt_counts'
    carrier rule, microcosm#1262)."""

    household, drawn, weights = _area_rows()
    single = ahv.localise_area_home_values(
        household, weights, drawn, resources=_resources()
    )
    extra = pd.DataFrame(
        {
            "household_id": np.array([4], dtype=np.int64),
            "region": ["LONDON"],
            "tenure_type": ["OWNED_OUTRIGHT"],
            "constituency_code": pd.array(["E14000002"], dtype="string"),
            "household_support_channel": ["frs"],
            "source_household_id": np.array([4], dtype=np.int64),
            "household_clone_index": np.array([0], dtype=np.int64),
        }
    )
    with_zero = pd.concat([household, extra], ignore_index=True)

    # The lowest draw in the region: a zero-weight owner at the top of a
    # region's rank would sit at rank 1, which owner_uniforms refuses.
    result = ahv.localise_area_home_values(
        with_zero,
        np.append(weights, 0.0),
        np.append(drawn, 250_000.0),
        resources=_resources(),
    )

    assert result.rewritten[-1]
    assert (
        result.evidence()["sources_rewritten_by_region"]
        == single.evidence()["sources_rewritten_by_region"]
        == {"E12000007": "<3"}
    )


# --- The refusals -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "constituency"),
    [
        # An England owner placed in a Scottish constituency.
        (0, "S14000001"),
        # A Scottish owner placed in an English one.
        (8, "E14000001"),
        # An England owner with no constituency.
        (4, pd.NA),
    ],
)
def test_an_owner_outside_its_regions_nation_is_refused(row, constituency) -> None:
    household, drawn, weights = _household()
    household.loc[row, "constituency_code"] = constituency

    with pytest.raises(ValueError, match="in its FRS region's nation") as caught:
        ahv.localise_area_home_values(household, weights, drawn, resources=_resources())
    assert _no_data_digit(str(caught.value))


def test_a_seat_without_knots_and_a_knot_at_zero_are_refused() -> None:
    household, drawn, weights = _household()
    seats = {code: knots for code, knots in _SEATS.items() if code != "W07000001"}
    with pytest.raises(ValueError, match="No value knots") as caught:
        ahv.localise_area_home_values(
            household, weights, drawn, resources=_resources(seats)
        )
    assert _no_data_digit(str(caught.value))
    zero = {**_SEATS, "E14000003": [0.0, 0.0, 400_000.0]}
    with pytest.raises(ValueError, match="not positive") as caught:
        ahv.localise_area_home_values(
            household, weights, drawn, resources=_resources(zero)
        )
    assert _no_data_digit(str(caught.value))


def test_missing_and_misaligned_inputs_are_refused_by_name() -> None:
    household, drawn, weights = _household()
    with pytest.raises(ValueError, match="constituency_code") as caught:
        ahv.localise_area_home_values(
            household.drop(columns=["constituency_code"]),
            weights,
            drawn,
            resources=_resources(),
        )
    assert _no_data_digit(str(caught.value))
    with pytest.raises(ValueError, match="source_household_id"):
        ahv.localise_area_home_values(
            household.drop(columns=["source_household_id"]),
            weights,
            drawn,
            resources=_resources(),
        )
    with pytest.raises(ValueError, match="one drawn value and one weight"):
        ahv.localise_area_home_values(
            household, weights[:-1], drawn, resources=_resources()
        )
    with pytest.raises(ValueError, match="finite"):
        ahv.localise_area_home_values(
            household, weights, np.where(drawn > 0, np.inf, 0.0), resources=_resources()
        )


# --- The localiser on a frame --------------------------------------------------------


def _frame(household: pd.DataFrame, weights: np.ndarray):
    person = pd.DataFrame(
        {
            "person_id": household["household_id"].to_numpy() * 10,
            "person_benunit_id": household["household_id"].to_numpy() * 10,
            "person_household_id": household["household_id"].to_numpy(),
        }
    )
    benunit = pd.DataFrame({"benunit_id": household["household_id"].to_numpy() * 10})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
        household_weights=weights,
    )


def test_the_localiser_reads_the_stages_frame_and_its_weights() -> None:
    household, drawn, weights = _household()
    resources = _resources()
    localiser = ahv.UKAreaHomeValuesLocaliser.from_frame(
        _frame(household, weights), resources=resources
    )

    result = localiser(drawn)
    direct = ahv.localise_area_home_values(
        household, weights, drawn, resources=resources
    )
    assert _bits(result.value).tolist() == _bits(direct.value).tolist()
    assert result.evidence() == direct.evidence()
    with pytest.raises(ValueError, match="constituency_code"):
        ahv.UKAreaHomeValuesLocaliser.from_frame(
            _frame(household.drop(columns=["constituency_code"]), weights),
            resources=resources,
        )


def test_the_committed_resources_load_with_their_digests() -> None:
    resources = ahv.load_committed_area_home_values_resources()

    assert resources.knots.quantiles
    assert all(code[0] in "EW" for code in resources.knots.quantiles)
    assert isinstance(resources.regional_reference["values"], list)
    for digest in (resources.knots_sha256, resources.regional_reference_sha256):
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
    # The knots' regions for the FRS regions are England's nine and Wales.
    assert sorted(ahv.FRS_REGION_TO_KNOTS_REGION.values()) == [
        *(f"E1200000{index}" for index in range(1, 10)),
        "W92000004",
    ]


# --- The declaration -----------------------------------------------------------------


def _was_stage(
    *,
    localise: bool = True,
    exempt: tuple[str, ...] | None = None,
    outputs: tuple[str, ...] | None = None,
    artifacts: tuple[dict, ...] | None = None,
    parameters: dict | None = None,
) -> SourceStageSpec:
    """A WAS stage carrying what the area home values need to be declared."""

    operations = [
        SourceOperationSpec(
            kind="support_clip",
            parameters={
                "range": "donor_realized",
                **(
                    {"exempt": list(exempt)}
                    if exempt is not None
                    else (
                        {"exempt": list(ahv.AREA_HOME_VALUES_EXEMPT_COLUMNS)}
                        if localise
                        else {}
                    )
                ),
            },
        )
    ]
    if localise:
        operations.append(
            SourceOperationSpec(
                kind=ahv.AREA_HOME_VALUES_OPERATION_KIND,
                parameters=parameters or ahv.area_home_values_operation_parameters(),
            )
        )
    declared_artifacts = (
        artifacts
        if artifacts is not None
        else tuple(
            {
                "role": role,
                "kind": "public_aggregate_reference",
                "resource": resource,
                "format": "json",
                "runtime_sha256_required": True,
            }
            for role, resource in (
                (ahv.KNOTS_ARTIFACT_ROLE, ahv.KNOTS_RESOURCE),
                (
                    ahv.REGIONAL_REFERENCE_ARTIFACT_ROLE,
                    UK_REGIONAL_LAND_VALUES_RESOURCE,
                ),
            )
        )
        if localise
        else ()
    )
    return SourceStageSpec(
        stage="was_wealth",
        survey="test",
        source="test",
        grain="household",
        artifacts=declared_artifacts,
        operations=tuple(operations),
        outputs=outputs
        if outputs is not None
        else (("main_residence_value", RANK_BASIS_COLUMN) if localise else ()),
    )


def test_the_reviewed_operation_is_declarative() -> None:
    parameters = ahv.area_home_values_operation_parameters()
    _reject_executable_parameter_keys(
        parameters, context=ahv.AREA_HOME_VALUES_OPERATION_KIND
    )
    _reject_incumbent_dependencies(
        parameters, context=ahv.AREA_HOME_VALUES_OPERATION_KIND
    )
    assert json.loads(json.dumps(parameters)) == parameters
    assert parameters["knots"] == ahv.KNOTS_RESOURCE
    assert parameters["regional_reference"] == UK_REGIONAL_LAND_VALUES_RESOURCE
    assert parameters["knots_level"] == "constituency"
    assert parameters["rank_basis_column"] == RANK_BASIS_COLUMN
    assert parameters["region_map"] == dict(
        sorted(ahv.FRS_REGION_TO_KNOTS_REGION.items())
    )


def test_a_stage_declares_the_step_whole_or_not_at_all() -> None:
    assert ahv.area_home_values_declared(_was_stage()) is True
    assert ahv.area_home_values_declared(_was_stage(localise=False)) is False
    assert ahv.support_clip_exemptions(_was_stage()) == frozenset(
        ahv.AREA_HOME_VALUES_EXEMPT_COLUMNS
    )
    assert ahv.support_clip_exemptions(_was_stage(localise=False)) == frozenset()
    # The packaged WAS stage does not declare it yet.
    packaged = load_country_spec("uk").sources.stage_map()["was_wealth"]
    assert ahv.area_home_values_declared(packaged) is False
    assert ahv.support_clip_exemptions(packaged) == frozenset()


@pytest.mark.parametrize(
    ("stage", "fragment"),
    [
        (
            lambda: _was_stage(localise=False, exempt=("main_residence_value",)),
            "without",
        ),
        (
            lambda: _was_stage(localise=False, outputs=(RANK_BASIS_COLUMN,)),
            "without",
        ),
        (lambda: _was_stage(exempt=()), "exempt exactly"),
        (lambda: _was_stage(exempt=("main_residence_value",)), "exempt exactly"),
        (lambda: _was_stage(outputs=("main_residence_value",)), "among its outputs"),
        (lambda: _was_stage(artifacts=()), "artifacts"),
        (
            lambda: _was_stage(
                artifacts=(
                    {
                        "role": ahv.KNOTS_ARTIFACT_ROLE,
                        "resource": ahv.KNOTS_RESOURCE,
                        "runtime_sha256_required": True,
                    },
                    {
                        "role": ahv.REGIONAL_REFERENCE_ARTIFACT_ROLE,
                        "resource": UK_REGIONAL_LAND_VALUES_RESOURCE,
                    },
                )
            ),
            "artifacts",
        ),
        (
            lambda: _was_stage(
                parameters={
                    **ahv.area_home_values_operation_parameters(),
                    "knots_level": "region",
                }
            ),
            "reviewed operation",
        ),
    ],
)
def test_a_half_declaration_is_refused(stage, fragment) -> None:
    with pytest.raises(ValueError, match=fragment) as caught:
        ahv.area_home_values_declared(stage())
    assert _no_data_digit(str(caught.value))


def test_a_repeated_declaration_is_refused() -> None:
    stage = _was_stage()
    repeated = replace(
        stage,
        operations=(
            *stage.operations,
            SourceOperationSpec(
                kind=ahv.AREA_HOME_VALUES_OPERATION_KIND,
                parameters=ahv.area_home_values_operation_parameters(),
            ),
        ),
    )
    with pytest.raises(ValueError, match="more than once"):
        ahv.area_home_values_declared(repeated)


def test_the_dense_roster_names_the_localisers_operation() -> None:
    """The dense line builds only on a spine whose WAS stage runs this
    operation (spine_posture), so the roster's name is the localiser's."""

    from microcosm.build.uk_runtime.geography_first_roster import (
        GEOGRAPHY_FIRST_OPERATIONS,
    )
    from microcosm.build.uk_runtime.was_wealth import UK_WAS_WEALTH_STAGE_NAME

    assert dict(GEOGRAPHY_FIRST_OPERATIONS) == {
        UK_WAS_WEALTH_STAGE_NAME: (ahv.AREA_HOME_VALUES_OPERATION_KIND,)
    }


def test_the_packaged_spec_pairs_the_localiser_with_the_upratings_denominator() -> None:
    """Form F (C-15) holds only where the WAS stage localises and the regional
    uprating divides by the kept draw: either alone lands England and Wales
    homes off their knots. Neither is declared yet; the declaration commit
    declares both."""

    from microcosm.build.uk_runtime.regional_uprating import (
        declared_regional_reference_denominator,
    )

    stages = load_country_spec("uk").sources.stage_map()
    assert ahv.area_home_values_declared(stages["was_wealth"]) == (
        declared_regional_reference_denominator(stages["regional_property_uprating"])
        == RANK_BASIS_COLUMN
    )
