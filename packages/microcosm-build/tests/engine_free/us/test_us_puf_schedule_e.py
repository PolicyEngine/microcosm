"""Schedule E line 32 by entity and its passive subset on the PUF donor.

microcosm#1239 put line 32 and its passive subset on the donor; microcosm#1258
splits line 32 into its partnership and S corporation shares. The invariants
below hold for every input the PUF can carry:

1. Line-32 identity: E26270 equals its Schedule E line 28 column fields,
   (g) + (j) - (f) - (h) - (i) for both entity types, to within the PUF's
   documented per-field rounding.
2. Conditional dominance: where a return reports no nonpassive loss and no
   section 179 deduction, its passive amount never exceeds a positive line 32
   by more than that rounding.
3. Leg bounds: the passive amount lies between minus the loss columns and the
   income columns.
4. Donor alignment: every regular processed tax unit carries its source
   record's entity shares and passive amount exactly, every aggregate-row
   bucket carries that row's weighted partnership, S-corporation and passive
   totals, and the processed nonpassive column must equal the archived sum
   before anything is replaced.
5. Entity partition: ``partnership + s_corp`` equals ``passive + nonpassive``
   exactly (two partitions of the same ten fields), so the passive subset is
   unchanged by the split and the entity sum meets invariant 1.
6. Entity support: an entity share is zero whenever that entity's five columns
   are zero, so the split never puts a rounding residual on a return with no
   activity of that type, and each share lies between minus its own loss
   columns and its own income columns.

"passive <= total where both are positive" is deliberately not an invariant:
a passive gain offset by a nonpassive loss is a lawful return whose passive
amount exceeds its line 32 (``test_passive_can_exceed_a_positive_line_32``).
On the pinned TY2015 file 2,745 records (0.20M weighted returns) do so.
"""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime.puf_source_schedule_e import (
    PUF_SCHEDULE_E_INCOME_LEGS,
    PUF_SCHEDULE_E_LOSS_LEGS,
    PUF_SCHEDULE_E_PART_II_LEGS,
    PUF_SCHEDULE_E_PARTNERSHIP_LEGS,
    PUF_SCHEDULE_E_S_CORP_LEGS,
    SCHEDULE_E_DONOR_OUTPUTS,
    processed_puf_tax_unit_partnership_s_corp_income,
    schedule_e_line_32_rounding_bound,
    schedule_e_partnership_s_corp_columns,
    source_year_puf_partnership_s_corp_income,
)

_PASSIVE_INCOME = ("E25940", "E26170")
_PASSIVE_LOSS = ("E25920", "E26160")
_NONPASSIVE_INCOME = ("E25980", "E26190")
_NONPASSIVE_LOSS = ("E25960", "E26180")
_SECTION_179 = ("E26110", "E26100")
# Booklet roles by entity, written out independently of the module constants.
_PARTNERSHIP_INCOME = ("E25940", "E25980")
_PARTNERSHIP_LOSSES = ("E25920", "E25960", "E26110")
_S_CORP_INCOME = ("E26170", "E26190")
_S_CORP_LOSSES = ("E26160", "E26180", "E26100")

# The restricted pinned inputs live in the archived pipeline's storage folder,
# as for the E00100 alignment test; MICROCOSM_RESTRICTED_PUF_DIR points
# elsewhere for a one-off local check.
_REAL_PUF_STORAGE = Path(
    os.environ.get(
        "MICROCOSM_RESTRICTED_PUF_DIR",
        Path.home()
        / "PolicyEngine"
        / ("policyengine" + "-us-data")
        / ("policyengine" + "_us_data")
        / "storage",
    )
)
_REAL_PROCESSED_PUF = _REAL_PUF_STORAGE / "puf_2024.h5"
_REAL_SOURCE_PUF = _REAL_PUF_STORAGE / "puf_2015.csv"
_REAL_PROCESSED_PUF_SHA256 = (
    "7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df"
)
_REAL_SOURCE_PUF_SHA256 = (
    "0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df"
)
_REAL_PARTNERSHIP_VECTOR_SHA256 = (
    "b710efb7c1e273d75b0dc0146bd10c5265794688bd9e29ab07452fcdf4d64d37"
)
_REAL_S_CORP_VECTOR_SHA256 = (
    "d0ce9c9dc009545e4af7c1654812625e86b9caa283aca009b70684c42cf5d0c7"
)
# Unchanged from microcosm#1239: the entity split does not touch the passive
# subset, byte for byte.
_REAL_PASSIVE_VECTOR_SHA256 = (
    "6c76e077304567ff5b499142012aae8c7863153d95c41ddbff385b968c4eca53"
)


def _puf_round(value: float) -> float:
    """Apply the 2014 booklet's disclosure step 9 to one amount."""

    magnitude = abs(value)
    if magnitude == 0:
        return 0.0
    if magnitude < 5:
        return float(np.copysign(2.0, value))
    if magnitude <= 10_000:
        unit = 10.0
    elif magnitude <= 100_000:
        unit = 100.0
    else:
        unit = 10.0 ** (int(np.floor(np.log10(magnitude))) - 3)
    return float(np.copysign(np.floor(magnitude / unit + 0.5) * unit, value))


def _line_32(legs: dict[str, float]) -> float:
    return sum(legs[field] for field in PUF_SCHEDULE_E_INCOME_LEGS) - sum(
        legs[field] for field in PUF_SCHEDULE_E_LOSS_LEGS
    )


def _row(**legs: float) -> dict[str, float]:
    row = {field: 0.0 for field in PUF_SCHEDULE_E_PART_II_LEGS}
    row.update(legs)
    row["E26270"] = _line_32(row)
    return row


_amount = st.integers(min_value=0, max_value=200_000_000).map(float)
_maybe_zero_amount = st.one_of(st.just(0.0), _amount)
_leg_rows = st.fixed_dictionaries(
    {field: _maybe_zero_amount for field in PUF_SCHEDULE_E_PART_II_LEGS}
)


def test_booklet_field_roles_define_line_32_and_the_passive_subset() -> None:
    frame = pd.DataFrame(
        [
            # Passive gain only: all of line 32 is passive.
            _row(E25940=1_000.0),
            # Passive loss allowed only.
            _row(E26160=400.0),
            # Nonpassive only, with a section 179 deduction (nonpassive).
            _row(E26190=5_000.0, E26100=700.0),
            # Every column on both entities.
            _row(
                E25940=300.0,
                E25980=2_000.0,
                E25920=50.0,
                E25960=600.0,
                E26110=40.0,
                E26170=90.0,
                E26190=800.0,
                E26160=10.0,
                E26180=120.0,
                E26100=30.0,
            ),
        ]
    )

    columns = schedule_e_partnership_s_corp_columns(frame)

    np.testing.assert_array_equal(
        columns["line_32"], [1_000.0, -400.0, 4_300.0, 2_340.0]
    )
    # (g) + (j) - (f) - (h) - (i) of each entity's own columns.
    np.testing.assert_array_equal(columns["partnership"], [1_000.0, 0.0, 0.0, 1_610.0])
    np.testing.assert_array_equal(columns["s_corp"], [0.0, -400.0, 4_300.0, 730.0])
    np.testing.assert_array_equal(columns["passive"], [1_000.0, -400.0, 0.0, 330.0])
    np.testing.assert_array_equal(columns["nonpassive"], [0.0, 0.0, 4_300.0, 2_010.0])
    # The processed artifact's archived column omits the passive legs and
    # section 179 (us-data datasets/puf/puf.py, "active" income).
    np.testing.assert_array_equal(
        columns["archived_nonpassive"], [0.0, 0.0, 5_000.0, 2_080.0]
    )


def test_passive_can_exceed_a_positive_line_32() -> None:
    """The intended violation of "passive <= total where both are positive".

    A $300 passive gain and a $100 nonpassive loss put $200 on line 32. Form
    8960 counts the $300 (the loss is a non-section-1411 adjustment), so the
    donor must carry passive $300 against a $200 total, not clip it.
    """

    columns = schedule_e_partnership_s_corp_columns(
        pd.DataFrame([_row(E25940=300.0, E25960=100.0)])
    )

    assert columns["line_32"][0] == 200.0
    assert columns["passive"][0] == 300.0
    assert columns["passive"][0] > columns["line_32"][0]


@pytest.mark.parametrize(
    ("stored", "bound"),
    [
        (0.0, 0.0),
        (2.0, 3.0),
        (-2.0, 3.0),
        (5.0, 5.0),
        (9_990.0, 5.0),
        (10_000.0, 50.0),
        (10_100.0, 50.0),
        (99_900.0, 50.0),
        (100_000.0, 50.0),
        (228_900.0, 50.0),
        (1_158_000.0, 500.0),
        (-90_550_000.0, 5_000.0),
    ],
)
def test_rounding_bound_matches_the_booklet_bands(stored: float, bound: float) -> None:
    assert schedule_e_line_32_rounding_bound([stored])[0] == bound


@settings(max_examples=300, deadline=None)
@given(_leg_rows)
def test_identity_holds_within_the_bound_after_disclosure_rounding(
    legs: dict[str, float],
) -> None:
    """Invariant 1: independently rounded fields still pass the identity."""

    stored = {field: _puf_round(value) for field, value in legs.items()}
    stored["E26270"] = _puf_round(_line_32(legs))

    columns = schedule_e_partnership_s_corp_columns(pd.DataFrame([stored]))

    residual = columns["line_32"][0] - (
        columns["passive"][0] + columns["nonpassive"][0]
    )
    assert abs(residual) <= columns["rounding_bound"][0]


@settings(max_examples=300, deadline=None)
@given(_leg_rows)
def test_identity_refuses_a_line_32_beyond_the_rounding_bound(
    legs: dict[str, float],
) -> None:
    stored = dict(legs)
    exact = _line_32(legs)
    bound = schedule_e_line_32_rounding_bound([exact])[0] + sum(
        schedule_e_line_32_rounding_bound([legs[field]])[0]
        for field in PUF_SCHEDULE_E_PART_II_LEGS
    )
    stored["E26270"] = exact + 2.0 * bound + 1_000_000.0

    with pytest.raises(ValueError, match="per-field rounding"):
        schedule_e_partnership_s_corp_columns(pd.DataFrame([stored]))


@settings(max_examples=300, deadline=None)
@given(
    st.fixed_dictionaries(
        {
            **{field: _maybe_zero_amount for field in _PASSIVE_INCOME},
            **{field: _maybe_zero_amount for field in _PASSIVE_LOSS},
            **{field: _maybe_zero_amount for field in _NONPASSIVE_INCOME},
        }
    )
)
def test_passive_never_exceeds_line_32_without_nonpassive_losses(
    legs: dict[str, float],
) -> None:
    """Invariant 2, on rounded fields: true whenever (h) and (i) are zero."""

    full = {field: 0.0 for field in PUF_SCHEDULE_E_PART_II_LEGS}
    full.update(legs)
    stored = {field: _puf_round(value) for field, value in full.items()}
    stored["E26270"] = _puf_round(_line_32(full))

    columns = schedule_e_partnership_s_corp_columns(pd.DataFrame([stored]))

    passive, line_32 = columns["passive"][0], columns["line_32"][0]
    if passive > 0 and line_32 > 0:
        assert passive <= line_32 + columns["rounding_bound"][0]


@settings(max_examples=300, deadline=None)
@given(_leg_rows)
def test_passive_lies_between_the_loss_and_income_columns(
    legs: dict[str, float],
) -> None:
    """Invariant 3: the passive subset is bounded by the line 28 columns."""

    row = dict(legs)
    row["E26270"] = _line_32(legs)

    columns = schedule_e_partnership_s_corp_columns(pd.DataFrame([row]))

    passive = columns["passive"][0]
    assert passive <= sum(legs[field] for field in PUF_SCHEDULE_E_INCOME_LEGS)
    assert passive >= -sum(legs[field] for field in PUF_SCHEDULE_E_LOSS_LEGS)
    assert passive <= sum(legs[field] for field in _PASSIVE_INCOME)
    assert passive >= -sum(legs[field] for field in _PASSIVE_LOSS)


def test_module_entity_legs_match_the_booklet_roles() -> None:
    """The entity constants name exactly the booklet's columns for each type."""

    assert PUF_SCHEDULE_E_PARTNERSHIP_LEGS == (_PARTNERSHIP_INCOME, _PARTNERSHIP_LOSSES)
    assert PUF_SCHEDULE_E_S_CORP_LEGS == (_S_CORP_INCOME, _S_CORP_LOSSES)
    assert sorted(
        (*_PARTNERSHIP_INCOME, *_PARTNERSHIP_LOSSES, *_S_CORP_INCOME, *_S_CORP_LOSSES)
    ) == sorted(PUF_SCHEDULE_E_PART_II_LEGS)
    assert SCHEDULE_E_DONOR_OUTPUTS == (
        "partnership_income",
        "s_corp_income",
        "passive_partnership_s_corp_income",
    )


@settings(max_examples=500, deadline=None)
@given(_leg_rows)
def test_entity_shares_partition_line_32_exactly(legs: dict[str, float]) -> None:
    """Invariant 5: partnership + S corp == passive + nonpassive, bit for bit.

    PUF amounts are whole dollars far below 2**53, so both partitions of the
    ten fields sum without rounding.
    """

    row = dict(legs)
    row["E26270"] = _line_32(legs)

    columns = schedule_e_partnership_s_corp_columns(pd.DataFrame([row]))

    entity_total = columns["partnership"][0] + columns["s_corp"][0]
    assert entity_total == columns["passive"][0] + columns["nonpassive"][0]
    assert entity_total == _line_32(legs)
    assert columns["partnership"][0] == sum(
        legs[field] for field in _PARTNERSHIP_INCOME
    ) - sum(legs[field] for field in _PARTNERSHIP_LOSSES)
    assert columns["s_corp"][0] == sum(legs[field] for field in _S_CORP_INCOME) - sum(
        legs[field] for field in _S_CORP_LOSSES
    )


@settings(max_examples=500, deadline=None)
@given(_leg_rows, st.sampled_from(("partnership", "s_corp")))
def test_an_entity_without_columns_carries_nothing(
    legs: dict[str, float],
    silent: str,
) -> None:
    """Invariant 6: no rounding residual lands on an entity with no columns.

    The stored E26270 is rounded independently of the fields, yet the entity
    with five zero columns stays exactly zero, so a return reporting only S
    corporation activity never becomes a partnership carrier (or vice versa).
    """

    silenced = (
        (*_PARTNERSHIP_INCOME, *_PARTNERSHIP_LOSSES)
        if silent == "partnership"
        else (*_S_CORP_INCOME, *_S_CORP_LOSSES)
    )
    full = {field: 0.0 if field in silenced else value for field, value in legs.items()}
    stored = {field: _puf_round(value) for field, value in full.items()}
    stored["E26270"] = _puf_round(_line_32(full))

    columns = schedule_e_partnership_s_corp_columns(pd.DataFrame([stored]))

    assert columns[silent][0] == 0.0
    other = "s_corp" if silent == "partnership" else "partnership"
    assert abs(stored["E26270"] - columns[other][0]) <= columns["rounding_bound"][0]


@settings(max_examples=500, deadline=None)
@given(_leg_rows)
def test_each_entity_share_lies_between_its_own_loss_and_income_columns(
    legs: dict[str, float],
) -> None:
    """Invariant 6: an entity share is bounded by that entity's columns only."""

    row = dict(legs)
    row["E26270"] = _line_32(legs)

    columns = schedule_e_partnership_s_corp_columns(pd.DataFrame([row]))

    for column, income, losses in (
        ("partnership", _PARTNERSHIP_INCOME, _PARTNERSHIP_LOSSES),
        ("s_corp", _S_CORP_INCOME, _S_CORP_LOSSES),
    ):
        share = columns[column][0]
        assert share <= sum(legs[field] for field in income)
        assert share >= -sum(legs[field] for field in losses)


@settings(max_examples=300, deadline=None)
@given(_leg_rows)
def test_raw_array_path_agrees_with_the_source_year_columns(
    legs: dict[str, float],
) -> None:
    """Differential: puf_support's raw-field derivation equals this module's.

    A raw PUF array (one carrying the Part II fields) derives the three leaves
    inside ``puf_support``; the processed donor takes them from
    :func:`schedule_e_partnership_s_corp_columns`. Both must agree exactly.
    """

    from microcosm.build.us_runtime.puf_support import (
        _raw_schedule_e_line_32_values,
    )

    row = dict(legs)
    row["E26270"] = _line_32(legs)
    columns = schedule_e_partnership_s_corp_columns(pd.DataFrame([row]))
    arrays = {field: np.asarray([value]) for field, value in legs.items()}

    for output, column in (
        ("partnership_income", "partnership"),
        ("s_corp_income", "s_corp"),
        ("passive_partnership_s_corp_income", "passive"),
    ):
        raw = _raw_schedule_e_line_32_values(arrays, output)
        assert raw is not None
        np.testing.assert_array_equal(raw, columns[column])


@pytest.mark.parametrize("field", PUF_SCHEDULE_E_PART_II_LEGS)
def test_columns_refuse_a_missing_or_negative_leg(field: str) -> None:
    row = _row(E25940=10.0, E25980=20.0)
    missing = pd.DataFrame([row]).drop(columns=[field])
    with pytest.raises(ValueError, match="missing column"):
        schedule_e_partnership_s_corp_columns(missing)

    negative = dict(row)
    negative[field] = -1.0
    negative["E26270"] = row["E26270"]
    with pytest.raises(ValueError, match="nonnegative"):
        schedule_e_partnership_s_corp_columns(pd.DataFrame([negative]))


# ---------------------------------------------------------------------------
# Donor alignment
# ---------------------------------------------------------------------------

_AGGREGATE_LEGS = {
    999_996: {"E25960": 3_000_000.0, "E25920": 600_000.0, "E25940": 100_000.0},
    999_997: {"E26180": 400_000.0, "E26160": 50_000.0, "E26190": 100_000.0},
    999_998: {"E25980": 4_000_000.0, "E25940": 3_500_000.0, "E26110": 20_000.0},
    999_999: {
        "E26190": 25_000_000.0,
        "E26170": 2_000_000.0,
        "E26180": 500_000.0,
        "E26100": 30_000.0,
    },
}


def _source_and_processed() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """A regular-plus-aggregate TY2015 source and its processed layout.

    Thirty regular records and the four aggregate rows; the processed side has
    the archived bucket counts (20, 23, 39 and a 20-record open tail) and a
    nonpassive column equal to the archived sum, with every bucket carrying
    its aggregate row's weighted total.
    """

    rng = np.random.default_rng(1239)
    regular_rows = []
    for _ in range(30):
        legs = {
            field: float(rng.choice([0.0, rng.integers(1, 5_000_000)]))
            for field in PUF_SCHEDULE_E_PART_II_LEGS
        }
        regular_rows.append(_row(**legs))
    aggregate_rows = [_row(**legs) for legs in _AGGREGATE_LEGS.values()]
    source = pd.DataFrame(regular_rows + aggregate_rows)
    source.insert(0, "RECID", [*range(1, 31), *_AGGREGATE_LEGS])
    source.insert(1, "MARS", [1.0] * 30 + [0.0] * 4)
    source.insert(2, "S006", [100.0] * 30 + [14_000.0, 23_000.0, 39_000.0, 10_000.0])

    columns = schedule_e_partnership_s_corp_columns(source)
    regular_nonpassive = columns["archived_nonpassive"][:30]
    bucket_weights = [
        np.full(20, 7.0),
        np.full(23, 10.0),
        np.full(39, 10.0),
        np.full(20, 5.0),
    ]
    bucket_values = []
    for index, weights in enumerate(bucket_weights):
        population = float(source["S006"].iloc[30 + index]) / 100.0
        target = population * columns["archived_nonpassive"][30 + index]
        base = rng.uniform(0.2, 1.8, size=len(weights))
        bucket_values.append(base * target / float(weights @ base))
    processed = {
        "tax_unit_id": np.concatenate(
            (np.arange(1, 31), np.arange(1_000_000, 1_000_102))
        ),
        "household_weight": np.concatenate((np.ones(30), *bucket_weights)),
        "nonpassive": np.concatenate((regular_nonpassive, *bucket_values)),
    }
    return source, processed


def _align(source: pd.DataFrame, processed: dict[str, np.ndarray], **kwargs):
    return source_year_puf_partnership_s_corp_income(
        source,
        processed_tax_unit_ids=processed["tax_unit_id"],
        processed_tax_unit_weights=processed["household_weight"],
        processed_nonpassive_partnership_s_corp_income=processed["nonpassive"],
        **kwargs,
    )


def test_alignment_carries_entity_shares_and_passive_on_regular_records() -> None:
    source, processed = _source_and_processed()

    aligned = _align(source, processed)

    assert tuple(aligned) == SCHEDULE_E_DONOR_OUTPUTS
    expected_partnership = (
        source[list(_PARTNERSHIP_INCOME)].sum(axis=1)
        - source[list(_PARTNERSHIP_LOSSES)].sum(axis=1)
    ).to_numpy()[:30]
    expected_s_corp = (
        source[list(_S_CORP_INCOME)].sum(axis=1)
        - source[list(_S_CORP_LOSSES)].sum(axis=1)
    ).to_numpy()[:30]
    np.testing.assert_array_equal(
        aligned["partnership_income"][:30], expected_partnership
    )
    np.testing.assert_array_equal(aligned["s_corp_income"][:30], expected_s_corp)
    # The fixture's E26270 is the exact field sum, so the entities meet it.
    np.testing.assert_array_equal(
        aligned["partnership_income"][:30] + aligned["s_corp_income"][:30],
        source["E26270"].to_numpy()[:30],
    )
    expected_passive = (
        source["E25940"] - source["E25920"] + source["E26170"] - source["E26160"]
    ).to_numpy()[:30]
    np.testing.assert_array_equal(
        aligned["passive_partnership_s_corp_income"][:30], expected_passive
    )


def test_alignment_keeps_entity_shares_when_e26270_carries_a_rounding_residual() -> (
    None
):
    """The donor carries the field sums, not a residual-bearing E26270."""

    source, processed = _source_and_processed()
    rounded = source.copy()
    # A rounding residual inside the booklet bound on a regular record whose
    # fields are untouched (so the archived nonpassive check still passes).
    row = int(np.flatnonzero(rounded["E26270"].to_numpy()[:30] != 0)[0])
    rounded.loc[row, "E26270"] = rounded.loc[row, "E26270"] + 3.0

    aligned = _align(rounded, processed)
    baseline = _align(source, processed)

    for output in SCHEDULE_E_DONOR_OUTPUTS:
        np.testing.assert_array_equal(aligned[output], baseline[output])


def test_alignment_preserves_each_aggregate_rows_weighted_totals() -> None:
    """Invariant 4 on the synthetic buckets: the signed totals match line 32."""

    source, processed = _source_and_processed()
    receipt: dict[str, object] = {}

    aligned = _align(source, processed, receipt=receipt)

    weights = processed["household_weight"]
    starts = np.cumsum([30, 20, 23, 39])
    stops = np.cumsum([50, 23, 39, 20])
    for index, recid in enumerate(_AGGREGATE_LEGS):
        rows = slice(int(starts[index]), int(stops[index]))
        population = float(source["S006"].iloc[30 + index]) / 100.0
        row = source.iloc[30 + index]
        partnership = sum(row[f] for f in _PARTNERSHIP_INCOME) - sum(
            row[f] for f in _PARTNERSHIP_LOSSES
        )
        s_corp = sum(row[f] for f in _S_CORP_INCOME) - sum(
            row[f] for f in _S_CORP_LOSSES
        )
        passive = row["E25940"] - row["E25920"] + row["E26170"] - row["E26160"]
        archived = row["E25980"] - row["E25960"] + row["E26190"] - row["E26180"]
        for output, per_return in (
            ("partnership_income", partnership),
            ("s_corp_income", s_corp),
            ("passive_partnership_s_corp_income", passive),
        ):
            # Each bucket carries its aggregate row's weighted total by entity.
            assert float(weights[rows] @ aligned[output][rows]) == pytest.approx(
                population * per_return, rel=1e-12, abs=1e-6
            )
            # Each synthetic record keeps its processed sign pattern.
            np.testing.assert_allclose(
                aligned[output][rows],
                processed["nonpassive"][rows] * (per_return / archived),
                rtol=1e-12,
            )
        assert float(
            weights[rows]
            @ (aligned["partnership_income"][rows] + aligned["s_corp_income"][rows])
        ) == pytest.approx(population * row["E26270"], rel=1e-12)
        bucket = receipt["synthetic_buckets"][str(recid)]
        assert bucket["partnership_per_return"] == partnership
        assert bucket["s_corp_per_return"] == s_corp
    line_32 = aligned["partnership_income"] + aligned["s_corp_income"]
    assert receipt["weighted_line_32"] == pytest.approx(float(weights @ line_32))
    for output in ("partnership_income", "s_corp_income"):
        legs = receipt["entities"][output]
        values = aligned[output]
        assert legs["weighted_net"] == pytest.approx(float(weights @ values))
        assert legs["weighted_income"] - legs["weighted_loss"] == pytest.approx(
            legs["weighted_net"]
        )
        assert legs["weighted_income_returns"] == pytest.approx(
            float(weights[values > 0].sum())
        )
        assert legs["weighted_loss_returns"] == pytest.approx(
            float(weights[values < 0].sum())
        )


def test_alignment_is_bit_deterministic() -> None:
    source, processed = _source_and_processed()

    first = _align(source, processed)
    second = _align(source, processed)

    for key in first:
        np.testing.assert_array_equal(
            first[key].view(np.uint64), second[key].view(np.uint64)
        )


def test_alignment_refuses_a_processed_column_that_is_not_the_archived_sum() -> None:
    source, processed = _source_and_processed()
    drifted = dict(processed)
    drifted["nonpassive"] = processed["nonpassive"].copy()
    # A processed artifact that already carried line 32 (or uprated it) must
    # not have line 32 layered on again.
    drifted["nonpassive"][:30] = source["E26270"].to_numpy()[:30] + 1.0

    with pytest.raises(ValueError, match="archived nonpassive sum"):
        _align(source, drifted)


def test_alignment_refuses_a_bucket_without_its_aggregate_total() -> None:
    source, processed = _source_and_processed()
    broken = dict(processed)
    broken["nonpassive"] = processed["nonpassive"].copy()
    broken["nonpassive"][30:50] *= 2.0

    with pytest.raises(ValueError, match="aggregate RECID 999996"):
        _align(source, broken)


def test_alignment_refuses_regular_record_drift() -> None:
    source, processed = _source_and_processed()
    ids = processed["tax_unit_id"].copy()
    ids[[3, 4]] = ids[[4, 3]]

    with pytest.raises(ValueError, match="regular RECID order"):
        source_year_puf_partnership_s_corp_income(
            source,
            processed_tax_unit_ids=ids,
            processed_tax_unit_weights=processed["household_weight"],
            processed_nonpassive_partnership_s_corp_income=processed["nonpassive"],
        )


def test_processed_nonpassive_sums_person_rows_to_tax_units() -> None:
    totals = processed_puf_tax_unit_partnership_s_corp_income(
        {
            "tax_unit_id": np.asarray([20, 10]),
            "person_tax_unit_id": np.asarray([10, 20, 20]),
            "partnership_s_corp_income": np.asarray([5.0, -2.0, 7.0]),
        }
    )

    np.testing.assert_array_equal(totals, [5.0, 5.0])

    with pytest.raises(ValueError, match="no person rows"):
        processed_puf_tax_unit_partnership_s_corp_income(
            {
                "tax_unit_id": np.asarray([10, 30]),
                "person_tax_unit_id": np.asarray([10]),
                "partnership_s_corp_income": np.asarray([1.0]),
            }
        )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_alignment_matches_the_pinned_flat_array_artifact() -> None:
    if not _REAL_PROCESSED_PUF.exists() or not _REAL_SOURCE_PUF.exists():
        pytest.skip("Pinned restricted PUF artifacts are not available.")
    assert _file_sha256(_REAL_PROCESSED_PUF) == _REAL_PROCESSED_PUF_SHA256
    assert _file_sha256(_REAL_SOURCE_PUF) == _REAL_SOURCE_PUF_SHA256

    import h5py

    with h5py.File(_REAL_PROCESSED_PUF, "r") as h5:
        arrays = {
            name: np.asarray(h5[name])
            for name in (
                "tax_unit_id",
                "household_weight",
                "person_tax_unit_id",
                "partnership_s_corp_income",
            )
        }
    receipt: dict[str, object] = {}
    aligned = source_year_puf_partnership_s_corp_income(
        _REAL_SOURCE_PUF,
        processed_tax_unit_ids=arrays["tax_unit_id"],
        processed_tax_unit_weights=arrays["household_weight"],
        processed_nonpassive_partnership_s_corp_income=(
            processed_puf_tax_unit_partnership_s_corp_income(arrays)
        ),
        receipt=receipt,
    )

    partnership = aligned["partnership_income"]
    s_corp = aligned["s_corp_income"]
    passive = aligned["passive_partnership_s_corp_income"]
    assert len(partnership) == 211_677
    assert sha256(partnership.astype("<f8").tobytes()).hexdigest() == (
        _REAL_PARTNERSHIP_VECTOR_SHA256
    )
    assert sha256(s_corp.astype("<f8").tobytes()).hexdigest() == (
        _REAL_S_CORP_VECTOR_SHA256
    )
    assert sha256(passive.astype("<f8").tobytes()).hexdigest() == (
        _REAL_PASSIVE_VECTOR_SHA256
    )
    # TY2015 dollars on the processed 2024 weights (microcosm#1239 and #1258
    # evidence). The entity sum keeps #1239's line-32 total to the dollar
    # shown; the stored E26270 differs from the fields on 38,378 regular
    # records, by $0.145 billion weighted in absolute value.
    assert receipt["weighted_line_32"] / 1e9 == pytest.approx(676.562, abs=1e-3)
    assert receipt["weighted_passive"] / 1e9 == pytest.approx(82.990, abs=1e-3)
    assert receipt["weighted_processed_nonpassive"] / 1e9 == pytest.approx(
        642.391, abs=1e-3
    )
    entities = receipt["entities"]
    assert entities["partnership_income"]["weighted_net"] / 1e9 == pytest.approx(
        250.584, abs=1e-3
    )
    assert entities["s_corp_income"]["weighted_net"] / 1e9 == pytest.approx(
        425.978, abs=1e-3
    )
    residual = receipt["regular_stored_e26270_residual"]
    assert residual["records_differing"] == 38_378
    assert residual["max_share_of_rounding_bound"] <= 1.0
