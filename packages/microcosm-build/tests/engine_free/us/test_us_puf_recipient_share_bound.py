"""The PUF recipient-share bound: one draw cannot outweigh the slice it stands for.

A PUF-clone recipient whose weight is the share ``s`` of the recipient
population stands for ``s`` of the population. The bound limits its raw draw
to the weighted mean of the donor slice holding the same share at that end of
the distribution, so ``w_r * value <= (W_r / W_d) * mass(slice)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import microcosm.build.us_runtime.puf_support as puf_support_module

_bounds = puf_support_module._recipient_share_slice_bounds
_grid_within = puf_support_module._largest_grid_values_within
_snap = puf_support_module._snap_to_observed_values

_PROPERTY_SETTINGS = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_amounts = st.floats(
    min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False
).map(lambda value: float(np.round(value, 2)))
_positive_weights = st.floats(
    min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False
)


@st.composite
def _donor_and_recipients(draw: st.DrawFn) -> tuple[np.ndarray, ...]:
    donors = draw(st.integers(min_value=1, max_value=40))
    values = np.asarray(draw(st.lists(_amounts, min_size=donors, max_size=donors)))
    weights = np.asarray(
        draw(st.lists(_positive_weights, min_size=donors, max_size=donors))
    )
    recipients = draw(st.integers(min_value=1, max_value=25))
    recipient_weights = np.asarray(
        draw(st.lists(_positive_weights, min_size=recipients, max_size=recipients))
    )
    return values, weights, recipient_weights


def _reference_slice_mean(
    values: np.ndarray, weights: np.ndarray, take: float, *, top: bool
) -> float:
    """One slice's same-sign mass per unit weight, donor by donor."""

    order = sorted(range(len(values)), key=lambda i: -values[i] if top else values[i])
    remaining = take
    mass = 0.0
    for index in order:
        used = min(weights[index], remaining)
        part = max(values[index], 0.0) if top else min(values[index], 0.0)
        mass += used * part
        remaining -= used
        if remaining <= 0.0:
            break
    return mass / take


def _reference_bounds(
    values: np.ndarray, weights: np.ndarray, recipient_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    donor_total = weights.sum()
    recipient_total = recipient_weights.sum()
    floors, caps = [], []
    for weight in recipient_weights:
        take = weight / recipient_total * donor_total
        top = _reference_slice_mean(values, weights, take, top=True)
        bottom = _reference_slice_mean(values, weights, take, top=False)
        caps.append(top if top > 0.0 else np.inf)
        floors.append(bottom if bottom < 0.0 else -np.inf)
    return np.asarray(floors), np.asarray(caps)


def _close(actual: np.ndarray, expected: np.ndarray) -> None:
    """Equal infinities, and finite values within float accumulation error."""

    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-9, atol=1e-6)


def _top_slice_edge(values: np.ndarray, weights: np.ndarray, take: float) -> float:
    """The value at the inner edge of the top slice holding ``take`` weight."""

    order = np.argsort(-values, kind="stable")
    cumulative = np.cumsum(weights[order])
    edge = min(int(np.searchsorted(cumulative, take, side="left")), len(values) - 1)
    return float(values[order][edge])


def test_bound_is_the_slice_mean_on_a_worked_example() -> None:
    # Donor population of weight 1,000: 990 at 10, 9 at 1,000, 1 at 100,000.
    values = np.asarray([10.0, 1_000.0, 100_000.0, -50.0])
    weights = np.asarray([980.0, 9.0, 1.0, 10.0])
    # Recipients of share 0.001, 0.01 and 0.989 of their own population.
    recipient_weights = np.asarray([1.0, 10.0, 989.0])

    floor, cap = _bounds(values, weights, recipient_weights)

    # Share 0.001 of 1,000 donor weight is the 100,000 record alone.
    assert cap[0] == 100_000.0
    # Share 0.01 is weight 10: the 100,000 record and all 9 at 1,000.
    assert cap[1] == pytest.approx((100_000.0 + 9 * 1_000.0) / 10)
    # The bottom slice of weight 10 is the ten records at -50.
    assert floor[1] == -50.0
    # Share 0.989 is weight 989: on top, everything but 11 of the 980 at 10.
    top_mass = 100_000.0 + 9_000.0 + 979.0 * 10.0
    assert cap[2] == pytest.approx(top_mass / 989.0)
    # The bottom slice of weight 989 holds every loss, so the heaviest
    # recipient may carry at most the population's whole loss mass.
    assert floor[2] == pytest.approx(-500.0 / 989.0)
    assert floor[0] == -50.0


def test_one_recipient_is_bounded_by_the_donor_mean() -> None:
    values = np.asarray([1.0, 2.0, 1_000.0])
    weights = np.asarray([10.0, 10.0, 1.0])

    floor, cap = _bounds(values, weights, np.asarray([7.0]))

    assert cap[0] == pytest.approx(1_030.0 / 21.0)
    assert floor[0] == -np.inf


def test_a_slice_of_tied_donors_caps_exactly_at_their_value() -> None:
    # Statutory caps tie many donors at one value: a slice made only of them
    # must cap at that value exactly, or draws equal to it would be clipped.
    weights = np.asarray([0.1, 3.7, 1234.5, 0.3, 77.0] * 40)
    values = np.full(len(weights), 2_500.0)
    values[::7] = 1_000.0
    recipient_weights = np.linspace(0.5, 900.0, 64)

    _, cap = _bounds(values, weights, recipient_weights / recipient_weights.sum())

    top_tie_weight = weights[values == 2_500.0].sum()
    within = recipient_weights / recipient_weights.sum() * weights.sum() <= (
        top_tie_weight
    )
    assert within.any()
    assert (cap[within] == 2_500.0).all()
    assert (cap <= 2_500.0).all()


def test_a_column_with_no_positive_donor_has_no_cap() -> None:
    floor, cap = _bounds(
        np.asarray([0.0, -5.0, -1.0]),
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([1.0, 2.0]),
    )
    assert np.isinf(cap).all()
    assert (floor < 0.0).all()


@pytest.mark.parametrize(
    ("donor_values", "donor_weights", "recipient_weights", "message"),
    [
        ([1.0], [1.0, 2.0], [1.0], "aligned"),
        ([], [], [1.0], "at least one donor"),
        ([np.nan], [1.0], [1.0], "finite"),
        ([1.0], [1.0], [np.inf], "finite"),
        ([1.0], [-1.0], [1.0], "nonnegative"),
        ([1.0], [1.0], [-1.0], "nonnegative"),
        ([1.0], [0.0], [1.0], "positive mass"),
        ([1.0], [1.0], [0.0], "positive mass"),
    ],
)
def test_bound_refuses_malformed_inputs(
    donor_values: list[float],
    donor_weights: list[float],
    recipient_weights: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _bounds(
            np.asarray(donor_values, dtype=np.float64),
            np.asarray(donor_weights, dtype=np.float64),
            np.asarray(recipient_weights, dtype=np.float64),
        )


@_PROPERTY_SETTINGS
@given(_donor_and_recipients())
def test_vectorized_bound_matches_the_donor_by_donor_reference(
    case: tuple[np.ndarray, ...],
) -> None:
    values, weights, recipient_weights = case

    floor, cap = _bounds(values, weights, recipient_weights)
    expected_floor, expected_cap = _reference_bounds(values, weights, recipient_weights)

    _close(cap, expected_cap)
    _close(floor, expected_floor)


@_PROPERTY_SETTINGS
@given(_donor_and_recipients())
def test_one_record_never_carries_more_than_its_slice_mass(
    case: tuple[np.ndarray, ...],
) -> None:
    values, weights, recipient_weights = case
    donor_total = weights.sum()
    recipient_total = recipient_weights.sum()

    floor, cap = _bounds(values, weights, recipient_weights)

    for index, weight in enumerate(recipient_weights):
        take = weight / recipient_total * donor_total
        scale = recipient_total / donor_total
        if np.isfinite(cap[index]):
            top_mass = take * _reference_slice_mean(values, weights, take, top=True)
            assert weight * cap[index] == pytest.approx(scale * top_mass, rel=1e-9)
        if np.isfinite(floor[index]):
            bottom_mass = take * _reference_slice_mean(values, weights, take, top=False)
            assert weight * floor[index] == pytest.approx(scale * bottom_mass, rel=1e-9)


@_PROPERTY_SETTINGS
@given(_donor_and_recipients())
def test_bound_stays_inside_the_donor_support_and_slice(
    case: tuple[np.ndarray, ...],
) -> None:
    values, weights, recipient_weights = case
    donor_total = weights.sum()
    recipient_total = recipient_weights.sum()

    floor, cap = _bounds(values, weights, recipient_weights)

    for index, weight in enumerate(recipient_weights):
        take = weight / recipient_total * donor_total
        if np.isfinite(cap[index]):
            assert 0.0 < cap[index] <= values.max()
            # The mean of the slice is at least its inner edge, exactly.
            edge = _top_slice_edge(values, weights, take)
            assert cap[index] >= edge
        if np.isfinite(floor[index]):
            assert values.min() <= floor[index] < 0.0
            edge = -_top_slice_edge(-values, weights, take)
            assert floor[index] <= edge


@_PROPERTY_SETTINGS
@given(_donor_and_recipients())
def test_heavier_recipients_get_tighter_bounds(case: tuple[np.ndarray, ...]) -> None:
    values, weights, recipient_weights = case

    floor, cap = _bounds(values, weights, recipient_weights)

    order = np.argsort(recipient_weights, kind="stable")
    ordered_cap = cap[order]
    ordered_floor = floor[order]
    tolerance = 1e-9 * max(1.0, float(np.abs(values).max()))
    assert (np.diff(ordered_cap[np.isfinite(ordered_cap)]) <= tolerance).all()
    assert (np.diff(ordered_floor[np.isfinite(ordered_floor)]) >= -tolerance).all()
    # A side applies to every recipient or to none: it is infinite only when
    # the donor has no value of that sign.
    assert np.isfinite(cap).all() == bool((values > 0.0).any())
    assert np.isinf(cap).all() == (not (values > 0.0).any())
    assert np.isfinite(floor).all() == bool((values < 0.0).any())
    assert np.isinf(floor).all() == (not (values < 0.0).any())


@_PROPERTY_SETTINGS
@given(
    _donor_and_recipients(),
    st.floats(min_value=1e-3, max_value=1e3),
    st.floats(min_value=1e-3, max_value=1e3),
)
def test_bound_depends_only_on_weight_shares(
    case: tuple[np.ndarray, ...], donor_scale: float, recipient_scale: float
) -> None:
    values, weights, recipient_weights = case

    floor, cap = _bounds(values, weights, recipient_weights)
    scaled_floor, scaled_cap = _bounds(
        values, weights * donor_scale, recipient_weights * recipient_scale
    )

    _close(scaled_cap, cap)
    _close(scaled_floor, floor)


@_PROPERTY_SETTINGS
@given(_donor_and_recipients(), st.randoms(use_true_random=False))
def test_bound_ignores_donor_order_and_follows_recipient_order(
    case: tuple[np.ndarray, ...], random: object
) -> None:
    values, weights, recipient_weights = case
    donor_order = list(range(len(values)))
    recipient_order = list(range(len(recipient_weights)))
    random.shuffle(donor_order)  # type: ignore[attr-defined]
    random.shuffle(recipient_order)  # type: ignore[attr-defined]

    floor, cap = _bounds(values, weights, recipient_weights)
    shuffled_floor, shuffled_cap = _bounds(
        values[donor_order], weights[donor_order], recipient_weights[recipient_order]
    )

    _close(shuffled_cap, cap[recipient_order])
    _close(shuffled_floor, floor[recipient_order])


@_PROPERTY_SETTINGS
@given(_donor_and_recipients(), st.floats(min_value=0.01, max_value=0.99))
def test_splitting_a_donor_record_leaves_the_bound_unchanged(
    case: tuple[np.ndarray, ...], fraction: float
) -> None:
    values, weights, recipient_weights = case
    split_values = np.concatenate([values, values[:1]])
    split_weights = np.concatenate(
        [[weights[0] * fraction], weights[1:], [weights[0] * (1.0 - fraction)]]
    )

    floor, cap = _bounds(values, weights, recipient_weights)
    split_floor, split_cap = _bounds(split_values, split_weights, recipient_weights)

    _close(split_cap, cap)
    _close(split_floor, floor)


@_PROPERTY_SETTINGS
@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1e8, allow_nan=False).map(round),
        min_size=1,
        max_size=30,
    ),
    st.lists(
        st.floats(min_value=-1e8, max_value=1e9, allow_nan=False),
        min_size=1,
        max_size=30,
    ),
    st.floats(min_value=0.0, max_value=1e9, allow_nan=False),
)
def test_snapping_after_the_lowered_cap_never_crosses_it(
    observed: list[float], draws: list[float], cap: float
) -> None:
    grid = np.unique(np.rint(np.asarray(observed, dtype=np.float64)).clip(min=0.0))
    caps = np.full(len(draws), cap)
    lowered = _grid_within(caps, grid)
    bounded = np.minimum(np.asarray(draws, dtype=np.float64), lowered)

    snapped = _snap(bounded, observed)

    assert (lowered <= np.maximum(cap, grid[0])).all()
    assert np.isin(lowered, grid).all()
    assert (snapped <= lowered).all()


def test_lowering_keeps_infinite_caps_infinite() -> None:
    grid = np.asarray([0.0, 5.0, 10.0])
    lowered = _grid_within(np.asarray([np.inf, 7.0, 10.0, 3.0]), grid)
    np.testing.assert_array_equal(lowered, [np.inf, 5.0, 10.0, 0.0])


# --------------------------------------------------------------------------- #
# The bound at the finalize seam                                              #
# --------------------------------------------------------------------------- #

import pandas as pd  # noqa: E402

from microcosm.build.us_runtime import (  # noqa: E402
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    support_channel_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights  # noqa: E402

_PASSTHROUGH = "partnership_income"
_SNAPPED = "taxable_interest_income"
_CALIBRATED = "farm_operations_income"
_BOOLEAN = "business_is_sstb"


def _recipient_frame(household_weights: list[float]) -> Frame:
    count = len(household_weights)
    ids = np.arange(1, count + 1, dtype=np.int64)
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": ids,
                "person_household_id": ids,
                "person_tax_unit_id": ids,
                "person_spm_unit_id": ids,
                "person_family_id": ids,
                "person_marital_unit_id": ids,
            }
        ),
        "household": pd.DataFrame(
            {"household_id": ids, "state_fips": np.full(count, 6, dtype=np.int64)}
        ),
        "tax_unit": pd.DataFrame(
            {"tax_unit_id": ids, "filing_status_input": ["SINGLE"] * count}
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    return clone_us_frame_for_puf_support(
        Frame(
            tables,
            US_SCHEMA,
            {
                "household": Weights(
                    np.asarray(household_weights, dtype=np.float64),
                    WeightKind.DESIGN,
                )
            },
        )
    )


def _recipient_index(frame: Frame) -> pd.Index:
    tax_unit = frame.table("tax_unit")
    return tax_unit.index[
        tax_unit[support_channel_column("tax_unit")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]


def _puf_values(frame: Frame, column: str) -> np.ndarray:
    person = frame.table("person")
    puf = person[
        person[support_channel_column("person")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ].sort_values(support_source_id_column("person"))
    return puf[column].to_numpy(dtype=np.float64)


def _recipient_weights(frame: Frame) -> np.ndarray:
    ids = frame.table("tax_unit").loc[_recipient_index(frame), "tax_unit_id"]
    return puf_support_module._tax_unit_household_weights(frame, ids)


def _finalize(
    frame: Frame,
    donor: pd.DataFrame,
    raw: dict[str, np.ndarray],
    *,
    diagnostics: list[dict[str, object]] | None,
    **kwargs: object,
) -> Frame:
    predictions = pd.DataFrame(
        {
            column: np.asarray(values, dtype=np.float64)
            for column, values in raw.items()
        },
        index=_recipient_index(frame),
    )
    return puf_support_module.finalize_us_puf_tax_detail_predictions(
        frame,
        donor,
        predictions,
        person_outputs=tuple(raw),
        tax_unit_outputs=(),
        tail_bound_diagnostics=diagnostics,
        **kwargs,
    )


def _heavy_tail_donor(column: str, *, losses: bool = True) -> pd.DataFrame:
    # 1,000 donor units: most small, nine large, one extreme record of weight
    # 0.13 (the Forbes-tail synthetic records' weight), and a loss tail.
    values = [100.0, 5_000.0, 2.0e8, -40_000.0 if losses else 0.0, 0.0]
    weights = [900.0, 9.0, 0.13, 10.0, 80.87]
    return pd.DataFrame({column: values, "weight": weights})


def test_one_extreme_draw_is_bounded_by_its_slice_and_recorded() -> None:
    # Recipient shares 0.01, 0.49, 0.01 and 0.49. The light recipients' slices
    # hold weight 10: the top one reaches the extreme record, the bottom one is
    # exactly the loss records, so neither of their draws moves.
    frame = _recipient_frame([20.0, 980.0, 20.0, 980.0])
    donor = _heavy_tail_donor(_PASSTHROUGH)
    raw = np.asarray([150.0, 2.0e8, -40_000.0, 4_000.0])
    diagnostics: list[dict[str, object]] = []

    finalized = _finalize(frame, donor, {_PASSTHROUGH: raw}, diagnostics=diagnostics)

    weights = _recipient_weights(frame)
    floor, cap = _bounds(donor[_PASSTHROUGH], donor["weight"], weights)
    expected = np.minimum(np.maximum(raw, floor), cap)
    actual = _puf_values(finalized, _PASSTHROUGH)
    np.testing.assert_array_equal(actual, expected)
    # Only the extreme draw moved; its weighted mass fell to its slice's mass.
    assert actual[1] < raw[1]
    np.testing.assert_array_equal(actual[[0, 2, 3]], raw[[0, 2, 3]])
    assert diagnostics == [
        {
            "output": _PASSTHROUGH,
            "rule": puf_support_module.PUF_TAX_DETAIL_RECIPIENT_SHARE_BOUND_RULE,
            "clipped_row_count": 1,
            "capped_row_count": 1,
            "floored_row_count": 0,
            "clipped_mass_before": float(raw[1] * weights[1]),
            "clipped_mass_after": float(expected[1] * weights[1]),
            "largest_weighted_draw_before": float(np.abs(raw * weights).max()),
            "largest_weighted_draw_after": float(np.abs(expected * weights).max()),
        }
    ]


_FINALIZE_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@st.composite
def _finalize_case(draw: st.DrawFn) -> tuple[list[float], pd.DataFrame, np.ndarray]:
    recipients = draw(st.integers(min_value=1, max_value=8))
    household_weights = draw(
        st.lists(
            st.floats(min_value=0.5, max_value=5e3),
            min_size=recipients,
            max_size=recipients,
        )
    )
    donors = draw(st.integers(min_value=1, max_value=12))
    donor = pd.DataFrame(
        {
            _PASSTHROUGH: draw(st.lists(_amounts, min_size=donors, max_size=donors)),
            "weight": draw(
                st.lists(_positive_weights, min_size=donors, max_size=donors)
            ),
        }
    )
    raw = np.asarray(draw(st.lists(_amounts, min_size=recipients, max_size=recipients)))
    return household_weights, donor, raw


@_FINALIZE_SETTINGS
@given(_finalize_case())
def test_finalized_draws_obey_the_bound_and_move_only_beyond_it(
    case: tuple[list[float], pd.DataFrame, np.ndarray],
) -> None:
    household_weights, donor, raw = case
    frame = _recipient_frame(household_weights)

    finalized = _finalize(frame, donor, {_PASSTHROUGH: raw}, diagnostics=[])

    floor, cap = _bounds(
        donor[_PASSTHROUGH], donor["weight"], _recipient_weights(frame)
    )
    actual = _puf_values(finalized, _PASSTHROUGH)
    # Bound.
    assert (actual <= cap).all() and (actual >= floor).all()
    # Untouched inside the bound, bit for bit.
    inside = (raw <= cap) & (raw >= floor)
    np.testing.assert_array_equal(
        actual[inside].view(np.uint64), raw[inside].view(np.uint64)
    )
    # Sign and zero preserving, shrink only.
    np.testing.assert_array_equal(np.sign(actual), np.sign(raw))
    assert (np.abs(actual) <= np.abs(raw)).all()
    # Moved draws land exactly on their own bound.
    np.testing.assert_array_equal(actual[raw > cap], cap[raw > cap])
    np.testing.assert_array_equal(actual[raw < floor], floor[raw < floor])


@_FINALIZE_SETTINGS
@given(_finalize_case())
def test_diagnostics_account_exactly_for_the_moved_mass(
    case: tuple[list[float], pd.DataFrame, np.ndarray],
) -> None:
    household_weights, donor, raw = case
    frame = _recipient_frame(household_weights)
    diagnostics: list[dict[str, object]] = []

    finalized = _finalize(frame, donor, {_PASSTHROUGH: raw}, diagnostics=diagnostics)

    weights = _recipient_weights(frame)
    actual = _puf_values(finalized, _PASSTHROUGH)
    (record,) = diagnostics
    moved = actual != raw
    assert record["clipped_row_count"] == int(moved.sum())
    assert record["capped_row_count"] + record["floored_row_count"] == int(moved.sum())
    assert record["clipped_mass_before"] == pytest.approx(
        float((raw[moved] * weights[moved]).sum()), rel=1e-12, abs=1e-6
    )
    assert record["clipped_mass_after"] == pytest.approx(
        float((actual[moved] * weights[moved]).sum()), rel=1e-12, abs=1e-6
    )
    total_before = float((raw * weights).sum())
    total_after = float((actual * weights).sum())
    assert total_before - total_after == pytest.approx(
        record["clipped_mass_before"] - record["clipped_mass_after"],
        rel=1e-9,
        abs=1e-3,
    )


def test_snapped_outputs_stay_on_observed_values_within_the_bound() -> None:
    # A donor positive rate above the recipients' keeps the donor-rate
    # sparsification idle, so the finalized values are the snapped bounded draws.
    frame = _recipient_frame([20.0, 400.0, 600.0, 980.0])
    donor = pd.DataFrame(
        {_SNAPPED: [100.0, 5_000.0, 2.0e8, 0.0], "weight": [990.0, 9.0, 0.13, 0.87]}
    )
    raw = np.asarray([0.0, 1.9e8, 7_000.4, 4_000.0])

    finalized = _finalize(frame, donor, {_SNAPPED: raw}, diagnostics=[])

    _, cap = _bounds(donor[_SNAPPED], donor["weight"], _recipient_weights(frame))
    actual = _puf_values(finalized, _SNAPPED)
    grid = puf_support_module._observed_snap_grid(donor[_SNAPPED])
    assert (actual <= cap).all()
    assert np.isin(actual, grid).all()
    assert actual[1] < raw[1]
    # The cap of the heaviest-share draw is not itself a donor value; the draw
    # lands on the largest donor value within it.
    assert actual[1] == grid[grid <= cap[1]].max()


def test_sparsification_rescales_bounded_draws_by_one_common_factor() -> None:
    # Every recipient draws positive while the donor is 90.9% positive, so the
    # donor-rate sparsification zeroes the smallest draw and rescales the rest
    # to keep the column's mass: the bound holds before that, and after it the
    # kept draws all carry the same factor.
    frame = _recipient_frame([20.0, 400.0, 600.0, 980.0])
    donor = _heavy_tail_donor(_SNAPPED, losses=False)
    raw = np.asarray([150.0, 1.9e8, 7_000.0, 4_000.0])

    finalized = _finalize(frame, donor, {_SNAPPED: raw}, diagnostics=[])

    weights = _recipient_weights(frame)
    floor, cap = _bounds(donor[_SNAPPED], donor["weight"], weights)
    grid = puf_support_module._observed_snap_grid(donor[_SNAPPED])
    lowered = puf_support_module._largest_grid_values_within(cap, grid)
    before = puf_support_module._snap_to_observed_values(
        np.minimum(np.maximum(raw, floor), lowered), donor[_SNAPPED]
    )
    actual = _puf_values(finalized, _SNAPPED)
    assert (before <= cap).all()
    kept = actual > 0.0
    assert kept.any() and not kept.all()
    factors = actual[kept] / before[kept]
    np.testing.assert_allclose(factors, factors[0], rtol=1e-12)
    assert float((actual * weights).sum()) == pytest.approx(
        float((before * weights).sum()), rel=1e-12
    )


def test_signed_mass_calibration_still_pins_both_legs_after_the_bound() -> None:
    frame = _recipient_frame([20.0, 400.0, 600.0, 980.0, 300.0])
    donor = _heavy_tail_donor(_CALIBRATED)
    raw = np.asarray([150.0, 2.0e8, -40_000.0, 4_000.0, -100.0])

    finalized = _finalize(frame, donor, {_CALIBRATED: raw}, diagnostics=[])

    weights = _recipient_weights(frame)
    actual = _puf_values(finalized, _CALIBRATED)
    values = donor[_CALIBRATED].to_numpy()
    donor_weights = donor["weight"].to_numpy()
    for leg, donor_leg in ((actual > 0.0, values > 0.0), (actual < 0.0, values < 0.0)):
        assert float((actual[leg] * weights[leg]).sum()) / weights.sum() == (
            pytest.approx(
                float((values[donor_leg] * donor_weights[donor_leg]).sum())
                / donor_weights.sum(),
                rel=1e-9,
            )
        )
    # The calibration rescales whole legs, so the bound's effect survives it:
    # the extreme draw no longer holds nearly all of the positive leg.
    positive = actual > 0.0
    leg_mass = actual[positive] * weights[positive]
    assert leg_mass.max() < 0.99 * leg_mass.sum()


def test_boolean_outputs_are_not_bounded() -> None:
    frame = _recipient_frame([20.0, 400.0])
    donor = pd.DataFrame({_BOOLEAN: [0.0, 1.0], "weight": [999.0, 1.0]})
    diagnostics: list[dict[str, object]] = []

    finalized = _finalize(frame, donor, {_BOOLEAN: [1.0, 1.0]}, diagnostics=diagnostics)

    assert diagnostics == []
    assert _puf_values(finalized, _BOOLEAN).tolist() == [1.0, 1.0]


def test_amount_outputs_require_a_diagnostics_sink() -> None:
    frame = _recipient_frame([20.0, 400.0])
    with pytest.raises(ValueError, match="diagnostics sink"):
        _finalize(
            frame,
            _heavy_tail_donor(_PASSTHROUGH),
            {_PASSTHROUGH: [1.0, 2.0]},
            diagnostics=None,
        )


def test_explicit_narrowing_is_validated_and_can_empty_the_bound() -> None:
    frame = _recipient_frame([20.0, 400.0])
    donor = _heavy_tail_donor(_PASSTHROUGH)
    raw = np.asarray([150.0, 2.0e8])

    unbounded = _finalize(
        frame,
        donor,
        {_PASSTHROUGH: raw},
        diagnostics=None,
        recipient_share_bound_outputs=(),
    )
    np.testing.assert_array_equal(_puf_values(unbounded, _PASSTHROUGH), raw)

    for bad in ((_BOOLEAN,), ("not_an_output",), (_PASSTHROUGH, _PASSTHROUGH)):
        with pytest.raises(ValueError, match="distinct amount outputs"):
            _finalize(
                frame,
                donor,
                {_PASSTHROUGH: raw},
                diagnostics=[],
                recipient_share_bound_outputs=bad,
            )


def test_bound_identity_is_exact_and_names_its_rule() -> None:
    identity = puf_support_module.puf_tax_detail_recipient_share_bound_identity()
    assert identity["rule"] == (
        puf_support_module.PUF_TAX_DETAIL_RECIPIENT_SHARE_BOUND_RULE
    )
    assert (
        identity == puf_support_module.puf_tax_detail_recipient_share_bound_identity()
    )
    canonical = (
        *puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
        *puf_support_module.PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    )
    bounded = puf_support_module._recipient_share_bound_amount_outputs(canonical)
    assert "partnership_income" in bounded
    assert "first_home_mortgage_origination_year" not in bounded
    assert not set(bounded) & set(puf_support_module.US_QBI_BOOLEAN_OUTPUT_COLUMNS)
