"""The PUF recipient-share bound: one draw cannot outweigh the slice it stands for.

A PUF-clone recipient whose weight is the share ``s`` of the recipient
population stands for ``s`` of the population. The bound limits its raw draw
to the same-sign mass, per unit of slice weight, of the donor slice holding the
same share at that end of the distribution, so
``w_r * value <= (W_r / W_d) * mass(slice)``.
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
# Zero weights occur on both sides: the donor fill is fillna(0.0), and a
# recipient household can carry no weight.
_weights = st.one_of(st.just(0.0), _positive_weights)


def _with_positive_mass(weights: list[float]) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float64)
    if not (array > 0.0).any():
        array[0] = 1.0
    return array


@st.composite
def _donor_and_recipients(draw: st.DrawFn) -> tuple[np.ndarray, ...]:
    donors = draw(st.integers(min_value=1, max_value=40))
    values = np.asarray(draw(st.lists(_amounts, min_size=donors, max_size=donors)))
    weights = _with_positive_mass(
        draw(st.lists(_weights, min_size=donors, max_size=donors))
    )
    recipients = draw(st.integers(min_value=1, max_value=25))
    recipient_weights = _with_positive_mass(
        draw(st.lists(_weights, min_size=recipients, max_size=recipients))
    )
    return values, weights, recipient_weights


def _reference_slice_mean(
    values: np.ndarray, weights: np.ndarray, take: float, *, top: bool
) -> float:
    """One slice's same-sign mass per unit weight, donor by donor."""

    weighted = [i for i in range(len(values)) if weights[i] > 0.0]
    order = sorted(weighted, key=lambda i: -values[i] if top else values[i])
    if take <= 0.0:
        # An empty slice reads the most extreme weighted value on its side.
        extreme = values[order[0]]
        return max(extreme, 0.0) if top else min(extreme, 0.0)
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

    weighted = weights > 0.0
    values, weights = values[weighted], weights[weighted]
    order = np.argsort(-values, kind="stable")
    cumulative = np.cumsum(weights[order])
    edge = min(int(np.searchsorted(cumulative, take, side="left")), len(values) - 1)
    return float(values[order][edge])


def test_bound_is_the_slice_mean_on_a_worked_example() -> None:
    # Donor population of weight 1,000: 980 at 10, 9 at 1,000, 1 at 100,000
    # and 10 at -50.
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


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_donor_and_recipients())
def test_cap_mass_is_the_most_any_donor_subpopulation_of_that_size_holds(
    case: tuple[np.ndarray, ...],
) -> None:
    # Independent of the slice construction: a linear program picks any
    # fractional subpopulation of donor weight `take` to maximize its positive
    # mass (or minimize its negative mass). The problem is scaled to unit
    # weight and unit value (y_d = x_d / take, values / max |value|), so the
    # solver's tolerances apply to an objective in [0, 1]. The bound per unit
    # of slice weight must equal the optimum.
    from scipy.optimize import linprog

    values, weights, recipient_weights = case
    donor_total = weights.sum()
    recipient_total = recipient_weights.sum()
    scale = max(float(np.abs(values).max()), 1.0)

    floor, cap = _bounds(values, weights, recipient_weights)

    for index, weight in enumerate(recipient_weights):
        take = weight / recipient_total * donor_total
        if take <= 0.0:
            continue
        for side, bound in ((1.0, cap[index]), (-1.0, floor[index])):
            part = np.maximum(side * values, 0.0) / scale
            if not (part[weights > 0.0] > 0.0).any():
                assert np.isinf(bound)
                continue
            result = linprog(
                -part,
                A_eq=np.ones((1, len(values))),
                b_eq=[1.0],
                # The shares sum to one, so no share exceeds one; capping the
                # bound there keeps the problem well conditioned.
                bounds=[(0.0, min(1.0, w / take)) for w in weights],
                method="highs",
            )
            assert result.success
            assert side * bound / scale == pytest.approx(-result.fun, abs=1e-6)


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
        weighted_values = values[weights > 0.0]
        if np.isfinite(cap[index]):
            assert 0.0 < cap[index] <= weighted_values.max()
            # The mean of the slice is at least its inner edge, exactly.
            edge = _top_slice_edge(values, weights, take)
            assert cap[index] >= edge
        if np.isfinite(floor[index]):
            assert weighted_values.min() <= floor[index] < 0.0
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
    weighted_values = values[weights > 0.0]
    assert np.isfinite(cap).all() == bool((weighted_values > 0.0).any())
    assert np.isinf(cap).all() == (not (weighted_values > 0.0).any())
    assert np.isfinite(floor).all() == bool((weighted_values < 0.0).any())
    assert np.isinf(floor).all() == (not (weighted_values < 0.0).any())


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
    positive_grid = grid[grid > 0.0]
    caps = np.full(len(draws), cap)
    lowered = _grid_within(caps, grid)
    raw = np.asarray(draws, dtype=np.float64)
    bounded = np.minimum(raw, lowered)

    snapped = _snap(bounded, observed)
    unbounded = _snap(raw, observed)

    if len(positive_grid) == 0:
        # Nothing positive to snap onto: the cap is left alone and every draw
        # snaps to zero with or without it.
        np.testing.assert_array_equal(lowered, caps)
        assert (snapped == 0.0).all()
        return
    # The lowered cap is a positive grid value: the largest within the cap, or
    # the smallest positive one when the cap is below them all.
    assert np.isin(lowered, positive_grid).all()
    assert (lowered <= np.maximum(cap, positive_grid[0])).all()
    assert (snapped <= lowered).all()
    # The bound never turns a draw that would have snapped positive into zero.
    assert (snapped[unbounded > 0.0] > 0.0).all()


def test_lowering_keeps_infinite_caps_infinite_and_stays_positive() -> None:
    grid = np.asarray([0.0, 5.0, 10.0])
    lowered = _grid_within(np.asarray([np.inf, 7.0, 10.0, 3.0]), grid)
    np.testing.assert_array_equal(lowered, [np.inf, 5.0, 10.0, 5.0])


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
    tax_unit_raw: dict[str, np.ndarray] | None = None,
    **kwargs: object,
) -> Frame:
    tax_unit_raw = tax_unit_raw or {}
    predictions = pd.DataFrame(
        {
            column: np.asarray(values, dtype=np.float64)
            for column, values in {**raw, **tax_unit_raw}.items()
        },
        index=_recipient_index(frame),
    )
    return puf_support_module.finalize_us_puf_tax_detail_predictions(
        frame,
        donor,
        predictions,
        person_outputs=tuple(raw),
        tax_unit_outputs=tuple(tax_unit_raw),
        tail_bound_diagnostics=diagnostics,
        **kwargs,
    )


def _puf_tax_unit_values(frame: Frame, column: str) -> np.ndarray:
    tax_unit = frame.table("tax_unit")
    puf = tax_unit[
        tax_unit[support_channel_column("tax_unit")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ].sort_values(support_source_id_column("tax_unit"))
    return puf[column].to_numpy(dtype=np.float64)


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
            "recipient_count": 4,
            "recipient_weight_total": float(weights.sum()),
            "donor_weight_total": float(donor["weight"].sum()),
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

    grid = puf_support_module._observed_snap_grid(donor[_SNAPPED])
    _, cap = _bounds(
        puf_support_module._snap_grid_values(donor[_SNAPPED]),
        donor["weight"],
        _recipient_weights(frame),
    )
    actual = _puf_values(finalized, _SNAPPED)
    assert (actual <= cap).all()
    assert np.isin(actual, grid).all()
    assert actual[1] < raw[1]
    # The cap of the heaviest-share draw is not itself a donor value; the draw
    # lands on the largest donor value within it.
    assert actual[1] == grid[grid <= cap[1]].max()


def test_the_bound_never_zeroes_a_carrier_of_a_snapped_output() -> None:
    # Half the donor holds 10 and half holds nothing, so a recipient standing
    # for two thirds of the population has a slice mass of 7.5 per unit, below
    # the only positive donor value. Lowering that cap to the grid value 0
    # would erase the carrier; the cap stays at the smallest positive value and
    # the carrier count is left to the donor-rate sparsification.
    frame = _recipient_frame([20.0, 40.0])
    donor = pd.DataFrame({_SNAPPED: [0.0, 10.0], "weight": [1.0, 1.0]})
    raw = np.asarray([0.25, 9.6])
    diagnostics: list[dict[str, object]] = []

    finalized = _finalize(frame, donor, {_SNAPPED: raw}, diagnostics=diagnostics)

    _, cap = _bounds(
        puf_support_module._snap_grid_values(donor[_SNAPPED]),
        donor["weight"],
        _recipient_weights(frame),
    )
    assert cap[1] < 10.0
    np.testing.assert_array_equal(_puf_values(finalized, _SNAPPED), [0.0, 10.0])
    assert diagnostics[0]["clipped_row_count"] == 0


def test_draws_at_a_tied_maximum_on_the_snap_grid_are_not_clipped() -> None:
    # Statutory caps aged by a factor leave the donor's maximum off the integer
    # grid (4,504.8 here) yet tied across many returns. The bound is computed
    # on the grid the column snaps to, so draws at that maximum snap to it
    # unmoved instead of being clipped to the grid value below.
    frame = _recipient_frame([20.0, 400.0, 600.0, 980.0])
    donor = pd.DataFrame(
        {_SNAPPED: [4_504.8, 1_000.4, 0.0], "weight": [500.0, 400.0, 100.0]}
    )
    raw = np.asarray([4_504.8, 4_504.8, 4_504.8, 0.0])
    diagnostics: list[dict[str, object]] = []

    finalized = _finalize(frame, donor, {_SNAPPED: raw}, diagnostics=diagnostics)

    assert diagnostics[0]["clipped_row_count"] == 0
    np.testing.assert_array_equal(
        _puf_values(finalized, _SNAPPED), [4_505.0, 4_505.0, 4_505.0, 0.0]
    )


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
    floor, cap = _bounds(
        puf_support_module._snap_grid_values(donor[_SNAPPED]), donor["weight"], weights
    )
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


def test_signed_mass_calibration_rescales_each_bounded_leg_by_one_factor() -> None:
    frame = _recipient_frame([20.0, 400.0, 600.0, 980.0, 300.0])
    donor = _heavy_tail_donor(_CALIBRATED)
    raw = np.asarray([150.0, 2.0e8, -40_000.0, 4_000.0, -100.0])

    finalized = _finalize(frame, donor, {_CALIBRATED: raw}, diagnostics=[])

    weights = _recipient_weights(frame)
    floor, cap = _bounds(donor[_CALIBRATED], donor["weight"], weights)
    bounded = np.minimum(np.maximum(raw, floor), cap)
    assert bounded[1] < raw[1]
    actual = _puf_values(finalized, _CALIBRATED)
    values = donor[_CALIBRATED].to_numpy()
    donor_weights = donor["weight"].to_numpy()
    for leg, donor_leg in (
        (bounded > 0.0, values > 0.0),
        (bounded < 0.0, values < 0.0),
    ):
        # Each leg is the bounded draws times one factor ...
        factors = actual[leg] / bounded[leg]
        np.testing.assert_allclose(factors, factors[0], rtol=1e-12)
        # ... chosen so the leg's mass per unit weight is the donor's.
        assert float((actual[leg] * weights[leg]).sum()) / weights.sum() == (
            pytest.approx(
                float((values[donor_leg] * donor_weights[donor_leg]).sum())
                / donor_weights.sum(),
                rel=1e-9,
            )
        )


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


def test_bound_identity_is_pinned() -> None:
    assert puf_support_module.puf_tax_detail_recipient_share_bound_identity() == {
        "rule": "recipient_share_slice_mass",
        "schema_version": 1,
        "scope": "every_amount_output_of_the_invocation",
        "excluded_output_classes": [
            "boolean_person_outputs",
            "discrete_tax_unit_outputs",
        ],
        "recipient_share": "household_design_weight_over_puf_recipient_total",
        "cap": "top_slice_positive_mass_per_slice_weight",
        "floor": "bottom_slice_negative_mass_per_slice_weight",
        "snapped_outputs": "bounded_on_snap_grid_values_cap_lowered_onto_positive_grid",
        "seam": "raw_tax_unit_draws_after_tail_bound_quantiles",
    }


def test_every_canonical_amount_output_is_bounded_and_nothing_else() -> None:
    canonical = (
        *puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
        *puf_support_module.PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    )
    bounded = puf_support_module._recipient_share_bound_amount_outputs(canonical)
    excluded = set(canonical) - set(bounded)
    assert excluded == set(puf_support_module.US_QBI_BOOLEAN_OUTPUT_COLUMNS) | {
        "first_home_mortgage_origination_year",
        "second_home_mortgage_origination_year",
    }
    assert "partnership_income" in bounded
    assert "first_home_mortgage_balance" in bounded
    assert list(bounded) == [output for output in canonical if output in bounded]


def test_realistic_weight_ratios_cap_a_forbes_tail_draw() -> None:
    # The production geometry: ~10^5 donors including 3,900 synthetic records
    # of weight 0.13 with Pareto values, and recipients of weight ~600 among
    # ~9e7 of recipient weight. Vectorized bounds match the donor-by-donor
    # reference, and a 218.9M draw on a weight-644.7 unit is capped at the
    # mass of the slice it stands for.
    rng = np.random.default_rng(1259)
    regular = 100_000
    values = np.concatenate(
        [
            np.where(
                rng.random(regular) < 0.06, rng.lognormal(10.0, 2.0, regular), 0.0
            ),
            (rng.pareto(1.2, 3_900) + 1.0) * 2.0e6,
        ]
    )
    values[-1] = 2.189e8
    weights = np.concatenate(
        [rng.uniform(200.0, 2_400.0, regular), np.full(3_900, 0.13)]
    )
    recipient_weights = np.concatenate(
        [rng.uniform(300.0, 1_000.0, 4_000), [644.7], [8.97e7]]
    )

    floor, cap = _bounds(values, weights, recipient_weights)

    probe = np.r_[np.arange(0, 4_000, 400), 4_000, 4_001]
    reference_cap = np.asarray(
        [
            _reference_slice_mean(
                values,
                weights,
                recipient_weights[i] / recipient_weights.sum() * weights.sum(),
                top=True,
            )
            for i in probe
        ]
    )
    np.testing.assert_allclose(cap[probe], reference_cap, rtol=1e-9)
    assert np.isinf(floor).all()
    heavy = 4_000
    assert cap[heavy] < 2.189e8


def test_non_finite_bounded_draws_are_refused_before_any_write() -> None:
    # non_sch_d_capital_gains has the q0.999 quantile cap, whose write runs
    # before the recipient-share bound; a NaN in another bounded output must
    # stop finalization before that write or any other.
    frame = _recipient_frame([20.0, 400.0])
    donor = pd.DataFrame(
        {
            "non_sch_d_capital_gains": [100.0, 200.0, 300.0],
            _PASSTHROUGH: [100.0, 5_000.0, -10.0],
            "weight": [999.0, 0.5, 0.5],
        }
    )
    predictions = pd.DataFrame(
        {
            "non_sch_d_capital_gains": [50.0, 400.0],
            _PASSTHROUGH: [np.nan, 1.0],
        },
        index=_recipient_index(frame),
    )
    before = predictions.copy(deep=True)
    diagnostics: list[dict[str, object]] = []

    with pytest.raises(ValueError, match="must be finite"):
        puf_support_module.finalize_us_puf_tax_detail_predictions(
            frame,
            donor,
            predictions,
            person_outputs=("non_sch_d_capital_gains", _PASSTHROUGH),
            tax_unit_outputs=(),
            tail_bound_diagnostics=diagnostics,
        )

    pd.testing.assert_frame_equal(predictions, before, check_exact=True)
    assert diagnostics == []


def test_tax_unit_amount_outputs_are_bounded_on_the_tax_unit_table() -> None:
    # A passthrough tax-unit amount (mortgage balance) and a snapped,
    # sparsified one (domestic production ALD) through the tax-unit write path.
    frame = _recipient_frame([20.0, 400.0, 600.0, 980.0])
    balance = "first_home_mortgage_balance"
    dpad = "domestic_production_ald"
    donor = pd.DataFrame(
        {
            balance: [200_000.0, 900_000.0, 5.0e7, 0.0],
            dpad: [0.0, 1_000.0, 3.0e6, 0.0],
            "weight": [900.0, 9.0, 0.13, 90.87],
        }
    )
    raw_balance = np.asarray([150_000.0, 5.0e7, 0.0, 300_000.0])
    raw_dpad = np.asarray([0.0, 3.0e6, 0.0, 0.0])
    diagnostics: list[dict[str, object]] = []

    finalized = _finalize(
        frame,
        donor,
        {},
        tax_unit_raw={balance: raw_balance, dpad: raw_dpad},
        diagnostics=diagnostics,
    )

    weights = _recipient_weights(frame)
    _, balance_cap = _bounds(donor[balance], donor["weight"], weights)
    actual_balance = _puf_tax_unit_values(finalized, balance)
    np.testing.assert_array_equal(actual_balance, np.minimum(raw_balance, balance_cap))
    assert actual_balance[1] < raw_balance[1]
    _, dpad_cap = _bounds(
        puf_support_module._snap_grid_values(donor[dpad]), donor["weight"], weights
    )
    actual_dpad = _puf_tax_unit_values(finalized, dpad)
    assert actual_dpad[1] < raw_dpad[1]
    assert actual_dpad[1] <= dpad_cap[1]
    assert [record["output"] for record in diagnostics] == [balance, dpad]


def test_social_security_reconciliation_keeps_the_bounded_total() -> None:
    # The four components are re-split by CPS shares after the bound, so a
    # component can move, but each unit's total is the bounded draws' total.
    frame = _recipient_frame([20.0, 980.0, 300.0])
    components = puf_support_module.PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS
    donor = pd.DataFrame(
        {
            "social_security_retirement": [20_000.0, 60_000.0, 4.0e6, 0.0],
            "social_security_disability": [0.0, 15_000.0, 0.0, 0.0],
            "social_security_dependents": [0.0, 0.0, 0.0, 0.0],
            "social_security_survivors": [0.0, 0.0, 0.0, 0.0],
            "weight": [600.0, 300.0, 0.13, 99.87],
        }
    )
    raw = {
        "social_security_retirement": np.asarray([20_000.0, 4.0e6, 60_000.0]),
        "social_security_disability": np.asarray([0.0, 15_000.0, 0.0]),
        "social_security_dependents": np.zeros(3),
        "social_security_survivors": np.zeros(3),
    }

    finalized = _finalize(frame, donor, raw, diagnostics=[])

    weights = _recipient_weights(frame)
    bounded_total = np.zeros(3)
    for component in components:
        floor, cap = _bounds(donor[component], donor["weight"], weights)
        bounded_total += np.minimum(np.maximum(raw[component], floor), cap)
    finalized_total = sum(_puf_values(finalized, component) for component in components)
    np.testing.assert_allclose(finalized_total, bounded_total, rtol=1e-12)
    assert finalized_total[1] < 4.0e6 + 15_000.0
