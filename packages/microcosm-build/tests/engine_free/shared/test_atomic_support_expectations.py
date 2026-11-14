"""The exact expectation of an area code's value under each household's
conditioned draw cell. Every support, household and value below is invented."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build import atomic_geography as geo
from microcosm.build.atomic_support_expectations import draw_cell_expectations

#: Each household input to the support code column it must match.
CONSTRAINTS = {"region": "region", "draw_decile": "decile"}
RULE = {"input": "band", "map": {"x": "households_x", "y": "households_y"}}
#: Two codes that share one weight column.
SHARED_RULE = {"input": "band", "map": {"x": "households_y", "y": "households_y"}}
#: Invented values for the authority codes, exact and not dyadic, so a float
#: anywhere in the sums shows.
VALUES = {"p": 3, "q": Fraction(7, 3), "r": 5, "s": Fraction(1, 2)}


def _u(*values: str) -> np.ndarray:
    return np.asarray(values, dtype="U")


def _support(region, decile, authority, **weights) -> geo.AtomicSupport:
    """Invented areas, each with a region, a decile, an authority code and the
    given weights."""
    code = {"kind": "code", "source": "invented", "vintage": "v1", "relation": "exact"}
    return geo.decode_atomic_support(
        geo.encode_atomic_support(
            {
                "version": 1,
                "system": "toy",
                "level": "area",
                "code_system": "toy",
                "vintage": "v1",
                "columns": {
                    "area": code,
                    "region": code,
                    "decile": code,
                    "authority": code,
                    **{
                        name: {"kind": "weight", "source": "invented", "basis": name}
                        for name in weights
                    },
                },
            },
            {
                "area": _u(*(f"a{index}" for index in range(len(region)))),
                "region": _u(*region),
                "decile": _u(*decile),
                "authority": _u(*authority),
                **{
                    name: np.asarray(values, dtype=np.int64)
                    for name, values in weights.items()
                },
            },
        )
    )


#: Five areas: a0 and a1 (north, decile 1), a2 (north, 2), a3 (south, 1) and
#: a4 (south, 2), the last with no households.
SUPPORT = _support(
    region=("north", "north", "north", "south", "south"),
    decile=("1", "1", "2", "1", "2"),
    authority=("p", "q", "s", "r", "r"),
    households=(2, 1, 3, 4, 0),
    households_x=(0, 5, 1, 0, 2),
    households_y=(1, 0, 0, 0, 3),
)


def _households(*rows) -> pd.DataFrame:
    """Households as (region, decile, band); None is a missing input."""
    return pd.DataFrame(
        list(rows), columns=["region", "draw_decile", "band"], dtype=object
    )


def _expect(households, support=SUPPORT, **overrides) -> list[Fraction]:
    arguments = {
        "constraint_columns": CONSTRAINTS,
        "weight_by": RULE,
        "default_weight": "households",
        "value_column": "authority",
        "values": VALUES,
        **overrides,
    }
    return draw_cell_expectations(support, households, **arguments)


def test_each_household_takes_the_weighted_mean_over_its_own_cell():
    """Worked by hand from SUPPORT and VALUES. Each line differs from the
    region's mean, the cell's unweighted mean or the mean by another weight
    column, and every result is an exact Fraction."""
    rows = _households(
        ("north", "1", "x"),  # a0, a1 by households_x 0, 5: q
        ("north", "1", None),  # a0, a1 by households 2, 1: (2 x 3 + 7/3) / 3
        ("north", None, "x"),  # a0 to a2 by households_x 0, 5, 1: (5 x 7/3 + 1/2) / 6
        ("north", None, None),  # a0 to a2 by households 2, 1, 3
        ("south", "1", None),  # a3 alone: r
        ("south", None, "y"),  # a3, a4 by households_y 0, 3: r
    )
    expected = [
        Fraction(7, 3),
        Fraction(25, 9),
        Fraction(73, 36),
        Fraction(59, 36),  # (2 x 3 + 7/3 + 3 x 1/2) / 6
        Fraction(5),
        Fraction(5),
    ]
    result = _expect(rows)
    assert result == expected
    assert all(type(value) is Fraction for value in result)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA], ids=["None", "nan", "NA"])
@pytest.mark.parametrize(
    "dtype",
    [object, "string", pd.StringDtype(storage="python"), "str", "category"],
    ids=["object", "string", "string_python", "str", "category"],
)
def test_a_missing_input_is_skipped_and_samples_by_the_default_weight(missing, dtype):
    """A missing decile widens the cell to the region, and a missing band
    samples by the default weight, however the frame holds the missing value:
    None, NaN or pd.NA, in object, StringDtype (either storage), pandas' str
    or category columns."""
    rows = _households(
        ("north", missing, "x"),
        ("north", "1", missing),
        ("north", missing, missing),
    ).astype(dtype)
    assert _expect(rows) == [Fraction(73, 36), Fraction(25, 9), Fraction(59, 36)]


@pytest.mark.parametrize("spelled", ["nan", "None", "<NA>"])
def test_a_missing_input_never_matches_a_code_spelled_like_missing(spelled):
    support = _support(
        region=("north", "north"),
        decile=(spelled, "2"),
        authority=("p", "r"),
        households=(1, 3),
    )
    rows = _households(
        ("north", None, None), ("north", np.nan, None), ("north", pd.NA, None)
    )
    # The region's two areas, by households 1 and 3: (3 + 3 x 5) / 4.
    assert _expect(rows, support, weight_by=None) == [Fraction(9, 2)] * 3


def test_inputs_follow_constraint_columns_not_their_own_names():
    # The input "region" matches the support's decile column and "draw_decile"
    # its region column: (decile 2, north) is a2 alone.
    swapped = {"region": "decile", "draw_decile": "region"}
    rows = _households(("2", "north", None))
    assert _expect(rows, weight_by=None, constraint_columns=swapped) == [Fraction(1, 2)]


def test_the_default_weight_is_the_declared_column():
    # By households_y, the default here, the north's areas weigh 1, 0, 0: p.
    rows = _households(("north", None, None))
    assert _expect(rows, weight_by=None, default_weight="households_y") == [3]


def test_two_codes_sharing_a_weight_column_both_read_it():
    # households_y gives a3 and a4 in the south 0 and 3: r for both codes.
    rows = _households(("south", None, "x"), ("south", None, "y"))
    assert _expect(rows, weight_by=SHARED_RULE) == [Fraction(5)] * 2


def test_a_code_without_mass_in_the_cell_is_not_read():
    """Only a code that carries mass in the household's weight column enters
    its expectation, so it alone needs a value."""
    rows = _households(("north", "1", "x"))  # a0 (p) has no households_x
    without_p = {code: value for code, value in VALUES.items() if code != "p"}
    assert _expect(rows, values=without_p) == [Fraction(7, 3)]
    with pytest.raises(ValueError, match="has no value"):
        _expect(_households(("north", "1", None)), values=without_p)


def test_integer_values_give_exact_fractions():
    values = {"p": 2, "q": 1, "r": 4, "s": 1}
    # (2 x 2 + 1 x 1) / 3 by households over a0 and a1.
    result = _expect(_households(("north", "1", None)), values=values)
    assert result == [Fraction(5, 3)]
    assert type(result[0]) is Fraction


def test_the_expectation_stays_exact_at_the_decoders_weight_range():
    """The north's two areas hold the decoder's whole exact sampling range,
    2**53, so their mean is (2**52 + 1) x 1/3 + (2**52 - 1) x 2/7 over 2**53:
    (13 x 2**52 + 1) over 21 x 2**53, in lowest terms. No float equals it,
    since its denominator is not a power of two, and neither does a fraction
    recovered from a float with a smaller denominator."""
    support = _support(
        region=("north", "north", "south"),
        decile=("1", "1", "1"),
        authority=("p", "q", "r"),
        households=(2**52 + 1, 2**52 - 1, 0),
    )
    values = {"p": Fraction(1, 3), "q": Fraction(2, 7), "r": 5}
    rows = _households(("north", None, None))
    assert _expect(rows, support, weight_by=None, values=values) == [
        Fraction(13 * 2**52 + 1, 21 * 2**53)
    ]


def test_without_households_the_expectation_is_empty():
    assert _expect(_households()) == []


def _brute_force(support, household, *, rule, values, default="households"):
    """One household's expectation by a loop over the support's areas, or
    None where the draw would refuse the household (a cell without a support
    row or without mass in its weight, or a code the rule does not map) or a
    code with mass in its cell has no value."""
    area_count = len(support.arrays["area"])
    cell = [
        area
        for area in range(area_count)
        if all(
            pd.isna(household[column])
            or support.arrays[matched][area] == household[column]
            for column, matched in CONSTRAINTS.items()
        )
    ]
    band = household[rule["input"]] if rule is not None else None
    if rule is not None and not pd.isna(band):
        if band not in rule["map"]:
            return None
        weight = rule["map"][band]
    else:
        weight = default
    total, numerator = 0, Fraction(0)
    for area in cell:
        amount = int(support.arrays[weight][area])
        if amount:
            code = str(support.arrays["authority"][area])
            if code not in values:
                return None
            total += amount
            numerator += amount * values[code]
    return numerator / total if total else None


def _operator_table(support, household, *, rule, default="households"):
    """The shared draw's own integer cumulative table for the household's
    cell and weight: its areas and each area's width, or None where the
    draw refuses the cell."""
    system = {
        "constraints": [
            {"input": column, "support": matched, "required": False}
            for column, matched in CONSTRAINTS.items()
        ]
    }
    stage = (
        {"level": "area", "weight": default}
        if rule is None
        else {"level": "area", "weight_by": {**rule, "default": default}}
    )
    index = geo._SupportIndex(support)
    try:
        constraints = geo._constraints(household, system)
        sampled = geo._household_stage(household, stage)
        index.pick(constraints, sampled, 0.5)
    except ValueError:
        return None
    areas, cumulative = index.draw_cells[(constraints, "area", sampled["weight"])]
    widths = np.diff([0, *cumulative]).tolist()
    return [str(area) for area in areas], widths, cumulative[-1]


def _authority_of(support) -> dict[str, str]:
    return dict(
        zip(
            support.arrays["area"].tolist(),
            support.arrays["authority"].tolist(),
            strict=True,
        )
    )


@st.composite
def _invented(draw):
    """An invented support, households, rule and values over code domains so
    small that empty, absent and missing cells are common: region r9 and
    decile 9 are in no support, band z is in no rule, an input may be
    missing, and a code may have no value."""
    areas = draw(st.integers(1, 6))

    def per_area(elements):
        return draw(st.lists(elements, min_size=areas, max_size=areas))

    weights = {}
    for name in ("households", "households_x", "households_y"):
        amounts = per_area(st.integers(0, 3))
        if not any(amounts):
            amounts[0] = 1  # the decoder requires mass in every weight column
        weights[name] = amounts
    support = _support(
        region=per_area(st.sampled_from(("r1", "r2"))),
        decile=per_area(st.sampled_from(("1", "2"))),
        authority=per_area(st.sampled_from(("p", "q", "s"))),
        **weights,
    )
    rows = draw(st.integers(0, 8))

    def per_household(codes, stray):
        # A code a support or rule holds is twice as likely as a stray one or
        # a missing input, so most households have a cell.
        choices = (*codes, *codes, stray, None)
        return draw(st.lists(st.sampled_from(choices), min_size=rows, max_size=rows))

    households = pd.DataFrame(
        {
            "region": per_household(("r1", "r2"), "r9"),
            "draw_decile": per_household(("1", "2"), "9"),
            "band": per_household(("x", "y"), "z"),
        },
        dtype=object,
    )
    rule = draw(st.sampled_from((None, RULE, SHARED_RULE)))
    value = st.one_of(
        st.integers(-50, 50),
        st.fractions(min_value=-50, max_value=50, max_denominator=97),
    )
    values = {code: draw(value) for code in ("p", "q", "s") if draw(st.integers(0, 5))}
    return support, households, rule, values


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_invented())
def test_on_invented_supports_the_expectation_equals_a_brute_force_loop(invented):
    """Equal to a loop over the support's areas for every household, or
    refused, with no digit, where the loop finds some household's cell
    unsampleable or a code with mass but no value."""
    support, households, rule, values = invented
    expected = [
        _brute_force(support, household, rule=rule, values=values)
        for _, household in households.iterrows()
    ]
    if any(value is None for value in expected):
        with pytest.raises(ValueError) as refused:
            _expect(households, support, weight_by=rule, values=values)
        assert not any(character.isdigit() for character in str(refused.value))
    else:
        assert _expect(households, support, weight_by=rule, values=values) == expected


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_invented())
def test_on_invented_supports_the_expectation_is_the_draws_own(invented):
    """Built from the shared draw's own integer cumulative table for the
    household's cell and weight column, with each area's probability its
    width over the table's total, the expectation is the helper's, and,
    where it is not zero, the drawn area's value over it averages exactly
    one. Where the draw refuses a household's cell, the helper refuses it
    too."""
    support, households, rule, values = invented
    authority = _authority_of(support)
    for row in range(len(households)):
        household = households.iloc[[row]]
        table = _operator_table(support, household.iloc[0], rule=rule)
        if table is None:
            with pytest.raises(ValueError):
                _expect(household, support, weight_by=rule, values=values)
            continue
        areas, widths, total = table
        drawn = [
            (authority[area], width)
            for area, width in zip(areas, widths, strict=True)
            if width
        ]
        if any(code not in values for code, _ in drawn):
            with pytest.raises(ValueError, match="has no value"):
                _expect(household, support, weight_by=rule, values=values)
            continue
        expectation = sum(
            (Fraction(width, total) * values[code] for code, width in drawn),
            Fraction(0),
        )
        assert _expect(household, support, weight_by=rule, values=values) == [
            expectation
        ]
        if expectation:
            ratio = sum(
                (
                    Fraction(width, total) * values[code] / expectation
                    for code, width in drawn
                ),
                Fraction(0),
            )
            assert ratio == 1


def test_a_draw_of_the_code_first_by_the_same_column_has_the_same_expectation():
    """A draw whose first stage takes the value code by the household's
    weight column, and whose next takes an area within it by the same
    column, gives each code the cell's mass in it. The shared draw's own
    cumulative table at the code's level reproduces each expectation, so
    the weights grouped by cell and code are the draw's."""
    rows = _households(
        ("north", "1", "x"),
        ("north", "1", None),
        ("north", None, "x"),
        ("north", None, None),
        ("south", "1", None),
        ("south", None, "y"),
    )
    system = {
        "constraints": [
            {"input": column, "support": matched, "required": False}
            for column, matched in CONSTRAINTS.items()
        ]
    }
    stage = {"level": "authority", "weight_by": {**RULE, "default": "households"}}
    index = geo._SupportIndex(SUPPORT)
    expected = []
    for _, household in rows.iterrows():
        constraints = geo._constraints(household, system)
        sampled = geo._household_stage(household, stage)
        index.pick(constraints, sampled, 0.5)
        codes, cumulative = index.draw_cells[
            (constraints, "authority", sampled["weight"])
        ]
        widths = np.diff([0, *cumulative]).tolist()
        expected.append(
            sum(
                (
                    Fraction(width, cumulative[-1]) * VALUES[str(code)]
                    for code, width in zip(codes, widths, strict=True)
                ),
                Fraction(0),
            )
        )
    assert _expect(rows) == expected


def _random_case(seed: int, rows: int):
    rng = np.random.default_rng(seed)
    areas = 60
    households, band_x, band_y = (rng.integers(0, 4, areas) for _ in range(3))
    households[:6] += 1  # every region and decile keeps households
    band_x[0] += 1
    band_y[1] += 1
    support = _support(
        region=np.tile(["r1", "r2", "r3"], areas // 3),
        decile=np.tile(["1", "2"], areas // 2),
        authority=rng.choice(["p", "q", "r", "s"], areas),
        households=households,
        households_x=band_x,
        households_y=band_y,
    )
    frame = pd.DataFrame(
        {
            "region": rng.choice(["r1", "r2", "r3"], rows),
            "draw_decile": rng.choice(["1", "2", None], rows),
            "band": rng.choice(["x", "y", None], rows),
        },
        dtype=object,
    )
    return rng, support, frame


def _without_empty_cells(support, frame):
    """The frame's households whose cell has mass in their weight column."""
    keep = [
        _brute_force(support, household, rule=RULE, values=VALUES) is not None
        for _, household in frame.iterrows()
    ]
    return frame[keep]


def test_the_vectorised_expectation_equals_one_household_at_a_time():
    _, support, frame = _random_case(931, 400)
    frame = _without_empty_cells(support, frame)
    assert len(frame) > 300
    expected = [_expect(frame.iloc[[row]], support)[0] for row in range(len(frame))]
    assert len(set(expected)) > 6
    assert _expect(frame, support) == expected


def test_the_expectation_does_not_depend_on_row_order():
    """Shuffling the households shuffles their expectations, and shuffling
    the support's areas changes none."""
    rng, support, frame = _random_case(1931, 2000)
    frame = _without_empty_cells(support, frame)
    expected = _expect(frame, support)
    order = rng.permutation(len(frame))
    assert _expect(frame.iloc[order], support) == [expected[i] for i in order]
    shuffle = rng.permutation(len(support.arrays["area"]))
    metadata = {
        **support.metadata,
        "columns": {
            name: dict(column) for name, column in support.metadata["columns"].items()
        },
    }
    shuffled = geo.decode_atomic_support(
        geo.encode_atomic_support(
            metadata,
            {
                name: np.asarray(array)[shuffle]
                for name, array in support.arrays.items()
            },
        )
    )
    assert shuffled.sha256 != support.sha256
    assert _expect(frame, shuffled) == expected


def _refusal(households=None, support=SUPPORT, **overrides) -> str:
    if households is None:
        households = _households(("north", "1", "x"))
    with pytest.raises(ValueError) as refused:
        _expect(households, support, **overrides)
    text = str(refused.value)
    assert text.startswith("Atomic support expectations: ")
    assert not any(character.isdigit() for character in text)
    return text


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"constraint_columns": {"region": "households"}}, "support code columns"),
        ({"constraint_columns": {"region": "area"}}, "support code columns"),
        (
            {"constraint_columns": {"region": "region", "draw_decile": "region"}},
            "support code columns",
        ),
        ({"default_weight": "decile"}, "default_weight must be"),
        ({"weight_by": {"input": "band", "map": {"x": "decile"}}}, "weight_by must"),
        ({"weight_by": {**RULE, "default": "households"}}, "weight_by must"),
        ({"weight_by": {"input": "band", "map": {}}}, "weight_by must"),
        (
            {"weight_by": {"input": "region", "map": {"north": "households_x"}}},
            "must not also be a constraint input",
        ),
        ({"value_column": "households"}, "value_column must be"),
        ({"value_column": "tenure"}, "value_column must be"),
        ({"values": {**VALUES, "q": 2.5}}, "ints or Fractions"),
        ({"values": {**VALUES, "q": True}}, "ints or Fractions"),
        ({"values": {**VALUES, "q": np.int64(2)}}, "ints or Fractions"),
        ({"values": {**VALUES, "t": 0.1}}, "ints or Fractions"),
        ({"values": {**VALUES, 7: 1}}, "ints or Fractions"),
        ({"values": [("p", 3)]}, "ints or Fractions"),
    ],
)
def test_a_declaration_it_cannot_read_is_refused_naming_no_data(overrides, message):
    assert message in _refusal(**overrides)


@pytest.mark.parametrize(
    ("households", "message"),
    [
        (
            _households(("north", "1", "x")).drop(columns="band"),
            "lack input columns",
        ),
        (_households(("north", 1, "x")), "nonempty strings"),
        (_households(("north", " 1", "x")), "nonempty strings"),
        (_households(("north", "1", "")), "nonempty strings"),
        # Band z has no weight column: the draw refuses it too.
        (_households(("north", "1", "x"), ("north", "1", "z")), "no declared weight"),
        # No area lies in the moor, or in the south's decile 3.
        (_households(("moor", "1", None)), "no support row"),
        (_households(("north", "1", None), ("south", "3", None)), "no support row"),
        # a4, the south's decile 2 area, has no households.
        (_households(("south", "2", None)), "no mass"),
        # a2, the north's decile 2 area, has no households_y.
        (_households(("north", "2", "y")), "no mass"),
    ],
)
def test_households_it_cannot_read_are_refused_naming_no_data(households, message):
    assert message in _refusal(households)


def test_a_repeated_input_column_is_refused_naming_no_data():
    rows = _households(("north", "1", "x"), ("south", "1", "y"))
    rows = pd.concat([rows, rows[["draw_decile"]]], axis=1)
    assert _refusal(rows) == (
        "Atomic support expectations: households repeat an input column"
    )


def test_a_code_with_mass_but_no_value_is_refused_naming_no_data():
    values = {code: value for code, value in VALUES.items() if code != "s"}
    assert _refusal(_households(("north", "2", None)), values=values) == (
        "Atomic support expectations: a code with mass in a household's cell "
        "has no value"
    )


def test_a_support_that_is_not_decoded_is_refused_naming_no_data():
    assert _refusal(support={"area": _u("a1")}) == (
        "Atomic support expectations: support must be a decoded atomic support"
    )
