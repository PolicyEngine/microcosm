"""Property tests for the proposed PUF donor rebase.

Run from the repository root:
    uv run pytest docs/evidence/us-puf-donor-uprating/proposal/test_rebase_properties.py

These test ``rebase_reference.py``, a standalone statement of the proposal's
semantics, and check the carried factors it relies on against the archived
generator's own ``uprate_puf`` (copied unchanged into ``../archived/``). They
do not exercise a production stage or the live build, and the repository's
``testpaths = ["packages"]`` keeps them out of ordinary collection.

The last test states the source-aging invariant for the ``uprate`` that
``us/spec/sources.yaml`` declares today, applied to the processed donor. It is
expected to fail and is marked xfail(strict=True) to record the counterexample.
"""

import gzip
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rebase_reference import (  # noqa: E402
    ARCHIVED_LINEAGE,
    Column,
    Donor,
    Leg,
    LevelTable,
    RebaseError,
    naive_declared_uprate,
    rebase_column,
    rebase_dividends,
    rebase_donor,
    verify_carried,
)

SETTINGS = settings(max_examples=500, deadline=None)
YEARS = range(2015, 2027)

positive = st.floats(min_value=0.05, max_value=20.0, allow_nan=False)
# Whole dollars, as PUF amounts are.
amount = st.integers(min_value=-5_000_000, max_value=50_000_000).map(float)
amounts = st.lists(amount, min_size=1, max_size=60).map(np.array)
build_years = st.integers(min_value=2016, max_value=2026)


@st.composite
def level_tables(draw, series=("wages", "interest", "dividends", "qualified")):
    levels = {name: {year: draw(positive) for year in YEARS} for name in series}
    returns = {year: draw(positive) for year in YEARS}
    return LevelTable(levels=levels, returns=returns)


def close(a, b, rtol=1e-12):
    return np.all(np.abs(a - b) <= rtol * np.maximum(1.0, np.abs(b)))


@SETTINGS
@given(amounts, positive, level_tables(), build_years)
def test_rebase_ages_the_source_value_once(source, carried, table, year):
    """Invariant 5: a verified donor value rebases to field × growth(2015, Y)."""

    column = Column("x", (Leg("F", "wages", carried),))
    out = rebase_column(source * carried, column, table, year)
    assert close(out, source * table.growth("wages", 2015, year), rtol=1e-9)


@SETTINGS
@given(amounts, positive, st.integers(min_value=0, max_value=59), st.floats(1e-6, 1.0))
def test_verify_gate_rejects_any_perturbed_record(source, carried, index, delta):
    """Invariant 1: one record off its carried factor fails verification."""

    assume(index < len(source) and abs(source[index]) >= 1.0)
    donor = source * carried
    assert verify_carried(donor, source, carried)
    donor[index] *= 1.0 + delta
    assert not verify_carried(donor, source, carried)


@SETTINGS
@given(amounts, positive, level_tables(), build_years)
def test_a_rebased_donor_refuses_a_second_rebase(source, carried, table, year):
    """Invariant 2: the stage runs once; its output cannot be rebased again."""

    columns = (Column("x", (Leg("F", "wages", carried),)),)
    donor = Donor(columns={"x": source * carried})
    once = rebase_donor(donor, columns, table, year, source={"F": source})
    with pytest.raises(RebaseError):
        rebase_donor(once, columns, table, year)


@SETTINGS
@given(amounts, positive, level_tables(), build_years)
def test_rebase_refuses_a_donor_that_does_not_verify(source, carried, table, year):
    """Invariant 1 at the stage: a donor already aged further does not verify."""

    assume(np.any(np.abs(source) >= 1.0))
    columns = (Column("x", (Leg("F", "wages", carried),)),)
    donor = Donor(columns={"x": source * carried * 1.05})
    with pytest.raises(RebaseError):
        rebase_donor(donor, columns, table, year, source={"F": source})


@SETTINGS
@given(amounts, positive, level_tables(), build_years, build_years)
def test_factors_chain(source, carried, table, first, second):
    """Invariant 4: rebasing to Y1 then aging to Y2 equals rebasing to Y2."""

    column = Column("x", (Leg("F", "wages", carried),))
    donor = source * carried
    via = rebase_column(donor, column, table, first) * table.growth(
        "wages", first, second
    )
    direct = rebase_column(donor, column, table, second)
    assert close(via, direct, rtol=1e-9)


@SETTINGS
@given(amounts, level_tables(), build_years)
def test_identity_when_the_series_matches_the_carried_factor(source, table, year):
    """Invariant 3: a column already at its series level is unchanged."""

    carried = table.growth("wages", 2015, year)
    column = Column("x", (Leg("F", "wages", carried),))
    donor = source * carried
    assert close(rebase_column(donor, column, table, year), donor, rtol=1e-12)


@SETTINGS
@given(amounts, positive, positive, level_tables(), build_years)
def test_zero_sign_support_and_rank_are_preserved(source, pos, neg, table, year):
    """Invariants 6 and 7, including a sign-split column like Schedule C."""

    column = Column(
        "c",
        (Leg("F", "wages", pos, sign=1), Leg("F", "interest", neg, sign=-1)),
    )
    donor = np.where(source > 0, source * pos, source * neg)
    out = rebase_column(donor, column, table, year)
    assert np.array_equal(out == 0, donor == 0)
    assert np.array_equal(np.sign(out), np.sign(donor))
    for side in (donor > 0, donor < 0):
        order = np.argsort(donor[side], kind="stable")
        before, after = donor[side][order], out[side][order]
        # Strictly larger donor values stay strictly larger; ties stay ties.
        assert np.all(np.diff(after)[np.diff(before) > 0] > 0)
        assert np.all(np.diff(after)[np.diff(before) == 0] == 0)


@SETTINGS
@given(amounts, positive, positive, level_tables(), build_years, st.booleans())
def test_sign_split_legs_follow_their_own_series(
    source, pos, neg, table, year, positive_first
):
    """Invariant 5 for a sign-split column, whatever order the legs are listed in."""

    legs = (Leg("F", "wages", pos, sign=1), Leg("F", "interest", neg, sign=-1))
    column = Column("c", legs if positive_first else legs[::-1])
    donor = np.where(source > 0, source * pos, source * neg)
    out = rebase_column(donor, column, table, year)
    expected = np.where(
        source > 0,
        source * table.growth("wages", 2015, year),
        source * table.growth("interest", 2015, year),
    )
    assert close(out, expected, rtol=1e-9)


@SETTINGS
@given(amounts, positive, level_tables(), build_years, st.data())
def test_weighted_totals_scale_by_the_leg_factor(source, carried, table, year, data):
    """Invariant 8: weighted totals move by exactly the leg's factor."""

    weights = np.array(
        data.draw(
            st.lists(
                st.floats(0.0, 1e4, allow_nan=False),
                min_size=len(source),
                max_size=len(source),
            )
        )
    )
    column = Column("x", (Leg("F", "wages", carried),))
    donor = source * carried
    out = rebase_column(donor, column, table, year)
    factor = table.growth("wages", 2015, year) / carried
    assert np.isclose(weights @ out, factor * (weights @ donor), rtol=1e-9, atol=1e-6)


@SETTINGS
@given(amounts, positive, level_tables(), build_years)
def test_untouched_columns_and_determinism(source, carried, table, year):
    """Invariants 9 and 11: other columns are copied; one input, one output."""

    columns = (Column("x", (Leg("F", "wages", carried),)),)
    untouched = {"weight": np.abs(source) + 1.0, "head_age": np.full(len(source), 40.0)}
    donor = Donor(columns={"x": source * carried}, untouched=untouched)
    first = rebase_donor(donor, columns, table, year, source={"F": source})
    second = rebase_donor(donor, columns, table, year, source={"F": source})
    for name, values in untouched.items():
        assert np.array_equal(first.untouched[name], values)
    assert np.array_equal(first.columns["x"], second.columns["x"])
    assert first.price_year == year and donor.lineage == ARCHIVED_LINEAGE


@SETTINGS
@given(
    st.lists(st.integers(0, 10_000_000), min_size=1, max_size=60),
    st.data(),
    positive,
    positive,
    level_tables(),
    build_years,
)
def test_dividends_add_up_and_stay_nonnegative(
    ordinary_raw, data, g_ordinary, g_qualified, table, year
):
    """Invariant 10: qualified + non-qualified = ordinary; non-qualified >= 0."""

    ordinary_raw = np.array(ordinary_raw, dtype=float)
    shares = np.array(
        data.draw(
            st.lists(
                st.floats(0.0, 1.0),
                min_size=len(ordinary_raw),
                max_size=len(ordinary_raw),
            )
        )
    )
    qualified_raw = np.floor(ordinary_raw * shares)
    qualified = qualified_raw * g_qualified
    non_qualified = ordinary_raw * g_ordinary - qualified
    assume(np.all(non_qualified >= 0))
    q_out, nq_out, clipped = rebase_dividends(
        qualified,
        non_qualified,
        ordinary_leg=Leg("E00600", "dividends", g_ordinary),
        qualified_leg=Leg("E00650", "qualified", g_qualified),
        table=table,
        build_year=year,
    )
    assert np.all(nq_out >= 0)
    ordinary_out = ordinary_raw * table.growth("dividends", 2015, year)
    if clipped == 0:
        assert close(q_out + nq_out, ordinary_out, rtol=1e-9)
    else:
        assert np.all(q_out + nq_out >= ordinary_out - 1e-6)


# --- Differential: the stated carried factors against the archived code -----

ARCHIVED = HERE.parent / "archived"


def _archived_uprate_namespace():
    calibration = Path(tempfile.mkdtemp(prefix="soi_"))
    (calibration / "soi_targets.csv").write_bytes(
        gzip.decompress((ARCHIVED / "soi_targets_42ed5d45.csv.gz").read_bytes())
    )
    storage = types.ModuleType("policyengine_us_data.storage")
    storage.CALIBRATION_FOLDER = calibration
    sys.modules.setdefault("policyengine_us_data", types.ModuleType("pud"))
    sys.modules["policyengine_us_data.storage"] = storage
    namespace: dict = {}
    source = ARCHIVED / "uprate_puf_42ed5d45.py.txt"
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    return namespace


NS = _archived_uprate_namespace()
GROWTH = NS["get_growth"]
AGI_2021 = GROWTH("adjusted_gross_income", 2015, 2021)
# The carried factors the proposal's lineage manifest would state, written
# here independently of the archived code.
STATED_CARRIED = {
    "E00200": lambda v: np.full_like(v, GROWTH("employment_income", 2015, 2021)),
    "E00300": lambda v: np.full_like(v, GROWTH("taxable_interest_income", 2015, 2021)),
    "E19800": lambda v: np.full_like(v, 1.02**6),
    "P23250": lambda v: np.full_like(v, AGI_2021),
    "E02100": lambda v: np.ones_like(v),
    "E26190": lambda v: np.ones_like(v),
    "E00900": lambda v: np.where(
        v > 0,
        GROWTH("business_net_profits", 2015, 2021) * AGI_2021,
        GROWTH("business_net_losses", 2015, 2021) * AGI_2021,
    ),
}


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(*[amount] * len(STATED_CARRIED)), min_size=1, max_size=40))
def test_stated_carried_factors_match_archived_uprate_puf(rows):
    """Invariant 12: the lineage manifest agrees with the archived generator.

    Includes the two cases a single 2021 price year would get wrong: Schedule C
    (E00900, aged by its own sign-split series and then by AGI growth again)
    and fields the archived lists never name (E02100, E26190), left at TY2015.
    """

    frame = pd.DataFrame(rows, columns=list(STATED_CARRIED), dtype=float)
    frame["S006"] = 100.0
    # uprate_puf indexes every field its three rename tables name.
    for table in ("STRAIGHT", "POS_ONLY", "NEG_ONLY"):
        for name in NS[f"SOI_TO_PUF_{table}_RENAMES"].values():
            if name not in frame:
                frame[name] = 0.0
    aged = NS["uprate_puf"](frame, 2015, 2021)
    for column, carried in STATED_CARRIED.items():
        raw = frame[column].to_numpy()
        assert close(aged[column].to_numpy(), raw * carried(raw), rtol=1e-12)


# --- Today's declared uprate, applied to the processed donor -----------------


@pytest.mark.xfail(strict=True, reason="the declared 2015-to-Y uprate double-counts")
@settings(max_examples=50, deadline=None)
@given(amounts, level_tables(), build_years)
def test_declared_uprate_on_the_donor_ages_the_source_once(source, table, year):
    """The ``uprate`` in sources.yaml runs from 2015; the donor is past 2015.

    Minimal case: a $1 TY2015 wage is stored as $1.187 (2015 to 2021). Aging
    that by the 2015-to-Y factor again gives $1.187 × g where $1 × g is right.
    """

    assume(np.any(np.abs(source) >= 1.0))
    carried = GROWTH("employment_income", 2015, 2021)
    out = naive_declared_uprate(source * carried, "wages", table, year)
    assert close(out, source * table.growth("wages", 2015, year), rtol=1e-9)
