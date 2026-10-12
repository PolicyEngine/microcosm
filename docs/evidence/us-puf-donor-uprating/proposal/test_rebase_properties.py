"""Property tests for the proposed PUF donor rebase.

Run from the repository root:
    uv run pytest docs/evidence/us-puf-donor-uprating/proposal/test_rebase_properties.py

These test ``rebase_reference.py``, a standalone statement of the proposal's
semantics. Two are differential tests against other implementations of the
same semantics: the carried-factor manifest against the archived generator's
own ``uprate_puf`` (copied unchanged into ``../archived/``), and the spliced
growth series against ``microcosm.build.us_runtime.target_aging``. They do not
exercise a production stage or the live build, and the repository's
``testpaths = ["packages"]`` keeps them out of ordinary collection.

The last test states the source-aging invariant for the ``uprate`` that
``us/spec/sources.yaml`` declares today, applied to the processed donor. It is
expected to fail and is marked xfail(strict=True) to record the counterexample.
"""

import gzip
import io
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
    FLOAT32_RTOL,
    Column,
    Donor,
    Leg,
    LevelTable,
    RebaseError,
    naive_declared_uprate,
    rebase_column,
    rebase_dividends,
    rebase_donor,
    rebase_synthetic_dividends,
    recover_source_dividends,
    scale_derived,
    splice,
    verify_carried,
)

SETTINGS = settings(max_examples=500, deadline=None)
YEARS = range(2015, 2027)

positive = st.floats(min_value=0.05, max_value=20.0, allow_nan=False)
# Whole dollars, as PUF amounts are.
amount = st.integers(min_value=-5_000_000, max_value=50_000_000).map(float)
amounts = st.lists(amount, min_size=1, max_size=60).map(np.array)
build_years = st.integers(min_value=2016, max_value=2026)
SERIES = (
    "wages",
    "interest",
    "qualified_dividends",
    "non_qualified_dividends",
)


@st.composite
def level_tables(draw, series=SERIES):
    levels = {name: {year: draw(positive) for year in YEARS} for name in series}
    returns = {year: draw(positive) for year in YEARS}
    return LevelTable(levels=levels, returns=returns)


def close(a, b, rtol=1e-12):
    return np.all(np.abs(a - b) <= rtol * np.maximum(1.0, np.abs(b)))


# --- Single-expression columns ---------------------------------------------


@SETTINGS
@given(amounts, positive, level_tables(), build_years)
def test_rebase_ages_the_source_value_once(source, carried, table, year):
    """Invariant 5: a verified donor value rebases to field × growth(2015, Y)."""

    column = Column("x", (Leg("F", "wages", carried),))
    out = rebase_column(source * carried, column, table, year)
    assert close(out, source * table.growth("wages", 2015, year), rtol=1e-9)


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
@given(amounts, positive, st.integers(min_value=0, max_value=59), st.floats(1e-6, 1.0))
def test_verify_gate_rejects_any_perturbed_record(source, carried, index, delta):
    """Invariant 1: one record off its carried factor fails verification."""

    assume(index < len(source) and abs(source[index]) >= 1.0)
    donor = source * carried
    assert verify_carried(donor, source, carried)
    donor[index] *= 1.0 + delta
    assert not verify_carried(donor, source, carried)


@SETTINGS
@given(amounts, positive)
def test_float32_columns_verify_at_their_own_tolerance(source, carried):
    """Invariant 1 for float32 arrays: they pass at 1e-6 and may fail at 1e-9."""

    donor = (source * carried).astype(np.float32).astype(np.float64)
    assert verify_carried(donor, source, carried, rtol=FLOAT32_RTOL)


@SETTINGS
@given(amounts, positive, level_tables(), build_years)
def test_a_rebased_donor_refuses_a_second_rebase(source, carried, table, year):
    """Invariant 2: the stage runs once; its output cannot be rebased again."""

    columns = (Column("x", (Leg("F", "wages", carried),)),)
    donor = Donor(columns={"x": source * carried})
    once = rebase_donor(donor, columns, table, year, source={"F": source})
    with pytest.raises(RebaseError):
        rebase_donor(once, columns, table, year, source={"F": source})


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
@given(
    amounts,
    positive,
    positive,
    level_tables(),
    build_years,
    st.integers(0, 59),
    st.floats(1.0, 1e7),
)
def test_a_nonzero_donor_value_on_a_zero_source_record_fails_the_gate(
    source, pos, neg, table, year, index, planted
):
    """Invariant 1 on a sign-split column: a source zero is checked too."""

    assume(index < len(source))
    source = source.copy()
    source[index] = 0.0
    column = Column(
        "c", (Leg("F", "wages", pos, sign=1), Leg("F", "interest", neg, sign=-1))
    )
    donor_values = np.where(source > 0, source * pos, source * neg)
    donor_values[index] = planted
    with pytest.raises(RebaseError):
        rebase_donor(
            Donor(columns={"c": donor_values}),
            (column,),
            table,
            year,
            source={"F": source},
        )


def test_the_stage_has_no_unverified_path():
    """Invariant 1 at the stage: the source is a required argument."""

    donor = Donor(columns={"x": np.array([1.0])})
    table = LevelTable({"wages": {2015: 1.0, 2024: 2.0}}, {2015: 1.0, 2024: 1.0})
    with pytest.raises(TypeError):
        rebase_donor(donor, (Column("x", (Leg("F", "wages", 1.0),)),), table, 2024)


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
    """Invariants 6 and 7, on a sign-split column like Schedule C. Rank
    preservation is claimed only for multiplicative single-expression columns;
    reconstructed non-qualified dividends can reorder."""

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
@given(amounts, positive, positive, level_tables(), build_years, st.data())
def test_each_leg_total_scales_by_its_factor(source, pos, neg, table, year, data):
    """Invariant 8: per leg, the weighted total moves by exactly that leg's factor."""

    weights = np.array(
        data.draw(
            st.lists(
                st.floats(0.0, 1e4, allow_nan=False),
                min_size=len(source),
                max_size=len(source),
            )
        )
    )
    column = Column(
        "c",
        (Leg("F", "wages", pos, sign=1), Leg("F", "interest", neg, sign=-1)),
    )
    donor = np.where(source > 0, source * pos, source * neg)
    out = rebase_column(donor, column, table, year)
    for mask, series, carried in (
        (donor > 0, "wages", pos),
        (donor < 0, "interest", neg),
    ):
        factor = table.growth(series, 2015, year) / carried
        assert np.isclose(
            weights[mask] @ out[mask],
            factor * (weights[mask] @ donor[mask]),
            rtol=1e-9,
            atol=1e-6,
        )


@SETTINGS
@given(amounts, positive, level_tables(), build_years)
def test_untouched_columns_and_determinism(source, carried, table, year):
    """Invariants 9 and 12: other columns are copied; one input, one output."""

    columns = (Column("x", (Leg("F", "wages", carried),)),)
    untouched = {"weight": np.abs(source) + 1.0, "head_age": np.full(len(source), 40.0)}
    donor = Donor(columns={"x": source * carried}, untouched=untouched)
    first = rebase_donor(donor, columns, table, year, source={"F": source})
    second = rebase_donor(donor, columns, table, year, source={"F": source})
    for name, values in untouched.items():
        assert np.array_equal(first.untouched[name], values)
    assert np.array_equal(first.columns["x"], second.columns["x"])
    assert first.price_year == year and donor.lineage == ARCHIVED_LINEAGE


# --- Factors chain across the splice ----------------------------------------


@st.composite
def spliced_series(draw):
    """Observed actuals to a pivot year, a projection index from the pivot on."""

    pivot = draw(st.integers(min_value=2017, max_value=2024))
    actuals = {year: draw(positive) for year in range(2015, pivot + 1)}
    projection = {year: draw(positive) for year in range(pivot, 2027)}
    return pivot, actuals, projection


@SETTINGS
@given(spliced_series(), st.integers(2015, 2026), st.integers(2015, 2026))
def test_growth_across_the_pivot_counts_it_once(series, start, end):
    """Invariant 4: observed growth to the pivot times projected growth from it."""

    pivot, actuals, projection = series
    assume(start < end)
    level = splice(actuals, projection)
    growth = level[end] / level[start]
    if end <= pivot:
        expected = actuals[end] / actuals[start]
    elif start >= pivot:
        expected = projection[end] / projection[start]
    else:
        expected = (actuals[pivot] / actuals[start]) * (
            projection[end] / projection[pivot]
        )
    assert np.isclose(growth, expected, rtol=1e-12)
    # Chained through any intermediate year, the result is the same.
    for middle in range(start + 1, end):
        assert np.isclose(
            growth,
            (level[middle] / level[start]) * (level[end] / level[middle]),
            rtol=1e-12,
        )


@SETTINGS
@given(
    spliced_series(), spliced_series(), st.integers(2015, 2026), st.integers(2015, 2026)
)
def test_per_return_growth_across_the_pivot(series, counts, start, end):
    """Invariant 4 per return: growth divides the spliced aggregate by the
    spliced return count, each counted once across its own pivot."""

    _, actuals, projection = series
    _, count_actuals, count_projection = counts
    assume(start < end)
    levels = splice(actuals, projection)
    returns = splice(count_actuals, count_projection)
    table = LevelTable({"line": levels}, returns)
    assert np.isclose(
        table.growth("line", start, end),
        (levels[end] / levels[start]) / (returns[end] / returns[start]),
        rtol=1e-12,
    )


@SETTINGS
@given(spliced_series(), st.integers(2015, 2026), st.integers(2015, 2026))
def test_splice_agrees_with_target_aging(series, start, end):
    """Invariant 4, differential: given the same series, the reference's
    aggregate splice equals the factor ``target_aging._projection_factor``
    gives targets for the same span. Return-count division is tested above."""

    target_aging = pytest.importorskip("microcosm.build.us_runtime.target_aging")
    pivot, actuals, projection = series
    assume(start < end and end >= pivot)
    result = target_aging._projection_factor(
        {"line": {year: (value, f"cbo.{year}") for year, value in projection.items()}},
        {"line": {year: (value, f"soi.{year}") for year, value in actuals.items()}},
        income_source="line",
        build_period_key=end,
        source_period_key=start,
    )
    assert result is not None
    level = splice(actuals, projection)
    assert np.isclose(result[0], level[end] / level[start], rtol=1e-12)


# --- Dividends ---------------------------------------------------------------

dividend_records = st.lists(
    st.tuples(st.integers(0, 10_000_000), st.floats(0.0, 1.0)),
    min_size=1,
    max_size=60,
)


@SETTINGS
@given(dividend_records, positive, positive, level_tables(), build_years)
def test_dividends_follow_two_series_and_keep_the_source_support(
    records, carried_ordinary, carried_qualified, table, year
):
    """Invariant 10: qualified and non-qualified each follow a series, and
    non-qualified is zero exactly where source ordinary equals qualified."""

    ordinary = np.array([o for o, _ in records], dtype=float)
    qualified = np.floor(ordinary * np.array([share for _, share in records]))
    stored_qualified = qualified * carried_qualified
    stored_non_qualified = ordinary * carried_ordinary - stored_qualified
    recovered_ordinary, recovered_qualified = recover_source_dividends(
        stored_qualified,
        stored_non_qualified,
        carried_ordinary=carried_ordinary,
        carried_qualified=carried_qualified,
    )
    assert np.array_equal(recovered_ordinary, ordinary)
    assert np.array_equal(recovered_qualified, qualified)
    q_out, nq_out = rebase_dividends(
        stored_qualified,
        stored_non_qualified,
        carried_ordinary=carried_ordinary,
        carried_qualified=carried_qualified,
        table=table,
        build_year=year,
    )
    assert close(
        q_out, qualified * table.growth("qualified_dividends", 2015, year), rtol=1e-9
    )
    assert close(
        nq_out,
        (ordinary - qualified) * table.growth("non_qualified_dividends", 2015, year),
        rtol=1e-9,
    )
    assert np.array_equal(nq_out == 0, ordinary == qualified)
    assert np.all(nq_out >= 0)


def test_dividends_off_the_dollar_grid_are_refused():
    """Invariant 10: a stored pair that is not whole TY2015 dollars does not rebase."""

    with pytest.raises(RebaseError):
        recover_source_dividends(
            np.array([100.4]),
            np.array([10.0]),
            carried_ordinary=1.0,
            carried_qualified=1.0,
        )


@pytest.mark.parametrize("value", [1_000_000.4, 1e9 + 0.3, 123.5, 7.25e10 + 0.5])
def test_large_off_grid_dividends_are_refused(value):
    """Invariant 10: an absolute tolerance refuses off-grid values at any size."""

    with pytest.raises(RebaseError):
        recover_source_dividends(
            np.array([value]),
            np.array([0.0]),
            carried_ordinary=1.0,
            carried_qualified=1.0,
        )


# Synthetic amounts are arbitrary floats, not whole dollars; cents keep them
# away from subnormal values that underflow under any factor below one.
cents = st.integers(min_value=-10_000_000_000, max_value=100_000_000_000).map(
    lambda c: c / 100
)
synthetic_pairs = st.lists(
    st.tuples(cents.map(abs), cents),
    min_size=1,
    max_size=60,
)


@SETTINGS
@given(synthetic_pairs, positive, positive, level_tables(), build_years)
def test_synthetic_dividends_scale_each_column_and_keep_signs(
    pairs, carried_ordinary, carried_qualified, table, year
):
    """Invariant 10 on synthetic records: fractional pairs, qualified above
    ordinary and negative non-qualified amounts all rebase multiplicatively."""

    qualified = np.array([q for q, _ in pairs])
    non_qualified = np.array([n for _, n in pairs])
    q_out, nq_out = rebase_synthetic_dividends(
        qualified,
        non_qualified,
        carried_ordinary=carried_ordinary,
        carried_qualified=carried_qualified,
        table=table,
        build_year=year,
    )
    q_factor = table.growth("qualified_dividends", 2015, year) / carried_qualified
    nq_factor = table.growth("non_qualified_dividends", 2015, year) / carried_ordinary
    assert close(q_out, qualified * q_factor, rtol=1e-12)
    assert close(nq_out, non_qualified * nq_factor, rtol=1e-12)
    assert np.array_equal(np.sign(q_out), np.sign(qualified))
    assert np.array_equal(np.sign(nq_out), np.sign(non_qualified))


def test_synthetic_pairs_do_not_pass_the_regular_recovery():
    """The regular rule refuses a pair whose qualified exceeds ordinary."""

    with pytest.raises(RebaseError):
        recover_source_dividends(
            np.array([200.0]),
            np.array([-100.0]),
            carried_ordinary=1.0,
            carried_qualified=1.0,
        )


# --- Derived columns ---------------------------------------------------------


@SETTINGS
@given(amounts, positive, positive, st.data())
def test_derived_columns_keep_their_ratio_to_the_base(base, carried, factor, data):
    """Invariant 11: a derived record keeps its ratio to its base; a zero base
    leaves it alone."""

    shares = np.array(
        data.draw(st.lists(st.floats(0.0, 3.0), min_size=len(base), max_size=len(base)))
    )
    base_carried = base * carried
    derived = np.abs(base_carried) * shares
    out = scale_derived(derived, base_carried, base_carried * factor)
    nonzero = base_carried != 0
    assert close(out[nonzero], derived[nonzero] * factor, rtol=1e-9)
    assert np.array_equal(out[~nonzero], derived[~nonzero])


@SETTINGS
@given(st.lists(st.floats(1e-3, 1e9), min_size=1, max_size=60), positive)
def test_a_derived_value_on_a_zero_base_is_left_unchanged(derived, factor):
    """Invariant 11: a nonzero derived value whose carried base is zero."""

    derived = np.array(derived)
    zero = np.zeros_like(derived)
    assert np.array_equal(scale_derived(derived, zero, zero * factor), derived)


# --- Differential: the carried-factor manifest against the archived code ----

ARCHIVED = HERE.parent / "archived"
SOI_BYTES = gzip.decompress((ARCHIVED / "soi_targets_42ed5d45.csv.gz").read_bytes())


def _archived_uprate_namespace():
    calibration = Path(tempfile.mkdtemp(prefix="soi_"))
    (calibration / "soi_targets.csv").write_bytes(SOI_BYTES)
    storage = types.ModuleType("policyengine_us_data.storage")
    storage.CALIBRATION_FOLDER = calibration
    sys.modules.setdefault("policyengine_us_data", types.ModuleType("pud"))
    sys.modules["policyengine_us_data.storage"] = storage
    namespace: dict = {}
    source = ARCHIVED / "uprate_puf_42ed5d45.py.txt"
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    return namespace


NS = _archived_uprate_namespace()
SOI = pd.read_csv(io.BytesIO(SOI_BYTES))


def soi_total(variable: str, year: int, count: bool) -> float:
    """One all-returns national SOI row, read here without the archived code."""

    rows = SOI[
        (SOI["Variable"] == variable)
        & (SOI["Year"] == year)
        & (SOI["Filing status"] == "All")
        & (SOI["AGI lower bound"] == -np.inf)
        & (SOI["AGI upper bound"] == np.inf)
        & (SOI["Count"] == count)
        & ~SOI["Taxable only"]
    ]
    return float(rows["Value"].iloc[0])


def per_return(variable: str) -> float:
    aggregate = soi_total(variable, 2021, False) / soi_total(variable, 2015, False)
    returns = soi_total("count", 2021, True) / soi_total("count", 2015, True)
    return aggregate / returns


AGI = per_return("adjusted_gross_income")
FLAT = 1.02**6
# The lineage manifest the proposal's gate would hold: for each TY2015 field,
# the factor the processed donor carries. Written from the audit's findings,
# with the factors recomputed above from the SOI table.
OWN_LINE = {
    "E00200": "employment_income",
    "E01100": "capital_gains_distributions",
    "E00300": "taxable_interest_income",
    "E00400": "exempt_interest",
    "E00600": "ordinary_dividends",
    "E00650": "qualified_dividends",
    "E01400": "ira_distributions",
    "E01500": "total_pension_income",
    "E01700": "taxable_pension_income",
    "E02300": "unemployment_compensation",
    "E02400": "total_social_security",
    "E02500": "taxable_social_security",
}
FLAT_TWO_PERCENT = ("E17500", "E18400", "E18500", "E19200", "E19800")
AGI_DEFAULT = (
    "E03500 E00800 E20500 E32800 E20100 E03240 E03400 E03220 E26390 E26400 "
    "T27800 E27200 E03290 P23250 E24518 E20400 E03230 E87530 E25850 E25860 "
    "E03270 E03300 P22250 E03210 E03150 E24515 E07300 E62900 E01200 E00700 "
    "E58990 E07400 E07600 E11200 E87521 E07260 E09900 P08000 E07240 E09700 "
    "E09800"
).split()
# (profits line, losses line, whether AGI growth is applied on top)
SIGN_SPLIT = {
    "E00900": ("business_net_profits", "business_net_losses", True),
    "E26270": ("partnership_and_s_corp_income", "partnership_and_s_corp_losses", True),
    "E01000": ("capital_gains_gross", "capital_gains_losses", False),
}
NOT_AGED = ("E02100", "E25960", "E25980", "E26180", "E26190", "E30400", "E30500")
ALL_FIELDS = (
    *OWN_LINE,
    *FLAT_TWO_PERCENT,
    *AGI_DEFAULT,
    *SIGN_SPLIT,
    *NOT_AGED,
)


def stated_carried(field: str, values: np.ndarray) -> np.ndarray:
    if field in OWN_LINE:
        return np.full_like(values, per_return(OWN_LINE[field]))
    if field in FLAT_TWO_PERCENT:
        return np.full_like(values, FLAT)
    if field in AGI_DEFAULT:
        return np.full_like(values, AGI)
    if field in SIGN_SPLIT:
        profits, losses, then_agi = SIGN_SPLIT[field]
        factor = np.where(
            values > 0,
            per_return(profits),
            np.where(values < 0, per_return(losses), 1.0),
        )
        return factor * (AGI if then_agi else 1.0)
    return np.ones_like(values)


def test_the_manifest_names_every_field_the_archived_lists_name():
    """Invariant 13, coverage: no archived field is missing from the manifest."""

    archived = set(NS["REMAINING_VARIABLES"])
    for table in ("STRAIGHT", "POS_ONLY", "NEG_ONLY"):
        archived |= set(NS[f"SOI_TO_PUF_{table}_RENAMES"].values())
    assert archived == set(ALL_FIELDS) - set(NOT_AGED)
    assert len(set(ALL_FIELDS)) == len(ALL_FIELDS)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(*[amount] * len(ALL_FIELDS)), min_size=1, max_size=20))
def test_stated_carried_factors_match_archived_uprate_puf(rows):
    """Invariant 13: the manifest agrees with the archived generator on every
    field, with the factors recomputed here from the SOI table.

    It covers the cases a single 2021 price year would get wrong: Schedule C
    and partnership income (aged by their own sign-split lines and then by AGI
    growth again) and the fields no archived list names, left at TY2015.
    """

    frame = pd.DataFrame(rows, columns=list(ALL_FIELDS), dtype=float)
    frame["S006"] = 100.0
    aged = NS["uprate_puf"](frame, 2015, 2021)
    for field in ALL_FIELDS:
        raw = frame[field].to_numpy()
        assert close(
            aged[field].to_numpy(), raw * stated_carried(field, raw), rtol=1e-12
        ), field
    returns = soi_total("count", 2021, True) / soi_total("count", 2015, True)
    assert close(aged["S006"].to_numpy(), frame["S006"].to_numpy() * returns)


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
    carried = per_return("employment_income")
    out = naive_declared_uprate(source * carried, "wages", table, year)
    assert close(out, source * table.growth("wages", 2015, year), rtol=1e-9)
