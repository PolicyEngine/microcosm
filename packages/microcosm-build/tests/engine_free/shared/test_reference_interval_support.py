"""Per-band source-household support in interval diagnostics (G8-r2c hunk D).

A table that declares ``diagnostics.support_household_entity`` reports, per
band, the distinct source households and the source-household Kish ESS of
its count and amount contributions. Contributions are summed per source
household first, so donor clones add no support. A table without the field
parses, classifies and reports exactly as before.
"""

from __future__ import annotations

import copy
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.reference_intervals import (
    band_support,
    boundary_diagnostics,
    classify_records,
    frame_boundary_diagnostics,
    interval_indicators,
    interval_table,
    kish_ess,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from test_support.microcosm_build.reference_intervals import interval_table_fixture

_SETTINGS = settings(
    max_examples=60,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _declaration(entity="person", support="household", **kwargs):
    declaration = interval_table_fixture(entity=entity, companions=True, **kwargs)
    if support is not None:
        declaration["diagnostics"]["support_household_entity"] = support
    return declaration


def _rational(value):
    return Fraction(value["numerator"], value["denominator"])


def test_a_table_without_support_fields_parses_and_reports_as_before():
    legacy = _declaration(support=None)
    supported = _declaration()
    old, new = interval_table(legacy), interval_table(supported)
    assert old.support_household_entity is None
    assert new.support_household_entity == "household"
    amounts = [0, 50_000, 100_000, 100_001, 2_000_000, -5]
    universe = ["included"] * 5 + ["excluded"]
    assert classify_records(old, amounts, universe) == classify_records(
        new, amounts, universe
    )
    weights = [1.0, 2.0, 0.5, 4.0, 3.0, 1.0]
    ids = list(range(6))
    report = boundary_diagnostics(
        old,
        ids,
        amounts,
        universe,
        weights,
        weight_record_ids=ids,
        weight_kind="design",
    )
    assert "support" not in report
    assert all(
        "support" not in row and "references" not in row for row in report["bands"]
    )
    assert set(report) == {
        "schema_version",
        "kind",
        "table",
        "entity",
        "currency",
        "input_unit",
        "positive_band_closure",
        "alternative_positive_band_closure",
        "nil_band",
        "negative_income_policy",
        "boundary_window_cents",
        "weight_kind",
        "records",
        "included_records",
        "digests",
        "cuts",
        "bands",
        "below_first_cut",
        "floored_negative_income",
        "publisher_reclassification",
        "limitation",
    }
    with pytest.raises(ValueError, match="support_household_entity"):
        boundary_diagnostics(
            old,
            ids,
            amounts,
            universe,
            weights,
            weight_record_ids=ids,
            weight_kind="design",
            source_households=ids,
        )
    supported_report = boundary_diagnostics(
        new,
        ids,
        amounts,
        universe,
        weights,
        weight_record_ids=ids,
        weight_kind="design",
        source_households=["h0", "h0", "h1", "h2", "h3", "h4"],
    )
    for before, after in zip(report["bands"], supported_report["bands"], strict=True):
        assert {key: after[key] for key in before} == before
    with pytest.raises(ValueError, match="support_household_entity"):
        boundary_diagnostics(
            new, ids, amounts, universe, weights, weight_record_ids=ids, weight_kind="d"
        )


def _tax_unit_frame(household_of_unit=(0, 0, 1), sources=("s0", "s1")):
    persons = []
    for unit, household in enumerate(household_of_unit):
        persons.append((len(persons), household, unit))
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [row[0] for row in persons],
                    "person_household_id": [row[1] for row in persons],
                    "person_tax_unit_id": [row[2] for row in persons],
                }
            ),
            "household": pd.DataFrame(
                {"household_id": [0, 1], "source_household_id": list(sources)}
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": list(range(len(household_of_unit))),
                    "unit_income": [50_000, 60_000, 70_000],
                    "unit_in": ["included"] * 3,
                }
            ),
        },
        EntitySchema(group_entities=("household", "tax_unit")),
        {"household": Weights(np.asarray([4.0, 6.0]), WeightKind.DESIGN)},
    )


def _tax_unit_table(support="household"):
    declaration = _declaration(entity="tax_unit", support=support)
    declaration.update(income_column="unit_income", universe_column="unit_in")
    return interval_table(declaration)


def test_two_tax_units_in_one_household_count_once_with_source_household_ess():
    report = frame_boundary_diagnostics(_tax_unit_table(), _tax_unit_frame())
    bands = {row["band"]: row for row in report["bands"]}
    support = bands["b0"]["support"]
    # Units 0 and 1 share household 0 (weight 4 each); unit 2 is household 1
    # (weight 6). Count contributions per source household: 8 and 6.
    assert support["count"]["source_households"] == 2
    assert _rational(support["count"]["kish_ess"]) == Fraction(14 * 14, 64 + 36)
    amount = [Fraction(4 * 50_000 + 4 * 60_000), Fraction(6 * 70_000)]
    assert _rational(support["amount"]["kish_ess"]) == kish_ess(amount)
    assert bands["b0"]["references"] == {
        "count": "toy_count_b0",
        "amount": "toy_money_b0",
    }
    assert report["support"]["household_entity"] == "household"


def test_clone_households_share_one_source_household():
    frame = _tax_unit_frame(sources=("s0", "s0"))
    report = frame_boundary_diagnostics(_tax_unit_table(), frame)
    support = {row["band"]: row for row in report["bands"]}["b0"]["support"]
    assert support["count"]["source_households"] == 1
    assert _rational(support["count"]["kish_ess"]) == 1


@pytest.mark.parametrize(
    ("frame", "table", "message"),
    [
        (
            _tax_unit_frame(household_of_unit=(0, 1, 1)),
            None,
            None,
        ),
        (None, "missing_household", "no household membership in 'missing_household'"),
    ],
)
def test_an_entity_without_household_membership_refuses_by_name(frame, table, message):
    if frame is not None:
        # A unit whose members sit in two households has no one household.
        persons = frame.table("person").copy()
        persons.loc[len(persons)] = [3, 1, 0]
        frame = Frame(
            {
                "person": persons,
                "household": frame.table("household"),
                "tax_unit": frame.table("tax_unit"),
            },
            frame.schema,
            # Explicit unit weights, so the frame derives none from members
            # that span two households.
            {
                "household": Weights(np.asarray([4.0, 6.0]), WeightKind.DESIGN),
                "tax_unit": Weights(np.asarray([4.0, 6.0, 6.0]), WeightKind.DESIGN),
            },
        )
        with pytest.raises(
            ValueError,
            match="entity 'tax_unit' has no household membership in 'household'; "
            "unit 0 has members in 2 households",
        ):
            frame_boundary_diagnostics(_tax_unit_table(), frame)
        return
    with pytest.raises(ValueError, match=message):
        frame_boundary_diagnostics(_tax_unit_table(table), _tax_unit_frame())


def test_support_needs_the_source_household_column():
    frame = _tax_unit_frame()
    stripped = Frame(
        {
            "person": frame.table("person"),
            "household": frame.table("household").drop(columns="source_household_id"),
            "tax_unit": frame.table("tax_unit"),
        },
        frame.schema,
        {"household": Weights(np.asarray([4.0, 6.0]), WeightKind.DESIGN)},
    )
    with pytest.raises(ValueError, match="household.source_household_id"):
        frame_boundary_diagnostics(_tax_unit_table(), stripped)


_records = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=6),
        st.integers(min_value=1, max_value=64),
        st.integers(min_value=0, max_value=3_000_000),
    ),
    min_size=1,
    max_size=40,
)


@_SETTINGS
@given(records=_records, clones=st.integers(min_value=2, max_value=4))
def test_ess_is_at_most_distinct_households_and_clones_leave_it_unchanged(
    records, clones
):
    table = interval_table(_declaration())
    floored = [amount for _, _, amount in records]
    weights = [float(weight) for _, weight, _ in records]
    sources = [f"s{source}" for source, _, _ in records]
    support = band_support(table, floored, weights, sources)
    for band in support.values():
        for role in ("count", "amount"):
            measured = band[role]
            assert _rational(measured["kish_ess"]) <= measured["source_households"]
    # Split every record into equal clones of one source household: the
    # summed contribution per household, and so the support, is unchanged.
    split = [
        (source, weight / clones, amount)
        for source, weight, amount in zip(sources, weights, floored, strict=True)
        for _ in range(clones)
    ]
    cloned = band_support(
        table,
        [row[2] for row in split],
        [row[1] for row in split],
        [row[0] for row in split],
    )
    if all(
        Fraction(weight / clones) * clones == Fraction(weight) for weight in weights
    ):
        assert cloned == support


@_SETTINGS
@given(records=_records)
def test_weighted_band_counts_equal_weighted_indicator_sums(records):
    table = interval_table(_declaration())
    amounts = [amount for _, _, amount in records]
    weights = [float(weight) / 8 for _, weight, _ in records]
    universe = ["included"] * len(records)
    ids = list(range(len(records)))
    report = boundary_diagnostics(
        table,
        ids,
        amounts,
        universe,
        weights,
        weight_record_ids=ids,
        weight_kind="design",
        source_households=[f"s{source}" for source, _, _ in records],
    )
    indicators = interval_indicators(table, amounts, universe)
    for band, indicator in zip(table.band_ids, table.indicators, strict=True):
        row = next(item for item in report["bands"] if item["band"] == band)
        expected = sum(
            (
                Fraction(weight)
                for weight, flag in zip(weights, indicators[indicator], strict=True)
                if flag
            ),
            Fraction(0),
        )
        assert _rational(row["declared"]["weighted_count"]) == expected
        households = {
            source
            for (source, _, _), flag in zip(records, indicators[indicator], strict=True)
            if flag
        }
        assert row["support"]["count"]["source_households"] == len(households)


def test_kish_ess_of_equal_contributions_is_their_number():
    assert kish_ess([Fraction(3)] * 7) == 7
    assert kish_ess([]) == 0
    assert kish_ess([Fraction(1), Fraction(3)]) == Fraction(16, 10)


def test_scaled_tables_keep_the_support_field():
    from microcosm.build.reference_intervals import scale_interval_table

    declaration = copy.deepcopy(_declaration())
    scaled = scale_interval_table(
        declaration,
        {"numerator": 11, "denominator": 10},
        factor_id="f",
        factor_declaration_sha256="a" * 64,
        factor_receipt_sha256="b" * 64,
    )
    assert interval_table(scaled).support_household_entity == "household"
