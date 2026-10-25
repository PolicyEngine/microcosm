"""Exact interval cuts, declared semantics, indicators and boundary diagnostics.

The Hypothesis oracle below reads cuts itself with ``fractions.Fraction`` and
never calls a production helper.
"""

from __future__ import annotations

import copy
import json
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.reference_intervals import (
    BELOW_FIRST_CUT,
    EXCLUDED,
    IntervalPreparationError,
    boundary_diagnostics,
    classify_records,
    frame_boundary_diagnostics,
    interval_document_sha256,
    interval_indicators,
    interval_input_fingerprint,
    interval_table,
    scale_interval_document,
    scale_interval_table,
    source_interval_table,
    validate_interval_document,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

CENTRAL = {"numerator": 1117109, "denominator": 1000000}
_SETTINGS = settings(
    max_examples=60,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def _table(
    cuts=(10_000, 100_000),
    *,
    closure="right",
    nil=True,
    first=0,
    entity="person",
    policy="floor_zero",
    table_id="table",
    window=100,
    income="income_cents",
    universe="universe_status",
):
    bands = []
    if nil:
        bands.append({"id": "nil", "kind": "nil", "indicator": f"{table_id}_nil"})
    for index, cut in enumerate((*cuts, None)):
        bands.append(
            {
                "id": f"b{index}",
                "kind": "positive",
                "upper_cent": cut,
                "indicator": f"{table_id}_b{index}",
            }
        )
    return {
        "id": table_id,
        "entity": entity,
        "currency": "XTS",
        "input_unit": "integer_cent",
        "income_column": income,
        "universe_column": universe,
        "source_period": 2001,
        "target_period": 2001,
        "positive_band_closure": closure,
        "nil_band": "present" if nil else "absent",
        "negative_income_policy": policy,
        "first_lower_cut_cents": first,
        "source_locator": "Toy!A1:A9",
        "diagnostics": {
            "boundary_window_cents": window,
            "alternative_positive_band_closure": "left"
            if closure == "right"
            else "right",
        },
        "bands": bands,
    }


def _scale(table, factor=CENTRAL):
    return scale_interval_table(
        table,
        factor,
        factor_id="toy_factor",
        factor_declaration_sha256="a" * 64,
        factor_receipt_sha256="b" * 64,
    )


def _ids(table, positions):
    parsed = interval_table(table)
    return [
        {BELOW_FIRST_CUT: "below", EXCLUDED: "excluded"}.get(
            position, parsed.band_ids[position] if position >= 0 else None
        )
        for position in positions
    ]


def _classify(table, amounts, universe=None, **kwargs):
    universe = ["included"] * len(amounts) if universe is None else universe
    return _ids(
        table, classify_records(interval_table(table), amounts, universe, **kwargs)
    )


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("closure", "expected"),
    [
        (
            "right",
            ["nil", "nil", "b0", "b0", "b0", "b1", "b1", "b1", "b2"],
        ),
        (
            "left",
            ["nil", "nil", "b0", "b0", "b1", "b1", "b1", "b2", "b2"],
        ),
    ],
)
def test_each_exact_cut_and_one_cent_either_side(closure, expected):
    table = _table(closure=closure)
    amounts = [-1, 0, 1, 9_999, 10_000, 10_001, 99_999, 100_000, 100_001]
    assert _classify(table, amounts) == expected


@pytest.mark.parametrize(
    ("closure", "zero", "last", "above"),
    [("right", "below", "b1", "b2"), ("left", "b0", "b2", "b2")],
)
def test_first_positive_band_and_open_tail_under_both_closures(
    closure, zero, last, above
):
    table = _table(closure=closure, nil=False)
    assert _classify(table, [0, 1, 100_000, 100_001]) == [zero, "b0", last, above]


@pytest.mark.parametrize("closure", ["right", "left"])
@pytest.mark.parametrize("nil", [False, True])
def test_a_table_starting_above_zero_reports_records_below_its_first_cut(closure, nil):
    table = _table((5_000_000,), closure=closure, nil=nil, first=4_000_000)
    amounts = [0, 3_999_999, 4_000_000, 4_000_001, 5_000_000, 5_000_001]
    at_first = "below" if closure == "right" else "b0"
    at_upper = "b0" if closure == "right" else "b1"
    assert _classify(table, amounts) == [
        "nil" if nil else "below",
        "below",
        at_first,
        "b0",
        at_upper,
        "b1",
    ]
    report = boundary_diagnostics(
        interval_table(table),
        list(range(6)),
        amounts,
        ["included"] * 6,
        [1.0] * 6,
        weight_record_ids=list(range(6)),
        weight_kind="design",
    )
    assert report["below_first_cut"]["declared"]["records"] == (
        2 if closure == "right" else 1
    ) + (0 if nil else 1)
    assert Fraction(**report["below_first_cut"]["declared"]["weighted_count"]) == (
        2 if closure == "right" else 1
    ) + (0 if nil else 1)
    assert Fraction(
        **report["below_first_cut"]["declared"]["weighted_amount_cents"]
    ) == 3_999_999 + (4_000_000 if closure == "right" else 0)
    assert report["cuts"][0]["cut"] == "first_lower_cut"
    assert report["cuts"][0]["source_cut_cents"] == 4_000_000


def test_nil_and_loss_records_follow_the_declared_negative_policy():
    assert _classify(_table(), [-123_456, 0, 1]) == ["nil", "nil", "b0"]
    with pytest.raises(IntervalPreparationError) as refused:
        classify_records(
            interval_table(_table(policy="refuse")), [-1, 0, 5], ["included"] * 3
        )
    assert refused.value.summary["reasons"] == {"negative_amount_refused": 1}
    # Excluded losses never enter a band under either policy.
    assert _classify(_table(policy="refuse"), [-1], ["excluded"]) == ["excluded"]


@pytest.mark.parametrize("scaled", [False, True])
def test_left_closed_nil_table_with_a_zero_first_cut_puts_zero_in_nil_only(scaled):
    # Zero is nil under either closure, so the first positive band is
    # 0 < z < upper. Scaling keeps the zero cut: one cent stays in b0 rather
    # than falling below a nonzero stand-in cut.
    table = _table((100, 1_000), closure="left")
    amounts = [-1, 0, 1, 99, 100, 999, 1_000, 1_001]
    if scaled:
        table = _scale(table, {"numerator": 2, "denominator": 1})
        assert table["first_lower_cut_cents"] == {"numerator": 0, "denominator": 1}
        amounts = [-1, 0, 1, 199, 200, 1_999, 2_000, 2_001]
    assert _classify(table, amounts) == [
        "nil",
        "nil",
        "b0",
        "b0",
        "b1",
        "b1",
        "b2",
        "b2",
    ]


def test_left_closed_nil_zero_first_cut_diagnostics_keep_zero_in_nil():
    table = interval_table(_table((100, 1_000), closure="left"))
    report = boundary_diagnostics(
        table,
        [0, 1, 2, 3],
        [-5, 0, 1, 100],
        ["included"] * 4,
        [2.0, 3.0, 5.0, 7.0],
        weight_record_ids=[0, 1, 2, 3],
        weight_kind="design",
    )
    bands = {band["band"]: band for band in report["bands"]}
    assert bands["nil"]["declared"]["records"] == 2
    assert Fraction(**bands["nil"]["declared"]["weighted_count"]) == 5
    assert "alternative" not in bands["nil"]
    # Declared left: 1 is b0 and 100 is b1. The right-closed alternative puts
    # both in b0; zero stays nil and nothing is below the first cut.
    assert [bands[band]["declared"]["records"] for band in ("b0", "b1", "b2")] == [
        1,
        1,
        0,
    ]
    assert [bands[band]["alternative"]["records"] for band in ("b0", "b1", "b2")] == [
        2,
        0,
        0,
    ]
    assert report["below_first_cut"]["declared"]["records"] == 0
    assert report["below_first_cut"]["alternative"]["records"] == 0


def test_left_closed_nil_table_keeps_a_positive_fractional_first_cut_after_scaling():
    effective = _scale(
        _table(closure="left", first=1), {"numerator": 1, "denominator": 2}
    )
    assert effective["first_lower_cut_cents"] == {"numerator": 1, "denominator": 2}
    assert _classify(effective, [-1, 0, 1]) == ["nil", "nil", "b0"]


def test_right_closed_nil_diagnostics_keep_zero_out_of_the_left_closed_alternative():
    table = interval_table(_table())
    report = boundary_diagnostics(
        table,
        [0, 1],
        [-5, 0],
        ["included", "included"],
        [2.0, 3.0],
        weight_record_ids=[0, 1],
        weight_kind="design",
    )
    bands = {band["band"]: band for band in report["bands"]}
    assert bands["nil"]["declared"]["records"] == 2
    for band in table.band_ids[1:]:
        assert bands[band]["declared"]["records"] == 0
        assert bands[band]["alternative"]["records"] == 0
    assert report["below_first_cut"]["alternative"]["records"] == 0


@pytest.mark.parametrize(
    ("closure", "nil", "first", "declared_destination", "alternative_destination"),
    [
        ("right", True, 0, "nil", "nil"),
        ("right", True, 10, "nil", "nil"),
        ("left", True, 1, "nil", "nil"),
        ("right", False, 1, "below_first_cut", "below_first_cut"),
        ("left", False, 1, "below_first_cut", "below_first_cut"),
        ("right", False, 0, "none", "b0"),
        ("left", False, 0, "b0", "none"),
    ],
)
def test_floored_negative_mass_reports_signed_amounts_by_actual_destination(
    closure, nil, first, declared_destination, alternative_destination
):
    report = boundary_diagnostics(
        interval_table(_table(closure=closure, nil=nil, first=first)),
        [0, 1, 2, 3, 4],
        [-120, -30, -999, 0, 1],
        ["included", "included", "excluded", "included", "included"],
        [2.0, 0.5, 99.0, 3.0, 4.0],
        weight_record_ids=[0, 1, 2, 3, 4],
        weight_kind="design",
    )
    floored = report["floored_negative_income"]
    expected = {
        "records": 2,
        "weighted_count": {"numerator": 5, "denominator": 2},
        "pre_floor_amount_cents": -150,
        "weighted_pre_floor_amount_cents": {"numerator": -255, "denominator": 1},
    }
    assert {key: floored[key] for key in expected} == expected
    for evaluation, destination in (
        ("declared", declared_destination),
        ("alternative", alternative_destination),
    ):
        destinations = floored["by_destination"][evaluation]
        actual = (
            destinations["positive_bands"][destination]
            if destination == "b0"
            else destinations[destination]
        )
        assert actual == expected
        all_destinations = [
            destinations["nil"],
            destinations["below_first_cut"],
            destinations["none"],
            *destinations["positive_bands"].values(),
        ]
        assert sum(row["records"] for row in all_destinations) == 2
        assert sum(row["pre_floor_amount_cents"] for row in all_destinations) == -150
        for measure in ("weighted_count", "weighted_pre_floor_amount_cents"):
            assert sum(
                Fraction(**row[measure]) for row in all_destinations
            ) == Fraction(**expected[measure])


def test_a_band_id_cannot_merge_with_the_below_first_cut_diagnostic_bucket():
    declaration = _table(first=100)
    declaration["bands"][1]["id"] = "below_first_cut"
    report = boundary_diagnostics(
        interval_table(declaration),
        [0, 1],
        [50, 101],
        ["included", "included"],
        [2.0, 3.0],
        weight_record_ids=[0, 1],
        weight_kind="design",
    )
    band = next(row for row in report["bands"] if row["band"] == "below_first_cut")
    assert band["declared"]["weighted_count"] == {"numerator": 3, "denominator": 1}
    assert report["below_first_cut"]["declared"]["weighted_count"] == {
        "numerator": 2,
        "denominator": 1,
    }


def test_fractional_cent_aged_cuts_stay_exact_and_are_never_rounded():
    effective = _scale(_table((10_000, 100_000, 4_000_000)))
    uppers = [band.get("upper_cent") for band in effective["bands"]]
    assert uppers == [
        None,
        {"numerator": 1117109, "denominator": 100},
        {"numerator": 1117109, "denominator": 10},
        {"numerator": 4468436, "denominator": 1},
        None,
    ]
    assert Fraction(1117109, 100) == Fraction("11171.09")
    assert Fraction(1117109, 10) == Fraction("111710.9")
    # $111.7109: 11,171 cents is inside the first band and 11,172 above it.
    # $1,117.109: 111,710 cents is inside, and 111,711 lies above the cut.
    assert _classify(effective, [11_171, 11_172, 111_710, 111_711]) == [
        "b0",
        "b1",
        "b1",
        "b2",
    ]
    assert _classify(effective, [4_468_436, 4_468_437]) == ["b2", "b3"]
    assert effective["scaling"]["operational_factor"] == CENTRAL
    assert source_interval_table(effective) == _table((10_000, 100_000, 4_000_000))


def test_included_excluded_and_unresolved_routes():
    table = interval_table(_table())
    indicators = interval_indicators(
        table, [0, 0, 500, 500_000], ["included", "excluded", "included", "excluded"]
    )
    assert {name: values.tolist() for name, values in indicators.items()} == {
        "table_nil": [True, False, False, False],
        "table_b0": [False, False, True, False],
        "table_b1": [False, False, False, False],
        "table_b2": [False, False, False, False],
    }
    with pytest.raises(IntervalPreparationError) as refused:
        interval_indicators(
            table, [1, 2, 3, 4], ["included", "unresolved", "maybe", None]
        )
    assert refused.value.summary == {
        "table": "table",
        "entity": "person",
        "records": 4,
        "refused_records": 3,
        "reasons": {"invalid_universe_status": 2, "unresolved_universe_status": 1},
        "universe_statuses": {"included": 1, "invalid": 2, "unresolved": 1},
    }


@pytest.mark.parametrize(
    ("amount", "reason"),
    [
        (True, "boolean_amount"),
        (np.bool_(False), "boolean_amount"),
        (1.0, "non_integer_amount"),
        (np.float64(250.0), "non_integer_amount"),
        ("100", "text_amount"),
        (None, "missing_amount"),
        (float("nan"), "missing_amount"),
        (float("inf"), "nonfinite_amount"),
        (pd.NA, "missing_amount"),
        (Decimal("100"), "non_integer_amount"),
        (Fraction(1, 2), "non_integer_amount"),
    ],
)
def test_booleans_floats_strings_and_missing_amounts_refuse(amount, reason):
    table = interval_table(_table())
    with pytest.raises(IntervalPreparationError) as refused:
        interval_indicators(table, [10, amount], ["included", "included"])
    assert refused.value.summary["reasons"] == {reason: 1}
    # An excluded record's amount is never read.
    assert interval_indicators(table, [10, amount], ["included", "excluded"])[
        "table_b0"
    ].tolist() == [True, False]


def test_signed_numpy_and_python_integers_classify_identically():
    table = interval_table(_table())
    expected = classify_records(table, [-5, 0, 10_000], ["included"] * 3)
    assert (
        classify_records(
            table, list(np.asarray([-5, 0, 10_000], dtype=np.int64)), ["included"] * 3
        )
        == expected
    )


@pytest.mark.parametrize(
    "field",
    [
        "positive_band_closure",
        "nil_band",
        "negative_income_policy",
        "first_lower_cut_cents",
        "input_unit",
        "income_column",
        "universe_column",
        "entity",
        "currency",
        "source_period",
        "target_period",
        "source_locator",
        "diagnostics",
    ],
)
def test_every_semantic_is_declared_and_none_has_a_default(field):
    table = _table()
    table.pop(field)
    with pytest.raises(ValueError, match="missing declared fields"):
        interval_table(table)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("positive_band_closure", "both"),
        ("positive_band_closure", True),
        ("nil_band", True),
        ("nil_band", "maybe"),
        ("negative_income_policy", "absolute"),
        ("input_unit", "float_dollar"),
        ("first_lower_cut_cents", -1),
        ("first_lower_cut_cents", 0.0),
        ("first_lower_cut_cents", False),
        ("source_period", True),
        ("income_column", "universe_status"),
    ],
)
def test_invalid_declared_semantics_refuse(field, value):
    table = _table()
    table[field] = value
    with pytest.raises(ValueError):
        interval_table(table)


def test_the_diagnostic_alternative_is_the_other_closure():
    table = _table(closure="left")
    assert table["diagnostics"]["alternative_positive_band_closure"] == "right"
    interval_table(table)
    table["diagnostics"]["alternative_positive_band_closure"] = "left"
    with pytest.raises(ValueError, match="other positive band closure"):
        interval_table(table)


@pytest.mark.parametrize(
    "cut",
    [
        10_000.0,
        float("inf"),
        float("nan"),
        True,
        "10000",
        json.loads('{"x": Infinity}')["x"],
        json.loads('{"x": NaN}')["x"],
        -5,
        0,
    ],
)
def test_floats_booleans_nonfinite_and_nonpositive_source_cuts_refuse(cut):
    table = _table()
    table["bands"][1]["upper_cent"] = cut
    with pytest.raises(ValueError):
        interval_table(table)


@pytest.mark.parametrize(
    "rational",
    [
        {"numerator": 2234218, "denominator": 200},
        {"numerator": 1117109, "denominator": 0},
        {"numerator": 1117109, "denominator": -100},
        {"numerator": 1117109.0, "denominator": 100},
        {"numerator": True, "denominator": 1},
        {"numerator": 1117109, "denominator": 100, "extra": 1},
    ],
)
def test_effective_cuts_must_be_reduced_integer_rationals(rational):
    effective = _scale(_table())
    effective["bands"][1]["upper_cent"] = rational
    with pytest.raises(ValueError):
        interval_table(effective)


@pytest.mark.parametrize(
    "defect",
    [
        "decreasing",
        "repeated",
        "below_first",
        "two_tails",
        "tail_not_last",
        "no_tail",
        "nil_not_first",
        "nil_declared_absent",
        "nil_missing",
        "duplicate_band",
        "duplicate_indicator",
        "unknown_kind",
        "partial_companions",
        "mixed_companions",
        "unknown_field",
    ],
)
def test_malformed_band_structures_refuse(defect):
    table = _table((10_000, 100_000, 200_000))
    bands = table["bands"]
    if defect == "decreasing":
        bands[2]["upper_cent"] = 5_000
    elif defect == "repeated":
        bands[2]["upper_cent"] = 10_000
    elif defect == "below_first":
        table["first_lower_cut_cents"] = 10_000
    elif defect == "two_tails":
        bands[2]["upper_cent"] = None
    elif defect == "tail_not_last":
        bands[-1], bands[-2] = bands[-2], bands[-1]
    elif defect == "no_tail":
        bands.pop()
    elif defect == "nil_not_first":
        bands.append(bands.pop(0))
    elif defect == "nil_declared_absent":
        table["nil_band"] = "absent"
    elif defect == "nil_missing":
        bands.pop(0)
    elif defect == "duplicate_band":
        bands[2]["id"] = bands[1]["id"]
    elif defect == "duplicate_indicator":
        bands[2]["indicator"] = bands[1]["indicator"]
    elif defect == "unknown_kind":
        bands[1]["kind"] = "finite"
    elif defect == "partial_companions":
        bands[1]["count_reference"] = "count_1"
    elif defect == "mixed_companions":
        bands[1].update(count_reference="count_1", amount_reference="amount_1")
    else:
        bands[1]["lower_cent"] = 0
    with pytest.raises(ValueError):
        interval_table(table)


def test_documents_require_unique_tables_indicators_and_references():
    first = _table(table_id="first")
    second = _table(table_id="second", entity="household")
    validate_interval_document({"schema_version": 1, "tables": [first, second]})
    for defect in ("table", "indicator", "reference", "schema", "unknown"):
        tables = [copy.deepcopy(first), copy.deepcopy(second)]
        document = {"schema_version": 1, "tables": tables}
        if defect == "table":
            tables[1]["id"] = "first"
        elif defect == "indicator":
            tables[1]["bands"][1]["indicator"] = tables[0]["bands"][1]["indicator"]
        elif defect == "reference":
            for table in tables:
                for index, band in enumerate(table["bands"]):
                    band.update(
                        count_reference=f"{table['id']}_count_{index}",
                        amount_reference=f"{table['id']}_amount_{index}",
                    )
            tables[1]["bands"][1]["count_reference"] = "first_count_1"
        elif defect == "schema":
            document["schema_version"] = True
        else:
            document["defaults"] = {"positive_band_closure": "right"}
        with pytest.raises(ValueError):
            validate_interval_document(document)


def test_scaling_twice_refuses_and_scaled_cuts_bind_their_source():
    source = _table()
    effective = _scale(source)
    with pytest.raises(ValueError, match="already scaled"):
        _scale(effective)
    with pytest.raises(ValueError, match="cannot contain scaled tables"):
        scale_interval_document(
            {"schema_version": 1, "tables": [effective]}, {"table": {}}
        )
    twice = copy.deepcopy(effective)
    factor = Fraction(1117109, 1000000)
    for band in twice["bands"]:
        if band.get("upper_cent") is not None:
            value = (
                Fraction(band["upper_cent"]["numerator"], 1)
                / band["upper_cent"]["denominator"]
            )
            value *= factor
            band["upper_cent"] = {
                "numerator": value.numerator,
                "denominator": value.denominator,
            }
    with pytest.raises(ValueError, match="exactly once"):
        interval_table(twice)
    for mutation in ("source_cut", "factor", "source_hash", "operation", "missing"):
        tampered = copy.deepcopy(effective)
        if mutation == "source_cut":
            tampered["bands"][1]["source_upper_cent"] = 10_001
        elif mutation == "factor":
            tampered["scaling"]["operational_factor"] = {
                "numerator": 1117110,
                "denominator": 1000000,
            }
        elif mutation == "source_hash":
            tampered["scaling"]["source_table_sha256"] = "0" * 64
        elif mutation == "operation":
            tampered["scaling"]["operation"] = "round_to_cents"
        else:
            tampered["bands"][1].pop("source_upper_cent")
        with pytest.raises(ValueError):
            interval_table(tampered)


def test_document_scaling_changes_only_named_tables():
    document = {
        "schema_version": 1,
        "tables": [_table(table_id="scaled"), _table(table_id="kept")],
    }
    effective = scale_interval_document(
        document,
        {
            "scaled": {
                "factor": CENTRAL,
                "factor_id": "toy_factor",
                "factor_declaration_sha256": "a" * 64,
                "factor_receipt_sha256": "b" * 64,
            }
        },
    )
    assert effective["tables"][1] == document["tables"][1]
    assert "scaling" in effective["tables"][0]
    assert interval_document_sha256(effective) != interval_document_sha256(document)
    with pytest.raises(ValueError, match="Unknown interval tables"):
        scale_interval_document(document, {"missing": {}})


def test_input_fingerprints_keep_value_types():
    digests = {
        interval_input_fingerprint(values)
        for values in ([1, 2], [True, 2], [1.0, 2], ["1", 2], [np.int64(1), 2])
    }
    assert len(digests) == 4
    assert interval_input_fingerprint([np.int64(1), 2]) == interval_input_fingerprint(
        [1, 2]
    )


# ---------------------------------------------------------------------------
# Diagnostics on frames
# ---------------------------------------------------------------------------


def test_household_entity_table_is_weighted_through_memberships():
    table = interval_table(
        _table(entity="household", income="household_income", universe="household_in")
    )
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": [0, 1, 2, 3], "person_household_id": [0, 0, 1, 2]}
            ),
            "household": pd.DataFrame(
                {
                    "household_id": [0, 1, 2],
                    "household_income": [0, 10_000, 10_001],
                    "household_in": ["included", "included", "excluded"],
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"person": Weights(np.asarray([2.0, 2.0, 3.0, 5.0]), WeightKind.CALIBRATED)},
    )
    report = frame_boundary_diagnostics(table, frame)
    assert report["entity"] == "household"
    assert report["weight_kind"] == "calibrated"
    bands = {row["band"]: row for row in report["bands"]}
    assert bands["nil"]["declared"]["weighted_count"] == {
        "numerator": 2,
        "denominator": 1,
    }
    assert bands["b0"]["declared"]["weighted_count"] == {
        "numerator": 3,
        "denominator": 1,
    }
    assert report["included_records"] == 2
    # The left-closed alternative moves the exact-cut household to the next band.
    assert bands["b0"]["delta"]["weighted_count"] == {
        "numerator": -3,
        "denominator": 1,
    }
    assert bands["b1"]["delta"]["weighted_amount_cents"] == {
        "numerator": 30_000,
        "denominator": 1,
    }


def test_group_entity_without_weights_resolves_from_the_weighted_household():
    table = interval_table(
        _table(entity="tax_unit", income="unit_income", universe="unit_in")
    )
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [0, 1, 2],
                    "person_household_id": [0, 0, 1],
                    "person_tax_unit_id": [0, 1, 2],
                }
            ),
            "household": pd.DataFrame({"household_id": [0, 1]}),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": [0, 1, 2],
                    "unit_income": [5, 0, 200_000],
                    "unit_in": ["included", "excluded", "included"],
                }
            ),
        },
        EntitySchema(group_entities=("household", "tax_unit")),
        {"household": Weights(np.asarray([4.0, 6.0]), WeightKind.DESIGN)},
    )
    report = frame_boundary_diagnostics(table, frame)
    bands = {row["band"]: row for row in report["bands"]}
    assert bands["b0"]["declared"]["weighted_count"]["numerator"] == 4
    assert bands["b2"]["declared"]["weighted_amount_cents"]["numerator"] == 1_200_000
    assert report["weight_kind"] == "design"


def test_diagnostics_report_near_cut_mass_digests_and_disclosure():
    effective = interval_table(_scale(_table()))
    amounts = [0, 11_071, 11_072, 11_271, 11_272, 111_711]
    weights = [1.0, 0.5, 0.25, 0.125, 2.0, 3.0]
    report = boundary_diagnostics(
        effective,
        list(range(6)),
        amounts,
        ["included"] * 6,
        weights,
        weight_record_ids=list(range(6)),
        weight_kind="design",
    )
    near = {row["cut"]: row for row in report["cuts"]}
    # |z - 11,171.09| <= 100 cents admits 11,072 and 11,271 only.
    assert near["b0"]["near_cut_records"] == 2
    assert Fraction(**near["b0"]["near_cut_weighted_count"]) == Fraction(3, 8)
    assert near["b0"]["cut_cents"] == {"numerator": 1117109, "denominator": 100}
    assert near["b0"]["source_cut_cents"] == 10_000
    assert near["first_lower_cut"]["near_cut_records"] == 1
    assert report["publisher_reclassification"] == "unavailable"
    assert "cannot be recovered" in report["limitation"]
    assert report["digests"]["factor"]["operational_factor"] == CENTRAL
    assert set(report["digests"]) == {
        "table_sha256",
        "source_cuts_sha256",
        "effective_cuts_sha256",
        "factor",
        "weights_sha256",
        "record_order_sha256",
    }
    assert (
        report["digests"]["source_cuts_sha256"]
        != report["digests"]["effective_cuts_sha256"]
    )


def test_misaligned_weights_and_nonfinite_weights_refuse():
    table = interval_table(_table())
    arguments = dict(
        record_ids=[0, 1],
        amounts=[1, 2],
        universe=["included", "included"],
        weight_kind="design",
    )
    with pytest.raises(ValueError, match="aligned"):
        boundary_diagnostics(
            table, weights=[1.0, 2.0], weight_record_ids=[1, 0], **arguments
        )
    with pytest.raises(ValueError, match="finite"):
        boundary_diagnostics(
            table, weights=[1.0, float("nan")], weight_record_ids=[0, 1], **arguments
        )


# ---------------------------------------------------------------------------
# Hypothesis: an independent Fraction oracle
# ---------------------------------------------------------------------------


def _oracle_cut(value) -> Fraction:
    if isinstance(value, dict):
        return Fraction(value["numerator"], value["denominator"])
    return Fraction(value)


def _oracle_cuts(table) -> list[Fraction]:
    cuts = [_oracle_cut(table["first_lower_cut_cents"])]
    for band in table["bands"]:
        if band["kind"] == "positive" and band["upper_cent"] is not None:
            cuts.append(_oracle_cut(band["upper_cent"]))
    return cuts


def _oracle_band(z: Fraction, table) -> str:
    """Band id for a nonnegative income under the table's own declaration."""
    if table["nil_band"] == "present" and z == 0:
        return "nil"
    cuts = _oracle_cuts(table)
    positive = [band["id"] for band in table["bands"] if band["kind"] == "positive"]
    for index, band_id in enumerate(positive):
        lower = cuts[index]
        upper = cuts[index + 1] if index + 1 < len(cuts) else None
        if table["positive_band_closure"] == "right":
            inside = lower < z and (upper is None or z <= upper)
        else:
            inside = lower <= z and (upper is None or z < upper)
        if inside:
            return band_id
    return "below"


@st.composite
def _tables(draw, big=False):
    count = draw(st.integers(min_value=1, max_value=6))
    step = st.integers(min_value=1, max_value=10**30 if big else 50_000)
    closure = draw(st.sampled_from(["left", "right"]))
    nil = draw(st.booleans())
    first = draw(st.one_of(st.just(0), step))
    cuts, previous = [], first
    for _ in range(count):
        previous += draw(step)
        cuts.append(previous)
    return _table(
        tuple(cuts),
        closure=closure,
        nil=nil,
        first=first,
    )


_FACTORS = st.builds(
    lambda numerator, denominator: {
        "numerator": numerator,
        "denominator": denominator,
    },
    st.integers(min_value=1, max_value=10**9),
    st.one_of(
        st.sampled_from([1, 10, 100, 1_000_000]),
        st.integers(min_value=1, max_value=10**9),
    ),
)


def _near_cuts(table, draw, count):
    cuts = [int(cut) for cut in _oracle_cuts(table)]
    values = st.one_of(
        st.integers(min_value=-(10**6), max_value=cuts[-1] * 2 + 10),
        st.sampled_from(cuts).flatmap(
            lambda cut: st.integers(min_value=cut - 2, max_value=cut + 2)
        ),
    )
    return draw(st.lists(values, min_size=1, max_size=count))


@given(data=st.data())
@_SETTINGS
def test_property_included_records_fall_in_exactly_one_band_excluded_in_none(data):
    table = data.draw(_tables())
    amounts = _near_cuts(table, data.draw, 24)
    universe = data.draw(
        st.lists(
            st.sampled_from(["included", "excluded"]),
            min_size=len(amounts),
            max_size=len(amounts),
        )
    )
    parsed = interval_table(table)
    indicators = interval_indicators(parsed, amounts, universe)
    stacked = np.vstack(list(indicators.values()))
    for index, (amount, status) in enumerate(zip(amounts, universe, strict=True)):
        column = stacked[:, index]
        expected = _oracle_band(Fraction(max(0, amount)), table)
        if status == "excluded":
            assert not column.any()
        elif expected == "below":
            assert not column.any()
            assert _oracle_cuts(table)[0] > 0 or (
                table["nil_band"] == "absent"
                and table["positive_band_closure"] == "right"
                and amount <= 0
            )
        else:
            assert column.sum() == 1
            assert parsed.band_ids[int(np.argmax(column))] == expected


@given(data=st.data())
@_SETTINGS
def test_property_unresolved_records_never_yield_indicators(data):
    table = data.draw(_tables())
    amounts = _near_cuts(table, data.draw, 12)
    universe = data.draw(
        st.lists(
            st.sampled_from(["included", "excluded", "unresolved"]),
            min_size=len(amounts),
            max_size=len(amounts),
        )
    )
    unresolved = universe.count("unresolved")
    if not unresolved:
        universe[data.draw(st.integers(0, len(universe) - 1))] = "unresolved"
        unresolved = 1
    with pytest.raises(IntervalPreparationError) as refused:
        interval_indicators(interval_table(table), amounts, universe)
    assert refused.value.summary["reasons"]["unresolved_universe_status"] == unresolved


@given(table=_tables(), factor=_FACTORS)
@_SETTINGS
def test_property_scaling_preserves_order_nil_closure_and_tails(table, factor):
    effective = _scale(table, factor)
    rational = Fraction(factor["numerator"], factor["denominator"])
    assert [_oracle_cut(cut) for cut in [effective["first_lower_cut_cents"]]] == [
        Fraction(table["first_lower_cut_cents"]) * rational
    ]
    assert _oracle_cuts(effective) == [cut * rational for cut in _oracle_cuts(table)]
    cuts = _oracle_cuts(effective)
    assert all(left < right for left, right in zip(cuts, cuts[1:], strict=False))
    for before, after in zip(table["bands"], effective["bands"], strict=True):
        assert (before["id"], before["kind"], before["indicator"]) == (
            after["id"],
            after["kind"],
            after["indicator"],
        )
    assert effective["bands"][-1]["upper_cent"] is None
    for key in ("positive_band_closure", "nil_band", "negative_income_policy"):
        assert effective[key] == table[key]
    with pytest.raises(ValueError, match="already scaled"):
        _scale(effective, factor)


@given(data=st.data(), factor=_FACTORS)
@_SETTINGS
def test_property_scaled_classification_equals_deflated_source_classification(
    data, factor
):
    table = data.draw(_tables())
    effective = _scale(table, factor)
    rational = Fraction(factor["numerator"], factor["denominator"])
    scaled_cuts = _oracle_cuts(effective)
    amounts = data.draw(
        st.lists(
            st.one_of(
                st.integers(min_value=-(10**6), max_value=int(scaled_cuts[-1]) * 2 + 2),
                st.sampled_from(scaled_cuts).flatmap(
                    lambda cut: st.integers(
                        min_value=int(cut) - 2, max_value=int(cut) + 2
                    )
                ),
            ),
            min_size=1,
            max_size=24,
        )
    )
    actual = _classify(effective, amounts)
    expected = [
        _oracle_band(Fraction(max(0, amount)) / rational, table) for amount in amounts
    ]
    assert actual == expected


@given(table=_tables(), factor=_FACTORS)
@_SETTINGS
def test_property_adjacent_attainable_cents_around_fractional_cuts(table, factor):
    effective = _scale(table, factor)
    for cut in _oracle_cuts(effective):
        floor = cut.numerator // cut.denominator
        amounts = [floor - 1, floor, floor + 1, floor + 2]
        actual = _classify(effective, amounts)
        assert actual == [
            _oracle_band(Fraction(max(0, amount)), effective) for amount in amounts
        ]
        if cut.denominator != 1:
            # No attainable cent sits on a fractional cut: floor and floor + 1
            # straddle it under either closure.
            assert floor < cut < floor + 1


@given(data=st.data())
@_SETTINGS
def test_property_permuting_records_and_weights_preserves_diagnostics(data):
    table = data.draw(_tables())
    amounts = _near_cuts(table, data.draw, 16)
    universe = data.draw(
        st.lists(
            st.sampled_from(["included", "excluded"]),
            min_size=len(amounts),
            max_size=len(amounts),
        )
    )
    weights = data.draw(
        st.lists(
            st.floats(min_value=0, max_value=1e6, allow_nan=False),
            min_size=len(amounts),
            max_size=len(amounts),
        )
    )
    ids = list(range(len(amounts)))
    order = data.draw(st.permutations(ids))
    parsed = interval_table(table)

    def report(indices, weight_ids=None):
        rows = [ids[index] for index in indices]
        return boundary_diagnostics(
            parsed,
            rows,
            [amounts[index] for index in indices],
            [universe[index] for index in indices],
            [weights[index] for index in indices],
            weight_record_ids=rows if weight_ids is None else weight_ids,
            weight_kind="design",
        )

    original, permuted = report(ids), report(order)
    for key in (
        "cuts",
        "bands",
        "below_first_cut",
        "floored_negative_income",
        "included_records",
    ):
        assert original[key] == permuted[key]
    if order != ids:
        with pytest.raises(ValueError, match="aligned"):
            report(order, weight_ids=ids)


@given(data=st.data())
@_SETTINGS
def test_property_alternative_closure_deltas_sum_to_zero_and_nil_is_unchanged(data):
    table = data.draw(_tables())
    amounts = _near_cuts(table, data.draw, 20)
    weights = data.draw(
        st.lists(
            st.floats(min_value=0, max_value=1e9, allow_nan=False),
            min_size=len(amounts),
            max_size=len(amounts),
        )
    )
    ids = list(range(len(amounts)))
    report = boundary_diagnostics(
        interval_table(table),
        ids,
        amounts,
        ["included"] * len(amounts),
        weights,
        weight_record_ids=ids,
        weight_kind="design",
    )
    rows = [row for row in report["bands"] if row["kind"] == "positive"]
    rows.append(report["below_first_cut"])
    for measure in ("weighted_count", "weighted_amount_cents"):
        assert sum(Fraction(**row["delta"][measure]) for row in rows) == 0
    assert sum(row["delta"]["records"] for row in rows) == 0
    nil = [row for row in report["bands"] if row["kind"] == "nil"]
    assert all(set(row) == {"band", "kind", "declared"} for row in nil)
    # The oracle's exact weighted count per declared band.
    for row in report["bands"]:
        expected = sum(
            (
                Fraction(weight)
                for amount, weight in zip(amounts, weights, strict=True)
                if _oracle_band(Fraction(max(0, amount)), table) == row["band"]
            ),
            Fraction(0),
        )
        assert Fraction(**row["declared"]["weighted_count"]) == expected


@given(data=st.data())
@_SETTINGS
def test_property_very_large_cross_products_do_not_overflow(data):
    table = data.draw(_tables(big=True))
    factor = data.draw(
        st.builds(
            lambda numerator, denominator: {
                "numerator": numerator,
                "denominator": denominator,
            },
            st.integers(min_value=1, max_value=10**40),
            st.integers(min_value=1, max_value=10**40),
        )
    )
    effective = _scale(table, factor)
    cuts = _oracle_cuts(effective)
    amounts = data.draw(
        st.lists(
            st.one_of(
                st.integers(min_value=-(10**45), max_value=10**75),
                st.sampled_from(cuts).flatmap(
                    lambda cut: st.integers(
                        min_value=int(cut) - 1, max_value=int(cut) + 1
                    )
                ),
            ),
            min_size=1,
            max_size=12,
        )
    )
    rational = Fraction(factor["numerator"], factor["denominator"])
    assert _classify(effective, amounts) == [
        _oracle_band(Fraction(max(0, amount)) / rational, table) for amount in amounts
    ]
