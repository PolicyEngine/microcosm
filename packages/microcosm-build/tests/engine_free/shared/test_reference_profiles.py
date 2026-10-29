"""Effective reference identities and the single extensible profile seam."""

from __future__ import annotations

import copy
import json
from datetime import date

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.reference_factors import resolve_reference_factor
from microcosm.build.reference_profiles import (
    REFERENCE_DOCUMENT_KEYS,
    reference_document_sha256,
    resolve_reference_profiles,
    validate_reference_resolution,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.reference_factors import (
    factor_context_fixture,
    factor_fixture,
)
from test_support.paths import paths_for


def _selector(key, category=None):
    selector = {
        "aggregate_fact_key": key,
        "source_name": "publisher",
        "source_measure_id": "count",
        "source_concept": "households",
        "entity_name": "household",
        "record_set_id": "households",
        "period_type": "calendar_year",
        "period_value": 2023,
        "geography_level": "region",
        "geography_id": "a",
        "geography_vintage": "source-vintage",
        "dimensions": [] if category is None else ["category"],
    }
    if category is not None:
        selector["dimension_values"] = {"category": category}
    return selector


def _inputs():
    references = {
        "country": "toy",
        "target_references": [
            {
                "name": "leaf",
                "entity": "household",
                "ledger_selector": _selector("atom", 1),
                "metadata": {
                    "activation_status": "requires_reference_profile_resolution"
                },
            },
            {
                "name": "parked",
                "entity": "household",
                "ledger_selector": _selector("other", 2),
                "metadata": {"activation_status": "requires_other_evidence"},
            },
        ],
    }
    transformation = {
        "reference_names": ["leaf"],
        "operation": "allocated_residual_by_ratio",
        "value_operands": [
            {"role": "base", "expected_member_count": 1},
            {"role": "total", **_selector("total")},
            {
                "role": "partition",
                "member_selectors": [_selector("atom", 1)],
                "allocation_member_keys": ["atom"],
            },
        ],
        "metadata": {"allocation_ruling": "consumer-ruling"},
        "notes": "Proportionally allocated residual.",
    }
    declarations = {
        "schema_version": 1,
        "dimensions": {
            "residual": {
                "default": "proportional",
                "profiles": {
                    "proportional": {"transformations": [transformation]},
                    "alternative": {"transformations": []},
                },
            },
            "aging_index": {
                "default": "central",
                "profiles": {
                    "central": {"deferred": True, "transformations": []},
                    "low": {"available": False, "transformations": []},
                    "high": {"available": False, "transformations": []},
                },
            },
        },
    }
    return references, declarations


def test_resolution_is_pure_and_binds_effective_documents():
    references, declarations = _inputs()
    original = copy.deepcopy((references, declarations))
    intervals = {"bands": [{"lower": 0, "upper": None, "lower_closed": True}]}
    effective, effective_intervals, receipt = resolve_reference_profiles(
        references, declarations, {}, intervals
    )
    assert (references, declarations) == original
    assert effective_intervals == intervals and effective_intervals is not intervals
    assert (
        effective["target_references"][0]["value_operation"]
        == "allocated_residual_by_ratio"
    )
    assert (
        effective["target_references"][0]["metadata"]["activation_status"] == "active"
    )
    assert effective["target_references"][1] == references["target_references"][1]
    assert receipt["transformed_names"] == ["leaf"]
    assert receipt["deferred_dimensions"] == ["aging_index"]
    assert receipt["selection"] == {
        "aging_index": "central",
        "residual": "proportional",
    }
    assert receipt["effective_intervals_sha256"] == reference_document_sha256(intervals)
    assert receipt["transformations"]["leaf"]["source_period"] == 2023
    assert (
        validate_reference_resolution(
            effective,
            receipt,
            references_sha256=reference_document_sha256(effective),
            interval_document=effective_intervals,
        )
        == receipt
    )
    with pytest.raises(ValueError, match="already been resolved"):
        resolve_reference_profiles(effective, declarations, {})


def test_unrelated_profiles_do_not_change_effective_identity():
    references, declarations = _inputs()
    first = resolve_reference_profiles(references, declarations, {})
    declarations["dimensions"]["residual"]["profiles"]["alternative"]["notes"] = (
        "edited"
    )
    assert resolve_reference_profiles(references, declarations, {}) == first
    # Map insertion order is descriptive to selection, not an execution order.
    declarations["dimensions"] = dict(
        reversed(list(declarations["dimensions"].items()))
    )
    assert resolve_reference_profiles(references, declarations, {}) == first


@pytest.mark.parametrize(
    "selection",
    [
        {"unknown": "x"},
        {"residual": "unknown"},
        {"aging_index": "low"},
        {"aging_index": "high"},
    ],
)
def test_unknown_or_unavailable_profiles_refuse(selection):
    references, declarations = _inputs()
    with pytest.raises(ValueError, match="Unknown|unavailable"):
        resolve_reference_profiles(references, declarations, selection)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "overlap",
        "observed",
        "factor",
        "interval",
        "reserved_identity",
        "deferred_transform",
    ],
)
def test_invalid_transformations_refuse(mutation):
    references, declarations = _inputs()
    dimension = declarations["dimensions"]["residual"]
    transformations = dimension["profiles"]["proportional"]["transformations"]
    transformation = transformations[0]
    if mutation == "missing":
        transformation["reference_names"] = ["missing"]
    elif mutation == "duplicate":
        transformation["reference_names"].append("leaf")
    elif mutation == "overlap":
        declarations["dimensions"]["other"] = copy.deepcopy(dimension)
    elif mutation == "observed":
        transformation["metadata"]["nested"] = {"observed_value": 17}
    elif mutation == "factor":
        transformation["factor"] = -1
    elif mutation == "interval":
        transformation["operation"] = "scale_money_and_intervals"
    elif mutation == "reserved_identity":
        transformation["metadata"]["reference_selection_sha256"] = "forged"
    elif mutation == "deferred_transform":
        dimension["profiles"]["proportional"]["deferred"] = True
    with pytest.raises(ValueError):
        resolve_reference_profiles(references, declarations, {})


@pytest.mark.parametrize(
    "mutation", ["effective", "supplied_hash", "receipt", "interval", "metadata"]
)
def test_stale_effective_documents_and_receipts_refuse(mutation):
    references, declarations = _inputs()
    effective, intervals, receipt = resolve_reference_profiles(
        references, declarations, {}, {"bands": []}
    )
    supplied_hash = reference_document_sha256(effective)
    if mutation == "effective":
        effective["target_references"][0]["notes"] = "stale"
    elif mutation == "supplied_hash":
        supplied_hash = "0" * 64
    elif mutation == "receipt":
        receipt["selection"]["residual"] = "alternative"
    elif mutation == "interval":
        intervals["bands"].append({"lower": 0})
    elif mutation == "metadata":
        effective["target_references"][0]["metadata"]["reference_selection_sha256"] = (
            "0" * 64
        )
    with pytest.raises(ValueError, match="hash mismatch|identity mismatch"):
        validate_reference_resolution(
            effective,
            receipt,
            references_sha256=supplied_hash,
            interval_document=intervals,
        )


def test_unsatisfied_prerequisites_remain_parked_after_transformation():
    references, declarations = _inputs()
    references["target_references"][0]["metadata"]["activation_status"] = (
        "requires_other_evidence"
    )
    effective, _, _ = resolve_reference_profiles(references, declarations, {})
    assert (
        effective["target_references"][0]["metadata"]["activation_status"]
        == "requires_other_evidence"
    )


@pytest.mark.parametrize("key", ["value", "values", "observed", "observed_value"])
def test_observed_values_cannot_enter_effective_references(key):
    references, declarations = _inputs()
    references["target_references"][0]["metadata"]["audit"] = {key: 19}
    with pytest.raises(ValueError, match="observed value"):
        resolve_reference_profiles(references, declarations, {})


def test_disjoint_transformations_are_deterministic():
    references, declarations = _inputs()
    another = copy.deepcopy(declarations["dimensions"]["residual"])
    transform = another["profiles"]["proportional"]["transformations"][0]
    transform["reference_names"] = ["parked"]
    transform["value_operands"][2]["member_selectors"] = [_selector("other", 2)]
    transform["value_operands"][2]["allocation_member_keys"] = ["other"]
    declarations["dimensions"]["another"] = another
    first = resolve_reference_profiles(references, declarations, {})
    declarations["dimensions"] = dict(
        reversed(list(declarations["dimensions"].items()))
    )
    assert resolve_reference_profiles(references, declarations, {}) == first
    assert first[2]["transformed_names"] == ["leaf", "parked"]


def test_transformation_must_satisfy_its_declared_profile_prerequisite():
    references, declarations = _inputs()
    references["target_references"][0]["metadata"]["required_reference_profile"] = (
        "other"
    )
    with pytest.raises(ValueError, match="prerequisite"):
        resolve_reference_profiles(references, declarations, {})


def test_unselected_profiles_cannot_carry_observed_values():
    references, declarations = _inputs()
    declarations["dimensions"]["residual"]["profiles"]["alternative"]["metadata"] = {
        "audit": {"observed_value": 19}
    }
    with pytest.raises(ValueError, match="observed values"):
        resolve_reference_profiles(references, declarations, {})


def test_effective_document_projects_resource_headers_and_binds_full_base():
    references, declarations = _inputs()
    references.update(description="Raw resource headers", reference_set="calibration")
    effective, _, receipt = resolve_reference_profiles(references, declarations, {})
    assert set(effective) <= REFERENCE_DOCUMENT_KEYS
    assert receipt["base_references_sha256"] == reference_document_sha256(references)
    assert receipt["effective_references_sha256"] == reference_document_sha256(
        effective
    )
    assert validate_reference_resolution(effective, receipt) == receipt


def _scaling_inputs(pair_count=2, anchor_level="110"):
    context, _, _ = factor_context_fixture(anchor_level=anchor_level)
    references = {"country": "toy", "target_references": []}
    money, counts = [], []
    for kind, names in (("count", counts), ("money", money)):
        for index in range(pair_count):
            name = f"{kind}_{index}"
            names.append(name)
            selector = _selector(f"{kind}-fact-{index}", index)
            selector.update(
                source_measure_id=kind,
                source_concept=f"households.{kind}",
                period_value=2001,
            )
            row = {
                "name": name,
                "entity": "household",
                "measure": f"prepared_{kind}",
                "filter": f"band_{index}",
                "period": 2002,
                "ledger_selector": selector,
                "metadata": {
                    "activation_status": "requires_prerequisites",
                    "reference_value_kind": kind,
                    "required_reference_profile": "aging_index",
                    "required_measurement_binding": "reviewed_income",
                    "reference_prerequisites": json.dumps(
                        ["reference_profile:aging_index", "measurement:reviewed_income"]
                    ),
                    "parking_reason": "Reviewed measurement remains unavailable.",
                },
            }
            if kind == "count":
                row.update(value_operation="identity", value_operands=[])
            references["target_references"].append(row)
    transformation = {
        "operation": "scale_money_and_intervals",
        "factor_id": "toy_factor",
        "reference_names": money,
        "count_reference_names": counts,
        "interval_table_ids": [],
        "required_measurement_binding": "reviewed_income",
    }
    declarations = {
        "schema_version": 1,
        "dimensions": {
            "aging_index": {
                "default": "central",
                "profiles": {
                    name: {"transformations": [copy.deepcopy(transformation)]}
                    for name in ("low", "central", "high")
                },
            }
        },
    }
    return references, declarations, context


def test_factor_scales_money_once_keeps_counts_and_measurement_prerequisite():
    references, declarations, context = _scaling_inputs()
    original = copy.deepcopy((references, declarations, context))
    effective, _, receipt = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    assert (references, declarations, context) == original
    assert receipt["resolver_version"] == "reference-profiles-v2"
    assert set(receipt["selected_factors"]) == {"toy_factor"}
    for source, row in zip(
        references["target_references"], effective["target_references"], strict=True
    ):
        metadata = row["metadata"]
        assert metadata["activation_status"] == "requires_prerequisites"
        assert metadata["parking_reason"] == source["metadata"]["parking_reason"]
        assert json.loads(metadata["reference_prerequisites_remaining"]) == [
            "measurement:reviewed_income"
        ]
        assert row["ledger_selector"] == source["ledger_selector"]
        if row["measure"] == "prepared_count":
            assert {key: value for key, value in row.items() if key != "metadata"} == {
                key: value for key, value in source.items() if key != "metadata"
            }
            assert receipt["transformations"][row["name"]]["operation"] == "identity"
        else:
            assert row["value_operation"] == "linear_combination"
            assert row["value_operands"] == [
                {
                    "aggregate_fact_key": source["ledger_selector"][
                        "aggregate_fact_key"
                    ],
                    "expected_member_count": 1,
                    "weight": 1.1,
                }
            ]
    assert validate_reference_resolution(effective, receipt) == receipt
    with pytest.raises(ValueError, match="already been resolved"):
        resolve_reference_profiles(effective, declarations, {}, factor_context=context)


@given(
    pair_count=st.integers(min_value=1, max_value=16),
    anchor=st.integers(min_value=1, max_value=500),
)
# Two integers are cheap to draw; a 31 s draw under host load tripped
# too_slow in review, so only wall-clock health is suppressed.
@settings(
    max_examples=16,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_counts_are_invariant_and_every_money_row_uses_one_common_factor(
    pair_count, anchor
):
    references, declarations, context = _scaling_inputs(pair_count, str(anchor))
    effective, _, _ = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    counts = references["target_references"][:pair_count]
    effective_counts = effective["target_references"][:pair_count]
    for before, after in zip(counts, effective_counts, strict=True):
        assert before["ledger_selector"] == after["ledger_selector"]
        assert before["value_operation"] == after["value_operation"]
        assert before["value_operands"] == after["value_operands"]
    weights = [
        row["value_operands"] for row in effective["target_references"][pair_count:]
    ]
    assert all(len(operands) == 1 for operands in weights)
    rational = context["factors"][0]["operational_factor"]
    assert {operands[0]["weight"] for operands in weights} == {
        rational["numerator"] / rational["denominator"]
    }
    with pytest.raises(ValueError, match="already been resolved"):
        resolve_reference_profiles(effective, declarations, {}, factor_context=context)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_count",
        "swapped_roles",
        "missing_both",
        "mismatched_filter",
        "overlap",
        "duplicate",
        "duplicate_count",
        "source_period",
        "target_period",
        "diagnostic_factor",
        "money_already_scaled",
        "arbitrary_parking",
        "missing_prerequisites",
        "wrong_measurement",
        "override_prerequisites",
        "override_measurement",
        "missing_context",
        "wrong_factor",
        "wrong_receipt",
        "wrong_profile_factor",
    ],
)
def test_invalid_factor_profiles_and_companions_refuse(mutation):
    references, declarations, context = _scaling_inputs()
    transform = declarations["dimensions"]["aging_index"]["profiles"]["central"][
        "transformations"
    ][0]
    if mutation == "missing_count":
        transform["count_reference_names"].pop()
    elif mutation == "swapped_roles":
        transform["count_reference_names"], transform["reference_names"] = (
            transform["reference_names"],
            transform["count_reference_names"],
        )
    elif mutation == "missing_both":
        transform["count_reference_names"].pop()
        transform["reference_names"].pop()
    elif mutation == "mismatched_filter":
        references["target_references"][0]["filter"] = "wrong_band"
    elif mutation == "overlap":
        declarations["dimensions"]["another"] = copy.deepcopy(
            declarations["dimensions"]["aging_index"]
        )
    elif mutation == "duplicate":
        transform["reference_names"].append(transform["reference_names"][0])
    elif mutation == "duplicate_count":
        transform["count_reference_names"].append(transform["count_reference_names"][0])
    elif mutation in {"source_period", "target_period"}:
        key = "ledger_selector" if mutation == "source_period" else "period"
        for row in references["target_references"]:
            if key == "ledger_selector":
                row[key]["period_value"] = 1999
            else:
                row[key] = 1999
    elif mutation == "diagnostic_factor":
        entry = context["factors"][0]
        entry["declaration"]["purpose"] = "diagnostic"
        from microcosm.build.reference_factors import factor_document_sha256

        entry["receipt"]["declaration_sha256"] = factor_document_sha256(
            entry["declaration"]
        )
        body = {
            key: value
            for key, value in entry["receipt"].items()
            if key != "receipt_sha256"
        }
        entry["receipt"]["receipt_sha256"] = factor_document_sha256(body)
    elif mutation == "money_already_scaled":
        row = references["target_references"][-1]
        row["value_operation"] = "linear_combination"
        row["value_operands"] = [
            {
                "aggregate_fact_key": row["ledger_selector"]["aggregate_fact_key"],
                "expected_member_count": 1,
                "weight": 1.1,
            }
        ]
    elif mutation == "arbitrary_parking":
        references["target_references"][0]["metadata"]["activation_status"] = (
            "requires_arbitrary_evidence"
        )
    elif mutation == "missing_prerequisites":
        references["target_references"][0]["metadata"].pop("reference_prerequisites")
    elif mutation == "wrong_measurement":
        references["target_references"][0]["metadata"][
            "required_measurement_binding"
        ] = "other"
    elif mutation in {"override_prerequisites", "override_measurement"}:
        key = (
            "reference_prerequisites"
            if mutation == "override_prerequisites"
            else "required_measurement_binding"
        )
        transform["metadata"] = {key: "forged"}
    elif mutation == "missing_context":
        context = None
    elif mutation == "wrong_factor":
        context["factors"][0]["operational_factor"]["numerator"] += 1
    elif mutation == "wrong_receipt":
        context["factors"][0]["receipt"]["receipt_sha256"] = "0" * 64
    elif mutation == "wrong_profile_factor":
        transform["factor_id"] = "missing"
    with pytest.raises(ValueError):
        resolve_reference_profiles(references, declarations, {}, factor_context=context)


def _interval_document(pair_count=2, *, table_id="reviewed_table", companions=True):
    bands = []
    for index in range(pair_count):
        band = {
            "id": f"band_{index}",
            "kind": "positive",
            "upper_cent": None if index == pair_count - 1 else 10_000 * (index + 1),
            "indicator": f"band_{index}",
        }
        if companions:
            band.update(
                count_reference=f"count_{index}", amount_reference=f"money_{index}"
            )
        bands.append(band)
    return {
        "schema_version": 1,
        "tables": [
            {
                "id": table_id,
                "entity": "household",
                "currency": "XTS",
                "input_unit": "integer_cent",
                "income_column": "household_income_cents",
                "universe_column": "household_income_universe",
                "source_period": 2001,
                "target_period": 2002,
                "positive_band_closure": "right",
                "nil_band": "absent",
                "negative_income_policy": "floor_zero",
                "first_lower_cut_cents": 0,
                "source_locator": "Toy!A1:A2",
                "diagnostics": {
                    "boundary_window_cents": 100,
                    "alternative_positive_band_closure": "left",
                },
                "bands": bands,
            }
        ],
    }


def _interval_inputs(pair_count=2):
    references, declarations, context = _scaling_inputs(pair_count)
    for profile in declarations["dimensions"]["aging_index"]["profiles"].values():
        profile["transformations"][0]["interval_table_ids"] = ["reviewed_table"]
    return references, declarations, context, _interval_document(pair_count)


def _effective_cuts(intervals, table_id="reviewed_table"):
    from fractions import Fraction

    table = next(table for table in intervals["tables"] if table["id"] == table_id)
    return [
        None
        if band["upper_cent"] is None
        else Fraction(
            band["upper_cent"]["numerator"], band["upper_cent"]["denominator"]
        )
        for band in table["bands"]
    ]


def test_named_interval_tables_scale_once_with_their_money_factor():
    references, declarations, context, intervals = _interval_inputs(3)
    original = copy.deepcopy((references, declarations, context, intervals))
    effective, effective_intervals, receipt = resolve_reference_profiles(
        references, declarations, {}, intervals, factor_context=context
    )
    assert (references, declarations, context, intervals) == original
    rational = context["factors"][0]["operational_factor"]
    factor = rational["numerator"] / rational["denominator"]
    # One factor covers each money/cut triple.
    money = [
        row
        for row in effective["target_references"]
        if row["name"] in {"money_0", "money_1", "money_2"}
    ]
    assert {row["value_operands"][0]["weight"] for row in money} == {factor}
    assert _effective_cuts(effective_intervals) == [11_000, 22_000, None]
    table = effective_intervals["tables"][0]
    assert table["scaling"]["factor_id"] == "toy_factor"
    assert table["scaling"]["operational_factor"] == rational
    assert [band.get("source_upper_cent") for band in table["bands"]] == [
        10_000,
        20_000,
        None,
    ]
    assert receipt["base_intervals_sha256"] == reference_document_sha256(intervals)
    assert receipt["effective_intervals_sha256"] == reference_document_sha256(
        effective_intervals
    )
    assert receipt["effective_intervals_sha256"] != receipt["base_intervals_sha256"]
    assert receipt["interval_tables"]["reviewed_table"]["factor_id"] == "toy_factor"
    assert receipt["interval_tables_sha256"] == reference_document_sha256(
        receipt["interval_tables"]
    )
    assert receipt["receipt_sha256"] not in json.dumps(effective_intervals)
    for row in effective["target_references"]:
        assert (
            row["metadata"]["effective_intervals_sha256"]
            == (receipt["effective_intervals_sha256"])
        )
        if row["metadata"]["reference_value_kind"] == "count":
            source = next(
                item
                for item in references["target_references"]
                if item["name"] == row["name"]
            )
            assert row["value_operation"] == source["value_operation"]
            assert row["value_operands"] == source["value_operands"]
    assert (
        validate_reference_resolution(
            effective, receipt, interval_document=effective_intervals
        )
        == receipt
    )
    with pytest.raises(ValueError, match="interval document hash mismatch"):
        validate_reference_resolution(effective, receipt, interval_document=intervals)
    with pytest.raises(ValueError, match="already been resolved"):
        resolve_reference_profiles(
            effective, declarations, {}, effective_intervals, factor_context=context
        )
    with pytest.raises(ValueError, match="cannot contain scaled tables"):
        resolve_reference_profiles(
            references, declarations, {}, effective_intervals, factor_context=context
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_document",
        "unknown_table",
        "duplicate_table_id",
        "omitted_band",
        "extra_band",
        "cross_pair",
        "swapped_roles",
        "duplicate_reference",
        "wrong_filter",
        "wrong_entity",
        "no_companions",
        "source_period",
        "target_period",
        "second_transformation",
    ],
)
def test_interval_companions_must_be_exactly_the_scaled_rosters(mutation):
    references, declarations, context, intervals = _interval_inputs(3)
    transform = declarations["dimensions"]["aging_index"]["profiles"]["central"][
        "transformations"
    ][0]
    table = intervals["tables"][0]
    bands = table["bands"]
    if mutation == "missing_document":
        intervals = None
    elif mutation == "unknown_table":
        transform["interval_table_ids"] = ["missing_table"]
    elif mutation == "duplicate_table_id":
        transform["interval_table_ids"] = ["reviewed_table", "reviewed_table"]
    elif mutation == "omitted_band":
        bands.pop(1)
    elif mutation == "extra_band":
        bands.insert(
            2,
            {
                "id": "band_extra",
                "kind": "positive",
                "upper_cent": 30_000,
                "indicator": "band_extra",
                "count_reference": "other_count",
                "amount_reference": "other_money",
            },
        )
    elif mutation == "cross_pair":
        bands[0]["amount_reference"], bands[1]["amount_reference"] = (
            bands[1]["amount_reference"],
            bands[0]["amount_reference"],
        )
    elif mutation == "swapped_roles":
        bands[0]["amount_reference"], bands[0]["count_reference"] = (
            bands[0]["count_reference"],
            bands[0]["amount_reference"],
        )
    elif mutation == "duplicate_reference":
        bands[1]["count_reference"] = bands[0]["count_reference"]
    elif mutation == "wrong_filter":
        bands[0]["indicator"] = "another_indicator"
    elif mutation == "wrong_entity":
        table["entity"] = "person"
    elif mutation == "no_companions":
        intervals = _interval_document(3, companions=False)
    elif mutation in {"source_period", "target_period"}:
        table[mutation] = 1999
    else:
        declarations["dimensions"]["another"] = {
            "default": "only",
            "profiles": {"only": {"transformations": [copy.deepcopy(transform)]}},
        }
    with pytest.raises(ValueError):
        resolve_reference_profiles(
            references, declarations, {}, intervals, factor_context=context
        )


def test_tables_without_companions_validate_but_profiles_do_not_scale_them():
    references, declarations, context, intervals = _interval_inputs(2)
    unscaled = _interval_document(3, table_id="declared_only", companions=False)[
        "tables"
    ][0]
    for band in unscaled["bands"]:
        band["indicator"] = f"declared_{band['indicator']}"
    intervals["tables"].append(unscaled)
    _, effective_intervals, receipt = resolve_reference_profiles(
        references, declarations, {}, intervals, factor_context=context
    )
    assert effective_intervals["tables"][1] == unscaled
    assert set(receipt["interval_tables"]) == {"reviewed_table"}


def test_unnamed_interval_tables_pass_through_and_effective_equals_base():
    references, declarations, context = _scaling_inputs()
    intervals = _interval_document(2)
    _, effective_intervals, receipt = resolve_reference_profiles(
        references, declarations, {}, intervals, factor_context=context
    )
    assert effective_intervals == intervals
    assert "interval_tables" not in receipt
    assert receipt["effective_intervals_sha256"] == receipt["base_intervals_sha256"]


def test_interval_scaling_is_local_to_the_selected_profile():
    references, declarations, context, intervals = _interval_inputs(2)
    declaration, facts, identity = factor_fixture(anchor_level="105")
    declaration["id"] = "toy_low"
    factor, factor_receipt = resolve_reference_factor(declaration, facts, identity)
    context["factors"].append(
        {
            "declaration": declaration,
            "operational_factor": factor,
            "receipt": factor_receipt,
        }
    )
    profiles = declarations["dimensions"]["aging_index"]["profiles"]
    profiles["low"]["transformations"][0]["factor_id"] = "toy_low"
    low = resolve_reference_profiles(
        references,
        declarations,
        {"aging_index": "low"},
        intervals,
        factor_context=context,
    )
    central = resolve_reference_profiles(
        references, declarations, {}, intervals, factor_context=context
    )
    assert _effective_cuts(low[1]) == [10_500, None]
    assert _effective_cuts(central[1]) == [11_000, None]
    profiles["high"]["notes"] = "edited unselected high declaration"
    assert (
        resolve_reference_profiles(
            references,
            declarations,
            {"aging_index": "low"},
            intervals,
            factor_context=context,
        )
        == low
    )
    assert (
        resolve_reference_profiles(
            references, declarations, {}, intervals, factor_context=context
        )
        == central
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "record_factor",
        "record_profile",
        "record_hash",
        "drop_records",
        "document_table",
        "unscaled_document",
    ],
)
def test_rehashed_interval_scaling_records_still_refuse(mutation):
    references, declarations, context, intervals = _interval_inputs(2)
    effective, effective_intervals, receipt = resolve_reference_profiles(
        references, declarations, {}, intervals, factor_context=context
    )
    receipt = copy.deepcopy(receipt)
    document = copy.deepcopy(effective_intervals)
    record = receipt["interval_tables"]["reviewed_table"]
    if mutation == "record_factor":
        record["operational_factor"] = {"numerator": 1, "denominator": 1}
        receipt["interval_tables_sha256"] = reference_document_sha256(
            receipt["interval_tables"]
        )
    elif mutation == "record_profile":
        record["profile"] = "high"
        receipt["interval_tables_sha256"] = reference_document_sha256(
            receipt["interval_tables"]
        )
    elif mutation == "record_hash":
        record["effective_table_sha256"] = "0" * 64
    elif mutation == "drop_records":
        receipt.pop("interval_tables")
        receipt.pop("interval_tables_sha256")
    elif mutation == "document_table":
        document["tables"][0]["scaling"]["factor_receipt_sha256"] = "0" * 64
        receipt["effective_intervals_sha256"] = reference_document_sha256(document)
    else:
        document = copy.deepcopy(intervals)
        receipt["effective_intervals_sha256"] = reference_document_sha256(document)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError):
        validate_reference_resolution(effective, receipt, interval_document=document)


def test_factor_evidence_and_unselected_profile_locality():
    references, declarations, context = _scaling_inputs()
    low = resolve_reference_profiles(
        references, declarations, {"aging_index": "low"}, factor_context=context
    )
    central = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    declarations["dimensions"]["aging_index"]["profiles"]["high"]["notes"] = (
        "edited unselected declaration"
    )
    unselected = copy.deepcopy(context["factors"][0])
    unselected["declaration"]["id"] = "unselected_invalid_factor"
    context["factors"].append(unselected)
    assert (
        resolve_reference_profiles(
            references, declarations, {"aging_index": "low"}, factor_context=context
        )
        == low
    )
    assert (
        resolve_reference_profiles(references, declarations, {}, factor_context=context)
        == central
    )
    # Factor evidence is part of selection identity, not merely an opaque attachment.
    receipt = copy.deepcopy(central[2])
    receipt["selected_factor_identities"]["toy_factor"]["receipt_sha256"] = "0" * 64
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="factor evidence identity"):
        validate_reference_resolution(central[0], receipt)


def test_legacy_receipt_remains_version_one_without_factor_fields():
    references, declarations = _inputs()
    _, _, receipt = resolve_reference_profiles(references, declarations, {})
    assert receipt["resolver_version"] == "reference-profiles-v1"
    assert "selected_factors" not in receipt
    assert "selected_factor_identities" not in receipt
    assert "factor_companions" not in receipt
    assert "factor_companions_sha256" not in receipt
    # Canonical bytes produced by the v1 resolver at 1d5fff46.
    assert reference_document_sha256(receipt) == (
        "753fe7ddb2bc5e00cc4d70a74e0cac8295af266ffbc05511101540e39ae21508"
    )


def test_legacy_receipt_without_transformed_names_refuses_as_a_value_error():
    references, declarations = _inputs()
    effective, _, receipt = resolve_reference_profiles(references, declarations, {})
    receipt.pop("transformed_names")
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="transformed names"):
        validate_reference_resolution(effective, receipt)


def test_unnamed_unresolved_rows_still_validate_without_factor_evidence():
    references, declarations = _inputs()
    effective, _, receipt = resolve_reference_profiles(references, declarations, {})
    effective["target_references"].extend(
        [
            {"entity": "household", "metadata": {}},
            {"entity": "household", "metadata": {}},
        ]
    )
    receipt.pop("receipt_sha256")
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    assert validate_reference_resolution(effective, receipt) == receipt


def test_duplicate_unresolved_factor_companion_refuses():
    references, declarations, context = _scaling_inputs()
    effective, _, receipt = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    duplicate = copy.deepcopy(effective["target_references"][0])
    assert duplicate["name"] == "count_0"
    duplicate["metadata"].pop("reference_profiles_resolved")
    effective["target_references"].append(duplicate)
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="Duplicate factor companion"):
        validate_reference_resolution(effective, receipt)


@pytest.mark.parametrize("with_factors", [False, True], ids=["v1", "v2"])
@pytest.mark.parametrize(
    "mutation",
    [
        "transformations_list",
        "transformations_null",
        "transformation_scalar",
        "resolved_name_missing",
        "resolved_name_null",
        "resolved_name_empty",
        "resolved_name_scalar",
        "resolved_name_list",
    ],
)
def test_malformed_resolution_shapes_refuse_as_value_errors(with_factors, mutation):
    if with_factors:
        references, declarations, context = _scaling_inputs()
        effective, _, receipt = resolve_reference_profiles(
            references, declarations, {}, factor_context=context
        )
    else:
        references, declarations = _inputs()
        effective, _, receipt = resolve_reference_profiles(references, declarations, {})
    row = effective["target_references"][0]
    if mutation == "transformations_list":
        receipt["transformations"] = []
    elif mutation == "transformations_null":
        receipt["transformations"] = None
    elif mutation == "transformation_scalar":
        receipt["transformations"][row["name"]] = 7
    elif mutation == "resolved_name_missing":
        row.pop("name")
    else:
        row["name"] = {
            "resolved_name_null": None,
            "resolved_name_empty": "",
            "resolved_name_scalar": 7,
            "resolved_name_list": [],
        }[mutation]
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="transformations|reference names"):
        validate_reference_resolution(effective, receipt)


@pytest.mark.parametrize(
    "name_fields",
    [{}, {"name": None}, {"name": ""}, {"name": 7}, {"name": []}],
    ids=["missing", "null", "empty", "int", "list"],
)
def test_unnamed_unresolved_factor_rows_refuse_as_value_errors(name_fields):
    references, declarations, context = _scaling_inputs()
    effective, _, receipt = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    effective["target_references"].append(
        {"entity": "household", "metadata": {}, **name_fields}
    )
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="reference names"):
        validate_reference_resolution(effective, receipt)


def test_nz_default_factor_receipt_is_compact_and_covers_the_exact_roster():
    spec = paths_for("microcosm-build").package / "src/microcosm/build/nz"
    base = json.loads((spec / "target_references.json").read_text())
    declarations = json.loads((spec / "reference_profiles.json").read_text())
    context = json.loads((spec / "reference_aging.json").read_text())
    intervals = json.loads((spec / "reference_intervals.json").read_text())
    _, _, receipt = resolve_reference_profiles(
        base, declarations, {}, intervals, factor_context=context
    )
    assert set(receipt["interval_tables"]) == {"ird_taxable_income_national_ty2024"}
    assert (
        receipt["interval_tables"]["ird_taxable_income_national_ty2024"]["bands"] == 237
    )
    assert "selected_declarations" not in receipt
    assert "base_reference_document" not in receipt
    companions = receipt["factor_companions"]
    expected_names = {
        f"ird_taxable_income_{kind}_by_band__row_{index}"
        for kind in ("people", "total")
        for index in range(8, 245)
    }
    assert len(companions) == 474
    assert set(companions) == expected_names
    source_rows = {row["name"]: row for row in base["target_references"]}
    for name, companion in companions.items():
        source = source_rows[name]
        role = source["metadata"]["reference_value_kind"]
        assert companion == {
            "role": role,
            "factor_id": "ird_ty2024_to_ty2027_central",
            "dimension": "aging_index",
            "profile": "central",
            "source": {
                "ledger_selector": source["ledger_selector"],
                "period": source.get("period"),
                "value_operation": source.get("value_operation", "identity"),
                "value_operands": source.get("value_operands", []),
                "metadata": {"reference_value_kind": role},
            },
        }
    assert receipt["factor_companions_sha256"] == reference_document_sha256(companions)
    assert len(canonical_json(receipt)) < 2_600_000


def test_nz_default_factor_resolution_validates_after_declaration_expiry(monkeypatch):
    from microcosm.build import reference_factors

    class LaterDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 5)

    monkeypatch.setattr(reference_factors, "date", LaterDate)
    spec = paths_for("microcosm-build").package / "src/microcosm/build/nz"
    base = json.loads((spec / "target_references.json").read_text())
    declarations = json.loads((spec / "reference_profiles.json").read_text())
    context = json.loads((spec / "reference_aging.json").read_text())
    intervals = json.loads((spec / "reference_intervals.json").read_text())
    effective, effective_intervals, receipt = resolve_reference_profiles(
        base, declarations, {}, intervals, factor_context=context
    )
    assert receipt["selected_factors"]
    assert (
        validate_reference_resolution(
            effective, receipt, interval_document=effective_intervals
        )
        == receipt
    )


def test_unrelated_direct_fact_references_are_outside_companion_rosters():
    references, declarations, context = _scaling_inputs()
    direct = {
        "name": "other_direct_fact",
        "ledger_fact_key": "other",
        "entity": "household",
    }
    references["target_references"].append(direct)
    effective, _, _ = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    assert effective["target_references"][-1] == direct


def test_self_consistent_effective_money_tampering_still_refuses():
    references, declarations, context = _scaling_inputs()
    effective, _, receipt = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    effective["target_references"][-1]["value_operands"][0]["weight"] *= 1.1
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="must be applied exactly once"):
        validate_reference_resolution(effective, receipt)


@pytest.mark.parametrize("target", ["count", "money"])
def test_rehashed_factor_application_cannot_bypass_companion_role(target):
    references, declarations, context = _scaling_inputs()
    effective, _, receipt = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    row = next(
        row for row in effective["target_references"] if row["name"] == f"{target}_0"
    )
    row["value_operation"] = "linear_combination"
    row["value_operands"] = [
        {
            "aggregate_fact_key": row["ledger_selector"]["aggregate_fact_key"],
            "expected_member_count": 1,
            "weight": 2,
        }
    ]
    if target == "money":
        receipt["transformations"][row["name"]]["operation"] = "identity"
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="roster role|count value semantics"):
        validate_reference_resolution(effective, receipt)


def test_rehashed_action_label_cannot_change_money_into_a_count():
    references, declarations, context = _scaling_inputs()
    effective, _, receipt = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    row = next(
        row for row in effective["target_references"] if row["name"] == "money_0"
    )
    source = receipt["factor_companions"][row["name"]]["source"]
    row["metadata"]["reference_value_kind"] = "count"
    row["value_operation"] = source["value_operation"]
    row["value_operands"] = copy.deepcopy(source["value_operands"])
    action = receipt["transformations"][row["name"]]
    action["operation"] = "identity"
    action["operand_selectors"] = copy.deepcopy(source["value_operands"])
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="roster role"):
        validate_reference_resolution(effective, receipt)


def test_count_preserves_original_weighted_operation_and_refuses_changed_weight():
    references, declarations, context = _scaling_inputs()
    source = references["target_references"][0]
    source["value_operation"] = "linear_combination"
    source["value_operands"] = [
        {
            "aggregate_fact_key": source["ledger_selector"]["aggregate_fact_key"],
            "expected_member_count": 1,
            "weight": 1,
        }
    ]
    effective, _, receipt = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    row = effective["target_references"][0]
    assert row["value_operation"] == source["value_operation"]
    assert row["value_operands"] == source["value_operands"]
    assert row["metadata"]["reference_value_kind"] == "count"
    action = receipt["transformations"][row["name"]]
    assert action["operation"] == "identity"
    assert action["operand_selectors"] == source["value_operands"]
    assert validate_reference_resolution(effective, receipt) == receipt

    row["value_operands"][0]["weight"] = 2
    action["operand_selectors"][0]["weight"] = 2
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="count value semantics"):
        validate_reference_resolution(effective, receipt)


@pytest.mark.parametrize(
    "mutation",
    [
        "count_action",
        "count_kind",
        "outside_roster",
        "outside_roster_without_factor",
        "outside_roster_with_residual_label",
        "source_values",
        "roster",
    ],
)
def test_factor_roles_and_count_semantics_use_hash_bound_companion_record(mutation):
    references, declarations, context = _scaling_inputs()
    effective, _, receipt = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    row = effective["target_references"][0]
    if mutation == "count_action":
        receipt["transformations"][row["name"]]["operation"] = "linear_combination"
    elif mutation == "count_kind":
        row["metadata"]["reference_value_kind"] = "money"
    elif mutation == "outside_roster":
        previous = row["name"]
        row["name"] = "unlisted_count"
        receipt["transformations"][row["name"]] = receipt["transformations"].pop(
            previous
        )
        receipt["transformed_names"] = sorted(receipt["transformations"])
    elif mutation in {
        "outside_roster_without_factor",
        "outside_roster_with_residual_label",
    }:
        extra = copy.deepcopy(row)
        extra["name"] = "unlisted_count"
        effective["target_references"].append(extra)
        receipt["transformations"][extra["name"]] = copy.deepcopy(
            receipt["transformations"][row["name"]]
        )
        receipt["transformations"][extra["name"]].pop("factor_id")
        if mutation == "outside_roster_with_residual_label":
            receipt["transformations"][extra["name"]]["operation"] = (
                "allocated_residual_by_ratio"
            )
        receipt["transformed_names"] = sorted(receipt["transformations"])
    elif mutation == "source_values":
        receipt["factor_companions"][row["name"]]["source"]["value_operands"] = [
            {"aggregate_fact_key": "count-fact-0", "weight": 2}
        ]
    elif mutation == "roster":
        receipt["factor_companions"]["unlisted_count"] = receipt[
            "factor_companions"
        ].pop(row["name"])
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(
        ValueError,
        match=(
            "roster|companion.*hash|source.*identity|source semantics"
            "|selection identity mismatch"
        ),
    ):
        validate_reference_resolution(effective, receipt)


@pytest.mark.parametrize("target", ["money", "count"])
def test_rehashed_forged_companion_role_requires_the_bound_selection_identity(target):
    references, declarations, context = _scaling_inputs()
    effective, _, receipt = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    row = next(
        row for row in effective["target_references"] if row["name"] == f"{target}_0"
    )
    companion = receipt["factor_companions"][row["name"]]
    forged_role = "count" if target == "money" else "money"
    companion["role"] = forged_role
    companion["source"]["metadata"]["reference_value_kind"] = forged_role
    row["metadata"]["reference_value_kind"] = forged_role
    action = receipt["transformations"][row["name"]]
    if forged_role == "money":
        rational = action["operational_factor"]
        operation = "linear_combination"
        operands = [
            {
                "aggregate_fact_key": row["ledger_selector"]["aggregate_fact_key"],
                "expected_member_count": 1,
                "weight": rational["numerator"] / rational["denominator"],
            }
        ]
    else:
        operation = companion["source"]["value_operation"]
        operands = companion["source"]["value_operands"]
    row["value_operation"] = operation
    row["value_operands"] = copy.deepcopy(operands)
    action["operation"] = "linear_combination" if forged_role == "money" else "identity"
    action["operand_selectors"] = copy.deepcopy(operands)
    # Rehash every mutable attachment while retaining the original selection.
    receipt["factor_companions_sha256"] = reference_document_sha256(
        receipt["factor_companions"]
    )
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="selection identity mismatch"):
        validate_reference_resolution(effective, receipt)


def test_factor_companion_hash_must_be_covered_by_the_selection_identity():
    references, declarations, context = _scaling_inputs()
    effective, _, receipt = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    incomplete_identity = {
        key: receipt[key]
        for key in (
            "resolver_version",
            "base_references_sha256",
            "base_intervals_sha256",
            "selected_declarations_sha256",
            "selection",
            "selected_factor_identities",
            "transformed_names",
        )
    }
    selection_hash = reference_document_sha256(incomplete_identity)
    receipt["reference_selection_sha256"] = selection_hash
    for row in effective["target_references"]:
        row["metadata"]["reference_selection_sha256"] = selection_hash
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="selection identity mismatch"):
        validate_reference_resolution(effective, receipt)


@pytest.mark.parametrize("mutation", ["missing_record", "untransformed_record"])
def test_factor_companion_record_and_transformed_rows_cover_each_other(mutation):
    references, declarations, context = _scaling_inputs()
    effective, _, receipt = resolve_reference_profiles(
        references, declarations, {}, factor_context=context
    )
    companions = receipt["factor_companions"]
    if mutation == "missing_record":
        companions.pop("count_0")
    else:
        companions["untransformed_count"] = copy.deepcopy(companions["count_0"])
    receipt["factor_companions_sha256"] = reference_document_sha256(companions)
    # Rebind the changed record so this checks roster coverage after identity.
    identity = {
        key: receipt[key]
        for key in (
            "resolver_version",
            "base_references_sha256",
            "base_intervals_sha256",
            "selected_declarations_sha256",
            "selection",
            "selected_factor_identities",
            "factor_companions_sha256",
            "transformed_names",
        )
    }
    selection_hash = reference_document_sha256(identity)
    receipt["reference_selection_sha256"] = selection_hash
    for row in effective["target_references"]:
        row["metadata"]["reference_selection_sha256"] = selection_hash
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="roster|missing factor companion"):
        validate_reference_resolution(effective, receipt)
