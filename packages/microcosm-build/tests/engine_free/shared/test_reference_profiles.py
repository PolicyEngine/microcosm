"""Effective reference identities and the single extensible profile seam."""

from __future__ import annotations

import copy
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from microcosm_build.reference_factors import factor_context_fixture

from microcosm.build.reference_profiles import (
    REFERENCE_DOCUMENT_KEYS,
    reference_document_sha256,
    resolve_reference_profiles,
    validate_reference_resolution,
)


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
@settings(max_examples=16, database=None, deadline=None)
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


def test_nonempty_interval_roster_refuses_until_interval_implementation():
    references, declarations, _ = _scaling_inputs()
    transform = declarations["dimensions"]["aging_index"]["profiles"]["central"][
        "transformations"
    ][0]
    transform["interval_table_ids"] = ["reviewed_table"]
    with pytest.raises(ValueError, match="interval scaling lands in r2b"):
        resolve_reference_profiles(references, declarations, {})


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
        "source_values",
        "roster",
    ],
)
def test_factor_roles_and_count_semantics_use_hash_bound_source_documents(mutation):
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
    elif mutation == "outside_roster_without_factor":
        extra = copy.deepcopy(row)
        extra["name"] = "unlisted_count"
        effective["target_references"].append(extra)
        receipt["transformations"][extra["name"]] = copy.deepcopy(
            receipt["transformations"][row["name"]]
        )
        receipt["transformations"][extra["name"]].pop("factor_id")
        receipt["transformed_names"] = sorted(receipt["transformations"])
    elif mutation == "source_values":
        receipt["base_reference_document"]["target_references"][0]["value_operands"] = [
            {"aggregate_fact_key": "count-fact-0", "weight": 2}
        ]
    elif mutation == "roster":
        transform = receipt["selected_declarations"]["aging_index"]["declaration"][
            "transformations"
        ][0]
        transform["count_reference_names"][0] = "unlisted_count"
    receipt["effective_references_sha256"] = reference_document_sha256(effective)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    with pytest.raises(ValueError, match="roster|source.*identity|source semantics"):
        validate_reference_resolution(effective, receipt)
