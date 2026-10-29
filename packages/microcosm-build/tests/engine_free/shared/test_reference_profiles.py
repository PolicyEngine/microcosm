"""Effective reference identities and the single extensible profile seam."""

from __future__ import annotations

import copy

import pytest

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
