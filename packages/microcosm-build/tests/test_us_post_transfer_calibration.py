"""Focused contracts for US post-transfer two-part calibration."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.post_transfer_calibration import (
    POST_TRANSFER_CALIBRATION_SPECS,
    PostTransferCalibrationSpec,
    apply_post_transfer_calibration,
    calibrate_post_transfer_values,
    post_transfer_calibration_policy_identity,
    post_transfer_calibration_spec,
    validate_post_transfer_calibration_receipt,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _mask(size: int, *positions: int) -> np.ndarray:
    result = np.zeros(size, dtype=bool)
    result[list(positions)] = True
    return result


def _match_spec():
    return post_transfer_calibration_spec(
        entity="person",
        family="source_operator_child_support",
        target="child_support_expense",
    )


def _preserve_spec():
    return post_transfer_calibration_spec(
        entity="person",
        family="model_required_numeric",
        target="unemployment_compensation",
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _assert_canonical_receipt(receipt: dict[str, object]) -> None:
    payload = dict(receipt)
    observed = payload.pop("sha256")
    assert observed == _canonical_sha256(payload)
    json.dumps(receipt, allow_nan=False)


def test_policy_identity_binds_exact_nine_specs_and_its_content_hash() -> None:
    identity = post_transfer_calibration_policy_identity()
    payload = dict(identity)
    observed_sha256 = payload.pop("sha256")

    assert observed_sha256 == _canonical_sha256(payload)
    assert len(POST_TRANSFER_CALIBRATION_SPECS) == 9
    assert [
        f"{target['entity']}/{target['family']}/{target['target']}"
        for target in identity["targets"]
    ] == sorted(POST_TRANSFER_CALIBRATION_SPECS)
    assert {
        spec.key
        for spec in POST_TRANSFER_CALIBRATION_SPECS.values()
        if spec.stage == "early_gap_fill"
    } == {
        "person/model_required_numeric/unemployment_compensation",
        ("person/source_operator_prior_year_income/self_employment_income_last_year"),
    }
    assert {
        spec.key
        for spec in POST_TRANSFER_CALIBRATION_SPECS.values()
        if spec.carrier_mode == "preserve_recipient"
    } == {
        "person/model_required_numeric/unemployment_compensation",
        "person/source_operator_disability_benefits/disability_benefits",
    }


def test_kernel_and_validator_reject_caller_constructed_undeclared_spec() -> None:
    declared = _match_spec()
    rogue = PostTransferCalibrationSpec(
        entity=declared.entity,
        family=declared.family,
        target=declared.target,
        stage=declared.stage,
        carrier_mode="preserve_recipient",
    )
    values = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    reference = _mask(len(values), 0, 1, 2, 3, 4)
    recipient = ~reference
    with pytest.raises(ValueError, match="exact live declared policy entry"):
        calibrate_post_transfer_values(
            values,
            np.ones(len(values), dtype=np.float64),
            np.arange(len(values)),
            spec=rogue,
            reference_rows=reference,
            recipient_rows=recipient,
            mutable_rows=recipient,
        )

    canonical = calibrate_post_transfer_values(
        values,
        np.ones(len(values), dtype=np.float64),
        np.arange(len(values)),
        spec=declared,
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )
    with pytest.raises(ValueError, match="exact live declared policy entry"):
        validate_post_transfer_calibration_receipt(
            canonical.receipt,
            spec=rogue,
            boundary="rogue spec validator regression",
        )


@pytest.mark.parametrize(
    "spec",
    (
        post_transfer_calibration_spec(
            entity="person",
            family="adult_care",
            target="pre_subsidy_care_expenses",
        ),
        post_transfer_calibration_spec(
            entity="person",
            family="source_operator_weeks_unemployed",
            target="weeks_unemployed",
        ),
    ),
)
@pytest.mark.parametrize(
    "supply_allowed,supply_additions", ((False, False), (True, False), (False, True))
)
def test_special_constraint_masks_are_mandatory(
    spec: PostTransferCalibrationSpec,
    supply_allowed: bool,
    supply_additions: bool,
) -> None:
    values = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    reference = _mask(len(values), 0, 1, 2, 3, 4)
    recipient = ~reference
    with pytest.raises(ValueError, match="requires explicit allowed_carrier_rows"):
        calibrate_post_transfer_values(
            values,
            np.ones(len(values), dtype=np.float64),
            np.arange(len(values)),
            spec=spec,
            reference_rows=reference,
            recipient_rows=recipient,
            mutable_rows=recipient,
            allowed_carrier_rows=recipient if supply_allowed else None,
            addition_candidate_rows=recipient if supply_additions else None,
        )


def test_match_reference_down_rake_keeps_strongest_weighted_prefix() -> None:
    values = np.asarray([0.0, 10.0, 100.0, 20.0, 50.0, 100.0, 0.0])
    weights = np.asarray([6.0, 2.0, 2.0, 1.0, 4.0, 5.0, 10.0])
    entity_ids = np.asarray([1, 2, 3, 10, 11, 12, 13])
    reference = _mask(len(values), 0, 1, 2)
    recipient = ~reference

    result = calibrate_post_transfer_values(
        values,
        weights,
        entity_ids,
        spec=_match_spec(),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )

    carriers = set(entity_ids[recipient & (result.values > 0.0)].tolist())
    assert carriers == {11, 12}
    assert set(result.values[recipient & (result.values > 0.0)]) <= {10.0, 100.0}
    carrier = result.receipt["carrier"]
    assert {
        key: carrier[key]
        for key in (
            "mode",
            "reference_positive_mass",
            "reference_positive_share",
            "target_positive_mass",
            "before_positive_mass",
            "before_positive_share",
            "after_positive_mass",
            "after_positive_share",
            "residual_after_minus_target",
            "absolute_residual",
            "removed_rows",
            "added_rows",
            "disallowed_cleared_rows",
            "capacity_limited",
        )
    } == {
        "mode": "match_reference",
        "reference_positive_mass": 4.0,
        "reference_positive_share": 0.4,
        "target_positive_mass": 8.0,
        "before_positive_mass": 10.0,
        "before_positive_share": 0.5,
        "after_positive_mass": 9.0,
        "after_positive_share": 0.45,
        "residual_after_minus_target": 1.0,
        "absolute_residual": 1.0,
        "removed_rows": 1,
        "added_rows": 0,
        "disallowed_cleared_rows": 0,
        "capacity_limited": False,
    }
    assert carrier["capacity"] == {
        "fixed_positive_rows": 0,
        "fixed_positive_mass": 0.0,
        "allowed_positive_rows_before": 3,
        "allowed_positive_mass_before": 10.0,
        "addition_candidate_rows": 1,
        "addition_candidate_mass": 10.0,
        "minimum_attainable_mass": 0.0,
        "maximum_attainable_mass": 20.0,
        "target_within_attainable_interval": True,
        "capacity_boundary_saturated": True,
    }
    assert carrier["selection"]["action"] == "retain_positive_prefix"
    assert carrier["selection"]["chosen_prefix_mass"] == 9.0
    _assert_canonical_receipt(result.receipt)


def test_match_reference_equal_distance_uses_lower_mass_and_entity_id_tie() -> None:
    # Reference incidence is 0.4 and recipient mass is 10, so the target is 4.
    # Equal-valued carriers have weights 3 and 2.  Their prefixes 3 and 5 are
    # equally distant; lower mass wins, and stable entity id 10 ranks first.
    values = np.asarray([0.0, 100.0, 100.0, 100.0, 0.0])
    weights = np.asarray([6.0, 4.0, 2.0, 3.0, 5.0])
    entity_ids = np.asarray([1, 2, 20, 10, 30])
    reference = _mask(len(values), 0, 1)
    recipient = ~reference

    result = calibrate_post_transfer_values(
        values,
        weights,
        entity_ids,
        spec=_match_spec(),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )

    assert set(entity_ids[recipient & (result.values > 0.0)]) == {10}
    assert result.receipt["carrier"]["after_positive_mass"] == 3.0
    assert result.receipt["carrier"]["residual_after_minus_target"] == -1.0


def _up_rake_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    values = np.asarray([0.0, 100.0, 50.0, 0.0, 0.0, 0.0])
    weights = np.asarray([4.0, 6.0, 2.0, 4.0, 1.0, 3.0])
    entity_ids = np.asarray([1, 2, 40, 30, 10, 20])
    reference = _mask(len(values), 0, 1)
    recipient = ~reference
    return values, weights, entity_ids, reference, recipient


def test_match_reference_up_rake_adds_weighted_prefix_in_entity_id_order() -> None:
    values, weights, entity_ids, reference, recipient = _up_rake_fixture()

    result = calibrate_post_transfer_values(
        values,
        weights,
        entity_ids,
        spec=_match_spec(),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )

    assert set(entity_ids[recipient & (result.values > 0.0)]) == {10, 20, 40}
    assert result.receipt["carrier"]["target_positive_mass"] == 6.0
    assert result.receipt["carrier"]["after_positive_mass"] == 6.0
    assert result.receipt["carrier"]["added_rows"] == 2
    assert result.receipt["carrier"]["absolute_residual"] == 0.0


def test_selection_and_amounts_are_invariant_to_input_row_order() -> None:
    values, weights, entity_ids, reference, recipient = _up_rake_fixture()
    baseline = calibrate_post_transfer_values(
        values,
        weights,
        entity_ids,
        spec=_match_spec(),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )
    permutation = np.asarray([4, 1, 5, 0, 3, 2])
    shuffled = calibrate_post_transfer_values(
        values[permutation],
        weights[permutation],
        entity_ids[permutation],
        spec=_match_spec(),
        reference_rows=reference[permutation],
        recipient_rows=recipient[permutation],
        mutable_rows=recipient[permutation],
    )

    baseline_by_id = dict(
        zip(entity_ids.tolist(), baseline.values.tolist(), strict=True)
    )
    shuffled_by_id = dict(
        zip(
            entity_ids[permutation].tolist(),
            shuffled.values.tolist(),
            strict=True,
        )
    )
    assert shuffled_by_id == baseline_by_id
    assert shuffled.receipt["carrier"] == baseline.receipt["carrier"]
    assert (
        shuffled.receipt["amount"]["anchor_rows"]
        == (baseline.receipt["amount"]["anchor_rows"])
    )


def test_preserve_mode_keeps_carriers_and_exactly_anchors_five_quantiles() -> None:
    # Donor CDF knots are exactly p10/p25/p50/p75/p100.  Recipient upper-CDF
    # ranks are p20/p40/p60/p80/p100, so a plain upper-CDF map would start at
    # 20 rather than 10.  The explicit five battery anchors restore all knots.
    values = np.asarray(
        [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    )
    weights = np.asarray([2.0, 3.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    entity_ids = np.arange(1, len(values) + 1)
    reference = _mask(len(values), 0, 1, 2, 3, 4)
    recipient = ~reference
    before_carriers = values[recipient] > 0.0

    result = calibrate_post_transfer_values(
        values,
        weights,
        entity_ids,
        spec=_preserve_spec(),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )

    np.testing.assert_array_equal(result.values[recipient] > 0.0, before_carriers)
    np.testing.assert_array_equal(
        result.values[recipient],
        np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]),
    )
    amount = result.receipt["amount"]
    assert amount["reference_quantiles"] == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert amount["recipient_after_quantiles"] == amount["reference_quantiles"]
    assert amount["qed_after"] == 0.0
    assert amount["exact_anchor_count"] == 5
    assert amount["anchor_conflicts"] == []
    assert amount["unanchored_quantiles"] == []
    assert set(result.values[recipient]) <= set(values[reference])
    assert result.receipt["invariants"]["preserve_carriers"] is True


def test_negative_negative_zero_zero_weight_and_immutable_bytes_are_exact() -> None:
    values = np.asarray(
        [0.0, 10.0, 20.0, -7.25, -0.0, 999.0, 123.0, 50.0, 0.0],
        dtype=np.float64,
    )
    weights = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0])
    entity_ids = np.arange(1, len(values) + 1)
    reference = _mask(len(values), 0, 1, 2)
    recipient = ~reference
    mutable = recipient.copy()
    mutable[6] = False
    byte_exact = _mask(len(values), 3, 4, 5, 6)
    before_bits = values.view(np.uint64).copy()

    result = calibrate_post_transfer_values(
        values,
        weights,
        entity_ids,
        spec=post_transfer_calibration_spec(
            entity="person",
            family="source_operator_prior_year_income",
            target="self_employment_income_last_year",
        ),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=mutable,
    )

    np.testing.assert_array_equal(
        result.values.view(np.uint64)[byte_exact], before_bits[byte_exact]
    )
    assert np.signbit(result.values[4])
    mapped = mutable & (weights > 0.0) & (result.values > 0.0)
    assert set(result.values[mapped]) <= {10.0, 20.0}
    assert result.receipt["carrier"]["after_positive_mass"] == 4.0
    assert result.receipt["invariants"] == {
        "immutable_bytes_preserved": True,
        "negative_bytes_preserved": True,
        "negative_zero_bytes_preserved": True,
        "zero_weight_bytes_preserved": True,
        "preserve_carriers": False,
        "allowed_carrier_violations": 0,
        "exact_quantile_anchors": False,
    }


def test_float32_values_fail_closed_before_byte_preservation_is_claimed() -> None:
    values = np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float32)
    reference = _mask(len(values), 0, 1)
    recipient = ~reference

    with pytest.raises(ValueError, match="exact float64 dtype"):
        calibrate_post_transfer_values(
            values,
            np.ones(len(values), dtype=np.float64),
            np.arange(1, len(values) + 1),
            spec=_match_spec(),
            reference_rows=reference,
            recipient_rows=recipient,
            mutable_rows=recipient,
        )


def test_allowed_masks_clear_forbidden_carrier_and_report_capacity_shortfall() -> None:
    values = np.asarray([0.0, 10.0, 100.0, 300.0, 0.0, 0.0])
    weights = np.asarray([2.0, 8.0, 1.0, 3.0, 2.0, 4.0])
    entity_ids = np.arange(1, len(values) + 1)
    reference = _mask(len(values), 0, 1)
    recipient = ~reference
    allowed = _mask(len(values), 2, 4)
    additions = _mask(len(values), 4)

    result = calibrate_post_transfer_values(
        values,
        weights,
        entity_ids,
        spec=post_transfer_calibration_spec(
            entity="person",
            family="source_operator_weeks_unemployed",
            target="weeks_unemployed",
        ),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
        allowed_carrier_rows=allowed,
        addition_candidate_rows=additions,
    )

    assert set(entity_ids[recipient & (result.values > 0.0)]) == {3, 5}
    carrier = result.receipt["carrier"]
    assert carrier["target_positive_mass"] == 8.0
    assert carrier["after_positive_mass"] == 3.0
    assert carrier["residual_after_minus_target"] == -5.0
    assert carrier["disallowed_cleared_rows"] == 1
    assert carrier["added_rows"] == 1
    assert carrier["capacity_limited"] is True
    assert carrier["capacity"] == {
        "fixed_positive_rows": 0,
        "fixed_positive_mass": 0.0,
        "allowed_positive_rows_before": 1,
        "allowed_positive_mass_before": 1.0,
        "addition_candidate_rows": 1,
        "addition_candidate_mass": 2.0,
        "minimum_attainable_mass": 0.0,
        "maximum_attainable_mass": 3.0,
        "target_within_attainable_interval": False,
        "capacity_boundary_saturated": True,
    }
    assert carrier["selection"]["action"] == "add_zero_prefix"
    assert carrier["selection"]["chosen_prefix_mass"] == 2.0
    assert result.receipt["invariants"]["allowed_carrier_violations"] == 0

    scope = result.receipt["scope"]
    assert (
        scope["allowed_carrier_rows_sha256"]
        == hashlib.sha256(np.ascontiguousarray(allowed).tobytes(order="C")).hexdigest()
    )
    assert (
        scope["addition_candidate_rows_sha256"]
        == hashlib.sha256(
            np.ascontiguousarray(additions).tobytes(order="C")
        ).hexdigest()
    )
    validate_post_transfer_calibration_receipt(
        result.receipt,
        spec=post_transfer_calibration_spec(
            entity="person",
            family="source_operator_weeks_unemployed",
            target="weeks_unemployed",
        ),
        boundary="allowed-mask binding regression",
    )

    stripped = {
        **result.receipt,
        "scope": dict(result.receipt["scope"]),
    }
    stripped["scope"].pop("addition_candidate_rows_sha256")
    stripped_payload = dict(stripped)
    stripped_payload.pop("sha256")
    stripped["sha256"] = _canonical_sha256(stripped_payload)
    with pytest.raises(ValueError, match="scope evidence is incomplete"):
        validate_post_transfer_calibration_receipt(
            stripped,
            spec=post_transfer_calibration_spec(
                entity="person",
                family="source_operator_weeks_unemployed",
                target="weeks_unemployed",
            ),
            boundary="stripped addition-mask binding regression",
        )


def test_match_reference_proves_immutable_positive_floor_saturation() -> None:
    values = np.asarray([0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0])
    weights = np.asarray([8.0, 0.5, 0.5, 0.5, 0.5] * 2)
    reference = _mask(len(values), 0, 1, 2, 3, 4)
    recipient = ~reference
    mutable = _mask(len(values), 6, 7, 8, 9)

    result = calibrate_post_transfer_values(
        values,
        weights,
        np.arange(len(values)),
        spec=_match_spec(),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=mutable,
    )

    carrier = result.receipt["carrier"]
    assert carrier["target_positive_mass"] == 2.0
    assert carrier["after_positive_mass"] == 8.0
    assert carrier["capacity_limited"] is True
    assert carrier["capacity"]["minimum_attainable_mass"] == 8.0
    assert carrier["capacity"]["maximum_attainable_mass"] == 10.0
    assert carrier["capacity"]["capacity_boundary_saturated"] is True
    assert carrier["selection"]["action"] == "retain_positive_prefix"
    assert carrier["selection"]["chosen_prefix_mass"] == 0.0
    assert result.values[5] == 10.0
    assert np.all(result.values[6:] == 0.0)
    validate_post_transfer_calibration_receipt(
        result.receipt,
        spec=_match_spec(),
        boundary="immutable positive floor saturation",
    )


def test_validator_rejects_rehashed_self_consistent_zero_carrier_forgery() -> None:
    values = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    reference = _mask(len(values), 0, 1, 2, 3, 4)
    recipient = ~reference
    result = calibrate_post_transfer_values(
        values,
        np.ones(len(values), dtype=np.float64),
        np.arange(len(values)),
        spec=_preserve_spec(),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )
    forged = {**result.receipt, "carrier": dict(result.receipt["carrier"])}
    for key in (
        "reference_positive_mass",
        "reference_positive_share",
        "target_positive_mass",
        "before_positive_mass",
        "before_positive_share",
        "after_positive_mass",
        "after_positive_share",
        "residual_after_minus_target",
        "absolute_residual",
    ):
        forged["carrier"][key] = 0.0
    payload = dict(forged)
    payload.pop("sha256")
    forged["sha256"] = _canonical_sha256(payload)

    with pytest.raises(ValueError, match="carrier relationships are invalid"):
        validate_post_transfer_calibration_receipt(
            forged,
            spec=_preserve_spec(),
            boundary="zero carrier forgery",
        )


def test_validator_rejects_rehashed_stripped_amount_evidence() -> None:
    values = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    reference = _mask(len(values), 0, 1, 2, 3, 4)
    recipient = ~reference
    result = calibrate_post_transfer_values(
        values,
        np.ones(len(values), dtype=np.float64),
        np.arange(len(values)),
        spec=_preserve_spec(),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )
    forged = {
        **result.receipt,
        "amount": {
            key: result.receipt["amount"][key]
            for key in (
                "donor_support_violations",
                "status",
                "exact_anchor_count",
                "anchor_conflicts",
                "unanchored_quantiles",
            )
        },
    }
    payload = dict(forged)
    payload.pop("sha256")
    forged["sha256"] = _canonical_sha256(payload)

    with pytest.raises(ValueError, match="amount schema is invalid"):
        validate_post_transfer_calibration_receipt(
            forged,
            spec=_preserve_spec(),
            boundary="stripped amount forgery",
        )


def test_validator_rejects_rehashed_verification_contract_tampering() -> None:
    values = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    reference = _mask(len(values), 0, 1, 2, 3, 4)
    recipient = ~reference
    result = calibrate_post_transfer_values(
        values,
        np.ones(len(values), dtype=np.float64),
        np.arange(len(values)),
        spec=_preserve_spec(),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )
    forged = {
        **result.receipt,
        "verification_contract": dict(result.receipt["verification_contract"]),
    }
    forged["verification_contract"]["terminal_pre_state_replay"] = True
    payload = dict(forged)
    payload.pop("sha256")
    forged["sha256"] = _canonical_sha256(payload)

    with pytest.raises(ValueError, match="policy/spec binding is invalid"):
        validate_post_transfer_calibration_receipt(
            forged,
            spec=_preserve_spec(),
            boundary="forged verification boundary",
        )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    (
        ("carrier", "carrier evidence is incomplete"),
        ("weights", "weight evidence is absent"),
        ("scope_count", "scope evidence is incomplete"),
    ),
)
def test_validator_rejects_rehashed_stripped_core_evidence(
    mutation: str,
    error_match: str,
) -> None:
    values = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    reference = _mask(len(values), 0, 1, 2, 3, 4)
    recipient = ~reference
    result = calibrate_post_transfer_values(
        values,
        np.asarray([2.0, 3.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
        np.arange(len(values)),
        spec=_preserve_spec(),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )
    stripped = {**result.receipt, "scope": dict(result.receipt["scope"])}
    if mutation == "scope_count":
        stripped["scope"].pop("reference_rows")
    else:
        stripped.pop(mutation)
    payload = dict(stripped)
    payload.pop("sha256")
    stripped["sha256"] = _canonical_sha256(payload)

    with pytest.raises(ValueError, match=error_match):
        validate_post_transfer_calibration_receipt(
            stripped,
            spec=_preserve_spec(),
            boundary=f"stripped {mutation} regression",
        )


def test_validator_rejects_same_count_different_mask_context_transplant() -> None:
    spec = _preserve_spec()
    values_a = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    values_b = np.asarray([11.0, 1.0, 21.0, 2.0, 31.0, 3.0, 41.0, 4.0, 51.0, 5.0])
    reference_a = _mask(len(values_a), 0, 1, 2, 3, 4)
    reference_b = _mask(len(values_b), 0, 2, 4, 6, 8)
    recipient_a = ~reference_a
    recipient_b = ~reference_b
    result_a = calibrate_post_transfer_values(
        values_a,
        np.ones(len(values_a), dtype=np.float64),
        np.arange(len(values_a)),
        spec=spec,
        reference_rows=reference_a,
        recipient_rows=recipient_a,
        mutable_rows=recipient_a,
    )
    result_b = calibrate_post_transfer_values(
        values_b,
        np.full(len(values_b), 2.0, dtype=np.float64),
        np.arange(100, 100 + len(values_b)),
        spec=spec,
        reference_rows=reference_b,
        recipient_rows=recipient_b,
        mutable_rows=recipient_b,
    )

    for count_key in ("reference_rows", "recipient_rows", "mutable_rows"):
        assert (
            result_a.receipt["scope"][count_key]
            == (result_b.receipt["scope"][count_key])
        )
    expected_mask_scope = {
        key: result_b.receipt["scope"][key]
        for key in (
            "reference_rows_sha256",
            "recipient_rows_sha256",
            "mutable_rows_sha256",
        )
    }
    with pytest.raises(ValueError, match="scope does not match.*live context"):
        validate_post_transfer_calibration_receipt(
            result_a.receipt,
            spec=spec,
            boundary="same-count context transplant",
            expected_scope=expected_mask_scope,
        )
    with pytest.raises(ValueError, match="weights do not match.*live context"):
        validate_post_transfer_calibration_receipt(
            result_a.receipt,
            spec=spec,
            boundary="weight context transplant",
            expected_scope=result_a.receipt["scope"],
            expected_weights_sha256=result_b.receipt["weights"]["sha256"],
        )


def test_validator_rejects_rehashed_forged_scope_against_expected_context() -> None:
    values = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    reference = _mask(len(values), 0, 1, 2, 3, 4)
    recipient = ~reference
    result = calibrate_post_transfer_values(
        values,
        np.ones(len(values), dtype=np.float64),
        np.arange(len(values)),
        spec=_preserve_spec(),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )
    expected_scope = dict(result.receipt["scope"])
    forged = {**result.receipt, "scope": dict(result.receipt["scope"])}
    forged["scope"]["reference_rows_sha256"] = "0" * 64
    payload = dict(forged)
    payload.pop("sha256")
    forged["sha256"] = _canonical_sha256(payload)

    with pytest.raises(ValueError, match="scope does not match.*live context"):
        validate_post_transfer_calibration_receipt(
            forged,
            spec=_preserve_spec(),
            boundary="rehashed forged scope",
            expected_scope=expected_scope,
            expected_weights_sha256=result.receipt["weights"]["sha256"],
        )


def test_sparse_full_recipient_cdf_marks_exact_anchors_infeasible() -> None:
    values = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 999.0])
    weights = np.asarray([2.0, 3.0, 5.0, 5.0, 5.0, 20.0])
    reference = _mask(len(values), 0, 1, 2, 3, 4)
    recipient = ~reference

    result = calibrate_post_transfer_values(
        values,
        weights,
        np.arange(1, len(values) + 1),
        spec=_match_spec(),
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )

    assert result.receipt["amount"]["status"] == "infeasible_exact_anchors"
    assert result.receipt["amount"]["exact_anchor_count"] < 5
    assert result.receipt["amount"]["anchor_conflicts"]
    assert result.receipt["invariants"]["exact_quantile_anchors"] is False


def _frame_without_support_provenance() -> Frame:
    ids = np.arange(1, 5, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids + 100,
            "person_household_id": ids,
            "person_tax_unit_id": ids + 10,
            "person_spm_unit_id": ids + 20,
            "person_family_id": ids + 30,
            "person_marital_unit_id": ids + 40,
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids + 10}),
        "spm_unit": pd.DataFrame(
            {
                "spm_unit_id": ids + 20,
                "spm_unit_energy_subsidy": [0.0, 10.0, 100.0, 200.0],
            }
        ),
        "family": pd.DataFrame({"family_id": ids + 30}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids + 40}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([6.0, 4.0, 1.0, 3.0]),
                WeightKind.DESIGN,
            )
        },
        metadata={"fixture": {"provenance_blind": True}},
    )


def test_frame_wrapper_uses_resolved_entity_weights_without_provenance_columns() -> (
    None
):
    frame = _frame_without_support_provenance()
    spm_before = frame.table("spm_unit").copy(deep=True)
    reference = _mask(4, 0, 1)
    recipient = _mask(4, 2, 3)

    result = apply_post_transfer_calibration(
        frame,
        entity="spm_unit",
        family="source_operator_energy_subsidy",
        target="spm_unit_energy_subsidy",
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
    )

    assert not any("support_" in column for column in frame.table("spm_unit"))
    assert result.receipt["weights"] == {
        "sha256": result.receipt["weights"]["sha256"],
        "reference_total": 10.0,
        "recipient_total": 4.0,
        "kind": WeightKind.DESIGN.value,
    }
    output_spm = result.frame.table("spm_unit")
    assert output_spm["spm_unit_energy_subsidy"].tolist() == [0.0, 10.0, 0.0, 10.0]
    pd.testing.assert_frame_equal(frame.table("spm_unit"), spm_before)
    for entity in ("person", "household", "tax_unit", "family", "marital_unit"):
        pd.testing.assert_frame_equal(result.frame.table(entity), frame.table(entity))
    assert result.frame.metadata == frame.metadata
    assert result.frame.mass_log == frame.mass_log
    _assert_canonical_receipt(result.receipt)


def test_frame_helper_rejects_sparse_infeasible_exact_anchors() -> None:
    frame = _frame_without_support_provenance()
    reference = _mask(4, 1, 2)
    recipient = _mask(4, 3)

    with pytest.raises(ValueError, match=r"exact quantile anchors.*infeasible"):
        apply_post_transfer_calibration(
            frame,
            entity="spm_unit",
            family="source_operator_energy_subsidy",
            target="spm_unit_energy_subsidy",
            reference_rows=reference,
            recipient_rows=recipient,
            mutable_rows=recipient,
        )
