"""Audit checkpointed pkg3 child-support receipts without starting a build.

The harness pins the assembled Frame and both child-support target checkpoints,
reconstructs only their native clone-0 post-transfer vectors, and runs the live
calibration kernel plus strict receipt validator.  It performs no fit, DAG
execution, artifact write, or build.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from reproduce_us_post_transfer_weeks_checkpoint import (
    _ASSEMBLED_FILENAME,
    _ASSEMBLED_SHA256,
    _BANK_IDENTITY,
    _FULL_POOL_CLONE_COUNT,
    _load_target_draw,
    _require_file_sha256,
)

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime.post_transfer_calibration import (
    calibrate_post_transfer_values,
    post_transfer_calibration_spec_for_target,
    validate_post_transfer_calibration_receipt,
)

_EXPECTED_NATIVE_ROWS = 38_604
_EXPECTED_REFERENCE_ROWS = 4_311
_EXPECTED_RECIPIENT_ROWS = 34_293
_EXPECTED_RECIPIENT_TOTAL = 79_926_522.10879111
_EXPECTED_INVALID_MAXIMUM = 79_926_522.10879174
_EXPECTED_MAXIMUM_RELATIONSHIP = "maximum_attainable_mass <= recipient_total"


@dataclass(frozen=True)
class _CheckpointCase:
    target: str
    source_column: str
    relative_path: Path
    file_sha256: str
    identity_sha256: str
    raw_sha256: str


_CASES = (
    _CheckpointCase(
        target="child_support_expense",
        source_column="CHSP_VAL",
        relative_path=Path(
            "late_producer_dag/person/source_operator_child_support/targets/"
            "000__child_support_expense.h5"
        ),
        file_sha256=(
            "d119075e19fb767f3d8d24c7c0149d0df1ed963774a4b93d96974a72b3ac9bfe"
        ),
        identity_sha256=(
            "41e3a6e3877fda23107b27bcd85aa6dd95e0f341d1e4b079defa6847f90b4cab"
        ),
        raw_sha256=("8b2845aff0aa0695d98ae30828523bf6bca9c5d4ed5d2d91d2d1a636bb917600"),
    ),
    _CheckpointCase(
        target="child_support_received",
        source_column="CSP_VAL",
        relative_path=Path(
            "late_producer_dag/person/source_operator_child_support/targets/"
            "001__child_support_received.h5"
        ),
        file_sha256=(
            "66120896d5793f3d737f9ffac2058e2196992e357f8d869f4b31b259d041b3aa"
        ),
        identity_sha256=(
            "41e3a6e3877fda23107b27bcd85aa6dd95e0f341d1e4b079defa6847f90b4cab"
        ),
        raw_sha256=("ea7f2eebb430b654acc639ef6ee6ed482207ffd74d54ba3a47cb55056813a381"),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-stage-root",
        required=True,
        type=Path,
        help="Path to the SHA-addressed stacked stage containing the assembled H5.",
    )
    parser.add_argument(
        "--expect",
        required=True,
        choices=("invalid", "valid"),
        help="Assert that both checkpointed child-support receipts have this state.",
    )
    return parser.parse_args()


def _audit_case(
    case: _CheckpointCase,
    *,
    bank_root: Path,
    person: pd.DataFrame,
    weights: np.ndarray,
    entity_ids: np.ndarray,
    reference: np.ndarray,
    recipient: np.ndarray,
) -> dict[str, object]:
    raw_draw = _load_target_draw(
        bank_root / case.relative_path,
        file_sha256=case.file_sha256,
        identity_sha256=case.identity_sha256,
        raw_sha256=case.raw_sha256,
    )
    native_draw = raw_draw[: len(person)]
    if (
        not np.isnan(native_draw[reference]).all()
        or not np.isfinite(native_draw[recipient]).all()
    ):
        raise ValueError(
            f"Pinned {case.target} draw no longer has null ASEC and finite ACS rows."
        )
    direct = pd.to_numeric(person[case.source_column], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(direct[reference]).all() or (direct[reference] < 0.0).any():
        raise ValueError(f"Pinned {case.source_column} reference values are invalid.")
    values = np.where(reference, direct, native_draw).astype(np.float64)
    mutable = recipient & np.isfinite(values)
    if int(mutable.sum()) != _EXPECTED_RECIPIENT_ROWS:
        raise ValueError(f"Pinned {case.target} mutable row count changed.")

    spec = post_transfer_calibration_spec_for_target(
        entity="person",
        target=case.target,
    )
    result = calibrate_post_transfer_values(
        values,
        weights,
        entity_ids,
        spec=spec,
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=mutable,
    )
    carrier = result.receipt["carrier"]
    capacity = carrier["capacity"]
    selection = carrier["selection"]
    recipient_total = float(result.receipt["weights"]["recipient_total"])
    partition_mass = float(
        capacity["fixed_positive_mass"]
        + capacity["allowed_positive_mass_before"]
        + capacity["addition_candidate_mass"]
    )
    maximum = float(capacity["maximum_attainable_mass"])
    candidate_mass = float(selection["candidate_mass"])
    upper_mass = float(selection["upper_prefix_mass"])
    failed_relationships: list[str] = []
    if maximum > recipient_total:
        failed_relationships.append("maximum_attainable_mass <= recipient_total")
    if upper_mass > candidate_mass:
        failed_relationships.append("upper_prefix_mass <= candidate_mass")

    validation_error: str | None = None
    try:
        validate_post_transfer_calibration_receipt(
            result.receipt,
            spec=spec,
            boundary=f"Frame post-transfer calibration {spec.key}",
        )
    except ValueError as error:
        validation_error = str(error)

    return {
        "artifact_sha256": {
            "file": case.file_sha256,
            "identity": case.identity_sha256,
            "raw_draw": case.raw_sha256,
        },
        "target": case.target,
        "reference_rows": int(reference.sum()),
        "recipient_rows": int(recipient.sum()),
        "mutable_rows": int(mutable.sum()),
        "reference_positive_mass": carrier["reference_positive_mass"],
        "target_positive_mass": carrier["target_positive_mass"],
        "allowed_positive_rows": capacity["allowed_positive_rows_before"],
        "allowed_positive_mass": capacity["allowed_positive_mass_before"],
        "addition_candidate_rows": capacity["addition_candidate_rows"],
        "addition_candidate_mass": capacity["addition_candidate_mass"],
        "partition_endpoint_mass": partition_mass,
        "maximum_attainable_mass": maximum,
        "recipient_total": recipient_total,
        "maximum_minus_recipient_total": maximum - recipient_total,
        "partition_minus_maximum": partition_mass - maximum,
        "selection_action": selection["action"],
        "selected_rows": selection["selected_rows"],
        "selected_mass": selection["selected_mass"],
        "lower_prefix_mass": selection["lower_prefix_mass"],
        "upper_prefix_mass": upper_mass,
        "failed_relationships": failed_relationships,
        "validation_error": validation_error,
        "receipt_valid": validation_error is None,
    }


def audit_checkpoints(checkpoint_stage_root: Path) -> dict[str, object]:
    """Return strict receipt results for both pinned child-support targets."""

    stage_root = checkpoint_stage_root.resolve()
    assembled_path = stage_root / _ASSEMBLED_FILENAME
    _require_file_sha256(assembled_path, _ASSEMBLED_SHA256)
    frame = load_frame_checkpoint(assembled_path).frame
    person = frame.table("person")
    if len(person) != _EXPECTED_NATIVE_ROWS:
        raise ValueError("Pinned assembled native person row count changed.")

    channels = person["person_support_channel"].astype(str)
    clone_index = pd.to_numeric(
        person["person_support_clone_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    reference = channels.eq("asec").to_numpy(dtype=bool) & (clone_index == 0)
    recipient = channels.eq("acs").to_numpy(dtype=bool) & (clone_index == 0)
    if (
        int(reference.sum()) != _EXPECTED_REFERENCE_ROWS
        or int(recipient.sum()) != _EXPECTED_RECIPIENT_ROWS
        or (reference & recipient).any()
        or not np.all(reference | recipient)
    ):
        raise ValueError("Pinned assembled clone-0 support masks changed.")

    weights = (
        np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
        / _FULL_POOL_CLONE_COUNT
    )
    entity_ids = person[frame.schema.entity_id_column("person")].to_numpy(copy=False)
    bank_root = stage_root / "acs-transfer" / _BANK_IDENTITY
    results = [
        _audit_case(
            case,
            bank_root=bank_root,
            person=person,
            weights=weights,
            entity_ids=entity_ids,
            reference=reference,
            recipient=recipient,
        )
        for case in _CASES
    ]
    return {
        "assembled_sha256": _ASSEMBLED_SHA256,
        "targets": results,
        "all_receipts_valid": all(result["receipt_valid"] for result in results),
    }


def main() -> int:
    args = _parse_args()
    audit = audit_checkpoints(args.checkpoint_stage_root)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    targets = audit["targets"]
    expected_valid = args.expect == "valid"
    states_match = all(result["receipt_valid"] is expected_valid for result in targets)
    if not states_match:
        raise SystemExit(
            f"Expected each child-support receipt to be {args.expect}, but got "
            f"{[result['receipt_valid'] for result in targets]}."
        )
    if expected_valid:
        proof_matches = all(
            not result["failed_relationships"]
            and result["validation_error"] is None
            and result["recipient_total"] == _EXPECTED_RECIPIENT_TOTAL
            and result["maximum_attainable_mass"] == _EXPECTED_RECIPIENT_TOTAL
            and result["partition_endpoint_mass"] == _EXPECTED_INVALID_MAXIMUM
            for result in targets
        )
    else:
        proof_matches = all(
            result["failed_relationships"] == [_EXPECTED_MAXIMUM_RELATIONSHIP]
            and result["validation_error"]
            == (
                "Frame post-transfer calibration "
                f"person/source_operator_child_support/{result['target']}: "
                "match-reference carrier capacity relationships are invalid."
            )
            and result["recipient_total"] == _EXPECTED_RECIPIENT_TOTAL
            and result["maximum_attainable_mass"] == _EXPECTED_INVALID_MAXIMUM
            for result in targets
        )
    if not proof_matches:
        raise SystemExit(
            f"The {args.expect} receipts no longer match the pinned relationship proof."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
