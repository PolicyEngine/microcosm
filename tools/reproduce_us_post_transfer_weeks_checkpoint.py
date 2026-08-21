"""Replay the pkg3 weeks receipt from checkpoints without starting a build.

The harness is intentionally pinned to the preserved 1% failure artifacts. It
loads the assembled frame and two identity-bound ACS target checkpoints, then
reconstructs only the clone-0 vectors consumed by the post-transfer kernel.
It never fits a model, executes the late-producer DAG, or writes an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime.post_transfer_calibration import (
    calibrate_post_transfer_values,
    post_transfer_calibration_spec_for_target,
    validate_post_transfer_calibration_receipt,
)

_ASSEMBLED_FILENAME = "assembled.checkpoint.h5"
_ASSEMBLED_SHA256 = "5ce1815fc44dc43c7c24ccf27526852b8f1bddbdfe371255410a22f9b56ac015"
_BANK_IDENTITY = "65c58b1c3fb282cbf7a814dad32481c1fe2de977ad747fe6576bdfc40b712019"
_UC_RELATIVE_PATH = Path("asec_survey_to_acs/targets/019__unemployment_compensation.h5")
_UC_FILE_SHA256 = "dc6637936ed4bd0322d38eaa3a4920fd137565f314387db3b3fdc7dfd6bc3086"
_UC_IDENTITY_SHA256 = "708722093ca610426175998d50bbb6663585b07ffef912899f17adc90520f51f"
_UC_RAW_SHA256 = "e32d1559668e10b24abad8e1d639e4dbade964a712925bfe8f56d3136b839840"
_WEEKS_RELATIVE_PATH = Path(
    "late_producer_dag/person/source_operator_weeks_unemployed/targets/"
    "000__weeks_unemployed.h5"
)
_WEEKS_FILE_SHA256 = "898397733aa3e5d8ec7d6679cb16a0504e826e25d23ca2c788f4397e0e061a43"
_WEEKS_IDENTITY_SHA256 = (
    "d0d554ba05045e39a07f0f9515c83bbf754f067df12b8247f4bf3866162c4bdd"
)
_WEEKS_RAW_SHA256 = "0214c8dcbc118676336069b906a07ee6145f2178542b6c5b4fb5899ad62d09f3"
_EXPECTED_NATIVE_ROWS = 38_604
_EXPECTED_REFERENCE_ROWS = 4_311
_EXPECTED_RECIPIENT_ROWS = 34_293
_EXPECTED_ALLOWED_ROWS = 32
_EXPECTED_REFERENCE_POSITIVE_ROWS = 134
_EXPECTED_RECIPIENT_POSITIVE_ROWS = 24
_FULL_POOL_CLONE_COUNT = 2
_EXPECTED_ERROR = (
    "Frame post-transfer calibration "
    "person/source_operator_weeks_unemployed/weeks_unemployed: "
    "match-reference carrier capacity relationships are invalid."
)
_EXPECTED_INVALID_CANDIDATE_MASS = 85_676.23791782455
_EXPECTED_PREFIX_MASS = 85_676.23791782456
_EXPECTED_VALID_MAXIMUM_MASS = 85_676.23791782453


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
        help="Assert the current kernel either reproduces or fixes the receipt.",
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file_sha256(path: Path, expected: str) -> None:
    observed = _file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"Checkpoint SHA-256 changed for {path}: {observed} != {expected}."
        )


def _load_target_draw(
    path: Path,
    *,
    file_sha256: str,
    identity_sha256: str,
    raw_sha256: str,
) -> np.ndarray:
    _require_file_sha256(path, file_sha256)
    with h5py.File(path, mode="r") as h5:
        metadata = json.loads(bytes(h5["metadata_json"][...]).decode("utf-8"))
        raw_bits = np.asarray(h5["raw_draw_bits"], dtype="<u8")
    observed_raw_sha256 = hashlib.sha256(
        np.ascontiguousarray(raw_bits).tobytes(order="C")
    ).hexdigest()
    if metadata.get("identity_sha256") != identity_sha256:
        raise ValueError(f"Target-checkpoint identity changed for {path}.")
    if metadata.get("raw_draw_sha256") != raw_sha256:
        raise ValueError(f"Target-checkpoint declared raw digest changed for {path}.")
    if observed_raw_sha256 != raw_sha256:
        raise ValueError(f"Target-checkpoint raw draw changed for {path}.")
    if metadata.get("recipient_rows") != len(raw_bits):
        raise ValueError(f"Target-checkpoint row count is invalid for {path}.")
    return raw_bits.view("<f8").astype(np.float64, copy=True)


def replay_checkpoint(checkpoint_stage_root: Path) -> dict[str, object]:
    """Return the exact capacity evidence and validator outcome for the replay."""

    stage_root = checkpoint_stage_root.resolve()
    assembled_path = stage_root / _ASSEMBLED_FILENAME
    _require_file_sha256(assembled_path, _ASSEMBLED_SHA256)
    frame = load_frame_checkpoint(assembled_path).frame
    person = frame.table("person")
    if len(person) != _EXPECTED_NATIVE_ROWS:
        raise ValueError(
            f"Assembled person rows changed: {len(person)} != {_EXPECTED_NATIVE_ROWS}."
        )

    bank_root = stage_root / "acs-transfer" / _BANK_IDENTITY
    unemployment = _load_target_draw(
        bank_root / _UC_RELATIVE_PATH,
        file_sha256=_UC_FILE_SHA256,
        identity_sha256=_UC_IDENTITY_SHA256,
        raw_sha256=_UC_RAW_SHA256,
    )
    weeks_full = _load_target_draw(
        bank_root / _WEEKS_RELATIVE_PATH,
        file_sha256=_WEEKS_FILE_SHA256,
        identity_sha256=_WEEKS_IDENTITY_SHA256,
        raw_sha256=_WEEKS_RAW_SHA256,
    )
    if len(unemployment) != len(person) or len(weeks_full) < len(person):
        raise ValueError("Target checkpoints do not cover the assembled native rows.")

    channels = person["person_support_channel"].astype(str)
    clone_index = pd.to_numeric(
        person["person_support_clone_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    reference = channels.eq("asec").to_numpy(dtype=bool) & (clone_index == 0)
    recipient = channels.eq("acs").to_numpy(dtype=bool) & (clone_index == 0)
    if (reference & recipient).any() or not np.all(reference | recipient):
        raise ValueError("Assembled clone-0 ASEC/ACS masks are not exhaustive.")

    direct_weeks = pd.to_numeric(person["LKWEEKS"], errors="raise").to_numpy(
        dtype=np.float64
    )
    direct_weeks = np.where(direct_weeks == -1.0, 0.0, direct_weeks)
    values = np.where(reference, direct_weeks, weeks_full[: len(person)]).astype(
        np.float64
    )
    mutable = recipient & np.isfinite(values)
    allowed = mutable & (unemployment > 0.0)
    weights = (
        np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
        / _FULL_POOL_CLONE_COUNT
    )
    entity_ids = person[frame.schema.entity_id_column("person")].to_numpy(copy=False)

    observed_counts = {
        "reference_rows": int(reference.sum()),
        "recipient_rows": int(recipient.sum()),
        "mutable_rows": int(mutable.sum()),
        "allowed_addition_rows": int(allowed.sum()),
        "reference_positive_rows": int((reference & (values > 0.0)).sum()),
        "recipient_positive_rows_before": int((recipient & (values > 0.0)).sum()),
        "disallowed_positive_rows_before": int(
            (recipient & (values > 0.0) & ~allowed).sum()
        ),
    }
    expected_counts = {
        "reference_rows": _EXPECTED_REFERENCE_ROWS,
        "recipient_rows": _EXPECTED_RECIPIENT_ROWS,
        "mutable_rows": _EXPECTED_RECIPIENT_ROWS,
        "allowed_addition_rows": _EXPECTED_ALLOWED_ROWS,
        "reference_positive_rows": _EXPECTED_REFERENCE_POSITIVE_ROWS,
        "recipient_positive_rows_before": _EXPECTED_RECIPIENT_POSITIVE_ROWS,
        "disallowed_positive_rows_before": _EXPECTED_RECIPIENT_POSITIVE_ROWS,
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"Pinned replay row counts changed: {observed_counts} != {expected_counts}."
        )

    spec = post_transfer_calibration_spec_for_target(
        entity="person", target="weeks_unemployed"
    )
    result = calibrate_post_transfer_values(
        values,
        weights,
        entity_ids,
        spec=spec,
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=mutable,
        allowed_carrier_rows=allowed,
        addition_candidate_rows=allowed,
    )
    carrier = result.receipt["carrier"]
    capacity = carrier["capacity"]
    selection = carrier["selection"]
    candidate_mass = float(capacity["addition_candidate_mass"])
    upper_mass = float(selection["upper_prefix_mass"])
    failed_relationships = []
    if not 0.0 <= float(selection["lower_prefix_mass"]) <= upper_mass:
        failed_relationships.append("0 <= lower_prefix_mass <= upper_prefix_mass")
    if not upper_mass <= candidate_mass:
        failed_relationships.append("upper_prefix_mass <= addition_candidate_mass")

    validation_error: str | None = None
    try:
        validate_post_transfer_calibration_receipt(
            result.receipt,
            spec=spec,
            boundary=f"Frame post-transfer calibration {spec.key}",
        )
    except ValueError as error:
        validation_error = str(error)
        if validation_error != _EXPECTED_ERROR:
            raise

    return {
        "artifact_sha256": {
            "assembled": _ASSEMBLED_SHA256,
            "unemployment_compensation": _UC_FILE_SHA256,
            "weeks_unemployed": _WEEKS_FILE_SHA256,
        },
        "counts": observed_counts,
        "weights": result.receipt["weights"],
        "reference_positive_mass": carrier["reference_positive_mass"],
        "target_positive_mass": carrier["target_positive_mass"],
        "before_positive_mass": carrier["before_positive_mass"],
        "after_positive_mass": carrier["after_positive_mass"],
        "addition_candidate_mass": candidate_mass,
        "maximum_attainable_mass": capacity["maximum_attainable_mass"],
        "selected_prefix_mass": selection["selected_mass"],
        "lower_prefix_mass": selection["lower_prefix_mass"],
        "upper_prefix_mass": upper_mass,
        "upper_minus_candidate_mass": upper_mass - candidate_mass,
        "failed_relationships": failed_relationships,
        "validation_error": validation_error,
        "receipt_valid": validation_error is None,
    }


def main() -> int:
    args = _parse_args()
    replay = replay_checkpoint(args.checkpoint_stage_root)
    print(json.dumps(replay, indent=2, sort_keys=True, allow_nan=False))
    observed = "valid" if replay["receipt_valid"] else "invalid"
    if observed != args.expect:
        raise SystemExit(
            f"Expected {args.expect} receipt, but replay produced {observed}."
        )
    if args.expect == "invalid" and (
        replay["failed_relationships"]
        != ["upper_prefix_mass <= addition_candidate_mass"]
        or replay["validation_error"] != _EXPECTED_ERROR
        or replay["addition_candidate_mass"] != _EXPECTED_INVALID_CANDIDATE_MASS
        or replay["upper_prefix_mass"] != _EXPECTED_PREFIX_MASS
    ):
        raise SystemExit("Invalid replay no longer matches the pinned weeks failure.")
    if args.expect == "valid" and (
        replay["failed_relationships"]
        or replay["validation_error"] is not None
        or replay["addition_candidate_mass"] != _EXPECTED_PREFIX_MASS
        or replay["upper_prefix_mass"] != _EXPECTED_PREFIX_MASS
        or replay["maximum_attainable_mass"] != _EXPECTED_VALID_MAXIMUM_MASS
    ):
        raise SystemExit("Valid replay no longer matches the pinned weeks repair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
