#!/usr/bin/env python3
"""Build replay-calibrated passive pass-through assignment assumptions.

The restricted PUF is a build-time input only.  This tool persists the
artifact identity, provisional bounds and midpoint, solved log-odds shift,
and seeded replay diagnostic.  The assignment runtime performs no solve and
does not open the replay artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.us_runtime.qbi_passive_passthrough import (
    build_qbi_passive_passthrough_assumptions_payload,
)
from microcosm.build.us_runtime.qbi_passive_passthrough_evidence import (
    load_qbi_passive_passthrough_resource,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/us"
DEFAULT_EVIDENCE = RESOURCE_DIR / "qbi_passive_passthrough_v1.json"
DEFAULT_OUTPUT = RESOURCE_DIR / "qbi_passive_passthrough_assumptions_v1.json"
PUF_REPLAY_ENVIRONMENT = "POPU" + "LACE_PUF_2024_H5"

_PERSON_KEYS = (
    "partnership_s_corp_income",
    "rental_income",
    "estate_income",
)
_MEMBERSHIP_KEYS = (
    "tax_unit_id",
    "household_weight",
    "person_tax_unit_id",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--puf-h5",
        type=Path,
        default=os.environ.get(PUF_REPLAY_ENVIRONMENT),
        help=(
            "Restricted full PUF H5 replay artifact. Defaults to the "
            f"{PUF_REPLAY_ENVIRONMENT} environment variable."
        ),
    )
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    if arguments.puf_h5 is None:
        parser.error(f"--puf-h5 or {PUF_REPLAY_ENVIRONMENT} is required")
    return arguments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_replay_artifact(
    path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read the exact person inputs and map tax-unit weights to people."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - the US extra supplies it.
        raise RuntimeError(
            "Building passive pass-through assumptions requires h5py; install "
            "the US extra."
        ) from error

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Restricted PUF replay artifact does not exist: {artifact_path}"
        )
    required = (*_MEMBERSHIP_KEYS, *_PERSON_KEYS)
    with h5py.File(artifact_path, mode="r") as artifact:
        missing = [key for key in required if key not in artifact]
        if missing:
            raise ValueError(
                f"Restricted PUF replay artifact is missing arrays {missing}."
            )
        values = {key: np.asarray(artifact[key][:]) for key in required}

    tax_unit_ids = np.asarray(values["tax_unit_id"])
    tax_unit_weights = np.asarray(values["household_weight"], dtype=np.float64)
    person_tax_unit_ids = np.asarray(values["person_tax_unit_id"])
    if any(
        array.ndim != 1
        for array in (tax_unit_ids, tax_unit_weights, person_tax_unit_ids)
    ):
        raise ValueError("Restricted replay membership arrays must be 1-D.")
    if len(tax_unit_ids) != len(tax_unit_weights) or len(tax_unit_ids) == 0:
        raise ValueError("Restricted replay tax-unit IDs and weights must align.")
    if len(person_tax_unit_ids) == 0:
        raise ValueError("Restricted replay person membership must be nonempty.")
    if np.any(tax_unit_ids[1:] <= tax_unit_ids[:-1]):
        raise ValueError("Restricted replay tax-unit IDs must strictly increase.")
    if not np.isfinite(tax_unit_weights).all() or np.any(tax_unit_weights < 0.0):
        raise ValueError("Restricted replay weights must be finite and nonnegative.")
    positions = np.searchsorted(tax_unit_ids, person_tax_unit_ids)
    if np.any(positions >= len(tax_unit_ids)) or not np.array_equal(
        tax_unit_ids[positions], person_tax_unit_ids
    ):
        raise ValueError("Restricted replay contains an unknown tax-unit ID.")
    person_weights = tax_unit_weights[positions]

    person_count = len(person_tax_unit_ids)
    person_values: dict[str, np.ndarray] = {}
    for key in _PERSON_KEYS:
        array = np.asarray(values[key], dtype=np.float64)
        if array.ndim != 1 or len(array) != person_count:
            raise ValueError(
                f"Restricted replay array {key!r} must have length {person_count}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Restricted replay array {key!r} must be finite.")
        person_values[key] = array

    passthrough = person_values["partnership_s_corp_income"]
    schedule_e = (
        passthrough + person_values["rental_income"] + person_values["estate_income"]
    )
    metadata = {
        "filename": artifact_path.name,
        "sha256": _sha256(artifact_path),
        "bytes": artifact_path.stat().st_size,
        "tax_unit_rows": len(tax_unit_ids),
    }
    return passthrough, schedule_e, person_weights, metadata


def build_assumptions(arguments: argparse.Namespace) -> dict[str, Any]:
    evidence_path = Path(arguments.evidence)
    evidence = load_qbi_passive_passthrough_resource(evidence_path)
    passthrough, schedule_e, weights, artifact = read_replay_artifact(arguments.puf_h5)
    return build_qbi_passive_passthrough_assumptions_payload(
        evidence=evidence,
        evidence_sha256=_sha256(evidence_path),
        partnership_s_corp_income=passthrough,
        schedule_e_income=schedule_e,
        person_weights=weights,
        replay_artifact=artifact,
    )


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(rendered + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    payload = build_assumptions(arguments)
    output = Path(arguments.output)
    _write_json(payload, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
