#!/usr/bin/env python3
"""Build the replay-calibrated QBI simulation v3 assumptions resource.

The restricted PUF is a build-time input only. This tool persists aggregate
calibration weights and solved parameters; the runtime never opens the PUF or
solves a calibration problem.

Reproduction:

    uv run python tools/build_us_qbi_v3_assumptions.py \
      --puf-h5 "$POPULACE_PUF_2024_H5"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from populace.build.us_runtime.qbi_v3_assumptions import (
    build_qbi_v3_assumptions_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = REPO_ROOT / "packages/populace-build/src/populace/build/us"
DEFAULT_V2_ASSUMPTIONS = RESOURCE_DIR / "qbi_assumptions_v2.json"
DEFAULT_EMPLOYER_STRUCTURE = RESOURCE_DIR / "qbi_employer_structure_v1.json"
DEFAULT_WAGE_CAPITAL_PRIORS = RESOURCE_DIR / "qbi_wage_capital_priors_v1.json"
DEFAULT_OUTPUT = RESOURCE_DIR / "qbi_assumptions_v3.json"

_REPLAY_PERSON_KEYS = (
    "self_employment_income",
    "farm_rent_income",
    "rental_income",
    "estate_income",
    "partnership_s_corp_income",
    "non_qualified_dividend_income",
)
_REPLAY_MEMBERSHIP_KEYS = (
    "tax_unit_id",
    "household_weight",
    "person_tax_unit_id",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--puf-h5",
        type=Path,
        default=os.environ.get("POPULACE_PUF_2024_H5"),
        help=(
            "Restricted full PUF H5 replay artifact. Defaults to POPULACE_PUF_2024_H5."
        ),
    )
    parser.add_argument(
        "--v2-assumptions",
        type=Path,
        default=DEFAULT_V2_ASSUMPTIONS,
    )
    parser.add_argument(
        "--employer-structure",
        type=Path,
        default=DEFAULT_EMPLOYER_STRUCTURE,
    )
    parser.add_argument(
        "--wage-capital-priors",
        type=Path,
        default=DEFAULT_WAGE_CAPITAL_PRIORS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args(argv)
    if args.puf_h5 is None:
        parser.error("--puf-h5 or POPULACE_PUF_2024_H5 is required")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} input does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} input must contain one JSON object.")
    return payload


def read_replay_artifact(
    path: Path | str,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Read the exact replay arrays and map tax-unit weights to person rows."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - the US extra supplies it.
        raise RuntimeError(
            "Building QBI v3 assumptions requires h5py; install the US extra."
        ) from error

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Restricted PUF replay artifact does not exist: {artifact_path}"
        )
    required_keys = (*_REPLAY_MEMBERSHIP_KEYS, *_REPLAY_PERSON_KEYS)
    with h5py.File(artifact_path, mode="r") as artifact:
        missing = [key for key in required_keys if key not in artifact]
        if missing:
            raise ValueError(
                f"Restricted PUF replay artifact is missing arrays {missing!r}."
            )
        values = {key: np.asarray(artifact[key][:]) for key in required_keys}

    tax_unit_ids = values["tax_unit_id"]
    tax_unit_weights = np.asarray(values["household_weight"], dtype=np.float64)
    person_tax_unit_ids = values["person_tax_unit_id"]
    if (
        tax_unit_ids.ndim != 1
        or tax_unit_weights.ndim != 1
        or person_tax_unit_ids.ndim != 1
    ):
        raise ValueError("Restricted replay membership arrays must be one-dimensional.")
    if len(tax_unit_ids) != len(tax_unit_weights):
        raise ValueError("Restricted replay tax-unit IDs and weights must align.")
    if len(tax_unit_ids) == 0 or len(person_tax_unit_ids) == 0:
        raise ValueError("Restricted replay membership arrays must be nonempty.")
    if np.any(tax_unit_ids[1:] <= tax_unit_ids[:-1]):
        raise ValueError("Restricted replay tax-unit IDs must be strictly increasing.")
    if np.any(~np.isfinite(tax_unit_weights)) or np.any(tax_unit_weights < 0.0):
        raise ValueError(
            "Restricted replay tax-unit weights must be finite and nonnegative."
        )

    positions = np.searchsorted(tax_unit_ids, person_tax_unit_ids)
    in_bounds = positions < len(tax_unit_ids)
    if not np.all(in_bounds):
        raise ValueError("Restricted replay contains an unknown person tax-unit ID.")
    if not np.array_equal(tax_unit_ids[positions], person_tax_unit_ids):
        raise ValueError("Restricted replay contains an unknown person tax-unit ID.")
    person_weights = tax_unit_weights[positions]

    replay_arrays = {key: np.asarray(values[key]) for key in _REPLAY_PERSON_KEYS}
    person_count = len(person_tax_unit_ids)
    for key, array in replay_arrays.items():
        if array.ndim != 1 or len(array) != person_count:
            raise ValueError(
                f"Restricted replay person array {key!r} must have "
                f"length {person_count}."
            )
    metadata = {
        "filename": artifact_path.name,
        "sha256": _sha256(artifact_path),
        "bytes": artifact_path.stat().st_size,
        "tax_unit_rows": len(tax_unit_ids),
    }
    return replay_arrays, person_weights, metadata


def build_assumptions(args: argparse.Namespace) -> dict[str, Any]:
    """Load inputs and return the deterministic v3 assumptions mapping."""

    v2_path = Path(args.v2_assumptions)
    employer_path = Path(args.employer_structure)
    wage_capital_path = Path(args.wage_capital_priors)
    replay_arrays, person_weights, replay_metadata = read_replay_artifact(
        Path(args.puf_h5)
    )
    v2_payload = _load_json(v2_path, label="QBI v2 assumptions")
    employer_resource = _load_json(
        employer_path,
        label="QBI employer structure",
    )
    wage_capital_resource = _load_json(
        wage_capital_path,
        label="QBI wage/capital priors",
    )
    return build_qbi_v3_assumptions_payload(
        v2_payload=v2_payload,
        employer_resource=employer_resource,
        wage_capital_resource=wage_capital_resource,
        replay_arrays=replay_arrays,
        person_weights=person_weights,
        replay_artifact=replay_metadata,
        resource_sha256={
            "v2_assumptions": _sha256(v2_path),
            "employer_structure": _sha256(employer_path),
            "wage_capital": _sha256(wage_capital_path),
        },
    )


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    path.write_text(rendered + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_assumptions(args)
    output = Path(args.output)
    _write_json(payload, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
