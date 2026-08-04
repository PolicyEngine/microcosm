"""Measure UK weighted-integrity baselines for the #609 threshold pass.

Increment 4 of the UK parity plan owes a measurement pass before any
threshold is written: weighted per-column totals for the certified compact,
the pinned eFRS incumbent, and the current staging candidate, plus top-k
weighted mass share and carrier count for every declared QRF output. This
tool produces those numbers so they can be posted on #578 and the gate
boundaries set at the measured edge with no discretionary headroom — the
same discipline that pinned ``UK_MAX_TO_MEDIAN_WEIGHT_RATIO``.

It is a diagnostic recorder only: it never gates, and release builds do not
run it. Each ``--h5`` must be a UK national single-year artifact (person,
benunit, household, time_period tables). For the pinned eFRS incumbent use
``tools/build_uk_efrs_parity_reference.py --emit-weighted-totals`` instead,
which verifies the pinned sha before reading.

The output contains weighted aggregates derived from licensed UKDS
microdata. Treat it under the UKDS End User Licence: post or commit derived
totals only once that disclosure class is confirmed (#609 open question).

Usage:

    python tools/measure_uk_weighted_integrity_baselines.py \
      --h5 certified_compact=/path/to/populace_uk_2023.h5 \
      --h5 staging=/path/to/staging.h5 \
      --output uk_weighted_integrity_baselines.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from populace.build.uk_runtime.hmrc_source_contract import (
    uk_hmrc_weighted_qrf_output_columns,
)
from populace.build.uk_runtime.national_build import load_uk_national_dataset
from populace.build.uk_runtime.weighted_integrity import (
    uk_dataset_input_mass_totals,
    uk_qrf_tail_concentration_columns,
)

DEFAULT_TOP_K_GRID = (10, 100, 500, 1000)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--h5",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help=(
            "A labelled UK national H5 to measure, e.g. "
            "certified_compact=/path/populace_uk_2023.h5. Repeatable."
        ),
    )
    parser.add_argument(
        "--top-k",
        default=",".join(str(k) for k in DEFAULT_TOP_K_GRID),
        help=(
            "Comma-separated tail sizes for the concentration measurement "
            f"(default: {','.join(str(k) for k in DEFAULT_TOP_K_GRID)})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the measurement JSON.",
    )
    return parser.parse_args()


def _labelled_paths(raw: list[str]) -> dict[str, Path]:
    labelled: dict[str, Path] = {}
    for entry in raw:
        label, separator, path = entry.partition("=")
        if not separator or not label.strip() or not path.strip():
            raise SystemExit(f"error: --h5 must be LABEL=PATH, got {entry!r}.")
        if label in labelled:
            raise SystemExit(f"error: duplicate --h5 label {label!r}.")
        labelled[label] = Path(path)
    return labelled


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail_measurements(
    values: np.ndarray,
    weights: np.ndarray,
    top_k_grid: tuple[int, ...],
) -> dict[str, object]:
    finite = np.isfinite(values) & np.isfinite(weights)
    mass = np.abs(values[finite]) * weights[finite]
    mass = mass[mass > 0.0]
    carriers = int(mass.size)
    total = float(mass.sum())
    records = int(values.size)
    top_shares: dict[str, float | None] = {}
    for top_k in top_k_grid:
        if carriers == 0 or total == 0.0:
            top_shares[str(top_k)] = None
        elif carriers <= top_k:
            top_shares[str(top_k)] = 1.0
        else:
            tail = float(np.partition(mass, -top_k)[-top_k:].sum())
            top_shares[str(top_k)] = tail / total
    return {
        "records": records,
        "carriers": carriers,
        "nonzero_share": (carriers / records) if records else 0.0,
        "total_weighted_abs_mass": total,
        "max_abs_value": float(np.abs(values[finite]).max()) if finite.any() else 0.0,
        "top_shares": top_shares,
    }


def _measure(path: Path, top_k_grid: tuple[int, ...]) -> dict[str, object]:
    dataset = load_uk_national_dataset(path)
    totals = uk_dataset_input_mass_totals(dataset)
    declared = uk_hmrc_weighted_qrf_output_columns()
    values, weights, surface = uk_qrf_tail_concentration_columns(
        dataset,
        output_columns=declared,
    )
    qrf_tail = {
        column: _tail_measurements(values[column], weights[column], top_k_grid)
        for column in sorted(values)
    }
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "entity_rows": {
            "person": len(dataset.person),
            "benunit": len(dataset.benunit),
            "household": len(dataset.household),
        },
        "input_mass_totals": dict(sorted(totals.items())),
        "qrf_surface": surface,
        "qrf_tail": qrf_tail,
        "input_mass_reference_template": {
            "schema_version": 1,
            "identity": {
                "filename": path.name,
                "revision": "<fill: pinned revision naming these bytes>",
                "sha256": _sha256(path),
                "vintage": "<fill: source vintage, e.g. 2023_24>",
            },
            "totals": dict(sorted(totals.items())),
        },
    }


def main() -> int:
    args = _parse_args()
    try:
        top_k_grid = tuple(int(part) for part in args.top_k.split(",") if part)
    except ValueError:
        raise SystemExit(
            f"error: --top-k must be integers, got {args.top_k!r}."
        ) from None
    if not top_k_grid or any(k < 1 for k in top_k_grid):
        raise SystemExit("error: --top-k values must be positive integers.")
    measurements = {
        label: _measure(path, top_k_grid)
        for label, path in _labelled_paths(args.h5).items()
    }
    payload = {
        "schema_version": 1,
        "purpose": (
            "UK weighted-integrity baseline measurement for the #609 "
            "threshold adjudication (#578). Diagnostic recorder only — "
            "never a gate."
        ),
        "top_k_grid": list(top_k_grid),
        "artifacts": measurements,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "UKDS licensing note: this file contains weighted aggregates derived "
        "from licensed microdata; confirm the EUL disclosure class before "
        "posting or committing it (#609).",
        file=sys.stderr,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
