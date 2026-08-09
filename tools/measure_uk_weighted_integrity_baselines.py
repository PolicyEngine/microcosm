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

The output is disclosure-controlled for publication. UKDS End User Licence
CD137 v16.00 clause 8 requires adherence to the statistical disclosure
control standards in CD171-ResearchDataHandling for "any outputs I produce
and publish", and §5.2.1 of that guide sets the rules this tool applies: no
unit-record values (maxima and minima are named explicitly), and nothing
reported from fewer cases than the minimum count. Per-column weighted totals
are population aggregates and pass those rules; the tail statistics are the
(n, k) dominance measure, so thin columns are suppressed. What the licence
does *not* waive: citation and acknowledgement (clauses 11 and 12), and each
study's Special Conditions (clause 3) — check whether a study requires a
threshold of 30 before posting.

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

from microcosm.build.uk_runtime.hmrc_source_contract import (
    uk_hmrc_weighted_qrf_output_columns,
)
from microcosm.build.uk_runtime.national_build import load_uk_national_frame
from microcosm.build.uk_runtime.weighted_integrity import (
    uk_dataset_input_mass_totals,
    uk_qrf_tail_concentration_columns,
)
from microcosm.frame import engine_tables

DEFAULT_TOP_K_GRID = (10, 100, 500, 1000)
# CD171-ResearchDataHandling §5.2.1: cells based on one or two cases are never
# reportable (minimum threshold of 3), and 10 is advised where several outputs
# come from the same source — which is exactly this file. Some studies impose
# 30; raise this with --sdc-minimum-count when a study's Special Conditions
# (EUL clause 3) require it.
DEFAULT_SDC_MINIMUM_COUNT = 10


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
        "--sdc-minimum-count",
        type=int,
        default=DEFAULT_SDC_MINIMUM_COUNT,
        help=(
            "Minimum carrier count before a column's concentration "
            "statistics are reportable, per CD171-ResearchDataHandling "
            f"§5.2.1 (default: {DEFAULT_SDC_MINIMUM_COUNT}; raise to 30 where "
            "a study's Special Conditions require it). Columns below it are "
            "suppressed, and --top-k values below it are refused."
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
    *,
    minimum_count: int = DEFAULT_SDC_MINIMUM_COUNT,
) -> dict[str, object]:
    """Concentration statistics under the UKDS output-disclosure standard.

    `CD171-ResearchDataHandling` §5.2.1 governs what may be published from
    Safeguarded data: never report cells based on one or two cases (minimum
    threshold of 3, with 10 advised against secondary disclosure), and "any
    output that refers to unit records, e.g. a maximum or minimum value, must
    be avoided".

    A per-column weighted total aggregates every carrier and is a population
    aggregate, so it is reported unconditionally. Tail concentration is
    different: it is the `(n, k)` dominance statistic, and at small `k` the
    share times the total recovers the mean of a handful of records. So a
    column with fewer than `minimum_count` carriers reports no shares at all,
    and `top_k` values below `minimum_count` are refused. No maximum,
    minimum, or other unit-record value is ever emitted.
    """

    finite = np.isfinite(values) & np.isfinite(weights)
    mass = np.abs(values[finite]) * weights[finite]
    mass = mass[mass > 0.0]
    carriers = int(mass.size)
    total = float(mass.sum())
    records = int(values.size)
    suppressed = carriers > 0 and carriers < minimum_count
    top_shares: dict[str, float | None] = {}
    if not suppressed:
        for top_k in top_k_grid:
            if carriers == 0 or total == 0.0:
                top_shares[str(top_k)] = None
            elif carriers <= top_k:
                # Every carrier is in the tail: the share is 1.0 by
                # construction and says nothing about individual records.
                top_shares[str(top_k)] = 1.0
            else:
                tail = float(np.partition(mass, -top_k)[-top_k:].sum())
                top_shares[str(top_k)] = tail / total
    return {
        "records": records,
        # A carrier count below the threshold is itself a small-cell
        # frequency, so it is withheld rather than reported.
        "carriers": None if suppressed else carriers,
        "nonzero_share": (
            None if suppressed else ((carriers / records) if records else 0.0)
        ),
        "total_weighted_abs_mass": total,
        "top_shares": top_shares,
        "disclosure_suppressed": suppressed,
    }


def _measure(
    path: Path,
    top_k_grid: tuple[int, ...],
    *,
    minimum_count: int,
) -> dict[str, object]:
    frame, _provenance = load_uk_national_frame(path)
    # The gate helpers are deliberately duck-typed (#611 owns their Frame
    # typing); the materialized mapping satisfies them today.
    tables = engine_tables(frame)
    totals = uk_dataset_input_mass_totals(tables)
    declared = uk_hmrc_weighted_qrf_output_columns()
    values, weights, surface = uk_qrf_tail_concentration_columns(
        tables,
        output_columns=declared,
    )
    qrf_tail = {
        column: _tail_measurements(
            values[column],
            weights[column],
            top_k_grid,
            minimum_count=minimum_count,
        )
        for column in sorted(values)
    }
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "entity_rows": {
            "person": len(tables["person"]),
            "benunit": len(tables["benunit"]),
            "household": len(tables["household"]),
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
    minimum_count = args.sdc_minimum_count
    if minimum_count < 3:
        raise SystemExit(
            "error: --sdc-minimum-count must be at least 3 "
            "(CD171-ResearchDataHandling §5.2.1 minimum threshold)."
        )
    refused = sorted(k for k in top_k_grid if k < minimum_count)
    if refused:
        raise SystemExit(
            f"error: --top-k values {refused} are below --sdc-minimum-count "
            f"{minimum_count}; a tail that narrow reports on too few records "
            "to publish (CD171-ResearchDataHandling §5.2.1)."
        )
    measurements = {
        label: _measure(path, top_k_grid, minimum_count=minimum_count)
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
        "disclosure_control": {
            "standard": (
                "UK Data Service CD171-ResearchDataHandling §5.2.1 "
                "(Safeguarded data output rules), applied under End User "
                "Licence CD137 clause 8."
            ),
            "minimum_count": minimum_count,
            "rules_applied": [
                "No maximum, minimum, or other unit-record value is emitted.",
                (
                    "Columns with fewer than minimum_count weighted carriers "
                    "report no concentration statistics and no carrier count."
                ),
                (
                    "top_k values below minimum_count are refused, so no "
                    "reported tail describes fewer records than the threshold."
                ),
                (
                    "Per-column weighted totals are population aggregates over "
                    "all carriers and are reported unconditionally."
                ),
            ],
            "caller_obligations": [
                (
                    "Cite and acknowledge the Data Collection(s) per EUL "
                    "clauses 11 and 12 wherever these numbers are posted."
                ),
                (
                    "Check each study's Special Conditions (EUL clause 3): a "
                    "study requiring a threshold of 30 needs "
                    "--sdc-minimum-count 30."
                ),
            ],
        },
        "artifacts": measurements,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "UKDS output note: disclosure control applied per "
        f"CD171-ResearchDataHandling §5.2.1 at minimum count {minimum_count} "
        "(no unit-record values; thin columns suppressed). Cite and "
        "acknowledge the Data Collection(s) per EUL clauses 11-12 wherever "
        "these numbers are posted, and confirm each study's Special "
        "Conditions first.",
        file=sys.stderr,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
