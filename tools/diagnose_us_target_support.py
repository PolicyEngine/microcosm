"""Per-target carrier/support forensics for a published US release.

Given a release's ``calibration_diagnostics.json``, its
``target_frame_checkpoint.h5`` (the compiled per-household target columns and
pre-solve design weights), and the published ``populace_us_2024.h5``, report
for each requested target: the target/design/final aggregates, the carrier
count, top-carrier concentration shares, per-carrier weight pressure against
the solver's weight-ratio cap, and household provenance (support channel,
clone index, source id, state FIPS) for the top carriers.

This is the read-only receipts tool behind the populace#462 / populace#451
support forensics (tips carrier deficit, six-state medical blowout). It never
modifies any artifact, never runs PolicyEngine, and never touches calibration:
it decomposes what a shipped solve already saw, so support defects can be
adjudicated to their owning stage with named records.

The checkpoint's compiled columns are verified consistent with the release
before any decomposition is reported: household ids must be integral, match
the published dataset's order exactly (compared losslessly as integers), every
vector must be household-length, and the recomputed final-weight aggregate
must reproduce the diagnostics' recorded ``final_estimate`` to within float
summation noise. Carriers are ranked by absolute weighted contribution (input
row order breaks ties deterministically), so signed columns report their
dominant carriers rather than whichever positive rows sort first.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

#: Refusal bounds for the recomputed-vs-recorded final aggregate. Measured
#: float64 re-summation noise on the Build N release is <= 7e-15 relative /
#: $0.41 absolute across 400 sampled targets; these sit orders of magnitude
#: above that while still rejecting any real cross-release pairing.
FINAL_ESTIMATE_RTOL = 1e-9
FINAL_ESTIMATE_ATOL = 0.5

#: Carriers whose final/design weight ratio exceeds this fraction of the
#: solve's realized maximum ratio are reported as pinned near the cap.
NEAR_CAP_FRACTION = 0.9

#: Float household ids at or above 2**53 cannot round-trip integers exactly;
#: refuse rather than compare lossy values.
_MAX_EXACT_FLOAT_ID = float(2**53)

_HOUSEHOLD_PROVENANCE_COLUMNS = (
    "household_support_channel",
    "household_support_clone_index",
    "household_source_id",
    "state_fips",
)


class TargetSupportError(RuntimeError):
    """Raised when the diagnostics/checkpoint/dataset triple is inconsistent."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help=(
            "Target name from the diagnostics (with or without the @period "
            "suffix), or a substring matching one or more household-entity "
            "targets. Repeatable."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top carriers to report per target (default 10).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write the full payload as JSON.",
    )
    return parser.parse_args()


def _load_household_target_rows(diagnostics: dict) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in diagnostics.get("targets", ()):
        if row.get("entity") != "household":
            continue
        rows[str(row["name"])] = row
    if not rows:
        raise TargetSupportError(
            "Diagnostics carry no household-entity targets; nothing to diagnose."
        )
    return rows


def _match_targets(rows: dict[str, dict], patterns: list[str]) -> list[str]:
    matched: list[str] = []
    for pattern in patterns:
        exact = [
            name for name in rows if name == pattern or name.split("@", 1)[0] == pattern
        ]
        hits = exact or [name for name in rows if pattern in name]
        if not hits:
            raise TargetSupportError(f"No household-entity target matches {pattern!r}.")
        for name in hits:
            if name not in matched:
                matched.append(name)
    return matched


def _checkpoint_household_columns(checkpoint: h5py.File) -> dict[str, str]:
    group = checkpoint["tables/household/columns"]
    return {
        str(group[key].attrs["name"]): f"tables/household/columns/{key}/values"
        for key in group
    }


def _read_checkpoint_column(
    checkpoint: h5py.File,
    columns: dict[str, str],
    name: str,
    *,
    n_households: int,
) -> np.ndarray:
    try:
        path = columns[name]
    except KeyError as error:
        raise TargetSupportError(
            f"Checkpoint has no compiled household column named {name!r}."
        ) from error
    values = np.asarray(checkpoint[path][:])
    if values.ndim != 1 or values.shape[0] != n_households:
        raise TargetSupportError(
            f"Checkpoint column {name!r} has shape {values.shape}; expected "
            f"({n_households},) to match the dataset's household count."
        )
    return values


def _integral_ids(values: np.ndarray, *, source: str) -> np.ndarray:
    """Return household ids as exact int64, refusing lossy representations."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise TargetSupportError(f"{source} household ids must be one-dimensional.")
    kind = array.dtype.kind
    if kind == "u":
        if array.size and int(array.max()) > np.iinfo(np.int64).max:
            raise TargetSupportError(
                f"{source} household ids exceed int64 range; refusing a lossy "
                "id comparison."
            )
        return array.astype(np.int64)
    if kind == "i":
        return array.astype(np.int64)
    if array.dtype != np.float64:
        raise TargetSupportError(
            f"{source} household ids have unsupported dtype {array.dtype}; "
            "expected an integer or float64 column."
        )
    if (
        not np.all(np.isfinite(array))
        or np.any(np.abs(array) >= _MAX_EXACT_FLOAT_ID)
        or np.any(array != np.trunc(array))
    ):
        raise TargetSupportError(
            f"{source} household ids are not exactly representable integers; "
            "refusing a lossy id comparison."
        )
    return array.astype(np.int64)


def diagnose_target_support(
    *,
    diagnostics_path: Path,
    checkpoint_path: Path,
    dataset_path: Path,
    target_patterns: list[str],
    top: int = 10,
) -> dict[str, object]:
    if top < 1:
        raise TargetSupportError(f"--top must be at least 1, got {top}.")
    diagnostics = json.loads(Path(diagnostics_path).read_text())
    rows = _load_household_target_rows(diagnostics)
    matched = _match_targets(rows, list(target_patterns))

    household = pd.read_hdf(dataset_path, "household")
    if "household_id" not in household or "household_weight" not in household:
        raise TargetSupportError(
            "Dataset household table must carry household_id and household_weight."
        )
    n_households = len(household)
    final_weights = household["household_weight"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(final_weights)):
        raise TargetSupportError("Dataset household weights are not all finite.")

    realized_max_ratio = diagnostics.get("realized_max_weight_ratio")
    near_cap_ratio = (
        float(realized_max_ratio) * NEAR_CAP_FRACTION
        if isinstance(realized_max_ratio, (int, float))
        and math.isfinite(realized_max_ratio)
        else None
    )

    missing_provenance = [
        column for column in _HOUSEHOLD_PROVENANCE_COLUMNS if column not in household
    ]

    reports: list[dict[str, object]] = []
    with h5py.File(checkpoint_path, "r") as checkpoint:
        columns = _checkpoint_household_columns(checkpoint)
        checkpoint_ids = _integral_ids(
            _read_checkpoint_column(
                checkpoint, columns, "household_id", n_households=n_households
            ),
            source="Checkpoint",
        )
        dataset_ids = _integral_ids(
            household["household_id"].to_numpy(), source="Dataset"
        )
        if not np.array_equal(checkpoint_ids, dataset_ids):
            raise TargetSupportError(
                "Checkpoint household_id order does not match the dataset; "
                "refusing to decompose against misaligned rows."
            )
        design_weights = np.asarray(
            checkpoint["weights/household/values"][:], dtype=np.float64
        )
        if design_weights.ndim != 1 or design_weights.shape[0] != n_households:
            raise TargetSupportError(
                f"Checkpoint design weights have shape {design_weights.shape}; "
                f"expected ({n_households},)."
            )
        if not np.all(np.isfinite(design_weights)):
            raise TargetSupportError("Checkpoint design weights are not all finite.")

        for name in matched:
            row = rows[name]
            compiled = np.asarray(
                _read_checkpoint_column(
                    checkpoint,
                    columns,
                    name.split("@", 1)[0],
                    n_households=n_households,
                ),
                dtype=np.float64,
            )
            reports.append(
                _diagnose_one(
                    name=name,
                    row=row,
                    compiled=compiled,
                    design_weights=design_weights,
                    final_weights=final_weights,
                    household=household,
                    top=top,
                    near_cap_ratio=near_cap_ratio,
                )
            )

    payload: dict[str, object] = {
        "diagnostics": str(diagnostics_path),
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "n_households": int(n_households),
        "realized_max_weight_ratio": realized_max_ratio,
        "targets": reports,
    }
    if missing_provenance:
        payload["missing_provenance_columns"] = missing_provenance
    return payload


def _diagnose_one(
    *,
    name: str,
    row: dict,
    compiled: np.ndarray,
    design_weights: np.ndarray,
    final_weights: np.ndarray,
    household: pd.DataFrame,
    top: int,
    near_cap_ratio: float | None,
) -> dict[str, object]:
    target_value = float(row["target"])
    if not math.isfinite(target_value):
        raise TargetSupportError(
            f"Diagnostics target value for {name!r} is not finite."
        )
    recorded_final = float(row["final_estimate"])
    final_estimate = float((final_weights * compiled).sum())
    if (
        not math.isfinite(recorded_final)
        or not math.isfinite(final_estimate)
        or not math.isclose(
            final_estimate,
            recorded_final,
            rel_tol=FINAL_ESTIMATE_RTOL,
            abs_tol=FINAL_ESTIMATE_ATOL,
        )
    ):
        raise TargetSupportError(
            f"Recomputed final for {name!r} ({final_estimate!r}) does not "
            f"reproduce the diagnostics final ({recorded_final!r}); the "
            "checkpoint does not belong to this release."
        )
    design_estimate = float((design_weights * compiled).sum())
    if not math.isfinite(design_estimate):
        raise TargetSupportError(
            f"Design-weight aggregate for {name!r} is not finite; the "
            "checkpoint column or design weights are corrupt."
        )

    carrier_mask = compiled != 0.0
    contributions = final_weights * compiled
    # Rank carriers by absolute weighted contribution (input row order as the
    # deterministic tiebreak via stable sort), so signed columns surface
    # their dominant carriers instead of whichever positive rows sort first.
    carrier_indices = np.flatnonzero(carrier_mask)
    ranked = carrier_indices[
        np.argsort(-np.abs(contributions[carrier_indices]), kind="stable")
    ]
    total = contributions.sum()

    def _share(count: int) -> float | None:
        if total == 0.0:
            return None
        return float(contributions[ranked[:count]].sum() / total)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(design_weights > 0.0, final_weights / design_weights, np.nan)
    carrier_ratios = ratios[carrier_mask]
    carrier_ratios = carrier_ratios[np.isfinite(carrier_ratios)]

    carriers: list[dict[str, object]] = []
    for index in ranked[:top]:
        entry: dict[str, object] = {
            "household_id": int(household["household_id"].iloc[index]),
            "compiled_value": float(compiled[index]),
            "design_weight": float(design_weights[index]),
            "final_weight": float(final_weights[index]),
            "weight_ratio": (
                float(ratios[index]) if np.isfinite(ratios[index]) else None
            ),
            "weighted_contribution": float(contributions[index]),
            "contribution_share": (
                float(contributions[index] / total) if total != 0.0 else None
            ),
        }
        for column in _HOUSEHOLD_PROVENANCE_COLUMNS:
            if column in household:
                value = household[column].iloc[index]
                entry[column] = value.item() if hasattr(value, "item") else value
        carriers.append(entry)

    report: dict[str, object] = {
        "name": name,
        "target": target_value,
        "design_estimate": design_estimate,
        "final_estimate": final_estimate,
        "design_relative_error": (
            design_estimate / target_value - 1.0 if target_value else None
        ),
        "final_relative_error": (
            final_estimate / target_value - 1.0 if target_value else None
        ),
        "carrier_count": int(carrier_mask.sum()),
        "carriers_with_nonpositive_design_weight": int(
            (design_weights[carrier_mask] <= 0.0).sum()
        ),
        "top_1_share": _share(1),
        "top_5_share": _share(5),
        f"top_{top}_share": _share(top),
        "carrier_weight_ratio": {
            "mean": (float(carrier_ratios.mean()) if carrier_ratios.size else None),
            "median": (
                float(np.median(carrier_ratios)) if carrier_ratios.size else None
            ),
            "p90": (
                float(np.quantile(carrier_ratios, 0.9)) if carrier_ratios.size else None
            ),
            "max": (float(carrier_ratios.max()) if carrier_ratios.size else None),
        },
        "top_carriers": carriers,
    }
    if near_cap_ratio is not None and carrier_ratios.size:
        report["carrier_share_near_cap"] = float(
            (carrier_ratios > near_cap_ratio).mean()
        )
    return report


def _percent_or_na(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _format_report(payload: dict[str, object]) -> str:
    lines: list[str] = []
    missing = payload.get("missing_provenance_columns")
    if missing:
        lines.append("NOTE: dataset lacks provenance column(s): " + ", ".join(missing))
        lines.append("")
    for report in payload["targets"]:
        lines.append(f"== {report['name']} ==")
        lines.append(
            "target {target:,.0f} | design {design:,.0f} ({design_rel}) | "
            "final {final:,.0f} ({final_rel})".format(
                target=report["target"],
                design=report["design_estimate"],
                design_rel=_percent_or_na(report["design_relative_error"], signed=True),
                final=report["final_estimate"],
                final_rel=_percent_or_na(report["final_relative_error"], signed=True),
            )
        )
        shares = [
            f"top-1 {_percent_or_na(report['top_1_share'])}",
            f"top-5 {_percent_or_na(report['top_5_share'])}",
        ]
        for key, value in report.items():
            if (
                key.startswith("top_")
                and key.endswith("_share")
                and key not in ("top_1_share", "top_5_share")
            ):
                shares.append(f"top-{key[4:-6]} {_percent_or_na(value)}")
        if "carrier_share_near_cap" in report:
            shares.append(
                f"near-cap {_percent_or_na(report['carrier_share_near_cap'])}"
            )
        lines.append(f"carriers {report['carrier_count']:,} | " + " | ".join(shares))
        nonpositive = report["carriers_with_nonpositive_design_weight"]
        if nonpositive:
            lines.append(
                f"NOTE: {nonpositive:,} carrier(s) have nonpositive design "
                "weight (excluded from ratio statistics)."
            )
        for carrier in report["top_carriers"]:
            provenance = ", ".join(
                f"{column}={carrier[column]}"
                for column in _HOUSEHOLD_PROVENANCE_COLUMNS
                if column in carrier
            )
            ratio = carrier["weight_ratio"]
            lines.append(
                "  hh {household_id}: value {compiled_value:,.0f} x "
                "w {final_weight:,.1f} (design {design_weight:,.1f}"
                "{ratio}) = {weighted_contribution:,.0f}"
                "{share} [{provenance}]".format(
                    household_id=carrier["household_id"],
                    compiled_value=carrier["compiled_value"],
                    final_weight=carrier["final_weight"],
                    design_weight=carrier["design_weight"],
                    ratio=f", x{ratio:.2f}" if ratio is not None else "",
                    weighted_contribution=carrier["weighted_contribution"],
                    share=(
                        f" ({carrier['contribution_share']:.1%})"
                        if carrier["contribution_share"] is not None
                        else ""
                    ),
                    provenance=provenance,
                )
            )
        lines.append("")
    return "\n".join(lines)


def _same_file(candidate: Path, reference: Path) -> bool:
    """True when two paths name the same inode (hard links, symlinks)."""

    try:
        left = candidate.stat()
        right = reference.stat()
    except OSError:
        return False
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def main() -> None:
    args = _parse_args()
    if args.json_output is not None:
        json_target = args.json_output.resolve()
        for input_path in (args.diagnostics, args.checkpoint, args.dataset):
            if json_target == Path(input_path).resolve() or _same_file(
                args.json_output, Path(input_path)
            ):
                raise TargetSupportError(
                    f"--json-output {args.json_output} names the same file as "
                    f"input {input_path}; refusing to overwrite a forensics "
                    "input."
                )
    payload = diagnose_target_support(
        diagnostics_path=args.diagnostics,
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        target_patterns=args.target,
        top=args.top,
    )
    print(_format_report(payload))
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(payload, indent=1))
        print(f"Wrote JSON payload to {args.json_output}.")


if __name__ == "__main__":
    main()
