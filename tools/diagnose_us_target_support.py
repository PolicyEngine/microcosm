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

The checkpoint's compiled columns are verified byte-consistent with the
release two ways before any decomposition is reported: household-id order must
match the published dataset exactly, and the recomputed final-weight aggregate
must reproduce the diagnostics' recorded ``final_estimate``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

#: Relative disagreement between the recomputed final aggregate and the
#: diagnostics' recorded final_estimate above which the checkpoint/dataset
#: pairing is rejected as inconsistent.
FINAL_ESTIMATE_RTOL = 1e-6

#: Carriers whose final/design weight ratio exceeds this fraction of the
#: solve's realized maximum ratio are reported as pinned near the cap.
NEAR_CAP_FRACTION = 0.9

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
    checkpoint: h5py.File, columns: dict[str, str], name: str
) -> np.ndarray:
    try:
        path = columns[name]
    except KeyError as error:
        raise TargetSupportError(
            f"Checkpoint has no compiled household column named {name!r}."
        ) from error
    return np.asarray(checkpoint[path][:], dtype=np.float64)


def diagnose_target_support(
    *,
    diagnostics_path: Path,
    checkpoint_path: Path,
    dataset_path: Path,
    target_patterns: list[str],
    top: int = 10,
) -> dict[str, object]:
    diagnostics = json.loads(Path(diagnostics_path).read_text())
    rows = _load_household_target_rows(diagnostics)
    matched = _match_targets(rows, list(target_patterns))

    household = pd.read_hdf(dataset_path, "household")
    if "household_id" not in household or "household_weight" not in household:
        raise TargetSupportError(
            "Dataset household table must carry household_id and household_weight."
        )
    final_weights = household["household_weight"].to_numpy(dtype=np.float64)

    realized_max_ratio = diagnostics.get("realized_max_weight_ratio")
    near_cap_ratio = (
        float(realized_max_ratio) * NEAR_CAP_FRACTION
        if isinstance(realized_max_ratio, (int, float))
        else None
    )

    reports: list[dict[str, object]] = []
    with h5py.File(checkpoint_path, "r") as checkpoint:
        columns = _checkpoint_household_columns(checkpoint)
        checkpoint_ids = _read_checkpoint_column(checkpoint, columns, "household_id")
        dataset_ids = household["household_id"].to_numpy(dtype=np.float64)
        if not np.array_equal(checkpoint_ids, dataset_ids):
            raise TargetSupportError(
                "Checkpoint household_id order does not match the dataset; "
                "refusing to decompose against misaligned rows."
            )
        design_weights = np.asarray(
            checkpoint["weights/household/values"][:], dtype=np.float64
        )

        for name in matched:
            row = rows[name]
            compiled = _read_checkpoint_column(
                checkpoint, columns, name.split("@", 1)[0]
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

    return {
        "diagnostics": str(diagnostics_path),
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "n_households": int(len(household)),
        "realized_max_weight_ratio": realized_max_ratio,
        "targets": reports,
    }


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
    recorded_final = float(row["final_estimate"])
    final_estimate = float((final_weights * compiled).sum())
    if not math.isclose(
        final_estimate,
        recorded_final,
        rel_tol=FINAL_ESTIMATE_RTOL,
        abs_tol=1.0,
    ):
        raise TargetSupportError(
            f"Recomputed final for {name!r} ({final_estimate!r}) does not "
            f"reproduce the diagnostics final ({recorded_final!r}); the "
            "checkpoint does not belong to this release."
        )
    design_estimate = float((design_weights * compiled).sum())

    carrier_mask = compiled != 0.0
    contributions = final_weights * compiled
    order = np.argsort(contributions)[::-1]
    top_order = order[: max(top, 0)]
    total = contributions.sum()

    def _share(count: int) -> float | None:
        if total == 0.0:
            return None
        return float(contributions[order[:count]].sum() / total)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(design_weights > 0.0, final_weights / design_weights, np.nan)
    carrier_ratios = ratios[carrier_mask]
    carrier_ratios = carrier_ratios[np.isfinite(carrier_ratios)]

    carriers: list[dict[str, object]] = []
    for index in top_order:
        if compiled[index] == 0.0:
            continue
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


def _format_report(payload: dict[str, object]) -> str:
    lines: list[str] = []
    for report in payload["targets"]:
        lines.append(f"== {report['name']} ==")
        lines.append(
            "target {target:,.0f} | design {design_estimate:,.0f} "
            "({design_relative_error:+.1%}) | final {final_estimate:,.0f} "
            "({final_relative_error:+.1%})".format(
                target=report["target"],
                design_estimate=report["design_estimate"],
                design_relative_error=report["design_relative_error"] or 0.0,
                final_estimate=report["final_estimate"],
                final_relative_error=report["final_relative_error"] or 0.0,
            )
        )
        shares = [
            f"top-1 {report['top_1_share']:.1%}"
            if report["top_1_share"] is not None
            else "top-1 n/a",
            f"top-5 {report['top_5_share']:.1%}"
            if report["top_5_share"] is not None
            else "top-5 n/a",
        ]
        lines.append(f"carriers {report['carrier_count']:,} | " + " | ".join(shares))
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


def main() -> None:
    args = _parse_args()
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
