"""Compare two Microcosm US calibration diagnostics on a shared target surface."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TARGET_VALUE_WEIGHT_POWER = 0.5
DEFAULT_TARGET_LOSS_CAP = 1.0


@dataclass(frozen=True)
class DiagnosticRow:
    key: str
    name: str
    target: float
    estimate: float
    relative_error: float
    family: str
    basis: str
    source_measure_id: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ComparedRow:
    key: str
    family: str
    basis: str
    weight: float
    target: float
    candidate_estimate: float
    incumbent_estimate: float
    candidate_relative_error: float
    incumbent_relative_error: float
    candidate_loss: float
    incumbent_loss: float
    weighted_delta: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of row-level improvements/regressions to include.",
    )
    parser.add_argument(
        "--target-loss-cap",
        type=float,
        default=None,
        help=(
            "Per-row capped percent-error loss cap. Defaults to the candidate "
            "diagnostics option when present, otherwise 1.0."
        ),
    )
    return parser.parse_args()


def _load_diagnostics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("targets"), list):
        raise ValueError(f"{path} does not look like calibration_diagnostics.json.")
    return payload


def _target_key(row: Mapping[str, Any]) -> str:
    key = row.get("target_name") or row.get("name")
    if not key:
        raise ValueError(f"Diagnostic row has no target key: {row!r}")
    return str(key)


def _family(row: Mapping[str, Any]) -> str:
    registry = row.get("registry")
    if isinstance(registry, Mapping):
        family = registry.get("family")
        if family:
            return str(family)
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        target_role = metadata.get("target_role")
        if target_role:
            return str(target_role)
    return "unknown"


def _basis(metadata: Mapping[str, Any]) -> str:
    measure_mode = str(metadata.get("measure_mode", ""))
    source_measure_id = str(metadata.get("source_measure_id", ""))
    if measure_mode in {"indicator_sum", "less_than_indicator_sum"}:
        return "count"
    if "enrollment" in source_measure_id or "recipients" in source_measure_id:
        return "count"
    if "return" in source_measure_id and "count" in source_measure_id:
        return "count"
    return "amount"


def _row_by_key(payload: Mapping[str, Any]) -> dict[str, DiagnosticRow]:
    rows: dict[str, DiagnosticRow] = {}
    for raw_row in payload["targets"]:
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"Unexpected diagnostic target row: {raw_row!r}")
        metadata = raw_row.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Diagnostic metadata is not an object: {raw_row!r}")
        key = _target_key(raw_row)
        if key in rows:
            raise ValueError(f"Duplicate diagnostic target key {key!r}.")
        rows[key] = DiagnosticRow(
            key=key,
            name=str(raw_row.get("name") or key),
            target=float(raw_row["target"]),
            estimate=float(raw_row["final_estimate"]),
            relative_error=float(raw_row["relative_error"]),
            family=_family(raw_row),
            basis=_basis(metadata),
            source_measure_id=str(metadata.get("source_measure_id", "")),
            metadata=metadata,
        )
    return rows


def _assert_matching_shared_targets(
    candidate_rows: Mapping[str, DiagnosticRow],
    incumbent_rows: Mapping[str, DiagnosticRow],
    shared_keys: Iterable[str],
) -> None:
    mismatches: list[str] = []
    for key in shared_keys:
        candidate = candidate_rows[key]
        incumbent = incumbent_rows[key]
        if not math.isclose(
            candidate.target,
            incumbent.target,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            mismatches.append(
                f"{key}: candidate={candidate.target!r}, incumbent={incumbent.target!r}"
            )
    if mismatches:
        preview = "; ".join(mismatches[:10])
        suffix = "" if len(mismatches) <= 10 else f"; +{len(mismatches) - 10} more"
        raise ValueError(
            "Cannot compare diagnostics with changed shared target values. "
            f"{preview}{suffix}"
        )


def _loss(target: float, estimate: float, *, cap: float) -> float:
    scale = max(abs(target), 1.0)
    return min(abs((estimate - target) / scale), cap)


def _row_weights(rows: Iterable[DiagnosticRow]) -> dict[str, float]:
    row_list = list(rows)
    if not row_list:
        return {}
    raw_weights = {
        row.key: max(abs(row.target), 1.0) ** TARGET_VALUE_WEIGHT_POWER
        for row in row_list
    }
    bases = sorted({row.basis for row in row_list})
    weights = {row.key: 1.0 for row in row_list}
    for basis in bases:
        keys = [row.key for row in row_list if row.basis == basis]
        mean_raw = sum(raw_weights[key] for key in keys) / len(keys)
        if mean_raw > 0:
            for key in keys:
                weights[key] = raw_weights[key] / mean_raw
    basis_total = len(row_list) / len(bases)
    for basis in bases:
        keys = [row.key for row in row_list if row.basis == basis]
        current_total = sum(weights[key] for key in keys)
        if current_total > 0:
            for key in keys:
                weights[key] *= basis_total / current_total
    mean_weight = sum(weights.values()) / len(weights)
    return {key: weight / mean_weight for key, weight in weights.items()}


def _target_loss_cap(payload: Mapping[str, Any], override: float | None) -> float:
    if override is not None:
        return float(override)
    options = payload.get("options")
    if isinstance(options, Mapping):
        scales = options.get("target_loss_scales")
        if isinstance(scales, Mapping) and scales.get("cap") is not None:
            return float(scales["cap"])
        if options.get("target_loss_cap") is not None:
            return float(options["target_loss_cap"])
    return DEFAULT_TARGET_LOSS_CAP


def compare_diagnostics(
    *,
    candidate_payload: Mapping[str, Any],
    incumbent_payload: Mapping[str, Any],
    target_loss_cap: float | None = None,
    top_n: int = 25,
) -> dict[str, Any]:
    candidate_rows = _row_by_key(candidate_payload)
    incumbent_rows = _row_by_key(incumbent_payload)
    shared_keys = sorted(set(candidate_rows) & set(incumbent_rows))
    _assert_matching_shared_targets(candidate_rows, incumbent_rows, shared_keys)
    cap = _target_loss_cap(candidate_payload, target_loss_cap)
    weights = _row_weights(candidate_rows[key] for key in shared_keys)
    compared: list[ComparedRow] = []
    for key in shared_keys:
        candidate = candidate_rows[key]
        incumbent = incumbent_rows[key]
        candidate_loss = _loss(candidate.target, candidate.estimate, cap=cap)
        incumbent_loss = _loss(incumbent.target, incumbent.estimate, cap=cap)
        weight = weights[key]
        compared.append(
            ComparedRow(
                key=key,
                family=candidate.family,
                basis=candidate.basis,
                weight=weight,
                target=candidate.target,
                candidate_estimate=candidate.estimate,
                incumbent_estimate=incumbent.estimate,
                candidate_relative_error=candidate.relative_error,
                incumbent_relative_error=incumbent.relative_error,
                candidate_loss=candidate_loss,
                incumbent_loss=incumbent_loss,
                weighted_delta=weight * (candidate_loss - incumbent_loss),
            )
        )
    total_weight = sum(row.weight for row in compared)
    if total_weight <= 0:
        raise ValueError("No positive shared target weights to compare.")
    summary = {
        "shared_targets": len(shared_keys),
        "candidate_only_targets": len(set(candidate_rows) - set(incumbent_rows)),
        "incumbent_only_targets": len(set(incumbent_rows) - set(candidate_rows)),
        "target_loss_cap": cap,
        "target_loss_weighting": (
            "sqrt_value_weighted_mape_50_50_amount_count_target_scale_cap_100pct"
        ),
        "candidate_weighted_loss": sum(
            row.weight * row.candidate_loss for row in compared
        )
        / total_weight,
        "incumbent_weighted_loss": sum(
            row.weight * row.incumbent_loss for row in compared
        )
        / total_weight,
        "candidate_within_10pct": sum(
            row.weight for row in compared if abs(row.candidate_relative_error) <= 0.10
        )
        / total_weight,
        "incumbent_within_10pct": sum(
            row.weight for row in compared if abs(row.incumbent_relative_error) <= 0.10
        )
        / total_weight,
    }
    summary["weighted_loss_delta"] = (
        summary["candidate_weighted_loss"] - summary["incumbent_weighted_loss"]
    )

    family_buckets: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "target_count": 0,
            "weight": 0.0,
            "candidate_weighted_loss_sum": 0.0,
            "incumbent_weighted_loss_sum": 0.0,
            "candidate_rows_over_10pct": 0,
            "incumbent_rows_over_10pct": 0,
        }
    )
    basis_buckets: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "target_count": 0,
            "weight": 0.0,
            "candidate_weighted_loss_sum": 0.0,
            "incumbent_weighted_loss_sum": 0.0,
        }
    )
    for row in compared:
        for bucket in (family_buckets[row.family], basis_buckets[row.basis]):
            bucket["target_count"] = int(bucket["target_count"]) + 1
            bucket["weight"] = float(bucket["weight"]) + row.weight
            bucket["candidate_weighted_loss_sum"] = (
                float(bucket["candidate_weighted_loss_sum"])
                + row.weight * row.candidate_loss
            )
            bucket["incumbent_weighted_loss_sum"] = (
                float(bucket["incumbent_weighted_loss_sum"])
                + row.weight * row.incumbent_loss
            )
        if abs(row.candidate_relative_error) > 0.10:
            family_buckets[row.family]["candidate_rows_over_10pct"] = (
                int(family_buckets[row.family]["candidate_rows_over_10pct"]) + 1
            )
        if abs(row.incumbent_relative_error) > 0.10:
            family_buckets[row.family]["incumbent_rows_over_10pct"] = (
                int(family_buckets[row.family]["incumbent_rows_over_10pct"]) + 1
            )

    return {
        "summary": summary,
        "families": _summarize_buckets(family_buckets, total_weight=total_weight),
        "bases": _summarize_buckets(basis_buckets, total_weight=total_weight),
        "top_candidate_regressions": [
            _row_payload(row, total_weight=total_weight)
            for row in sorted(
                (row for row in compared if row.weighted_delta > 0),
                key=lambda row: row.weighted_delta,
                reverse=True,
            )[:top_n]
        ],
        "top_candidate_improvements": [
            _row_payload(row, total_weight=total_weight)
            for row in sorted(
                (row for row in compared if row.weighted_delta < 0),
                key=lambda row: row.weighted_delta,
            )[:top_n]
        ],
        "candidate_only_target_examples": sorted(
            set(candidate_rows) - set(incumbent_rows)
        )[:top_n],
        "incumbent_only_target_examples": sorted(
            set(incumbent_rows) - set(candidate_rows)
        )[:top_n],
    }


def _summarize_buckets(
    buckets: Mapping[str, Mapping[str, float | int]], *, total_weight: float
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        weight = float(bucket["weight"])
        candidate_sum = float(bucket["candidate_weighted_loss_sum"])
        incumbent_sum = float(bucket["incumbent_weighted_loss_sum"])
        payload.append(
            {
                "name": key,
                "target_count": int(bucket["target_count"]),
                "weight_share": weight / total_weight,
                "candidate_loss_contribution": candidate_sum / total_weight,
                "incumbent_loss_contribution": incumbent_sum / total_weight,
                "loss_contribution_delta": (candidate_sum - incumbent_sum)
                / total_weight,
                "candidate_mean_loss": candidate_sum / weight if weight else 0.0,
                "incumbent_mean_loss": incumbent_sum / weight if weight else 0.0,
                **(
                    {
                        "candidate_rows_over_10pct": int(
                            bucket["candidate_rows_over_10pct"]
                        ),
                        "incumbent_rows_over_10pct": int(
                            bucket["incumbent_rows_over_10pct"]
                        ),
                    }
                    if "candidate_rows_over_10pct" in bucket
                    else {}
                ),
            }
        )
    return sorted(payload, key=lambda row: row["loss_contribution_delta"], reverse=True)


def _row_payload(row: ComparedRow, *, total_weight: float) -> dict[str, Any]:
    return {
        "target": row.key,
        "family": row.family,
        "basis": row.basis,
        "weight": row.weight,
        "weight_share": row.weight / total_weight,
        "target_value": row.target,
        "candidate_estimate": row.candidate_estimate,
        "incumbent_estimate": row.incumbent_estimate,
        "candidate_relative_error": row.candidate_relative_error,
        "incumbent_relative_error": row.incumbent_relative_error,
        "candidate_loss": row.candidate_loss,
        "incumbent_loss": row.incumbent_loss,
        "loss_contribution_delta": row.weighted_delta / total_weight,
    }


def _write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_loss(value: float) -> str:
    return f"{value:.6f}"


def _markdown_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Microcosm US Target Fit Comparison",
        "",
        "## Summary",
        "",
        f"- Shared targets: {summary['shared_targets']:,}",
        f"- Candidate-only targets: {summary['candidate_only_targets']:,}",
        f"- Incumbent-only targets: {summary['incumbent_only_targets']:,}",
        (
            "- Candidate weighted loss: "
            f"{_format_loss(summary['candidate_weighted_loss'])}"
        ),
        (
            "- Incumbent weighted loss: "
            f"{_format_loss(summary['incumbent_weighted_loss'])}"
        ),
        f"- Delta: {_format_loss(summary['weighted_loss_delta'])}",
        (
            "- Candidate weighted share within 10%: "
            f"{_format_pct(summary['candidate_within_10pct'])}"
        ),
        (
            "- Incumbent weighted share within 10%: "
            f"{_format_pct(summary['incumbent_within_10pct'])}"
        ),
        "",
        "## Families Worsening",
        "",
        "| Family | Targets | Weight share | Candidate contrib | Incumbent contrib | Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["families"][:12]:
        lines.append(
            "| {name} | {target_count:,} | {weight_share:.2%} | "
            "{candidate_loss_contribution:.6f} | "
            "{incumbent_loss_contribution:.6f} | "
            "{loss_contribution_delta:+.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Top Candidate Regressions",
            "",
            "| Delta | Family | Basis | Candidate error | Incumbent error | Target |",
            "|---:|---|---|---:|---:|---|",
        ]
    )
    for row in payload["top_candidate_regressions"][:15]:
        lines.append(
            "| {loss_contribution_delta:+.6f} | {family} | {basis} | "
            "{candidate_relative_error:+.1%} | {incumbent_relative_error:+.1%} | "
            "`{target}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Top Candidate Improvements",
            "",
            "| Delta | Family | Basis | Candidate error | Incumbent error | Target |",
            "|---:|---|---|---:|---:|---|",
        ]
    )
    for row in payload["top_candidate_improvements"][:15]:
        lines.append(
            "| {loss_contribution_delta:+.6f} | {family} | {basis} | "
            "{candidate_relative_error:+.1%} | {incumbent_relative_error:+.1%} | "
            "`{target}` |".format(**row)
        )
    return "\n".join(lines) + "\n"


def write_comparison(payload: Mapping[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "target_fit_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(payload["families"], out / "family_comparison.csv")
    _write_csv(payload["bases"], out / "basis_comparison.csv")
    _write_csv(payload["top_candidate_regressions"], out / "top_regressions.csv")
    _write_csv(payload["top_candidate_improvements"], out / "top_improvements.csv")
    (out / "target_fit_comparison.md").write_text(
        _markdown_report(payload),
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    candidate = _load_diagnostics(args.candidate)
    incumbent = _load_diagnostics(args.incumbent)
    payload = compare_diagnostics(
        candidate_payload=candidate,
        incumbent_payload=incumbent,
        target_loss_cap=args.target_loss_cap,
        top_n=args.top_n,
    )
    write_comparison(payload, args.out)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
