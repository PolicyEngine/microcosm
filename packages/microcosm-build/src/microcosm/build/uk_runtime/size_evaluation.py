"""Pure evaluation helpers for UK rowwise dataset-size experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.diagnostics import (
    uk_fit_by_family,
    uk_weight_summary,
)
from microcosm.build.uk_runtime.rowwise_geography import id_multiplier_for_values

__all__ = [
    "PRE_REGISTERED_OUTCOMES_V1",
    "RowwiseRun",
    "area_support_tables",
    "dense_reference_deltas",
    "fit_tables",
    "footprint",
    "frozen_vs_recomputed",
    "gate_table",
    "load_run",
    "paired_targets",
    "run_acceptance",
    "summarize",
    "weight_tables",
]

_MANIFEST = "rowwise_candidate_manifest.json"
_SOLVE_DIAGNOSTICS = "solve_diagnostics.csv"
_CALIBRATION_DIAGNOSTICS = "calibration_diagnostics.json"
_AREA_SUPPORT = "area_support_summary.csv"
_DENSE_DIAGNOSTICS = "dense_reference_diagnostics.csv"
_SELECTION = "dataset_size_selection.csv"
_GATE_IDS = (
    "uk_local_geography_ladder_post_calibration",
    "uk_local_area_support",
    "uk_local_target_fit",
    "uk_local_per_family_fit",
    "uk_local_weight_ratio",
    "uk_local_weight_ess",
)


@dataclass(frozen=True)
class RowwiseRun:
    """The evaluation inputs loaded from one rowwise-candidate run directory."""

    label: str
    path: Path
    manifest: dict[str, Any]
    targets: pd.DataFrame
    local_diagnostics: pd.DataFrame
    dense_reference_diagnostics: pd.DataFrame | None
    selection: pd.DataFrame | None
    area_support: pd.DataFrame
    gates: dict[str, Any]
    holdout: dict[str, Any] | None
    weights: pd.DataFrame | None
    runtime: dict[str, int | float | None]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def _require(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"required run file is missing: {path.name}")
    return path


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _target_grain(row: Mapping[str, Any]) -> str:
    metadata = _mapping(row.get("metadata"))
    family = str(metadata.get("family") or row.get("family") or "")
    name = str(row.get("name") or "")
    if not metadata.get("area_code"):
        return "national"
    if family == "census_households" and name.startswith("external:"):
        return "ladder"
    area_type = str(metadata.get("area_type") or "")
    return "local_authority" if area_type == "la" else area_type


def _targets_frame(rows: object) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for raw in rows if isinstance(rows, list) else []:
        if not isinstance(raw, Mapping):
            continue
        metadata = _mapping(raw.get("metadata"))
        relative_error = _optional_float(raw.get("relative_error"))
        records.append(
            {
                "name": str(raw.get("name") or ""),
                "target_name": str(raw.get("target_name") or ""),
                "grain": _target_grain(raw),
                "family": str(metadata.get("family") or raw.get("family") or ""),
                "area_type": metadata.get("area_type"),
                "area_code": metadata.get("area_code"),
                "metric": metadata.get("metric"),
                "target": _optional_float(raw.get("target")),
                "initial_estimate": _optional_float(raw.get("initial_estimate")),
                "final_estimate": _optional_float(raw.get("final_estimate")),
                "relative_error": relative_error,
                "abs_relative_error": (
                    abs(relative_error) if relative_error is not None else None
                ),
                "target_loss_scale": _optional_float(raw.get("target_loss_scale")),
            }
        )
    columns = [
        "name",
        "target_name",
        "grain",
        "family",
        "area_type",
        "area_code",
        "metric",
        "target",
        "initial_estimate",
        "final_estimate",
        "relative_error",
        "abs_relative_error",
        "target_loss_scale",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


_TIME_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s+real\s+"
    r"[0-9]+(?:\.[0-9]+)?\s+user\s+[0-9]+(?:\.[0-9]+)?\s+sys\s*$"
)
_RSS_RE = re.compile(r"^\s*([0-9]+)\s+maximum resident set size\s*$")


def _runtime(path: Path, h5_path: Path | None) -> dict[str, int | float | None]:
    wall: float | None = None
    rss: int | None = None
    log = path / "run.log"
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            time_match = _TIME_RE.match(line)
            if time_match:
                wall = float(time_match.group(1))
            rss_match = _RSS_RE.match(line)
            if rss_match:
                rss = int(rss_match.group(1))
    if wall is None:
        for receipt_path in sorted((path / "logbook-spool").glob("*.json")):
            try:
                receipt = _read_json(receipt_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            candidate = _optional_float(receipt.get("wall_seconds"))
            if candidate is not None:
                wall = candidate
                break
    return {
        "wall_seconds": wall,
        "peak_rss_bytes": rss,
        "h5_bytes": h5_path.stat().st_size if h5_path is not None else None,
    }


def _find_h5(path: Path, manifest: Mapping[str, Any]) -> Path | None:
    outputs = _mapping(manifest.get("outputs"))
    for record in outputs.values():
        candidate_name = Path(str(_mapping(record).get("path") or "")).name
        if candidate_name.endswith(".h5"):
            candidate = path / candidate_name
            if candidate.is_file():
                return candidate
    candidates = sorted(path.glob("microcosm_uk_*_local.h5"))
    return candidates[0] if candidates else None


def load_run(path: str | Path, *, label: str, skip_h5: bool = False) -> RowwiseRun:
    """Load one run using only the documented run-directory basenames."""

    run_path = Path(path).resolve()
    manifest = _read_json(_require(run_path / _MANIFEST))
    local = pd.read_csv(_require(run_path / _SOLVE_DIAGNOSTICS))
    calibration = _read_json(_require(run_path / _CALIBRATION_DIAGNOSTICS))
    support = pd.read_csv(_require(run_path / _AREA_SUPPORT))
    gate_paths = sorted(run_path.glob("*.local_gates.json"))
    if not gate_paths:
        raise ValueError("required run file is missing: *.local_gates.json")
    gates = _read_json(gate_paths[0])

    dense_path = run_path / _DENSE_DIAGNOSTICS
    selection_path = run_path / _SELECTION
    dense = pd.read_csv(dense_path) if dense_path.is_file() else None
    selection = pd.read_csv(selection_path) if selection_path.is_file() else None
    h5_path = _find_h5(run_path, manifest)
    weights = None
    if not skip_h5 and h5_path is not None:
        weights = pd.read_hdf(
            h5_path,
            "household",
            columns=["household_id", "household_weight", "clone_index"],
        )
    uk_diagnostics = _mapping(calibration.get("uk_diagnostics"))
    holdout_value = uk_diagnostics.get("rotated_holdout")
    holdout = dict(holdout_value) if isinstance(holdout_value, Mapping) else None
    return RowwiseRun(
        label=label,
        path=run_path,
        manifest=manifest,
        targets=_targets_frame(calibration.get("targets")),
        local_diagnostics=local,
        dense_reference_diagnostics=dense,
        selection=selection,
        area_support=support,
        gates=gates,
        holdout=holdout,
        weights=weights,
        runtime=_runtime(run_path, h5_path),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_acceptance(
    run: RowwiseRun,
    *,
    expected_households: int | None = None,
    expected_pool: int | None = None,
    expected_epochs: int | None = None,
) -> dict[str, Any]:
    """Evaluate the mechanical acceptance contract recorded by one run."""

    manifest = run.manifest
    parameters = _mapping(manifest.get("parameters"))
    solve = _mapping(manifest.get("solve"))
    size = solve.get("dataset_size")
    size_receipt = _mapping(size)
    is_size = isinstance(size, Mapping)
    weights_manifest = _mapping(manifest.get("weights"))
    release_posture = _mapping(manifest.get("release_posture"))
    checks: list[dict[str, Any]] = []

    def add(
        check_id: str,
        passed: bool | None,
        observed: object,
        expected: object,
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "status": (
                    "not_applicable" if passed is None else "pass" if passed else "fail"
                ),
                "observed": observed,
                "expected": expected,
            }
        )

    add(
        "releasable_false",
        manifest.get("releasable") is False if is_size else None,
        manifest.get("releasable"),
        False,
    )
    add(
        "size_certification_absent",
        release_posture.get("size_certification_present") is False if is_size else None,
        release_posture.get("size_certification_present"),
        False,
    )
    requested = size_receipt.get("requested_households")
    realized = size_receipt.get("realized_households")
    wanted_households = (
        expected_households if expected_households is not None else realized
    )
    add(
        "dataset_households",
        requested == realized == wanted_households if is_size else None,
        {"requested": requested, "realized": realized},
        wanted_households,
    )
    pool = size_receipt.get("pool_households", solve.get("pool_households"))
    add(
        "pool_households",
        pool == expected_pool if is_size and expected_pool is not None else None,
        pool,
        expected_pool,
    )
    add("n_clones", parameters.get("n_clones") == 15, parameters.get("n_clones"), 15)
    epochs = parameters.get("epochs")
    epoch_values = {
        "epochs": epochs,
        "selection_epochs": size_receipt.get("selection_epochs"),
        "refit_epochs": size_receipt.get("refit_epochs"),
    }
    epoch_expected = expected_epochs if expected_epochs is not None else epochs
    epoch_pass = epochs == epoch_expected
    if is_size:
        epoch_pass = epoch_pass and all(
            value == epochs for value in epoch_values.values()
        )
    add("epochs", epoch_pass, epoch_values if is_size else epochs, epoch_expected)
    measure = _mapping(solve.get("measure_resolution"))
    add("engine_blocks", measure.get("blocks") == 1, measure.get("blocks"), 1)
    stretch_reference = weights_manifest.get(
        "stretch_reference", "pool_design" if not is_size else None
    )
    # The refit's stretch reference is the normalised Horvitz-Thompson
    # baseline, trimmed or not (--baseline-pi-floor records which).
    accepted_stretch_references = (
        "normalized_horvitz_thompson_w_over_q",
        "normalized_horvitz_thompson_w_over_q_floored",
    )
    add(
        "stretch_reference",
        stretch_reference in accepted_stretch_references if is_size else None,
        stretch_reference,
        " | ".join(accepted_stretch_references),
    )
    selection_receipt = _mapping(size_receipt.get("selection_receipt"))
    add(
        "protected_carriers",
        selection_receipt.get("certainty_count")
        == size_receipt.get("protected_carriers")
        if is_size
        else None,
        selection_receipt.get("certainty_count"),
        size_receipt.get("protected_carriers"),
    )
    add(
        "selection_design",
        selection_receipt.get("design") == "sampford" if is_size else None,
        selection_receipt.get("design"),
        "sampford",
    )
    l0 = _optional_float(size_receipt.get("selection_l0_lambda"))
    add("selection_l0_lambda", l0 is not None if is_size else None, l0, "finite")
    dense = _mapping(size_receipt.get("dense_reference"))
    add(
        "dense_loss",
        dense.get("final_loss") == size_receipt.get("dense_loss") if is_size else None,
        dense.get("final_loss"),
        size_receipt.get("dense_loss"),
    )
    expected_rows = realized if is_size else solve.get("n_households")
    add(
        "h5_households",
        len(run.weights) == expected_rows if run.weights is not None else None,
        len(run.weights) if run.weights is not None else None,
        expected_rows,
    )
    add(
        "positive_weights",
        bool((run.weights["household_weight"] > 0).all())
        if run.weights is not None
        else None,
        (
            int((run.weights["household_weight"] <= 0).sum())
            if run.weights is not None
            else None
        ),
        0,
    )
    add(
        "selection_rows",
        (run.selection is not None and len(run.selection) == realized)
        if is_size
        else None,
        len(run.selection) if run.selection is not None else None,
        realized,
    )
    bad_outputs: list[str] = []
    for name, raw_record in _mapping(manifest.get("outputs")).items():
        record = _mapping(raw_record)
        output_path = run.path / Path(str(record.get("path") or name)).name
        if not output_path.is_file() or _sha256(output_path) != record.get("sha256"):
            bad_outputs.append(str(name))
    add("output_sha256", not bad_outputs, bad_outputs, [])
    return {
        "checks": checks,
        "passed": all(check["status"] != "fail" for check in checks),
    }


def _distribution(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {"min": None, "p10": None, "median": None}
    return {
        "min": float(numeric.min()),
        "p10": float(numeric.quantile(0.10)),
        "median": float(numeric.median()),
    }


def _fit_frame(frame: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_grain: dict[str, Any] = {}
    for grain, group in frame.groupby("grain", sort=True):
        errors = pd.to_numeric(group["abs_relative_error"], errors="coerce")
        valid = group.loc[errors.notna()].copy()
        valid_errors = errors.loc[errors.notna()]
        worst_name = None
        if not valid.empty:
            worst_name = str(valid.loc[valid_errors.idxmax(), "name"])
        by_grain[str(grain)] = {
            "n": int(len(valid_errors)),
            "within_10pct": int((valid_errors <= 0.10).sum()),
            "within_25pct": int((valid_errors <= 0.25).sum()),
            "median_abs": float(valid_errors.median()) if len(valid_errors) else None,
            "max_abs": float(valid_errors.max()) if len(valid_errors) else None,
            "worst_name": worst_name,
        }
    past = frame.loc[
        pd.to_numeric(frame["abs_relative_error"], errors="coerce") > 0.25,
        ["name", "abs_relative_error"],
    ].sort_values(["abs_relative_error", "name"], ascending=[False, True])
    return by_grain, past.to_dict("records")


def fit_tables(run: RowwiseRun) -> dict[str, Any]:
    """Return loss and target-fit summaries for one run."""

    solve = _mapping(run.manifest.get("solve"))
    by_grain, past = _fit_frame(run.targets)
    return {
        "loss": {
            "initial": solve.get("initial_loss"),
            "final": solve.get("final_loss"),
        },
        "by_grain": by_grain,
        "by_family": uk_fit_by_family(run.targets, name_column="name"),
        "rows_past_25pct": past,
        "past_cap": solve.get("past_cap"),
    }


def _source_multiplier(
    weights: pd.DataFrame,
    spine_weights: pd.DataFrame | None,
) -> tuple[int, str]:
    if spine_weights is not None:
        return (
            id_multiplier_for_values(spine_weights["household_id"].tolist()),
            "spine",
        )
    originals = weights.loc[weights["clone_index"] == 0, "household_id"]
    if originals.empty:
        raise ValueError("cannot infer source-id multiplier without clone_index == 0")
    return 10 ** len(str(int(originals.max()))), "inferred_from_clone_zero_ids"


def _ratio_summary(ratios: Sequence[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(ratios, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("weight stretch ratios must be finite and non-empty")
    return {
        "max": float(values.max()),
        "p99": float(np.quantile(values, 0.99)),
        "p50": float(np.quantile(values, 0.50)),
        "share_gt_10": float((values > 10).mean()),
        "share_gt_100": float((values > 100).mean()),
    }


def weight_tables(
    run: RowwiseRun,
    *,
    spine_weights: pd.DataFrame | None = None,
    n_clones: int | None = None,
) -> dict[str, Any]:
    """Summarize shipped weights, source support, and design stretch."""

    manifest_weights = _mapping(run.manifest.get("weights"))
    reference = manifest_weights.get("stretch_reference", "pool_design")
    result: dict[str, Any] = {
        "summary": None,
        "distinct_source_households": None,
        "source_id_multiplier": None,
        "source_id_multiplier_basis": None,
        "stretch": {
            "reference": reference,
            "vs_ht_baseline_max": manifest_weights.get(
                "realized_max_weight_ratio_vs_design"
            ),
            "vs_pool_design": None,
        },
        "clones_per_source": None,
    }
    weight_frame = run.weights
    if weight_frame is None and run.selection is not None:
        weight_frame = run.selection[
            ["household_id", "refit_weight", "clone_index"]
        ].rename(columns={"refit_weight": "household_weight"})
    if weight_frame is not None:
        values = weight_frame["household_weight"].to_numpy(dtype=np.float64)
        result["summary"] = uk_weight_summary(values)
        multiplier, basis = _source_multiplier(weight_frame, spine_weights)
        base_ids = (
            weight_frame["household_id"].to_numpy(dtype=np.int64)
            - weight_frame["clone_index"].to_numpy(dtype=np.int64) * multiplier
        )
        result["distinct_source_households"] = int(np.unique(base_ids).size)
        result["source_id_multiplier"] = multiplier
        result["source_id_multiplier_basis"] = basis
        if run.selection is not None:
            counts = pd.Series(base_ids).value_counts()
            result["clones_per_source"] = {
                "n_sources": int(len(counts)),
                "min": int(counts.min()),
                "p10": float(counts.quantile(0.10)),
                "median": float(counts.median()),
                "p90": float(counts.quantile(0.90)),
                "max": int(counts.max()),
                "by_clone_count": {
                    str(int(key)): int(value)
                    for key, value in counts.value_counts().sort_index().items()
                },
            }
    if run.selection is not None:
        design = run.selection["design_weight"].to_numpy(dtype=np.float64)
        if (design <= 0).any():
            raise ValueError(
                "dataset_size_selection.csv design weights must be positive"
            )
        result["stretch"]["vs_pool_design"] = _ratio_summary(
            run.selection["refit_weight"].to_numpy(dtype=np.float64) / design
        )
    elif spine_weights is not None and weight_frame is not None:
        clones = n_clones or int(
            _mapping(run.manifest.get("parameters")).get("n_clones")
        )
        multiplier = int(result["source_id_multiplier"])
        source_map = spine_weights.set_index("household_id")["household_weight"]
        base_ids = (
            weight_frame["household_id"].to_numpy(dtype=np.int64)
            - weight_frame["clone_index"].to_numpy(dtype=np.int64) * multiplier
        )
        try:
            design = source_map.loc[base_ids].to_numpy(dtype=np.float64) / clones
        except KeyError as exc:
            raise ValueError(
                "spine weights do not cover every source household"
            ) from exc
        ratios = weight_frame["household_weight"].to_numpy(dtype=np.float64) / design
        summary = _ratio_summary(ratios)
        recorded = _optional_float(
            manifest_weights.get("realized_max_weight_ratio_vs_design")
        )
        if recorded is not None and not math.isclose(
            summary["max"], recorded, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(
                "dense-run spine stretch does not match "
                "realized_max_weight_ratio_vs_design"
            )
        result["stretch"]["vs_pool_design"] = summary
    return result


def area_support_tables(run: RowwiseRun) -> dict[str, Any]:
    """Summarize area rows, ESS, source support, and floor breaches."""

    by_level: dict[str, Any] = {}
    for level, group in run.area_support.groupby("geography_level", sort=True):
        rows = pd.to_numeric(group["assigned_households"], errors="coerce")
        ess = pd.to_numeric(group["effective_sample_size"], errors="coerce")
        sources = pd.to_numeric(group["nonzero_source_households"], errors="coerce")
        rows_bad = rows < 50
        ess_bad = ess < 50
        sources_bad = sources < 50
        any_bad = rows_bad | ess_bad | sources_bad
        by_level[str(level)] = {
            "n_areas": int(len(group)),
            "rows": _distribution(rows),
            "ess": _distribution(ess),
            "sources": _distribution(sources),
            "breaches": {
                "rows_lt_50": int(rows_bad.sum()),
                "ess_lt_50": int(ess_bad.sum()),
                "sources_lt_50": int(sources_bad.sum()),
                "any": int(any_bad.sum()),
                "codes": sorted(group.loc[any_bad, "area_code"].astype(str).tolist()),
            },
        }
    support_gate = _mapping(
        _mapping(run.gates.get("gates")).get("uk_local_area_support")
    )
    reviewed = _mapping(support_gate.get("details")).get("reviewed_exclusions", {})
    return {"by_geography_level": by_level, "reviewed_exclusions": reviewed}


def gate_table(run: RowwiseRun) -> list[dict[str, Any]]:
    """Return the six pre-registered gates in a stable order."""

    gates = _mapping(run.gates.get("gates"))
    rows = []
    for gate_id in _GATE_IDS:
        payload = _mapping(gates.get(gate_id))
        rows.append(
            {
                "id": gate_id,
                "status": payload.get("status", "absent"),
                "criticality": payload.get("criticality"),
                "failures": payload.get("failures"),
            }
        )
    return rows


def _delta_summary(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "mean": float(numeric.mean()) if len(numeric) else None,
        "median": float(numeric.median()) if len(numeric) else None,
        "p90": float(numeric.quantile(0.90)) if len(numeric) else None,
    }


def _paired_breakdown(joined: pd.DataFrame, column: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, group in joined.groupby(column, dropna=False, sort=True):
        delta = group["_delta"]
        result[str(key)] = {
            "n": int(len(group)),
            "abs_err_delta": _delta_summary(delta),
            "wins": int((delta <= -1e-4).sum()),
            "ties": int((delta.abs() < 1e-4).sum()),
            "losses": int((delta >= 1e-4).sum()),
        }
    return result


def _pair_frames(
    run_rows: pd.DataFrame,
    reference_rows: pd.DataFrame,
    *,
    run_key: str = "name",
    reference_key: str = "name",
) -> tuple[dict[str, Any], pd.DataFrame]:
    left = run_rows.copy()
    right = reference_rows.copy()
    common_columns = [
        column
        for column in (
            "grain",
            "family",
            "target",
            "relative_error",
            "abs_relative_error",
        )
        if column in right.columns
    ]
    joined = left.merge(
        right[[reference_key, *common_columns]],
        left_on=run_key,
        right_on=reference_key,
        how="inner",
        suffixes=("_run", "_reference"),
    )
    joined["_delta"] = pd.to_numeric(
        joined["abs_relative_error_run"], errors="coerce"
    ) - pd.to_numeric(joined["abs_relative_error_reference"], errors="coerce")
    target_run = pd.to_numeric(joined["target_run"], errors="coerce")
    target_ref = pd.to_numeric(joined["target_reference"], errors="coerce")
    changed = ~np.isclose(target_run, target_ref, rtol=1e-9, atol=0.0, equal_nan=True)
    result = {
        "n_common": int(len(joined)),
        "n_only_run": int(len(set(left[run_key]) - set(right[reference_key]))),
        "n_only_reference": int(len(set(right[reference_key]) - set(left[run_key]))),
        "n_target_value_changed": int(changed.sum()),
        "abs_err_delta": _delta_summary(joined["_delta"]),
        "wins": int((joined["_delta"] <= -1e-4).sum()),
        "ties": int((joined["_delta"].abs() < 1e-4).sum()),
        "losses": int((joined["_delta"] >= 1e-4).sum()),
        "by_grain": _paired_breakdown(joined, "grain_run"),
        "by_family": _paired_breakdown(joined, "family_run"),
    }
    return result, joined


def _red_target_names(run: RowwiseRun) -> set[str]:
    gate = _mapping(_mapping(run.gates.get("gates")).get("uk_local_target_fit"))
    failures = _mapping(_mapping(gate.get("details")).get("failing_targets"))
    names: set[str] = set()
    diagnostics = run.local_diagnostics
    for key in failures:
        pieces = str(key).split("/", 2)
        if len(pieces) != 3:
            continue
        family, area_code, metric = pieces
        matched = diagnostics.loc[
            (diagnostics["family"].astype(str) == family)
            & (diagnostics["area_code"].astype(str) == area_code)
            & (diagnostics["metric"].astype(str) == metric)
        ]
        for target_name in matched["target_name"].astype(str):
            target_rows = run.targets.loc[run.targets["target_name"] == target_name]
            names.update(target_rows["name"].astype(str))
    return names


def paired_targets(run: RowwiseRun, reference: RowwiseRun) -> dict[str, Any]:
    """Compare matched target rows and preserve the reference's red-row ledger."""

    result, joined = _pair_frames(run.targets, reference.targets)
    reference_red = _red_target_names(reference)
    red_rows = joined.loc[joined["name"].isin(reference_red)]
    red_output = [
        {
            "name": str(row["name"]),
            "reference_error": float(row["relative_error_reference"]),
            "run_error": float(row["relative_error_run"]),
        }
        for _, row in red_rows.sort_values("name").iterrows()
    ]
    result["reference_red_rows"] = {
        "n": len(red_output),
        "still_red": sum(abs(row["run_error"]) > 0.25 for row in red_output),
        "now_green": sum(abs(row["run_error"]) <= 0.25 for row in red_output),
        "rows": red_output,
    }
    new_red = joined.loc[
        (pd.to_numeric(joined["abs_relative_error_run"], errors="coerce") > 0.25)
        & ~joined["name"].isin(reference_red)
    ].sort_values(["abs_relative_error_run", "name"], ascending=[False, True])
    result["new_red_rows"] = [
        {
            "name": str(row["name"]),
            "reference_error": float(row["relative_error_reference"]),
            "run_error": float(row["relative_error_run"]),
        }
        for _, row in new_red.head(50).iterrows()
    ]
    worst = joined.sort_values(
        ["abs_relative_error_run", "name"], ascending=[False, True]
    ).head(25)
    result["worst_rows"] = [
        {
            "name": str(row["name"]),
            "target_run": float(row["target_run"]),
            "target_reference": float(row["target_reference"]),
            "run_error": float(row["relative_error_run"]),
            "reference_error": float(row["relative_error_reference"]),
        }
        for _, row in worst.iterrows()
    ]
    return result


def _dense_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    dense = frame.copy()
    if "abs_relative_error" not in dense:
        dense["abs_relative_error"] = pd.to_numeric(
            dense["relative_error"], errors="coerce"
        ).abs()
    if "grain" not in dense:
        dense["grain"] = np.where(dense.get("area_code").isna(), "national", "")
    dense["grain"] = dense["grain"].replace({"la": "local_authority"})
    if "family" not in dense:
        dense["family"] = ""
    if "target" not in dense:
        dense["target"] = np.nan
    dense["_join"] = np.where(
        dense["grain"].astype(str) == "national",
        dense.get("name", pd.Series(index=dense.index, dtype=object)),
        dense.get("target_name", pd.Series(index=dense.index, dtype=object)),
    )
    return dense, "_join", "_join"


def dense_reference_deltas(run: RowwiseRun) -> dict[str, Any] | None:
    """Compare a compact run with the dense solve embedded in its size receipt."""

    if run.dense_reference_diagnostics is None:
        return None
    dense, _, _ = _dense_targets(run.dense_reference_diagnostics)
    compact = run.targets.copy()
    compact["_join"] = np.where(
        compact["grain"] == "national", compact["name"], compact["target_name"]
    )
    paired, joined = _pair_frames(
        compact,
        dense,
        run_key="_join",
        reference_key="_join",
    )
    paired["reference_red_rows"] = {"n": 0, "still_red": 0, "now_green": 0, "rows": []}
    paired["new_red_rows"] = []
    paired["worst_rows"] = [
        {
            "name": str(row["name"]),
            "run_error": float(row["relative_error_run"]),
            "reference_error": float(row["relative_error_reference"]),
        }
        for _, row in joined.sort_values("abs_relative_error_run", ascending=False)
        .head(25)
        .iterrows()
    ]
    size = _mapping(_mapping(run.manifest.get("solve")).get("dataset_size"))
    dense_summary = _mapping(size.get("dense_reference"))
    return {
        **paired,
        "dense_loss": size.get("dense_loss"),
        "compact_loss": size.get("compact_loss"),
        "max_target_scaled_change": size.get("max_target_scaled_change"),
        "weights": {
            "dense": dense_summary.get("weights"),
            "compact": (
                uk_weight_summary(
                    run.weights["household_weight"].to_numpy(dtype=np.float64)
                )
                if run.weights is not None
                else None
            ),
        },
    }


def _surface_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Find national recomputation rows.

    Current evaluator output uses ``national_rows`` with ``our_name`` and
    ``candidate_estimate``.  The fallback contract accepts ``rows`` entries
    with ``name``, ``estimate``, and ``target``.
    """

    for key in ("national_rows", "rows"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
    return []


def frozen_vs_recomputed(
    run: RowwiseRun,
    surface_evaluation: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Compare frozen national contributions with recomputed surface values."""

    payload = (
        _read_json(Path(surface_evaluation))
        if isinstance(surface_evaluation, (str, Path))
        else surface_evaluation
    )
    recomputed: dict[str, float] = {}
    for row in _surface_rows(payload):
        name = next(
            (
                str(row[key])
                for key in (
                    "name",
                    "our_name",
                    "contract_target_id",
                    "target_name",
                    "id",
                )
                if row.get(key)
            ),
            "",
        )
        estimate = next(
            (
                _optional_float(row.get(key))
                for key in (
                    "estimate",
                    "candidate_estimate",
                    "recomputed_estimate",
                    "final_estimate",
                )
                if row.get(key) is not None
            ),
            None,
        )
        if name and estimate is not None:
            recomputed[name] = recomputed.get(name, 0.0) + estimate
    national = run.targets.loc[run.targets["grain"] == "national"]
    output_rows: list[dict[str, Any]] = []
    divergences: list[float] = []
    for _, row in national.iterrows():
        candidates = (str(row["name"]), str(row["target_name"]))
        matched_name = next((name for name in candidates if name in recomputed), None)
        frozen = _optional_float(row["final_estimate"])
        if matched_name is None or frozen is None:
            continue
        value = recomputed[matched_name]
        divergence = abs(value - frozen) / max(abs(frozen), 1e-12)
        divergences.append(divergence)
        if divergence > 0.01:
            output_rows.append(
                {
                    "name": str(row["name"]),
                    "frozen": frozen,
                    "recomputed": value,
                }
            )
    return {
        "n_rows": int(len(national)),
        "n_matched": len(divergences),
        "max_abs_rel_divergence": max(divergences) if divergences else None,
        "rows_over_1pct": output_rows,
    }


def footprint(run: RowwiseRun) -> dict[str, int | float | None]:
    """Return the captured wall time, peak RSS, and household H5 size."""

    return dict(run.runtime)


PRE_REGISTERED_OUTCOMES_V1: tuple[dict[str, Any], ...] = (
    {
        "id": "dense_reference_loss",
        "label": "Dense reference loss",
        "reference": 0.01425,
        "expectation": "≤ 0.014",
        "red_flag": "> 0.0285",
    },
    {
        "id": "compact_loss_ratio",
        "label": "Compact / dense loss",
        "reference": None,
        "expectation": "≤ 2",
        "red_flag": "> 5",
    },
    {
        "id": "max_target_scaled_change",
        "label": "Maximum target-scaled change",
        "reference": None,
        "expectation": "< 0.5",
        "red_flag": "≥ 0.5",
    },
    {
        "id": "national_within_10pct",
        "label": "National rows within 10%",
        "reference": "340 of 359",
        "expectation": "≥ 335",
        "red_flag": "< 330",
    },
    {
        "id": "constituency_within_10pct",
        "label": "Constituency share within 10%",
        "reference": 0.9978,
        "expectation": "≥ 0.98",
        "red_flag": "< 0.90",
    },
    {
        "id": "local_authority_within_10pct",
        "label": "Local-authority share within 10%",
        "reference": 0.9774,
        "expectation": "≥ 0.95",
        "red_flag": "< 0.90",
    },
    {
        "id": "rows_past_25pct",
        "label": "Rows past 25%",
        "reference": 42,
        "expectation": "≤ 60",
        "red_flag": "> 60",
    },
    {
        "id": "kish_ess",
        "label": "Kish effective sample size",
        "reference": 129236,
        "expectation": "20000–40000",
        "red_flag": "< 10000",
    },
    {
        "id": "max_to_median_positive_weight",
        "label": "Maximum / median positive weight",
        "reference": 400.03,
        "expectation": "report against 100",
        "red_flag": None,
    },
    {
        "id": "stretch_vs_pool_design_share_gt_100",
        "label": "Pool-design stretch share above 100",
        "reference": 0,
        "expectation": "report",
        "red_flag": "> 0.01",
    },
    {
        "id": "constituency_ess_min",
        "label": "Minimum constituency ESS",
        "reference": 63.9,
        "expectation": "~20",
        "red_flag": None,
    },
    {
        "id": "area_support_breach_share",
        "label": "Area-support breach share",
        "reference": "0.002 (2 of 1011, both excluded)",
        "expectation": "> 0 likely",
        "red_flag": "> 0.25",
    },
    {
        "id": "gates_release_blocking_failed",
        "label": "Failed release-blocking gates",
        "reference": "2 (as diagnostic)",
        "expectation": "area_support + ratio",
        "red_flag": "ladder gate failed",
    },
    {
        "id": "frozen_vs_recomputed_max",
        "label": "Maximum frozen/recomputed divergence",
        "reference": None,
        "expectation": "≤ 0.01",
        "red_flag": "> 0.05",
    },
    {
        "id": "wall_seconds",
        "label": "Wall seconds",
        "reference": 11122,
        "expectation": "+1–5 h",
        "red_flag": None,
    },
    {
        "id": "peak_rss_bytes",
        "label": "Peak RSS bytes",
        "reference": 11103666176,
        "expectation": "+0.3 GB",
        "red_flag": "> 20e9",
    },
    {
        "id": "h5_bytes",
        "label": "H5 bytes",
        "reference": 2355789273,
        "expectation": "~165 MB",
        "red_flag": None,
    },
)


def _outcome_observed(
    metric_id: str,
    *,
    fit: Mapping[str, Any],
    weights: Mapping[str, Any],
    areas: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    dense: Mapping[str, Any],
    surface: Mapping[str, Any],
    run_footprint: Mapping[str, Any],
) -> object:
    grains = _mapping(fit.get("by_grain"))
    if metric_id == "dense_reference_loss":
        return dense.get("dense_loss")
    if metric_id == "compact_loss_ratio":
        dense_loss = _optional_float(dense.get("dense_loss"))
        compact_loss = _optional_float(dense.get("compact_loss"))
        return (
            compact_loss / dense_loss
            if dense_loss and compact_loss is not None
            else None
        )
    if metric_id == "max_target_scaled_change":
        return dense.get("max_target_scaled_change")
    if metric_id == "national_within_10pct":
        return _mapping(grains.get("national")).get("within_10pct")
    if metric_id in {"constituency_within_10pct", "local_authority_within_10pct"}:
        grain = metric_id.removesuffix("_within_10pct")
        row = _mapping(grains.get(grain))
        n = row.get("n")
        return row.get("within_10pct") / n if n else None
    if metric_id == "rows_past_25pct":
        return len(fit.get("rows_past_25pct") or [])
    if metric_id == "kish_ess":
        return _mapping(weights.get("summary")).get("effective_sample_size")
    if metric_id == "max_to_median_positive_weight":
        return _mapping(weights.get("summary")).get("max_to_median_positive_weight")
    if metric_id == "stretch_vs_pool_design_share_gt_100":
        return _mapping(_mapping(weights.get("stretch")).get("vs_pool_design")).get(
            "share_gt_100"
        )
    if metric_id == "constituency_ess_min":
        return _mapping(
            _mapping(_mapping(areas.get("by_geography_level")).get("constituency")).get(
                "ess"
            )
        ).get("min")
    if metric_id == "area_support_breach_share":
        levels = _mapping(areas.get("by_geography_level"))
        count = sum(
            int(_mapping(_mapping(row).get("breaches")).get("any") or 0)
            for row in levels.values()
        )
        total = sum(int(_mapping(row).get("n_areas") or 0) for row in levels.values())
        return count / total if total else None
    if metric_id == "gates_release_blocking_failed":
        return sum(
            row.get("criticality") == "release_blocking"
            and row.get("status") != "passed"
            for row in gates
        )
    if metric_id == "frozen_vs_recomputed_max":
        return surface.get("max_abs_rel_divergence")
    if metric_id in {"wall_seconds", "peak_rss_bytes", "h5_bytes"}:
        return run_footprint.get(metric_id)
    return None


def _flag(metric_id: str, value: object, gates: Sequence[Mapping[str, Any]]) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return "not_measured"
    if metric_id == "dense_reference_loss":
        return "red" if numeric > 0.0285 else "ok" if numeric <= 0.014 else "watch"
    if metric_id == "compact_loss_ratio":
        return "red" if numeric > 5 else "ok" if numeric <= 2 else "watch"
    if metric_id == "max_target_scaled_change":
        return "red" if numeric >= 0.5 else "ok"
    if metric_id == "national_within_10pct":
        return "red" if numeric < 330 else "ok" if numeric >= 335 else "watch"
    if metric_id == "constituency_within_10pct":
        return "red" if numeric < 0.90 else "ok" if numeric >= 0.98 else "watch"
    if metric_id == "local_authority_within_10pct":
        return "red" if numeric < 0.90 else "ok" if numeric >= 0.95 else "watch"
    if metric_id == "rows_past_25pct":
        return "red" if numeric > 60 else "ok"
    if metric_id == "kish_ess":
        return (
            "red" if numeric < 10000 else "ok" if 20000 <= numeric <= 40000 else "watch"
        )
    if metric_id == "max_to_median_positive_weight":
        return "ok" if numeric <= 100 else "watch"
    if metric_id == "stretch_vs_pool_design_share_gt_100":
        return "red" if numeric > 0.01 else "ok"
    if metric_id == "area_support_breach_share":
        return "red" if numeric > 0.25 else "watch" if numeric == 0 else "ok"
    if metric_id == "gates_release_blocking_failed":
        ladder = next((row for row in gates if row.get("id") == _GATE_IDS[0]), {})
        if ladder.get("status") not in {"passed", "absent"}:
            return "red"
        return "ok" if numeric == 2 else "watch"
    if metric_id == "frozen_vs_recomputed_max":
        return "red" if numeric > 0.05 else "ok" if numeric <= 0.01 else "watch"
    if metric_id == "peak_rss_bytes":
        return "red" if numeric > 20e9 else "ok"
    return "ok"


def summarize(
    run: RowwiseRun,
    *,
    references: Mapping[str, RowwiseRun],
    dense_deltas: Mapping[str, Any] | None,
    incumbent_score: Mapping[str, Any] | None = None,
    surface: Mapping[str, Any] | None = None,
    downstream: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the candidate-only scorecard without applying a release verdict."""

    fit = fit_tables(run)
    weights = weight_tables(run)
    areas = area_support_tables(run)
    gates = gate_table(run)
    dense = _mapping(dense_deltas)
    surface_map = _mapping(surface)
    run_footprint = footprint(run)
    outcomes = []
    for definition in PRE_REGISTERED_OUTCOMES_V1:
        observed = _outcome_observed(
            str(definition["id"]),
            fit=fit,
            weights=weights,
            areas=areas,
            gates=gates,
            dense=dense,
            surface=surface_map,
            run_footprint=run_footprint,
        )
        outcomes.append(
            {
                **definition,
                "observed": observed,
                "flag": _flag(str(definition["id"]), observed, gates),
            }
        )
    reference_tables: dict[str, Any] = {}
    reference_observed: dict[str, dict[str, object]] = {}
    for label, reference in references.items():
        reference_fit = fit_tables(reference)
        reference_weights = weight_tables(reference)
        reference_areas = area_support_tables(reference)
        reference_gates = gate_table(reference)
        reference_footprint = footprint(reference)
        paired = paired_targets(run, reference)
        reference_tables[label] = {
            "fit": reference_fit,
            "weights": reference_weights,
            "area_support": reference_areas,
            "gates": reference_gates,
            "footprint": reference_footprint,
            "paired_targets": paired,
        }
        reference_observed[label] = {
            str(definition["id"]): _outcome_observed(
                str(definition["id"]),
                fit=reference_fit,
                weights=reference_weights,
                areas=reference_areas,
                gates=reference_gates,
                dense={
                    "dense_loss": _mapping(reference.manifest.get("solve")).get(
                        "final_loss"
                    )
                },
                surface={},
                run_footprint=reference_footprint,
            )
            for definition in PRE_REGISTERED_OUTCOMES_V1
        }
    dense_observed: dict[str, object] = {}
    if run.dense_reference_diagnostics is not None:
        dense_frame, _, _ = _dense_targets(run.dense_reference_diagnostics)
        if "name" not in dense_frame:
            dense_frame["name"] = dense_frame["_join"]
        else:
            dense_frame["name"] = dense_frame["name"].fillna(dense_frame["_join"])
        dense_by_grain, dense_past = _fit_frame(dense_frame)
        dense_fit = {"by_grain": dense_by_grain, "rows_past_25pct": dense_past}
        dense_weights = {"summary": _mapping(dense.get("weights")).get("dense")}
        dense_observed = {
            str(definition["id"]): _outcome_observed(
                str(definition["id"]),
                fit=dense_fit,
                weights=dense_weights,
                areas={},
                gates=[],
                dense={"dense_loss": dense.get("dense_loss")},
                surface={},
                run_footprint={},
            )
            for definition in PRE_REGISTERED_OUTCOMES_V1
        }
    run_observed = {str(row["id"]): row["observed"] for row in outcomes}
    scorecard_columns = [run.label, "dense_reference", *references]
    scorecard = {
        "columns": scorecard_columns,
        "rows": [
            {
                "id": definition["id"],
                "label": definition["label"],
                "values": {
                    run.label: run_observed.get(str(definition["id"])),
                    "dense_reference": dense_observed.get(str(definition["id"])),
                    **{
                        label: values.get(str(definition["id"]))
                        for label, values in reference_observed.items()
                    },
                },
            }
            for definition in PRE_REGISTERED_OUTCOMES_V1
        ],
    }
    return {
        "identity": {
            "label": run.label,
            "path": str(run.path),
            "git_commit": run.manifest.get("git_commit"),
            "identity": run.manifest.get("identity"),
            "parameters": run.manifest.get("parameters"),
        },
        "fit": fit,
        "weights": weights,
        "area_support": areas,
        "gates": gates,
        "footprint": run_footprint,
        "dense_reference": dense_deltas,
        "references": {
            label: tables["paired_targets"]
            for label, tables in reference_tables.items()
        },
        "reference_tables": reference_tables,
        "scorecard": scorecard,
        "incumbent_score": incumbent_score,
        "surface": surface,
        "downstream": downstream,
        "pre_registered_outcomes": outcomes,
    }
