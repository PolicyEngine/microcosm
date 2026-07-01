"""Export a US Populace H5 from a base frame plus saved L0/refit weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from populace.frame import US_SCHEMA, Frame, MassChange, WeightKind, Weights
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

US_RELEASE_REQUIRED_TAX_UNIT_SOURCE_COLUMNS = (
    "takes_up_aca_if_eligible",
    "selected_marketplace_plan_benchmark_ratio",
)


@dataclass(frozen=True)
class L0RefitWeights:
    """A saved post-L0 refit solution aligned to candidate entity rows."""

    weights: np.ndarray
    selected_mask: np.ndarray
    metadata: dict[str, Any]

    @property
    def selected_weights(self) -> np.ndarray:
        return self.weights[self.selected_mask]


def _metadata_from_npz(value: np.ndarray | None) -> dict[str, Any]:
    if value is None:
        return {}
    if value.shape != ():
        raise ValueError("metadata_json must be a scalar JSON string.")
    raw = value.item()
    if isinstance(raw, bytes):
        raw = raw.decode()
    if not isinstance(raw, str):
        raise ValueError("metadata_json must be a scalar JSON string.")
    metadata = json.loads(raw)
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to an object.")
    return metadata


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return {
        "path": str(source),
        "size_bytes": int(source.stat().st_size),
        "sha256": _sha256(source),
    }


def load_l0_refit_npz(
    path: str | Path,
    *,
    expected_candidate_records: int,
    weight_key: str = "weights",
    zero_weight_tolerance: float = 0.0,
) -> L0RefitWeights:
    """Load and validate saved L0/refit weights.

    The returned full-length vector stays aligned to the candidate household
    table. Selection is the positive-weight support after ``zero_weight_tolerance``.
    """

    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        if weight_key not in payload.files:
            raise ValueError(f"{source} is missing required key {weight_key!r}.")
        weights = np.asarray(payload[weight_key], dtype=np.float64)
        metadata = _metadata_from_npz(
            payload["metadata_json"] if "metadata_json" in payload.files else None
        )

    if weights.shape != (expected_candidate_records,):
        raise ValueError(
            f"{weight_key!r} must have shape {(expected_candidate_records,)}, "
            f"got {weights.shape}."
        )
    if not np.isfinite(weights).all():
        raise ValueError(f"{weight_key!r} must be finite.")
    if (weights < 0).any():
        raise ValueError(f"{weight_key!r} must be non-negative.")
    if zero_weight_tolerance < 0:
        raise ValueError("zero_weight_tolerance must be non-negative.")

    selected_mask = weights > zero_weight_tolerance
    n_selected = int(selected_mask.sum())
    if n_selected == 0:
        raise ValueError("L0/refit weights select zero candidate rows.")

    metadata_candidate_records = metadata.get("candidate_records")
    if (
        metadata_candidate_records is not None
        and int(metadata_candidate_records) != expected_candidate_records
    ):
        raise ValueError(
            "metadata candidate_records does not match base frame: "
            f"{metadata_candidate_records} != {expected_candidate_records}."
        )
    metadata_selected = metadata.get("n_selected", metadata.get("budget_achieved"))
    if metadata_selected is not None and int(metadata_selected) != n_selected:
        raise ValueError(
            "metadata selected count does not match positive-weight support: "
            f"{metadata_selected} != {n_selected}."
        )
    metadata_weight_entity = metadata.get("weight_entity")
    if metadata_weight_entity is not None and metadata_weight_entity != "household":
        raise ValueError(
            "US L0/refit export currently supports household weights only, "
            f"got {metadata_weight_entity!r}."
        )

    return L0RefitWeights(
        weights=weights,
        selected_mask=selected_mask,
        metadata=metadata,
    )


def attach_l0_refit_weights(
    base_frame: Frame,
    solution: L0RefitWeights,
) -> Frame:
    """Return the selected base-frame support with post-L0 refit weights."""

    schema = base_frame.schema
    weight_entity = "household"
    if weight_entity not in schema.group_entities:
        raise ValueError("US L0/refit export requires a household group entity.")
    if solution.weights.shape != (base_frame.n(weight_entity),):
        raise ValueError(
            "L0/refit weights must align to household rows: "
            f"{solution.weights.shape} != {(base_frame.n(weight_entity),)}."
        )

    selected_ids = base_frame.table(weight_entity)[
        schema.id_column(weight_entity)
    ].to_numpy()[solution.selected_mask]
    return attach_l0_refit_entity_weights(
        base_frame,
        weight_entity=weight_entity,
        selected_entity_ids=selected_ids,
        selected_weights=solution.selected_weights,
        reason="US L0/refit saved-weight export",
    )


def attach_l0_refit_entity_weights(
    base_frame: Frame,
    *,
    weight_entity: str,
    selected_entity_ids: np.ndarray,
    selected_weights: np.ndarray,
    reason: str,
) -> Frame:
    """Return selected support with post-L0 refit weights for one entity."""

    schema = base_frame.schema
    selected_ids = np.asarray(selected_entity_ids)
    selected_weights = np.asarray(selected_weights, dtype=np.float64)
    if selected_ids.shape != selected_weights.shape:
        raise ValueError(
            "Selected entity ids and weights must have the same shape: "
            f"{selected_ids.shape} != {selected_weights.shape}."
        )
    if selected_weights.size == 0:
        raise ValueError("L0/refit selected support is empty.")
    if not np.isfinite(selected_weights).all():
        raise ValueError("L0/refit selected weights must be finite.")
    if (selected_weights < 0).any():
        raise ValueError("L0/refit selected weights must be non-negative.")
    if weight_entity == schema.person_entity:
        person_ids = base_frame.person[schema.person_id_column].to_numpy()
        person_mask = np.isin(person_ids, selected_ids)
    elif weight_entity in schema.group_entities:
        membership = base_frame.person[
            schema.membership_column(weight_entity)
        ].to_numpy()
        person_mask = np.isin(membership, selected_ids)
    else:
        raise ValueError(f"L0/refit export cannot map weight entity {weight_entity!r}.")
    selected_base = base_frame.select(person_mask)
    exported_ids = selected_base.table(weight_entity)[
        schema.id_column(weight_entity)
    ].to_numpy()
    if not np.array_equal(exported_ids, selected_ids):
        raise ValueError(
            "Selected support is not aligned with the base-frame export support "
            f"for {weight_entity!r}."
        )
    return selected_base.with_weights(
        weight_entity,
        Weights(selected_weights, WeightKind.CALIBRATED),
        mass=MassChange(
            factor=selected_weights.sum()
            / selected_base.weights_for(weight_entity).total,
            reason=reason,
        ),
    )


def assert_required_us_release_source_columns(
    frame: Frame,
    *,
    columns: tuple[str, ...] = US_RELEASE_REQUIRED_TAX_UNIT_SOURCE_COLUMNS,
) -> None:
    """Require source-stage tax-unit columns needed by US release gates."""

    tax_unit = frame.table("tax_unit")
    failures: list[str] = []
    for column in columns:
        if column not in tax_unit.columns:
            failures.append(f"{column}: missing")
            continue
        unique = tax_unit[column].dropna().unique()
        if len(unique) < 2:
            failures.append(f"{column}: not nonconstant")
    if failures:
        raise ValueError(
            "US L0/refit release export requires source-stage tax-unit columns: "
            + "; ".join(failures)
        )


def load_us_frame(path: str | Path) -> Frame:
    """Load a PolicyEngine-US single-year H5 into a Populace frame."""

    from policyengine_us.data import USSingleYearDataset

    dataset = USSingleYearDataset(file_path=str(path))
    tables = {
        "person": dataset.person.copy(),
        "household": dataset.household.copy(),
        "tax_unit": dataset.tax_unit.copy(),
        "spm_unit": dataset.spm_unit.copy(),
        "family": dataset.family.copy(),
        "marital_unit": dataset.marital_unit.copy(),
    }
    weights = tables["household"].pop("household_weight").to_numpy(dtype=np.float64)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
    )


def copy_populace_root_attrs(
    source_h5: str | Path,
    destination_h5: str | Path,
) -> tuple[str, ...]:
    """Copy Populace-owned root attrs from the base H5 to the exported H5."""

    import h5py

    copied: list[str] = []
    with (
        h5py.File(source_h5, "r") as source,
        h5py.File(destination_h5, "a") as destination,
    ):
        for name, value in source.attrs.items():
            if not str(name).startswith("populace_"):
                continue
            destination.attrs[name] = value
            copied.append(str(name))
    return tuple(copied)


def export_us_l0_refit_h5(
    *,
    base_h5: str | Path,
    weights_npz: str | Path,
    output_h5: str | Path,
    period: int = 2024,
    weight_key: str = "weights",
    zero_weight_tolerance: float = 0.0,
    summary_json: str | Path | None = None,
    require_source_columns: bool = True,
    root_attrs_h5: str | Path | None = None,
) -> dict[str, Any]:
    """Write a selected US H5 from a base H5 and saved L0/refit weights."""

    root_attrs_source = (
        Path(root_attrs_h5) if root_attrs_h5 is not None else Path(base_h5)
    )
    base_frame = load_us_frame(base_h5)
    solution = load_l0_refit_npz(
        weights_npz,
        expected_candidate_records=base_frame.n("household"),
        weight_key=weight_key,
        zero_weight_tolerance=zero_weight_tolerance,
    )
    export_frame = attach_l0_refit_weights(base_frame, solution)
    if require_source_columns:
        assert_required_us_release_source_columns(export_frame)
    destination = Path(output_h5)
    destination.parent.mkdir(parents=True, exist_ok=True)
    PolicyEngineUSEngine().write_dataset(export_frame, destination, period=period)
    copied_attrs = copy_populace_root_attrs(root_attrs_source, destination)
    summary = {
        "schema_version": 1,
        "kind": "us_l0_refit_h5_export",
        "base_h5": _file_manifest(base_h5),
        "root_attrs_h5": _file_manifest(root_attrs_source),
        "weights_npz": _file_manifest(weights_npz),
        "output_h5": _file_manifest(destination),
        "period": int(period),
        "weight_key": weight_key,
        "required_source_columns": list(US_RELEASE_REQUIRED_TAX_UNIT_SOURCE_COLUMNS),
        "required_source_columns_checked": bool(require_source_columns),
        "candidate_households": int(base_frame.n("household")),
        "selected_households": int(export_frame.n("household")),
        "selected_weight_sum": float(export_frame.weights_for("household").total),
        "copied_root_attrs": list(copied_attrs),
        "metadata": solution.metadata,
    }
    summary_destination = (
        Path(summary_json)
        if summary_json is not None
        else destination.with_suffix(".l0_refit_export_summary.json")
    )
    summary["summary_json_path"] = str(summary_destination)
    summary_destination.parent.mkdir(parents=True, exist_ok=True)
    summary_destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    summary["summary_json"] = _file_manifest(summary_destination)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a PolicyEngine-US H5 by attaching saved L0/refit weights "
            "to a Populace base support frame."
        )
    )
    parser.add_argument("--base-h5", required=True, type=Path)
    parser.add_argument("--weights-npz", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--period", type=int, default=2024)
    parser.add_argument("--weight-key", default="weights")
    parser.add_argument("--zero-weight-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--summary-json",
        type=Path,
        help=(
            "Path for the reconstruction manifest. Defaults to a "
            ".l0_refit_export_summary.json file beside --output-h5."
        ),
    )
    parser.add_argument(
        "--root-attrs-h5",
        type=Path,
        help=(
            "Optional H5 whose Populace-owned root attrs are copied to the "
            "exported dataset. Defaults to --base-h5."
        ),
    )
    parser.add_argument(
        "--allow-missing-source-columns",
        action="store_true",
        help=(
            "Diagnostic escape hatch. By default, reconstruction requires the "
            "US release source-stage tax-unit columns used by release gates."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    summary = export_us_l0_refit_h5(
        base_h5=args.base_h5,
        weights_npz=args.weights_npz,
        output_h5=args.output_h5,
        period=args.period,
        weight_key=args.weight_key,
        zero_weight_tolerance=args.zero_weight_tolerance,
        summary_json=args.summary_json,
        require_source_columns=not args.allow_missing_source_columns,
        root_attrs_h5=args.root_attrs_h5,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


__all__ = [
    "L0RefitWeights",
    "attach_l0_refit_entity_weights",
    "attach_l0_refit_weights",
    "assert_required_us_release_source_columns",
    "copy_populace_root_attrs",
    "export_us_l0_refit_h5",
    "load_l0_refit_npz",
    "load_us_frame",
    "main",
]


if __name__ == "__main__":
    main()
