"""Export a US Populace H5 from a base frame plus saved L0/refit weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from populace.build.gates import input_mass_parity_gate
from populace.build.us_runtime.alimony import US_ALIMONY_NONCONSTANT_PERSON_COLUMNS
from populace.build.us_runtime.casualty_losses import (
    US_CASUALTY_LOSS_NONCONSTANT_PERSON_COLUMNS,
)
from populace.build.us_runtime.childcare import US_CHILDCARE_OUTPUT_COLUMNS
from populace.build.us_runtime.congressional_district_geography import (
    CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
)
from populace.build.us_runtime.education_inputs import (
    US_EDUCATION_INPUTS_NONCONSTANT_PERSON_COLUMNS,
)
from populace.build.us_runtime.eligibility_inputs import (
    US_ELIGIBILITY_INPUTS_NONCONSTANT_PERSON_COLUMNS,
)
from populace.build.us_runtime.geography_ladder import (
    US_GEOGRAPHY_LADDER_COLUMNS,
    us_geography_ladder_gate,
)
from populace.build.us_runtime.hours_worked import (
    US_HOURS_WORKED_NONCONSTANT_PERSON_COLUMNS,
)
from populace.build.us_runtime.immigration import (
    US_IMMIGRATION_NONCONSTANT_PERSON_COLUMNS,
)
from populace.build.us_runtime.input_mass import us_input_mass_totals
from populace.build.us_runtime.misc_itemized import (
    US_MISC_ITEMIZED_NONCONSTANT_PERSON_COLUMNS,
)
from populace.build.us_runtime.org_wages import (
    US_ORG_WAGES_NONCONSTANT_PERSON_COLUMNS,
)
from populace.build.us_runtime.pregnancy import (
    US_PREGNANCY_NONCONSTANT_PERSON_COLUMNS,
)
from populace.build.us_runtime.qbi_inputs import (
    US_QBI_NONCONSTANT_PERSON_COLUMNS,
)
from populace.build.us_runtime.retirement_contributions import (
    US_RETIREMENT_CONTRIBUTION_NONCONSTANT_PERSON_COLUMNS,
)
from populace.build.us_runtime.scf_auto_loans import (
    US_SCF_AUTO_LOAN_NONCONSTANT_HOUSEHOLD_COLUMNS,
)
from populace.build.us_runtime.scf_wealth import (
    US_SCF_WEALTH_NONCONSTANT_PERSON_COLUMNS,
)
from populace.build.us_runtime.sipp_tips import (
    US_SIPP_TIPS_NONCONSTANT_PERSON_COLUMNS,
)
from populace.build.us_runtime.snap_discretionary_exemption import (
    US_SNAP_DISCRETIONARY_EXEMPTION_NONCONSTANT_PERSON_COLUMNS,
)
from populace.frame import US_SCHEMA, Frame, MassChange, WeightKind, Weights
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

US_RELEASE_REQUIRED_TAX_UNIT_SOURCE_COLUMNS = (
    "takes_up_aca_if_eligible",
    "selected_marketplace_plan_benchmark_ratio",
)

US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS = (
    *US_IMMIGRATION_NONCONSTANT_PERSON_COLUMNS,
    *US_HOURS_WORKED_NONCONSTANT_PERSON_COLUMNS,
    *US_ELIGIBILITY_INPUTS_NONCONSTANT_PERSON_COLUMNS,
    *US_PREGNANCY_NONCONSTANT_PERSON_COLUMNS,
    *US_SNAP_DISCRETIONARY_EXEMPTION_NONCONSTANT_PERSON_COLUMNS,
    *US_SCF_WEALTH_NONCONSTANT_PERSON_COLUMNS,
    *US_SIPP_TIPS_NONCONSTANT_PERSON_COLUMNS,
    *US_ALIMONY_NONCONSTANT_PERSON_COLUMNS,
    *US_CASUALTY_LOSS_NONCONSTANT_PERSON_COLUMNS,
    *US_MISC_ITEMIZED_NONCONSTANT_PERSON_COLUMNS,
    *US_EDUCATION_INPUTS_NONCONSTANT_PERSON_COLUMNS,
    *US_RETIREMENT_CONTRIBUTION_NONCONSTANT_PERSON_COLUMNS,
    *US_QBI_NONCONSTANT_PERSON_COLUMNS,
    *US_ORG_WAGES_NONCONSTANT_PERSON_COLUMNS,
)

US_RELEASE_REQUIRED_SPM_UNIT_SOURCE_COLUMNS = US_CHILDCARE_OUTPUT_COLUMNS

#: The geography spine a US release carries by default: state and district,
#: plus the block-anchored ladder (populace #275). A release missing or
#: constant on any of these silently loses local computability — the same
#: failure family as #225's everyone-is-a-citizen surface.
US_RELEASE_REQUIRED_HOUSEHOLD_SOURCE_COLUMNS = (
    "state_fips",
    CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
    *US_GEOGRAPHY_LADDER_COLUMNS,
)

# Unlike the geography spine above, these household inputs must carry signal:
# presence-only would accept the engine's broadcast-zero defaults and silently
# return the OBBBA auto-loan provision to a structural zero.
US_RELEASE_REQUIRED_HOUSEHOLD_NONCONSTANT_SOURCE_COLUMNS = (
    US_SCF_AUTO_LOAN_NONCONSTANT_HOUSEHOLD_COLUMNS
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
    person_columns: tuple[str, ...] = US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
    spm_unit_columns: tuple[str, ...] = US_RELEASE_REQUIRED_SPM_UNIT_SOURCE_COLUMNS,
    household_columns: tuple[str, ...] = (US_RELEASE_REQUIRED_HOUSEHOLD_SOURCE_COLUMNS),
    household_nonconstant_columns: tuple[str, ...] = (
        US_RELEASE_REQUIRED_HOUSEHOLD_NONCONSTANT_SOURCE_COLUMNS
    ),
) -> None:
    """Require source-stage columns needed by US release gates.

    Tax-unit columns come from the ACA Marketplace source stage; person
    columns are the SSN/immigration surface (a missing or constant
    ``ssn_card_type`` reproduces the everyone-is-a-citizen failure of
    populace issue #225); household columns are the geography spine (a
    release without the block-anchored ladder of populace #275 cannot be
    filtered below state or recompute county-driven programs). Household
    columns are presence-checked only — their value quality is the
    geography-ladder gate's job.
    """

    failures: list[str] = []
    for entity, required, check_nonconstant in (
        ("tax_unit", columns, True),
        ("person", person_columns, True),
        ("spm_unit", spm_unit_columns, True),
        ("household", household_columns, False),
        ("household", household_nonconstant_columns, True),
    ):
        table = frame.table(entity)
        for column in required:
            if column not in table.columns:
                failures.append(f"{entity}.{column}: missing")
                continue
            if not check_nonconstant:
                continue
            unique = table[column].dropna().unique()
            # A one-household test/export can still carry real (non-default)
            # auto-loan signal even though two distinct values are impossible.
            # For normal multi-household releases these columns must be truly
            # nonconstant; an all-zero broadcast always fails.
            single_nondefault_auto_value = (
                entity == "household"
                and column in household_nonconstant_columns
                and len(table) == 1
                and len(unique) == 1
                and bool(unique[0])
            )
            single_nondefault_spm_value = (
                entity == "spm_unit"
                and column in spm_unit_columns
                and len(table) == 1
                and len(unique) == 1
                and bool(unique[0])
            )
            if len(unique) < 2 and not (
                single_nondefault_auto_value or single_nondefault_spm_value
            ):
                failures.append(f"{entity}.{column}: not nonconstant")
    if failures:
        raise ValueError(
            "US L0/refit release export requires source-stage columns: "
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
    reference_h5: str | Path | None = None,
    input_mass_relative_tolerance: float = 0.5,
    input_mass_minimum_reference_total: float = 0.0,
    input_mass_reviewed_exclusions: Mapping[str, str] | None = None,
    require_input_mass_parity: bool = True,
    require_geography_ladder: bool = True,
) -> dict[str, Any]:
    """Write a selected US H5 from a base H5 and saved L0/refit weights.

    The selected support must keep the persisted input mass of its parent:
    per-column weighted totals of the export are gated against the base H5
    (or against ``reference_h5``, e.g. the certified dense release) via
    :func:`populace.build.gates.input_mass_parity_gate`, so a selection that
    zeroes an untargeted input base (populace issue #278) fails instead of
    shipping. The geography-ladder gate runs on the selected support with
    its calibrated weights (populace #275/#34): a release whose spine is
    inconsistent or whose NYC mass collapsed fails by default.
    """

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
    geography_ladder_gate = us_geography_ladder_gate(
        export_frame.table("household"),
        export_frame.weights_for("household").values,
    )
    if require_geography_ladder and not geography_ladder_gate.passed:
        raise ValueError(
            "US L0/refit release export failed the geography-ladder gate: "
            + "; ".join(geography_ladder_gate.failures)
        )
    reference_frame = (
        load_us_frame(reference_h5) if reference_h5 is not None else base_frame
    )
    input_mass_gate = input_mass_parity_gate(
        us_input_mass_totals(export_frame),
        us_input_mass_totals(reference_frame),
        candidate_name="l0_refit_export",
        reference_name="reference_h5" if reference_h5 is not None else "base_h5",
        relative_tolerance=input_mass_relative_tolerance,
        minimum_reference_total=input_mass_minimum_reference_total,
        reviewed_exclusions=input_mass_reviewed_exclusions,
    )
    if require_input_mass_parity and not input_mass_gate.passed:
        raise ValueError(
            "US L0/refit release export lost persisted input mass: "
            + "; ".join(input_mass_gate.failures)
        )
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
        "required_person_source_columns": list(
            US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS
        ),
        "required_spm_unit_source_columns": list(
            US_RELEASE_REQUIRED_SPM_UNIT_SOURCE_COLUMNS
        ),
        "required_household_source_columns": list(
            US_RELEASE_REQUIRED_HOUSEHOLD_SOURCE_COLUMNS
        ),
        "required_household_nonconstant_source_columns": list(
            US_RELEASE_REQUIRED_HOUSEHOLD_NONCONSTANT_SOURCE_COLUMNS
        ),
        "required_source_columns_checked": bool(require_source_columns),
        "geography_ladder_gate_enforced": bool(require_geography_ladder),
        "geography_ladder_gate": {
            "passed": geography_ladder_gate.passed,
            "failures": list(geography_ladder_gate.failures),
            "details": dict(geography_ladder_gate.details),
        },
        "input_mass_reference_h5": (
            _file_manifest(reference_h5) if reference_h5 is not None else None
        ),
        "input_mass_parity_enforced": bool(require_input_mass_parity),
        "input_mass_parity": {
            "passed": input_mass_gate.passed,
            "failures": list(input_mass_gate.failures),
            "details": dict(input_mass_gate.details),
        },
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
    parser.add_argument(
        "--reference-h5",
        type=Path,
        help=(
            "Optional H5 whose persisted input mass the export must preserve "
            "(e.g. the certified dense release). Defaults to --base-h5."
        ),
    )
    parser.add_argument(
        "--input-mass-relative-tolerance",
        type=float,
        default=0.5,
        help=(
            "Maximum allowed relative drift of a persisted input column's "
            "weighted total versus the reference before the export fails."
        ),
    )
    parser.add_argument(
        "--input-mass-minimum-reference-total",
        type=float,
        default=0.0,
        help=(
            "Reference-mass floor below which a column's drift is not "
            "checked (relative drift on near-zero totals is meaningless)."
        ),
    )
    parser.add_argument(
        "--allow-input-mass-drift",
        action="store_true",
        help=(
            "Diagnostic escape hatch. By default, the export fails when a "
            "persisted input column loses its reference mass (issue #278); "
            "the gate result is recorded in the summary either way."
        ),
    )
    parser.add_argument(
        "--allow-geography-ladder-gate-failures",
        action="store_true",
        help=(
            "Diagnostic escape hatch. By default, the export fails when the "
            "geography-ladder gate fails (issues #275/#34: spine "
            "inconsistency or NYC mass collapse); the gate result is "
            "recorded in the summary either way."
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
        reference_h5=args.reference_h5,
        input_mass_relative_tolerance=args.input_mass_relative_tolerance,
        input_mass_minimum_reference_total=args.input_mass_minimum_reference_total,
        require_input_mass_parity=not args.allow_input_mass_drift,
        require_geography_ladder=not args.allow_geography_ladder_gate_failures,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


__all__ = [
    "L0RefitWeights",
    "US_RELEASE_REQUIRED_HOUSEHOLD_SOURCE_COLUMNS",
    "US_RELEASE_REQUIRED_HOUSEHOLD_NONCONSTANT_SOURCE_COLUMNS",
    "US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS",
    "US_RELEASE_REQUIRED_SPM_UNIT_SOURCE_COLUMNS",
    "US_RELEASE_REQUIRED_TAX_UNIT_SOURCE_COLUMNS",
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
