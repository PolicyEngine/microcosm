"""Build a local US base H5 with a PUF tax-detail support channel.

This diagnostic builder starts from an existing Populace US H5, clones the
frame into ASEC and PUF-tax-detail support channels, imputes PUF-observed
inputs onto the PUF channel with Populace's weighted QRF, and writes a fresh
base H5 for the fiscal refresh calibration builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build import FitWeightRecord, weights_audit_gate
from populace.build.ledger_artifact import load_ledger_consumer_artifact
from populace.build.source_manifest import SupportSpineSpec, load_support_spine_manifest
from populace.build.us_runtime import (
    BASE_ASEC_SUPPORT_CHANNEL,
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
    GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR,
    GEOGRAPHY_LADDER_VINTAGES_ATTR,
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    US_SUPPORT_SPINE_SPEC,
    AsecSource,
    build_pooled_asec_unit_frame,
    clone_us_frame_for_puf_support,
    congressional_district_assignment_summary,
    congressional_district_distribution_from_ledger_facts,
    derive_us_cps_carried_inputs,
    impute_us_puf_tax_detail_support,
    load_congressional_district_vintage_crosswalk,
    load_us_block_ladder,
    puf_tax_unit_donor_from_arrays,
    support_channel_column,
    translate_congressional_district_facts_to_current_vintage,
    us_alimony_signal_gate,
    us_casualty_loss_signal_gate,
    us_childcare_signal_gate,
    us_education_inputs_signal_gate,
    us_geography_ladder_assignment_summary,
    us_geography_ladder_gate,
    us_immigration_composition_summary,
    us_misc_itemized_signal_gate,
    us_qbi_inputs_signal_gate,
    us_retirement_contributions_signal_gate,
    with_household_congressional_districts,
    with_household_us_geography_ladder,
    with_us_childcare_inputs,
    with_us_education_inputs,
    with_us_immigration_inputs,
    with_us_qbi_input_reconciliation,
    with_us_retirement_contribution_inputs,
)
from populace.build.us_runtime.puf_support import PUF_TAX_DETAIL_DEFAULT_PREDICTORS
from populace.frame import Frame, WeightKind, Weights
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine
from populace.frame.units import US_SCHEMA

PERIOD = 2024
DATASET_FILENAME = "base_populace_us_2024_puf_support.h5"
SUMMARY_FILENAME = "base_populace_us_2024_puf_support.summary.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-h5", type=Path)
    source.add_argument(
        "--asec-h5",
        action="append",
        help="Raw ASEC source as YEAR=PATH. Pass once per source year.",
    )
    parser.add_argument("--target-year", default=PERIOD, type=int)
    parser.add_argument(
        "--asec-max-households",
        type=int,
        help="Optional smoke limit applied to every raw ASEC source.",
    )
    parser.add_argument(
        "--support-spine-spec",
        type=Path,
        help=(
            "Optional support-spine manifest. When provided, --asec-h5 values "
            "are YEAR=PATH file mappings and the manifest owns source roles, "
            "relative years, and shares."
        ),
    )
    parser.add_argument("--puf-h5", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-estimators", default=32, type=int)
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        help=(
            "Ledger consumer facts JSONL. Required when assigning congressional "
            "district geography."
        ),
    )
    parser.add_argument(
        "--assign-congressional-districts",
        action="store_true",
        help=(
            "Assign household congressional_district_geoid values from SOI CD "
            "return-count Ledger facts, constrained by state_fips."
        ),
    )
    parser.add_argument(
        "--congressional-district-vintage-crosswalk",
        type=Path,
        help=(
            "Optional source-to-current congressional-district crosswalk "
            "artifact with source_geography_id, target_geography_id, and "
            "weight columns. When provided, SOI CD return-count facts are "
            "translated before support assignment."
        ),
    )
    parser.add_argument("--congressional-district-seed", default=0, type=int)
    parser.add_argument(
        "--block-ladder-artifact",
        type=Path,
        help=(
            "US block-ladder NPZ artifact "
            "(tools/build_us_block_ladder_artifact.py). Assigns each "
            "household a census block within its congressional district and "
            "derives the county/tract/place/SLD/CBSA spine columns. US bases "
            "carry the ladder by default; omit only with "
            "--without-block-ladder."
        ),
    )
    parser.add_argument(
        "--without-block-ladder",
        action="store_true",
        help=(
            "Explicitly build a base without the geography ladder "
            "(diagnostic builds only; a ladder-less base cannot become a "
            "release — the L0/refit export requires the spine columns)."
        ),
    )
    parser.add_argument("--geography-ladder-seed", default=0, type=int)
    parser.add_argument(
        "--allow-geography-ladder-gate-failures",
        action="store_true",
        help=(
            "Diagnostic escape hatch for partial-spine smoke builds. By "
            "default a failing geography-ladder gate (e.g. the NYC "
            "never-collapses-to-zero regression of populace #34) aborts the "
            "build."
        ),
    )
    args = parser.parse_args(argv)
    if (
        args.congressional_district_vintage_crosswalk is not None
        and not args.assign_congressional_districts
    ):
        parser.error(
            "--congressional-district-vintage-crosswalk requires "
            "--assign-congressional-districts"
        )
    if args.assign_congressional_districts and args.ledger_facts is None:
        parser.error("--assign-congressional-districts requires --ledger-facts")
    if args.block_ladder_artifact is not None and (
        args.congressional_district_vintage_crosswalk is None
    ):
        parser.error(
            "--block-ladder-artifact requires "
            "--congressional-district-vintage-crosswalk: block sampling is "
            "conditioned on households carrying current-vintage districts"
        )
    if args.block_ladder_artifact is not None and args.without_block_ladder:
        parser.error(
            "--block-ladder-artifact and --without-block-ladder are contradictory"
        )
    if args.block_ladder_artifact is None and not args.without_block_ladder:
        parser.error(
            "US bases carry the block-anchored geography ladder by default "
            "(populace #275): pass --block-ladder-artifact <npz> "
            "(tools/build_us_block_ladder_artifact.py) or opt out "
            "explicitly with --without-block-ladder"
        )
    if args.support_spine_spec is not None and args.asec_h5 is None:
        parser.error("--support-spine-spec requires --asec-h5")
    return args


def main() -> None:
    args = _parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = out_dir / _dataset_filename(args.target_year)
    summary_path = out_dir / _summary_filename(args.target_year)

    raw_base, base_source = _load_base_frame_from_args(args)
    base = derive_us_cps_carried_inputs(raw_base)
    base = with_us_childcare_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_retirement_contribution_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_immigration_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    expanded = clone_us_frame_for_puf_support(base)
    arrays = _read_h5_arrays(args.puf_h5)
    donor = puf_tax_unit_donor_from_arrays(arrays)
    imputed, weights_audit = impute_and_audit_us_puf_support(
        expanded,
        donor,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    imputed = with_us_qbi_input_reconciliation(imputed)
    qbi_inputs_gate = us_qbi_inputs_signal_gate(imputed)
    if not qbi_inputs_gate.passed:
        raise SystemExit(
            "QBI-input signal gate failed:\n  " + "\n  ".join(qbi_inputs_gate.failures)
        )
    imputed = with_us_childcare_inputs(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
    )
    childcare_gate = us_childcare_signal_gate(imputed)
    if not childcare_gate.passed:
        raise SystemExit(
            "Childcare-input signal gate failed:\n  "
            + "\n  ".join(childcare_gate.failures)
        )
    alimony_gate = us_alimony_signal_gate(imputed)
    if not alimony_gate.passed:
        raise SystemExit(
            "Alimony-input signal gate failed:\n  " + "\n  ".join(alimony_gate.failures)
        )
    casualty_loss_gate = us_casualty_loss_signal_gate(imputed)
    if not casualty_loss_gate.passed:
        raise SystemExit(
            "Casualty-loss signal gate failed:\n  "
            + "\n  ".join(casualty_loss_gate.failures)
        )
    misc_itemized_gate = us_misc_itemized_signal_gate(imputed)
    if not misc_itemized_gate.passed:
        raise SystemExit(
            "Miscellaneous-itemized signal gate failed:\n  "
            + "\n  ".join(misc_itemized_gate.failures)
        )
    imputed = with_us_retirement_contribution_inputs(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
    )
    retirement_contributions_gate = us_retirement_contributions_signal_gate(imputed)
    if not retirement_contributions_gate.passed:
        raise SystemExit(
            "Retirement-contribution signal gate failed:\n  "
            + "\n  ".join(retirement_contributions_gate.failures)
        )
    imputed = with_us_education_inputs(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
    )
    education_inputs_gate = us_education_inputs_signal_gate(imputed)
    if not education_inputs_gate.passed:
        raise SystemExit(
            "Education-input signal gate failed:\n  "
            + "\n  ".join(education_inputs_gate.failures)
        )
    congressional_district_assignment = {"applied": False}
    if args.assign_congressional_districts:
        ledger_facts = load_ledger_consumer_artifact(args.ledger_facts).facts
        if args.congressional_district_vintage_crosswalk is not None:
            ledger_facts = translate_congressional_district_facts_to_current_vintage(
                ledger_facts,
                load_congressional_district_vintage_crosswalk(
                    args.congressional_district_vintage_crosswalk
                ),
            )
        distribution = congressional_district_distribution_from_ledger_facts(
            ledger_facts
        )
        imputed = with_household_congressional_districts(
            imputed,
            distribution,
            seed=args.congressional_district_seed,
        )
        congressional_district_assignment = congressional_district_assignment_summary(
            imputed.table("household"),
            distribution,
        )
        congressional_district_assignment.update(
            {
                "ledger_facts": str(args.ledger_facts.resolve()),
                "ledger_facts_sha256": _sha256(args.ledger_facts),
                "congressional_district_vintage_crosswalk": (
                    str(args.congressional_district_vintage_crosswalk.resolve())
                    if args.congressional_district_vintage_crosswalk is not None
                    else None
                ),
                "congressional_district_vintage_crosswalk_sha256": (
                    _sha256(args.congressional_district_vintage_crosswalk)
                    if args.congressional_district_vintage_crosswalk is not None
                    else None
                ),
                "seed": args.congressional_district_seed,
            }
        )
    geography_ladder_assignment = {
        "applied": False,
        "opted_out": bool(args.without_block_ladder),
    }
    if args.block_ladder_artifact is not None:
        ladder = load_us_block_ladder(args.block_ladder_artifact)
        imputed = with_household_us_geography_ladder(
            imputed,
            ladder,
            seed=args.geography_ladder_seed,
            expected_congressional_district_vintage=(
                CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
            ),
        )
        household = imputed.table("household")
        household_weights = imputed.weights_for("household").values
        gate = us_geography_ladder_gate(household, household_weights)
        if not gate.passed and not args.allow_geography_ladder_gate_failures:
            raise SystemExit(
                "Geography-ladder gate failed:\n  " + "\n  ".join(gate.failures)
            )
        geography_ladder_assignment = us_geography_ladder_assignment_summary(
            household,
            ladder,
            weight_values=household_weights,
        )
        geography_ladder_assignment.update(
            {
                "artifact": str(args.block_ladder_artifact.resolve()),
                "artifact_sha256": _sha256(args.block_ladder_artifact),
                "seed": args.geography_ladder_seed,
                "gate": {
                    "passed": gate.passed,
                    "failures": list(gate.failures),
                    "details": dict(gate.details),
                },
            }
        )
    PolicyEngineUSEngine().write_dataset(imputed, output_h5, period=args.target_year)
    if (
        args.congressional_district_vintage_crosswalk is not None
        or args.block_ladder_artifact is not None
    ):
        import h5py

        with h5py.File(output_h5, "a") as h5:
            if args.congressional_district_vintage_crosswalk is not None:
                h5.attrs[CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR] = (
                    _sha256(args.congressional_district_vintage_crosswalk)
                )
                h5.attrs[CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR] = (
                    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
                )
            if args.block_ladder_artifact is not None:
                h5.attrs[GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR] = _sha256(
                    args.block_ladder_artifact
                )
                h5.attrs[GEOGRAPHY_LADDER_VINTAGES_ATTR] = json.dumps(
                    geography_ladder_assignment["layer_vintages"],
                    sort_keys=True,
                )

    summary = {
        "base_source": base_source,
        "base_h5": (str(args.base_h5.resolve()) if args.base_h5 is not None else None),
        "base_sha256": _sha256(args.base_h5) if args.base_h5 is not None else None,
        "puf_h5": str(args.puf_h5.resolve()),
        "puf_sha256": _sha256(args.puf_h5),
        "output_h5": str(output_h5),
        "output_sha256": _sha256(output_h5),
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "base_rows": _row_counts(base),
        "expanded_rows": _row_counts(imputed),
        "base_household_weight_total": float(base.weights_for("household").total),
        "expanded_household_weight_total": float(
            imputed.weights_for("household").total
        ),
        "channel_weight_totals": _channel_weight_totals(imputed),
        "puf_donor_rows": int(len(donor)),
        "puf_donor_columns": sorted(donor.columns.tolist()),
        "weights_audit": weights_audit,
        "qbi_inputs_signal": {
            "passed": qbi_inputs_gate.passed,
            "failures": list(qbi_inputs_gate.failures),
            "details": dict(qbi_inputs_gate.details),
        },
        "childcare_inputs_signal": {
            "passed": childcare_gate.passed,
            "failures": list(childcare_gate.failures),
            "details": dict(childcare_gate.details),
        },
        "alimony_inputs_signal": {
            "passed": alimony_gate.passed,
            "failures": list(alimony_gate.failures),
            "details": dict(alimony_gate.details),
        },
        "casualty_loss_signal": {
            "passed": casualty_loss_gate.passed,
            "failures": list(casualty_loss_gate.failures),
            "details": dict(casualty_loss_gate.details),
        },
        "misc_itemized_signal": {
            "passed": misc_itemized_gate.passed,
            "failures": list(misc_itemized_gate.failures),
            "details": dict(misc_itemized_gate.details),
        },
        "education_inputs_signal": {
            "passed": education_inputs_gate.passed,
            "failures": list(education_inputs_gate.failures),
            "details": dict(education_inputs_gate.details),
        },
        "retirement_contributions_signal": {
            "passed": retirement_contributions_gate.passed,
            "failures": list(retirement_contributions_gate.failures),
            "details": dict(retirement_contributions_gate.details),
        },
        "congressional_district_assignment": congressional_district_assignment,
        "geography_ladder_assignment": geography_ladder_assignment,
        "channel_output_totals": _channel_output_totals(imputed),
        "immigration_composition": us_immigration_composition_summary(imputed),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def impute_and_audit_us_puf_support(
    expanded: Frame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int,
    predictors: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    person_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    tax_unit_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
) -> tuple[Frame, dict]:
    """Impute the PUF support channel and audit the fit's resolved weight kind.

    Runs the production PUF tax-detail support imputation, capturing the kind the
    QRF *resolved* to via the build-level weights audit (populace #300): the fit
    emits one :class:`~populace.build.FitWeightRecord`, and
    :func:`~populace.build.weights_audit_gate` proves it did not silently resolve
    unweighted. A failing audit **aborts the build** with a non-zero exit, exactly
    as the geography-ladder gate does — a support channel imputed by an unweighted
    fit is a broken donor whose on-surface residuals can still look perfect.

    The audit is unconditional: the PUF support fit always runs, so there is no
    opt-out, and the seed allowlist is empty because the fit is design-weighted.

    Args:
        expanded: The channel-cloned US frame the fit imputes onto.
        donor: The PUF tax-unit donor table the fit trains on.
        seed: The imputation seed.
        n_estimators: Trees per QRF forest.
        predictors: Predictor columns for the fit; defaults to the production set.
        person_outputs: Person-grain outputs; defaults to the production set.
        tax_unit_outputs: Tax-unit-grain outputs; defaults to the production set.
            The three ``*_outputs``/``predictors`` arguments exist so the seam can
            be exercised on a small synthetic frame in an engine-free test; the
            build calls this with the defaults, so production behavior is
            unchanged.

    Returns:
        ``(imputed_frame, weights_audit)`` where ``weights_audit`` is the gate's
        publishable record — ``{"passed", "failures", "details"}`` — carrying
        ``details["resolved_weight_kinds"]`` (the fit-name -> resolved-kind map)
        for the release summary.

    Raises:
        SystemExit: If the weights audit fails (a fit resolved unweighted with no
            allowlist entry), naming the offending fit.
    """
    fit_records: list[FitWeightRecord] = []
    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=predictors,
        person_outputs=person_outputs,
        tax_unit_outputs=tax_unit_outputs,
        seed=seed,
        n_estimators=n_estimators,
        fit_records=fit_records,
    )
    report = weights_audit_gate(fit_records)
    if not report.passed:
        raise SystemExit("Weights audit failed:\n  " + "\n  ".join(report.failures))
    weights_audit = {
        "passed": report.passed,
        "failures": list(report.failures),
        "details": dict(report.details),
    }
    return imputed, weights_audit


def _dataset_filename(period: int) -> str:
    if period == PERIOD:
        return DATASET_FILENAME
    return f"base_populace_us_{period}_puf_support.h5"


def _summary_filename(period: int) -> str:
    if period == PERIOD:
        return SUMMARY_FILENAME
    return f"base_populace_us_{period}_puf_support.summary.json"


def _load_base_frame_from_args(args: argparse.Namespace) -> tuple[Frame, dict]:
    if args.base_h5 is not None:
        frame = _load_frame(args.base_h5)
        return frame, {
            "kind": "base_h5",
            "path": str(args.base_h5.resolve()),
            "sha256": _sha256(args.base_h5),
        }
    support_spine_spec = _support_spine_spec_from_args(args)
    sources = _asec_sources_from_args(
        args,
        support_spine_spec=support_spine_spec,
    )
    frame, metadata = build_pooled_asec_unit_frame(
        sources,
        target_year=args.target_year,
    )
    return frame, {
        "kind": "pooled_asec",
        "target_year": args.target_year,
        "sources": [
            {
                "year": source.year,
                "path": str(source.path.resolve()),
                "sha256": _sha256(source.path),
                "share": source.share,
                "max_households": source.max_households,
            }
            for source in sources
        ],
        "support_spine_spec": _support_spine_spec_metadata(
            args,
            support_spine_spec=support_spine_spec,
        ),
        "metadata": metadata,
    }


def _support_spine_spec_from_args(args: argparse.Namespace) -> SupportSpineSpec | None:
    if args.support_spine_spec is None:
        return None
    if args.support_spine_spec.name == "default":
        return US_SUPPORT_SPINE_SPEC
    return load_support_spine_manifest(args.support_spine_spec).support_spine


def _asec_sources_from_args(
    args: argparse.Namespace,
    *,
    support_spine_spec: SupportSpineSpec | None,
) -> tuple[AsecSource, ...]:
    if support_spine_spec is None:
        return tuple(
            _parse_asec_source(value, max_households=args.asec_max_households)
            for value in args.asec_h5
        )
    path_by_year = _parse_asec_source_paths(args.asec_h5)
    expected_years = {
        source_spec.resolved_year(args.target_year)
        for source_spec in support_spine_spec.sources
    }
    extra_years = sorted(set(path_by_year) - expected_years)
    if extra_years:
        expected = ", ".join(str(value) for value in sorted(expected_years))
        raise ValueError(
            "Support-spine spec mode received unused --asec-h5 mapping(s) for "
            f"year(s) {extra_years}. Expected year(s): {expected or 'none'}."
        )
    sources: list[AsecSource] = []
    for source_spec in support_spine_spec.sources:
        year = source_spec.resolved_year(args.target_year)
        if year not in path_by_year:
            available = ", ".join(str(value) for value in sorted(path_by_year))
            raise ValueError(
                f"Support-spine spec source {source_spec.role!r} resolves to "
                f"ASEC year {year}, but no --asec-h5 mapping was provided for "
                f"that year. Available year(s): {available or 'none'}."
            )
        sources.append(
            AsecSource(
                year=year,
                path=path_by_year[year],
                share=source_spec.share,
                max_households=args.asec_max_households,
            )
        )
    return tuple(sources)


def _support_spine_spec_metadata(
    args: argparse.Namespace,
    *,
    support_spine_spec: SupportSpineSpec | None,
) -> dict | None:
    if support_spine_spec is None:
        return None
    return {
        "path": (
            "package:populace.build.us/support_spine.json"
            if args.support_spine_spec is not None
            and args.support_spine_spec.name == "default"
            else str(args.support_spine_spec.resolve())
        ),
        "stage": support_spine_spec.stage,
        "method": support_spine_spec.method,
        "target_year_from_build_config": (
            support_spine_spec.target_year_from_build_config
        ),
        "sources": [
            {
                "role": source.role,
                "survey": source.survey,
                "source": source.source,
                "source_year_offset": source.source_year_offset,
                "resolved_year": source.resolved_year(args.target_year),
                "share": source.share,
                "notes": source.notes,
            }
            for source in support_spine_spec.sources
        ],
    }


def _parse_asec_source(value: str, *, max_households: int | None) -> AsecSource:
    if "=" not in value:
        raise ValueError(f"ASEC source must be YEAR=PATH, got {value!r}.")
    raw_year, raw_path = value.split("=", 1)
    return AsecSource(
        year=int(raw_year),
        path=Path(raw_path),
        max_households=max_households,
    )


def _parse_asec_source_paths(values: list[str]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"ASEC source must be YEAR=PATH, got {value!r}.")
        raw_year, raw_path = value.split("=", 1)
        year = int(raw_year)
        if year in paths:
            raise ValueError(f"Duplicate --asec-h5 mapping for year {year}.")
        paths[year] = Path(raw_path)
    return paths


def _load_frame(path: Path) -> Frame:
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


def _read_h5_arrays(path: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as h5:
        return {name: np.asarray(dataset) for name, dataset in h5.items()}


def _row_counts(frame: Frame) -> dict[str, int]:
    return {entity: frame.n(entity) for entity in frame.entities}


def _channel_weight_totals(frame: Frame) -> dict[str, float]:
    household = frame.table("household")
    channel = support_channel_column("household")
    weights = pd.Series(frame.weights_for("household").values, index=household.index)
    return {
        str(name): float(weights.loc[group.index].sum())
        for name, group in household.groupby(channel, sort=True)
    }


def _channel_output_totals(frame: Frame) -> dict[str, dict[str, float]]:
    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    person_outputs = PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    tax_unit_outputs = PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    result: dict[str, dict[str, float]] = {
        BASE_ASEC_SUPPORT_CHANNEL: {},
        PUF_TAX_DETAIL_SUPPORT_CHANNEL: {},
    }
    for channel in result:
        person_mask = person[support_channel_column("person")] == channel
        tax_unit_mask = tax_unit[support_channel_column("tax_unit")] == channel
        for column in person_outputs:
            if column in person:
                result[channel][column] = float(
                    pd.to_numeric(person.loc[person_mask, column], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
        for column in tax_unit_outputs:
            if column in tax_unit:
                result[channel][column] = float(
                    pd.to_numeric(tax_unit.loc[tax_unit_mask, column], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
