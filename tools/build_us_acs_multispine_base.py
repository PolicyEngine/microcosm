"""Build a nullable ASEC-by-PUF plus ACS multispine staging artifact.

This tool starts from the dense, input-complete ASEC-by-PUF H5 produced by
the US input-family pipeline. It acquires the byte-pinned ACS PUMS archives,
loads and maps the ACS spine, transfers model input leaves from the dense
donor, assigns the PUMA-anchored state/CD/county geography ladder, audits every
fit's resolved typed weight kind, and writes the combined base. Calibration is
deliberately downstream.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build import FitWeightRecord, weights_audit_gate
from populace.build.us_runtime import acs_sources
from populace.build.us_runtime.acs_multispine import (
    AcsMultispineResult,
    build_optional_acs_multispine,
)
from populace.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    DEFAULT_CHUNKSIZE,
    AcsPumsSource,
)
from populace.build.us_runtime.acs_transfer import (
    ACS_DONOR_CHANNEL_AUTO,
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    default_acs_transfer_target_families,
    resolve_acs_donor_channel,
)
from populace.build.us_runtime.base_pool import spine_column
from populace.build.us_runtime.puma_ladder import (
    UsPumaLadder,
    load_us_puma_ladder,
)
from populace.build.us_runtime.release_input_coverage import (
    us_release_input_coverage_gate,
)
from populace.frame import Frame, WeightKind, Weights
from populace.frame.units import US_SCHEMA

PERIOD = 2024
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS_DIR = _REPOSITORY_ROOT / "inputs" / "acs_2024_1yr"
DEFAULT_PUMA_LADDER = _REPOSITORY_ROOT / "build" / "us" / "us_puma_ladder_2020.npz"
DEFAULT_N_ESTIMATORS = 32
DEFAULT_STAGING_EXPORT_PEAK_LIMIT_BYTES = 30_000_000_000
_STAGING_EXPORT_FIXED_OVERHEAD_BYTES = 512 * 1024**2
_PACKAGED_MANIFEST_REFERENCE = (
    "package:populace.build.us_runtime/acs_2024_1yr_sources.json"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append the pinned ACS 2024 1-year PUMS spine to an already-built "
            "dense ASEC-by-PUF base. The nullable result is a pre-calibration "
            "staging artifact, not a simulation-ready dataset."
        )
    )
    parser.add_argument("--base-h5", required=True, type=Path)
    parser.add_argument("--out-h5", required=True, type=Path)
    parser.add_argument(
        "--summary",
        type=Path,
        help=(
            "JSON build summary path. Defaults to OUT-H5 with a .summary.json suffix."
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Strict ACS source manifest override; defaults to the packaged pin.",
    )
    parser.add_argument(
        "--inputs-dir",
        default=DEFAULT_INPUTS_DIR,
        type=Path,
        help="Hash-verified ACS archive cache (default: inputs/acs_2024_1yr).",
    )
    parser.add_argument(
        "--puma-ladder",
        default=DEFAULT_PUMA_LADDER,
        type=Path,
        help=(
            "Validated national PUMA-ladder NPZ (default: "
            "build/us/us_puma_ladder_2020.npz)."
        ),
    )
    parser.add_argument(
        "--max-households",
        type=_positive_int,
        help="Optional deterministic smoke limit applied after ACS household sort.",
    )
    parser.add_argument("--period", default=PERIOD, type=_positive_int)
    parser.add_argument("--chunksize", default=DEFAULT_CHUNKSIZE, type=_positive_int)
    parser.add_argument("--acs-share", default=0.5, type=_open_unit_interval)
    parser.add_argument("--seed", default=0, type=_nonnegative_int)
    parser.add_argument(
        "--geography-seed",
        default=0,
        type=_nonnegative_int,
        help="Deterministic PUMA/CD/county assignment seed (default: 0).",
    )
    parser.add_argument(
        "--n-estimators",
        default=DEFAULT_N_ESTIMATORS,
        type=_positive_int,
    )
    parser.add_argument(
        "--max-targets-per-fit",
        default=DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
        type=_positive_int,
        help="Maximum chained QRF targets retained at once (default: 8).",
    )
    parser.add_argument(
        "--donor-channel",
        default=ACS_DONOR_CHANNEL_AUTO,
        help=(
            "ASEC-by-PUF support channel used by transfer. The default selects "
            "the appropriate support channel for each input family."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary_path = args.summary or args.out_h5.with_suffix(".summary.json")
    _validate_artifact_paths(args, summary_path=summary_path)

    manifest = acs_sources.load_acs_source_manifest(args.source_manifest)
    manifest_file = _manifest_file(args.source_manifest)
    manifest_sha256 = _sha256(manifest_file)
    base_sha256 = _sha256(args.base_h5)
    base = _load_base_frame(args.base_h5)
    _require_benefit_participation_inputs(base)
    _require_dense_donor_coverage(base, donor_channel=args.donor_channel)
    base_rows = _row_counts(base)
    base_mass = float(base.weights_for("household").total)
    puma_ladder_sha256 = _sha256(args.puma_ladder)
    puma_ladder = load_us_puma_ladder(args.puma_ladder)

    source = acs_sources.fetch_acs_pums_sources(
        args.inputs_dir,
        manifest=manifest,
    )
    if args.max_households is not None:
        source = replace(source, max_households=args.max_households)

    result = build_optional_acs_multispine(
        base,
        source,
        chunksize=args.chunksize,
        acs_share=args.acs_share,
        donor_channel=args.donor_channel,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_targets_per_fit=args.max_targets_per_fit,
        puma_ladder=puma_ladder,
        geography_seed=args.geography_seed,
    )
    _require_puma_ladder_assignment(result)
    _require_benefit_participation_transfer(result)
    transfer_coverage = _require_default_transfer_coverage(result, base)
    weights_audit = _audit_fits(result)

    # The pooled frame owns its assembled blocks. Release the dense donor
    # before HDF serialization so export cannot retain both full spines plus
    # writer scratch at once.
    del base
    gc.collect()
    input_null_audit = _engine_input_null_audit(result.frame)
    gc.collect()

    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    staging_export_peak_bytes = _preflight_staging_export(result.frame)
    _write_dataset(result.frame, args.out_h5, period=args.period)

    summary = _build_summary(
        args=args,
        summary_path=summary_path,
        manifest=manifest,
        source=source,
        base_rows=base_rows,
        base_mass=base_mass,
        base_sha256=base_sha256,
        manifest_sha256=manifest_sha256,
        result=result,
        weights_audit=weights_audit,
        transfer_coverage=transfer_coverage,
        input_null_audit=input_null_audit,
        staging_export_peak_bytes=staging_export_peak_bytes,
        puma_ladder=puma_ladder,
        puma_ladder_sha256=puma_ladder_sha256,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    summary_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


def _audit_fits(result: AcsMultispineResult) -> dict[str, object]:
    records = result.fit_records
    if not records:
        raise SystemExit(
            "ACS input transfer produced no fit records. The dense donor must "
            "carry transferable tax-detail and benefit-participation inputs."
        )
    non_typed = [
        type(record).__name__
        for record in records
        if not isinstance(record, FitWeightRecord)
    ]
    if non_typed:
        raise TypeError(
            "ACS input transfer returned non-FitWeightRecord audit evidence: "
            f"{non_typed}."
        )
    gate = weights_audit_gate(records)
    if not gate.passed:
        raise SystemExit("Weights audit failed:\n  " + "\n  ".join(gate.failures))
    return {
        "passed": gate.passed,
        "failures": list(gate.failures),
        "details": dict(gate.details),
    }


def _require_benefit_participation_inputs(base: Frame) -> None:
    participation = sorted(
        column
        for entity in base.entities
        for column in base.table(entity).columns
        if column.startswith("takes_up_")
    )
    if not participation:
        raise SystemExit(
            "Dense ASEC-by-PUF donor has no takes_up_* benefit-participation "
            "inputs. This tool must run after the benefit input-family stages."
        )


def _require_dense_donor_coverage(
    base: Frame,
    engine: Any | None = None,
    *,
    donor_channel: str | None = ACS_DONOR_CHANNEL_AUTO,
) -> None:
    """Fail before download unless the selected donor passes the hard contract."""

    if engine is None:
        from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

        engine = PolicyEngineUSEngine()
    selected, resolved_channel = _coverage_donor_channel(base, donor_channel)
    gate = us_release_input_coverage_gate(selected, engine)
    if not gate.passed:
        raise SystemExit(
            "Dense ASEC-by-PUF donor failed the release input-coverage gate; "
            f"selected channel={resolved_channel!r}. Run this tool only after "
            "the coverage-owned input-family stages:\n  " + "\n  ".join(gate.failures)
        )


def _coverage_donor_channel(
    base: Frame,
    requested: str | None,
) -> tuple[Frame, str | None]:
    try:
        return resolve_acs_donor_channel(base, requested)
    except ValueError as exc:
        raise SystemExit(f"Invalid ACS transfer donor support metadata: {exc}") from exc


def _require_benefit_participation_transfer(result: AcsMultispineResult) -> None:
    imputed = result.provenance.get("imputed_inputs", [])
    transferred = isinstance(imputed, list) and any(
        isinstance(item, dict)
        and isinstance(item.get("family"), str)
        and item["family"].split("__batch_", 1)[0] == "benefit_participation"
        and isinstance(item.get("column"), str)
        and item["column"].startswith("takes_up_")
        for item in imputed
    )
    if not transferred:
        raise SystemExit(
            "ACS default transfer produced no takes_up_* benefit-participation "
            "input. Refusing to report a complete multispine base."
        )


def _require_puma_ladder_assignment(result: AcsMultispineResult) -> None:
    """Fail unless the enabled multispine resolved launch geography inputs."""

    raw = result.provenance.get("geography_ladder")
    if not isinstance(raw, dict) or raw.get("applied") is not True:
        raise SystemExit(
            "ACS multispine did not apply the required PUMA geography ladder."
        )
    household = result.frame.table("household")
    required = ("puma", "congressional_district_geoid", "county_fips")
    missing = [column for column in required if column not in household]
    if missing:
        raise SystemExit(
            "PUMA geography assignment omitted household column(s): "
            f"{missing}."
        )
    nulls = {
        column: int(household[column].isna().sum())
        for column in required
        if household[column].isna().any()
    }
    if nulls:
        raise SystemExit(
            f"PUMA geography assignment left null household values: {nulls}."
        )
    deferred = result.provenance.get("deferred_inputs", [])
    still_deferred = sorted(
        {"congressional_district_geoid", "county_fips"}.intersection(deferred)
        if isinstance(deferred, list)
        else {"congressional_district_geoid", "county_fips"}
    )
    if still_deferred:
        raise SystemExit(
            "PUMA geography assignment still reports resolved input(s) as "
            f"deferred: {still_deferred}."
        )


def _require_default_transfer_coverage(
    result: AcsMultispineResult,
    donor: Frame,
) -> dict[str, object]:
    """Prove every planned target is present and complete on its ACS universe."""

    expected: dict[str, str] = {}
    for entity, entity_families in default_acs_transfer_target_families(donor).items():
        for targets in entity_families.values():
            for target in targets:
                expected[target] = entity
    raw_imputed = result.provenance.get("imputed_inputs", [])
    if not isinstance(raw_imputed, list):
        raise SystemExit("ACS imputed-input provenance must be a JSON list.")
    entries = {
        item["column"]: item
        for item in raw_imputed
        if isinstance(item, dict) and isinstance(item.get("column"), str)
    }
    if len(entries) != len(
        [item for item in raw_imputed if isinstance(item, dict) and "column" in item]
    ):
        raise SystemExit("ACS imputed-input provenance contains duplicate columns.")
    missing = sorted(set(expected) - set(entries))
    if missing:
        raise SystemExit(
            "ACS default transfer omitted donor-observed model input(s): "
            f"{missing}. Refusing to report a complete multispine base."
        )

    structural_pending: list[dict[str, object]] = []
    for column, entity in sorted(expected.items()):
        table = result.frame.table(entity)
        if column not in table:
            raise SystemExit(
                f"ACS transfer registered {column!r} but the combined {entity!r} "
                "table does not contain it."
            )
        tag = spine_column(entity)
        if tag not in table:
            raise SystemExit(
                f"Combined staging frame lacks ACS spine tag {tag!r} on {entity!r}."
            )
        acs_mask = table[tag].eq(ACS_2024_1YR_SPINE)
        if not acs_mask.any():
            raise SystemExit(f"Combined staging frame has no ACS rows on {entity!r}.")
        missing_mask = table[column].isna() & acs_mask
        missing_rows = int(missing_mask.sum())
        raw_unmodeled = entries[column].get("unmodeled_recipient_rows", 0)
        if type(raw_unmodeled) is not int or raw_unmodeled < 0:
            raise SystemExit(
                f"ACS imputation provenance for {column!r} has invalid "
                f"unmodeled_recipient_rows={raw_unmodeled!r}."
            )
        if missing_rows != raw_unmodeled:
            raise SystemExit(
                f"ACS transfer provenance for {column!r} reports "
                f"{raw_unmodeled} unmodeled row(s), but the combined ACS spine "
                f"contains {missing_rows} missing row(s)."
            )
        if missing_rows == 0:
            continue
        if column != "pre_subsidy_rent" or entity != "person":
            raise SystemExit(
                f"ACS default transfer left {missing_rows} unmodeled row(s) in "
                f"required input {column!r}."
            )
        gq_mask = _acs_group_quarters_person_mask(result.frame)
        if not missing_mask.equals(gq_mask):
            raise SystemExit(
                "ACS pre_subsidy_rent may remain absent only for native "
                "group-quarters rows."
            )
        structural_pending.append(
            {
                "column": column,
                "entity": entity,
                "rows": missing_rows,
                "reason": "ACS group-quarters rows are outside the housing-tenure universe",
            }
        )

    return {
        "expected_inputs": sorted(expected),
        "registered_inputs": sorted(entries),
        "structural_pending": structural_pending,
    }


def _acs_group_quarters_person_mask(frame: Frame) -> pd.Series:
    household = frame.table("household")
    person = frame.table("person")
    required = {"household_id", "TYPEHUGQ", spine_column("household")}
    missing = sorted(required - set(household.columns))
    if missing:
        raise SystemExit(
            "Cannot validate ACS group-quarters transfer gaps; household table "
            f"lacks {missing}."
        )
    kind = pd.to_numeric(household["TYPEHUGQ"], errors="coerce")
    gq_households = household.loc[
        household[spine_column("household")].eq(ACS_2024_1YR_SPINE) & kind.isin([2, 3]),
        "household_id",
    ]
    return person[spine_column("person")].eq(ACS_2024_1YR_SPINE) & person[
        "person_household_id"
    ].isin(gq_households)


def _build_summary(
    *,
    args: argparse.Namespace,
    summary_path: Path,
    manifest: acs_sources.AcsSourceManifest,
    source: AcsPumsSource,
    base_rows: dict[str, int],
    base_mass: float,
    base_sha256: str,
    manifest_sha256: str,
    result: AcsMultispineResult,
    weights_audit: dict[str, object],
    transfer_coverage: dict[str, object],
    input_null_audit: list[dict[str, object]],
    staging_export_peak_bytes: int,
    puma_ladder: UsPumaLadder,
    puma_ladder_sha256: str,
) -> dict[str, object]:
    output_rows = _row_counts(result.frame)
    output_mass = float(result.frame.weights_for("household").total)
    return {
        "version": 1,
        "stage": "acs_2024_1yr_multispine_base",
        "artifact_kind": "nullable_precalibration_staging_h5",
        "calibration_applied": False,
        "simulation_ready": False,
        "simulation_readiness_blockers": [
            "calibration_not_applied",
            "sub_puma_geography_deferred",
            "native_acs_universe_blanks_preserved",
        ],
        "period": args.period,
        "base": {
            "path": str(args.base_h5.resolve()),
            "sha256": base_sha256,
            "rows": base_rows,
            "household_weight_total": base_mass,
        },
        "acs_sources": _source_provenance(
            manifest,
            source,
            manifest_path=args.source_manifest,
            manifest_sha256=manifest_sha256,
        ),
        "geography_ladder": {
            "path": str(args.puma_ladder.resolve()),
            "sha256": puma_ladder_sha256,
            "pumas": len(puma_ladder),
            "layer_vintages": puma_ladder.layer_vintages,
            "seed": args.geography_seed,
            "assignment": result.provenance["geography_ladder"],
        },
        "orchestration": {
            "chunksize": args.chunksize,
            "acs_share": args.acs_share,
            "max_households": args.max_households,
            "seed": args.seed,
            "geography_seed": args.geography_seed,
            "n_estimators": args.n_estimators,
            "max_targets_per_fit": args.max_targets_per_fit,
            "donor_channel": args.donor_channel,
            "provenance": result.provenance,
        },
        "weights_audit": weights_audit,
        "transfer_coverage": transfer_coverage,
        "pending_engine_input_nulls": input_null_audit,
        "staging_export_peak_estimate_bytes": staging_export_peak_bytes,
        "rows": {
            "base": base_rows,
            "combined": output_rows,
        },
        "household_weight_totals": {
            "base": base_mass,
            "combined": output_mass,
        },
        "spine_totals": _spine_totals(result.frame),
        "output": {
            "path": str(args.out_h5.resolve()),
            "sha256": _sha256(args.out_h5),
            "summary_path": str(summary_path.resolve()),
            "rows": output_rows,
            "household_weight_total": output_mass,
        },
    }


def _source_provenance(
    manifest: acs_sources.AcsSourceManifest,
    source: AcsPumsSource,
    *,
    manifest_path: Path | None,
    manifest_sha256: str,
) -> dict[str, object]:
    resolved_manifest_path = _manifest_file(manifest_path)
    local_paths = {
        "household": source.household_zip.resolve(),
        "person": source.person_zip.resolve(),
    }
    return {
        "manifest": (
            str(resolved_manifest_path)
            if manifest_path is not None
            else _PACKAGED_MANIFEST_REFERENCE
        ),
        "manifest_sha256": manifest_sha256,
        "version": manifest.version,
        "spine": manifest.spine,
        "vintage": manifest.vintage,
        "verified_on": manifest.verified_on,
        "source_directory": manifest.source_directory,
        "artifacts": [
            {
                "role": artifact.role,
                "filename": artifact.filename,
                "url": artifact.url,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "local_path": str(local_paths[artifact.role]),
            }
            for artifact in manifest.artifacts
        ],
    }


def _manifest_file(override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    return Path(acs_sources.__file__).with_name("acs_2024_1yr_sources.json")


def _validate_artifact_paths(
    args: argparse.Namespace,
    *,
    summary_path: Path,
) -> None:
    base = args.base_h5.resolve()
    output = args.out_h5.resolve()
    summary = summary_path.resolve()
    if args.out_h5.suffix != ".h5":
        raise SystemExit(f"--out-h5 must end with .h5, got {args.out_h5.name!r}.")
    if base == output:
        raise SystemExit("--out-h5 must differ from --base-h5.")
    if summary in {base, output}:
        raise SystemExit("--summary must differ from both --base-h5 and --out-h5.")
    ladder = args.puma_ladder.resolve()
    if ladder == output:
        raise SystemExit("--puma-ladder must differ from --out-h5.")


def _spine_totals(frame: Frame) -> dict[str, dict[str, Any]]:
    spine_values: set[str] = set()
    for entity in frame.entities:
        column = spine_column(entity)
        table = frame.table(entity)
        if column not in table:
            raise ValueError(f"Combined ACS base lacks required spine tag {column!r}.")
        if table[column].isna().any():
            raise ValueError(
                f"Combined ACS base carries missing values in spine tag {column!r}."
            )
        spine_values.update(map(str, table[column].dropna().unique()))

    household = frame.table("household")
    household_spine = household[spine_column("household")]
    weights = pd.Series(
        frame.weights_for("household").values,
        index=household.index,
    )
    return {
        spine: {
            "rows": {
                entity: int(frame.table(entity)[spine_column(entity)].eq(spine).sum())
                for entity in frame.entities
            },
            "household_weight_total": float(
                weights.loc[household_spine.eq(spine)].sum()
            ),
        }
        for spine in sorted(spine_values)
    }


def _load_base_frame(path: Path) -> Frame:
    """Load the dense donor H5 without importing PolicyEngine-US at tool import."""

    from policyengine_us.data import USSingleYearDataset

    dataset = USSingleYearDataset(file_path=str(path))
    tables = {
        "person": dataset.person,
        "household": dataset.household,
        "tax_unit": dataset.tax_unit,
        "spm_unit": dataset.spm_unit,
        "family": dataset.family,
        "marital_unit": dataset.marital_unit,
    }
    household_weights = (
        tables["household"].pop("household_weight").to_numpy(dtype=np.float64)
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                household_weights,
                WeightKind.CALIBRATED,
            )
        },
    )


def _write_dataset(frame: Frame, path: Path, *, period: int) -> None:
    """Write a nullable staging H5 with one-table-at-a-time verification."""

    output = Path(path)
    output.unlink(missing_ok=True)
    try:
        with pd.HDFStore(output, mode="w") as store:
            for entity in frame.entities:
                table = frame.table(entity)
                if entity == "household":
                    table = table.copy()
                    table["household_weight"] = frame.weights_for("household").values
                if len(table):
                    # Fixed format preserves mixed bool/null object columns
                    # losslessly. Table format rejects them, which would force
                    # an unauthorized fill or type rewrite on base-only inputs.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                        store.put(entity, table, format="fixed")
            store.put(
                "_time_period",
                pd.Series([int(period)]),
                format="table",
            )
            store.put(
                "_populace_staging_metadata",
                pd.Series(
                    [
                        json.dumps(
                            {
                                "artifact_kind": "nullable_precalibration_staging_h5",
                                "entity_hdf_format": "fixed_nullable",
                                "household_weight_kind": frame.weights_for(
                                    "household"
                                ).kind.value,
                            },
                            sort_keys=True,
                        )
                    ]
                ),
                format="table",
            )

        with pd.HDFStore(output, mode="r") as store:
            for entity in frame.entities:
                expected = frame.table(entity)
                if not len(expected):
                    continue
                stored = store[entity]
                expected_columns = list(expected.columns)
                if entity == "household":
                    expected_columns.append("household_weight")
                if (
                    len(stored) != len(expected)
                    or list(stored.columns) != expected_columns
                ):
                    raise RuntimeError(
                        f"Staging H5 round trip changed {entity!r}: expected "
                        f"{len(expected)} rows/{expected_columns}, got "
                        f"{len(stored)} rows/{list(stored.columns)}."
                    )
                del stored
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def _engine_input_null_audit(
    frame: Frame,
    engine: Any | None = None,
) -> list[dict[str, object]]:
    """Inventory why this truthful staging frame is not simulation-ready."""

    if engine is None:
        from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

        engine = PolicyEngineUSEngine()
    input_names = set(engine.variables())
    entries: list[dict[str, object]] = []
    for entity in frame.entities:
        table = frame.table(entity)
        tag = spine_column(entity)
        for column in sorted(set(table.columns).intersection(input_names)):
            missing = table[column].isna()
            if not missing.any():
                continue
            by_spine: dict[str, int] = {}
            if tag in table:
                by_spine = {
                    str(spine): int((missing & table[tag].eq(spine)).sum())
                    for spine in sorted(map(str, table[tag].dropna().unique()))
                    if int((missing & table[tag].eq(spine)).sum())
                }
            entries.append(
                {
                    "entity": entity,
                    "column": column,
                    "dtype": engine.variable_metadata(column).dtype,
                    "missing_rows": int(missing.sum()),
                    "rows": len(table),
                    "missing_rows_by_spine": by_spine,
                }
            )
    return entries


def _preflight_staging_export(
    frame: Frame,
    *,
    max_peak_bytes: int = DEFAULT_STAGING_EXPORT_PEAK_LIMIT_BYTES,
) -> int:
    if type(max_peak_bytes) is not int or max_peak_bytes <= 0:
        raise ValueError("max_peak_bytes must be a positive integer.")
    table_bytes = {
        entity: int(frame.table(entity).memory_usage(index=True, deep=True).sum())
        for entity in frame.entities
    }
    resident = sum(table_bytes.values())
    resident += sum(
        frame.weights_for(entity).values.nbytes for entity in frame.weighted_entities
    )
    resident += int(frame.strata.memory_usage(index=True, deep=True))
    largest_table = max(table_bytes.values(), default=0)
    household_copy = table_bytes.get("household", 0) + 8 * frame.n("household")
    estimate = int(
        resident
        + 2 * largest_table
        + household_copy
        + _STAGING_EXPORT_FIXED_OVERHEAD_BYTES
    )
    if estimate > max_peak_bytes:
        raise MemoryError(
            "Nullable ACS staging export is estimated to require "
            f"{estimate / 1_000_000_000:.2f} GB, above the "
            f"{max_peak_bytes / 1_000_000_000:.2f} GB limit."
        )
    return estimate


def _row_counts(frame: Frame) -> dict[str, int]:
    return {entity: frame.n(entity) for entity in frame.entities}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _open_unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be strictly between 0 and 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
