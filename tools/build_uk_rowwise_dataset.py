"""Build a Microcosm UK row-wise local-geography dataset.

This is the narrow build driver for the UK local replacement path. It starts
from an existing compact Microcosm UK single-year H5, builds or loads the
official-source geography crosswalk, clones the entity tables, assigns each
household a finest available geography row, and writes diagnostics that prove
coverage and weight preservation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.logbook_adoption import (
    AttemptState,
    append_phase,
    apply_error_verdict,
    error_receipt_path,
    git_code_pin,
    local_artifact_reference,
    preflight_digest,
    record_terminal_attempt,
    resolve_predecessor,
    role_pins_digest,
    sha256_argument,
    write_error_receipt,
)
from microcosm.build.uk_runtime import (
    MASS_CONSERVATION_RELATIVE_TOLERANCE,
    PERSON_ID_COLUMNS,
    POOL_SOURCE_LINEAGE_COLUMN,
    UK_GEOGRAPHY_LADDER_COLUMNS,
    UKLadderRowwiseDatasetResult,
    apply_uk_source_lineage_modulus,
    assign_household_geography,
    assign_uk_geography_ladder,
    build_official_uk_geography_crosswalk,
    clone_entity_frame,
    clone_uk_dataset_with_ladder_geography,
    clone_uk_dataset_with_rowwise_geography,
    expected_uk_rowwise_area_support,
    geography_coverage_summary,
    id_multiplier_for_values,
    ladder_clone_index_column,
    load_uk_local_area_crosswalk,
    load_uk_oa_ladder,
    read_uk_single_year_weight_metadata,
    uk_geography_ladder_gate,
    uk_household_weight_kind,
    uk_time_period,
    validate_geography_coverage,
    write_geography_crosswalk,
)
from microcosm.build.uk_runtime.rowwise_dataset import (
    UK_SPINE_LINEAGE_COLUMNS,
    _refuse_preassigned_geography,
)
from microcosm.frame import engine_tables

CROSSWALK_FILENAME = "uk_official_geography_crosswalk.csv.gz"
DATASET_FILENAME_TEMPLATE = "{input_stem}_rowwise.h5"
MANIFEST_FILENAME = "rowwise_build_manifest.json"
COVERAGE_FILENAME = "geography_coverage_summary.csv"
DRY_RUN_PLAN_FILENAME = "rowwise_dry_run_plan.json"
EXPECTED_SUPPORT_BOTTOM_AREAS = 15
_UK_ROWWISE_PIPELINE = "uk-local-rowwise"
_REPOSITORY = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-h5",
        type=Path,
        required=True,
        help="Compact Microcosm UK single-year H5 to clone.",
    )
    parser.add_argument(
        "--input-sha256",
        type=sha256_argument,
        help="Optional SHA-256 pin for --input-h5; mismatches fail before H5 parsing.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the row-wise H5 and diagnostics.",
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        help=(
            "Optional existing official geography crosswalk CSV/CSV.GZ. If omitted, "
            "the driver downloads public source tables and builds one."
        ),
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        help=(
            "UK OA-ladder NPZ artifact (tools/build_uk_oa_ladder_artifact.py). "
            "When set, geography is assigned through the ratified ladder route "
            "with its release-blocking gate, instead of the crosswalk sampler. "
            "Mutually exclusive with --crosswalk and the coverage-code checks."
        ),
    )
    parser.add_argument(
        "--ladder-sha256",
        type=sha256_argument,
        help="Optional SHA-256 pin for --ladder; mismatches fail before NPZ parsing.",
    )
    parser.add_argument(
        "--expected-constituency-vintage",
        default="2024_pcon",
        help=(
            "Constituency vintage the ladder artifact must declare "
            "(vintage_policy: error). Applies to the --ladder route."
        ),
    )
    parser.add_argument(
        "--constituency-codes",
        type=Path,
        help="Optional CSV containing a `code` column for constituency coverage checks.",
    )
    parser.add_argument(
        "--la-codes",
        type=Path,
        help="Optional CSV containing a `code` column for local-authority coverage checks.",
    )
    parser.add_argument("--n-clones", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-year",
        type=int,
        help="Source year for cloned household lineage. Defaults to the input H5 time_period.",
    )
    parser.add_argument(
        "--dataset-filename",
        help=(
            "Output H5 filename within --out. Defaults to the input H5 stem "
            "plus '_rowwise.h5'."
        ),
    )
    parser.add_argument(
        "--allow-missing-country",
        action="store_true",
        help="Do not require all UK countries to appear in the input H5.",
    )
    parser.add_argument(
        "--allow-blank-constituency",
        action="store_true",
        help="Allow blank constituency codes in the crosswalk.",
    )
    parser.add_argument(
        "--allow-cross-region-assignment",
        action="store_true",
        help="Allow households to draw geography from any UK region in their country.",
    )
    parser.add_argument(
        "--allow-constituency-collisions",
        action="store_true",
        help="Allow the same source household to be assigned to the same constituency across clones.",
    )
    parser.add_argument(
        "--source-lineage-modulus",
        type=int,
        help=(
            "Pool inputs only: derive pool_source_household_id before cloning "
            "from household_id = tier * 10**8 + base, leaving the immediate "
            "source_household_id untouched. Spine inputs instead use sernum + "
            "1{spi} * 10**d + 1{cgt_clone} * 10**(d+1) + 1{band_donor} * "
            "10**(d+2) and carry authoritative explicit lineage columns, so "
            "the modulus is refused for them. Also refused when the pool "
            "column exists or the mapping would be an identity."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Compute and write the clone plan (row/byte math, weight-kind "
            "chain, the realized per-area support of the real sampler at "
            "this seed, and the analytic collision-free expectation) as "
            f"{DRY_RUN_PLAN_FILENAME} without cloning or writing a dataset. "
            "When no --crosswalk is supplied, the freshly built crosswalk "
            "cache is still written to --out."
        ),
    )
    parser.add_argument(
        "--logbook-prev-row-digest",
        type=sha256_argument,
        help=(
            "Optional current Logbook chain head. If omitted, "
            "POPULACE_LOGBOOK_PREV_ROW_DIGEST is used, then genesis null."
        ),
    )
    return parser.parse_args(argv)


def _new_rowwise_build_id(
    *,
    route: str,
    seed: int,
    timestamp: datetime,
) -> str:
    instant = timestamp.astimezone(UTC)
    return (
        f"uk-local-rowwise-{route}-f100-s{seed}-"
        f"{instant.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )


def _pin_from_artifact(info: dict[str, Any]) -> dict[str, object]:
    return {
        "sha256": str(info["sha256"]),
        "size_bytes": int(info["bytes"]),
    }


def _rowwise_identity_digest(
    *,
    route: str,
    pins: dict[str, dict[str, object]],
    args: argparse.Namespace,
    source_year: int,
) -> str:
    payload = {
        "build_kind": "uk_rowwise_local_geography_dataset",
        "route": route,
        "inputs": pins,
        "parameters": _parameters(args, source_year=source_year),
        "source_year": source_year,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _record_rowwise_attempt(
    *,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    seed: int,
    code_pin: str,
    disposition: str,
    predecessor: str | None,
    spool_dir: Path,
) -> Path:
    return record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_UK_ROWWISE_PIPELINE,
        rung="f100",
        seed=seed,
        code_pin=code_pin,
        disposition=disposition,
        predecessor=predecessor,
        spool_dir=spool_dir,
    )


def _record_rowwise_error(
    *,
    error: BaseException,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    seed: int,
    code_pin: str,
    predecessor: str | None,
    base_dir: Path,
    spool_dir: Path,
) -> None:
    error_path = write_error_receipt(
        error_receipt_path(base_dir, build_id=state.build_id),
        state=state,
        pipeline=_UK_ROWWISE_PIPELINE,
        error=error,
    )
    apply_error_verdict(
        state,
        f"{local_artifact_reference(error_path, repository_hint=_REPOSITORY)}#/error_type",
    )
    _record_rowwise_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        seed=seed,
        code_pin=code_pin,
        disposition="failed",
        predecessor=predecessor,
        spool_dir=spool_dir,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        return _main_impl(args, attempt=None)

    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    digest = preflight_digest(_UK_ROWWISE_PIPELINE)
    state = AttemptState(
        build_id=_new_rowwise_build_id(
            route="attempt",
            seed=args.seed,
            timestamp=started_ts,
        ),
        identity_digest=digest,
        input_pins_digest=digest,
        phases_reached=["attempt_started"],
        gate_verdicts={
            "pipeline": {
                "verdict": "running",
                "receipt": "pending-build-scoped-terminal-receipt",
            }
        },
    )
    attempt: dict[str, object] = {
        "state": state,
        "started_at": started_at,
        "started_ts": started_ts,
        "code_pin": "unresolved-local-git-code-pin",
        # Logbook chain configuration is validated before any side effect: a
        # malformed or conflicting predecessor refuses the run here, before
        # the build can unlink stale coverage/crosswalk sidecars. Config
        # refusals record no row, like argparse refusals.
        "predecessor": resolve_predecessor(args.logbook_prev_row_digest),
    }
    try:
        return _main_impl(args, attempt=attempt)
    except Exception as error:
        _record_rowwise_error(
            error=error,
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            seed=args.seed,
            code_pin=str(attempt["code_pin"]),
            predecessor=attempt["predecessor"],
            base_dir=args.out,
            spool_dir=args.out / "logbook-spool",
        )
        raise


def _main_impl(
    args: argparse.Namespace,
    *,
    attempt: dict[str, object] | None,
) -> int:
    input_h5 = args.input_h5.resolve()
    input_artifact = _artifact_info(input_h5)
    _verify_artifact_pin(
        input_artifact,
        pinned_sha256=args.input_sha256,
        option="--input-h5",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    base_summary = _h5_summary(input_h5)
    source_year = _source_year(args.source_year, base_summary=base_summary)
    output_h5 = _dataset_output_path(
        args.out,
        dataset_filename=args.dataset_filename,
        input_stem=input_h5.stem,
        source_year=source_year,
    )
    _validate_output_paths(input_h5=input_h5, output_h5=output_h5, args=args)
    if args.crosswalk is not None and args.ladder_sha256 is not None:
        raise ValueError("--ladder-sha256 and --crosswalk are mutually exclusive.")
    if args.ladder is not None:
        if args.crosswalk is not None:
            raise ValueError("--ladder and --crosswalk are mutually exclusive.")
        if args.constituency_codes is not None or args.la_codes is not None:
            raise ValueError(
                "--ladder does not take coverage-code checks; the ladder gate "
                "validates coverage."
            )
        sidecars = {
            (args.out / MANIFEST_FILENAME).resolve(),
            (args.out / COVERAGE_FILENAME).resolve(),
            (args.out / DRY_RUN_PLAN_FILENAME).resolve(),
            (args.out / CROSSWALK_FILENAME).resolve(),
        }
        if args.ladder.resolve() in sidecars:
            raise ValueError(
                "--ladder must not point at a build sidecar path inside "
                "--out; it would be overwritten."
            )
        # A reused crosswalk output directory must not leave stale sidecars
        # beside a ladder manifest that reports no coverage output.
        (args.out / COVERAGE_FILENAME).unlink(missing_ok=True)
        (args.out / CROSSWALK_FILENAME).unlink(missing_ok=True)
        return _run_ladder_route(
            args,
            input_h5=input_h5,
            input_artifact=input_artifact,
            output_h5=output_h5,
            base_summary=base_summary,
            source_year=source_year,
            attempt=attempt,
        )
    crosswalk_source = _load_or_build_crosswalk(args)
    crosswalk = crosswalk_source.frame
    crosswalk_path = crosswalk_source.path
    area_codes_by_type = _area_codes_by_type(args)
    coverage = _validate_optional_coverage(crosswalk, area_codes_by_type)

    if args.dry_run:
        plan = _dry_run_plan(
            args,
            input_h5=input_h5,
            input_artifact=input_artifact,
            output_h5=output_h5,
            crosswalk=crosswalk,
            crosswalk_source=crosswalk_source,
            base_summary=base_summary,
            source_year=source_year,
            coverage=coverage,
        )
        plan_path = args.out / DRY_RUN_PLAN_FILENAME
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if attempt is not None:
        state = attempt["state"]
        assert isinstance(state, AttemptState)
        pins = {
            "dataset": _pin_from_artifact(input_artifact),
            "crosswalk": _pin_from_artifact(_artifact_info(crosswalk_path)),
        }
        attempt["code_pin"] = git_code_pin(_REPOSITORY)
        state.build_id = _new_rowwise_build_id(
            route="crosswalk",
            seed=args.seed,
            timestamp=attempt["started_ts"],
        )
        state.input_pins_digest = role_pins_digest(pins)
        state.identity_digest = _rowwise_identity_digest(
            route="crosswalk",
            pins=pins,
            args=args,
            source_year=source_year,
        )
        append_phase(state, "configured")
        append_phase(state, "inputs_pinned")

    result = clone_uk_dataset_with_rowwise_geography(
        input_h5,
        crosswalk,
        output_path=output_h5,
        n_clones=args.n_clones,
        seed=args.seed,
        source_year=source_year,
        require_all_countries=not args.allow_missing_country,
        require_constituency=not args.allow_blank_constituency,
        constrain_to_region=not args.allow_cross_region_assignment,
        avoid_constituency_collisions=not args.allow_constituency_collisions,
        source_lineage_modulus=args.source_lineage_modulus,
    )
    if attempt is not None:
        state = attempt["state"]
        assert isinstance(state, AttemptState)
        append_phase(state, "cloned")
    rowwise_summary = _rowwise_summary(
        result,
        base_summary=base_summary,
        source_lineage_modulus=args.source_lineage_modulus,
    )
    coverage_path = args.out / COVERAGE_FILENAME
    coverage_artifact = None
    if not coverage.empty:
        coverage.to_csv(coverage_path, index=False)
        coverage_artifact = _artifact_info(coverage_path)
    else:
        coverage_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": 1,
        "build_kind": "uk_rowwise_local_geography_dataset",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "parameters": _parameters(args, source_year=source_year),
        "inputs": {
            "dataset": input_artifact,
            "crosswalk": _artifact_info(crosswalk_path),
        },
        "outputs": {
            "dataset": _artifact_info(output_h5),
            "crosswalk": (
                _artifact_info(crosswalk_path) if crosswalk_source.generated else None
            ),
            "coverage_summary": coverage_artifact,
        },
        "base_dataset": base_summary,
        "rowwise_dataset": rowwise_summary,
        "coverage": coverage.to_dict("records") if not coverage.empty else [],
    }
    manifest_path = args.out / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if attempt is not None:
        state = attempt["state"]
        assert isinstance(state, AttemptState)
        append_phase(state, "manifest_written")
        state.gate_verdicts = {
            "uk_mass_conservation": {
                "verdict": (
                    "passed"
                    if rowwise_summary["weights"]["mass_conservation"]["passed"]
                    else "failed"
                ),
                "receipt": (
                    f"{local_artifact_reference(manifest_path, repository_hint=_REPOSITORY)}"
                    "#/rowwise_dataset/weights/mass_conservation"
                ),
            }
        }
        if coverage_artifact is not None:
            state.gate_verdicts["uk_coverage"] = {
                "verdict": "passed",
                "receipt": (
                    f"{local_artifact_reference(manifest_path, repository_hint=_REPOSITORY)}"
                    "#/rowwise_dataset/coverage"
                ),
            }
        state.artifact_location = local_artifact_reference(
            output_h5,
            repository_hint=_REPOSITORY,
        )
        spool_path = _record_rowwise_attempt(
            state=state,
            started_at=attempt["started_at"],
            started_ts=attempt["started_ts"],
            seed=args.seed,
            code_pin=str(attempt["code_pin"]),
            disposition="iterating",
            predecessor=attempt["predecessor"],
            spool_dir=args.out / "logbook-spool",
        )
        print(f"Wrote Logbook row: {spool_path}", file=sys.stderr)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


class CrosswalkSource:
    def __init__(self, frame: pd.DataFrame, path: Path, *, generated: bool) -> None:
        self.frame = frame
        self.path = path
        self.generated = generated


def _dataset_output_path(
    out_dir: Path,
    *,
    dataset_filename: str | None,
    input_stem: str,
    source_year: int,
) -> Path:
    filename = dataset_filename or DATASET_FILENAME_TEMPLATE.format(
        input_stem=input_stem,
        source_year=source_year,
    )
    path = Path(filename)
    if path.is_absolute() or path.name != filename or path.name in {"", ".", ".."}:
        raise ValueError("--dataset-filename must be a filename, not a path.")
    reserved = {
        CROSSWALK_FILENAME,
        MANIFEST_FILENAME,
        COVERAGE_FILENAME,
        DRY_RUN_PLAN_FILENAME,
    }
    if path.name in reserved:
        raise ValueError(
            f"--dataset-filename must not use reserved name {path.name!r}."
        )
    return out_dir / path.name


def _validate_output_paths(
    *,
    input_h5: Path,
    output_h5: Path,
    args: argparse.Namespace,
) -> None:
    output_sidecars = {
        (args.out / MANIFEST_FILENAME).resolve(),
        (args.out / COVERAGE_FILENAME).resolve(),
        (args.out / DRY_RUN_PLAN_FILENAME).resolve(),
    }
    generated_crosswalk_path = (args.out / CROSSWALK_FILENAME).resolve()
    reserved_paths = {
        input_h5,
        *output_sidecars,
    }
    if getattr(args, "ladder", None) is not None:
        reserved_paths.add(args.ladder.resolve())
    if args.crosswalk is None:
        reserved_paths.add(generated_crosswalk_path)
    else:
        crosswalk_path = args.crosswalk.resolve()
        if crosswalk_path in output_sidecars:
            raise ValueError("--crosswalk path must differ from output sidecars.")
        reserved_paths.add(crosswalk_path)
    if output_h5.resolve() in reserved_paths:
        raise ValueError("Output H5 path must differ from inputs and sidecars.")


def _source_year(cli_source_year: int | None, *, base_summary: dict[str, Any]) -> int:
    if cli_source_year is not None:
        return cli_source_year
    time_period = base_summary.get("time_period")
    if time_period is None:
        raise ValueError(
            "Could not infer source year from input H5 time_period; pass --source-year."
        )
    try:
        return int(str(time_period)[:4])
    except ValueError as exc:
        raise ValueError(
            "Could not infer source year from input H5 time_period; pass --source-year."
        ) from exc


def _load_or_build_crosswalk(args: argparse.Namespace) -> CrosswalkSource:
    if args.crosswalk is not None:
        path = args.crosswalk.resolve()
        generated_crosswalk_path = args.out / CROSSWALK_FILENAME
        if path != generated_crosswalk_path.resolve() and not getattr(
            args, "dry_run", False
        ):
            # A dry run must not delete build artifacts, including a
            # previously generated crosswalk cache.
            generated_crosswalk_path.unlink(missing_ok=True)
        return CrosswalkSource(_read_crosswalk(path), path, generated=False)
    crosswalk = build_official_uk_geography_crosswalk()
    path = args.out / CROSSWALK_FILENAME
    write_geography_crosswalk(crosswalk, path)
    return CrosswalkSource(crosswalk, path, generated=True)


def _read_crosswalk(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "oa_code": str,
            "lsoa_code": str,
            "msoa_code": str,
            "la_code": str,
            "constituency_code": str,
            "region_code": str,
            "country": str,
        },
    )


def _area_codes_by_type(args: argparse.Namespace) -> dict[str, list[str]]:
    area_codes: dict[str, list[str]] = {}
    if args.constituency_codes is not None:
        area_codes["constituency"] = _read_code_csv(args.constituency_codes)
    if args.la_codes is not None:
        area_codes["la"] = _read_code_csv(args.la_codes)
    return area_codes


def _read_code_csv(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype=str)
    if "code" not in frame.columns:
        raise ValueError(f"{path} must include a `code` column.")
    return frame["code"].dropna().astype(str).str.strip().tolist()


def _validate_optional_coverage(
    crosswalk: pd.DataFrame,
    area_codes_by_type: dict[str, list[str]],
) -> pd.DataFrame:
    if not area_codes_by_type:
        return pd.DataFrame()
    validate_geography_coverage(
        crosswalk,
        required_countries=["England", "Wales", "Scotland", "Northern Ireland"],
        area_codes_by_type=area_codes_by_type,
    )
    return geography_coverage_summary(crosswalk, area_codes_by_type)


def _parameters(args: argparse.Namespace, *, source_year: int) -> dict[str, Any]:
    ladder_route = getattr(args, "ladder", None) is not None
    return {
        "n_clones": args.n_clones,
        "seed": args.seed,
        "source_year": source_year,
        # Crosswalk-sampler knobs are meaningless on the ladder route and are
        # recorded as null rather than falsely claimed effective.
        "require_all_countries": (
            None if ladder_route else not args.allow_missing_country
        ),
        "require_constituency": (
            None if ladder_route else not args.allow_blank_constituency
        ),
        "constrain_to_region": (
            None if ladder_route else not args.allow_cross_region_assignment
        ),
        "avoid_constituency_collisions": (
            None if ladder_route else not args.allow_constituency_collisions
        ),
        "source_lineage_modulus": args.source_lineage_modulus,
        "assignment_route": "ladder" if ladder_route else "crosswalk",
        "expected_constituency_vintage": (
            args.expected_constituency_vintage if ladder_route else None
        ),
    }


def _h5_summary(path: Path) -> dict[str, Any]:
    weight_kind, mass_log = read_uk_single_year_weight_metadata(path)
    with pd.HDFStore(path, mode="r") as store:
        household = store["household"]
        return {
            "path": str(path),
            "tables": {key.strip("/"): list(store[key].shape) for key in store.keys()},
            "household_weight_sum": float(household["household_weight"].sum()),
            "time_period": str(store["time_period"].iloc[0]),
            "household_weight_kind": weight_kind.value,
            "mass_log_records": len(mass_log),
            "distinct_source_households": (
                int(household["source_household_id"].nunique())
                if "source_household_id" in household.columns
                else None
            ),
        }


def _dry_run_plan(
    args: argparse.Namespace,
    *,
    input_h5: Path,
    input_artifact: dict[str, Any],
    output_h5: Path,
    crosswalk: pd.DataFrame,
    crosswalk_source: CrosswalkSource,
    base_summary: dict[str, Any],
    source_year: int,
    coverage: pd.DataFrame,
) -> dict[str, Any]:
    """Compute the clone plan without cloning or writing a dataset."""

    with pd.HDFStore(input_h5, mode="r") as store:
        household = store["household"]
        person_ids = _select_h5_columns(store, "person", list(PERSON_ID_COLUMNS))
        benunit_ids = _select_h5_columns(store, "benunit", ["benunit_id"])
    _validate_dry_run_input(
        input_h5,
        household=household,
        person_ids=person_ids,
        benunit_ids=benunit_ids,
    )
    id_multiplier = id_multiplier_for_values(
        household["household_id"],
        person_ids["person_id"],
        person_ids["person_household_id"],
        person_ids["person_benunit_id"],
        benunit_ids["benunit_id"],
    )
    if args.source_lineage_modulus is not None:
        household = apply_uk_source_lineage_modulus(
            household,
            modulus=args.source_lineage_modulus,
        )
    # The realized assignment IS the real build's: the sampler consumes only
    # the household table, and identical inputs, flags, id multiplier, and
    # seed produce identical draws — so these per-area counts are exact for
    # the build this plan describes, collision avoidance included.
    assignment = assign_household_geography(
        household,
        crosswalk,
        n_clones=args.n_clones,
        seed=args.seed,
        id_multiplier=id_multiplier,
        source_year=source_year,
        require_all_countries=not args.allow_missing_country,
        require_constituency=not args.allow_blank_constituency,
        constrain_to_region=not args.allow_cross_region_assignment,
        avoid_constituency_collisions=not args.allow_constituency_collisions,
    )
    realized = _realized_area_support(assignment.household, crosswalk)
    collision_free = expected_uk_rowwise_area_support(
        household,
        crosswalk,
        n_clones=args.n_clones,
        source_year=source_year,
        require_all_countries=not args.allow_missing_country,
        require_constituency=not args.allow_blank_constituency,
        constrain_to_region=not args.allow_cross_region_assignment,
    )
    table_rows = {
        name: base_summary["tables"][name][0]
        for name in ("person", "benunit", "household")
    }
    input_bytes = input_h5.stat().st_size
    return {
        "schema_version": 1,
        "build_kind": "uk_rowwise_local_geography_dry_run",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "parameters": _parameters(args, source_year=source_year),
        "input": {
            "dataset": input_artifact,
            "crosswalk": _artifact_info(crosswalk_source.path),
            "tables": base_summary["tables"],
            "household_weight_sum": base_summary["household_weight_sum"],
            "time_period": base_summary["time_period"],
            "household_weight_kind": base_summary["household_weight_kind"],
            "mass_log_records": base_summary["mass_log_records"],
        },
        "plan": {
            "n_clones": args.n_clones,
            "id_multiplier": id_multiplier,
            "output_h5": str(output_h5),
            "rows": {name: rows * args.n_clones for name, rows in table_rows.items()},
            "output_bytes_estimate": input_bytes * args.n_clones,
            "output_bytes_estimate_basis": (
                "lower-bound estimate: linear scaling of the input H5 byte "
                "size by n_clones; added geography/lineage columns and HDF "
                "table overhead increase the actual size"
            ),
        },
        "realized_support": {
            "basis": (
                f"realized assignment at seed {args.seed} — identical draws "
                "to the real build under these parameters; zero-row "
                "sampleable areas included"
            ),
            **{
                area_type: _support_summary(realized, area_type)
                for area_type in ("constituency", "la")
            },
        },
        "collision_free_expected_support": {
            "basis": (
                "analytic collision-free expectation; diverges from realized "
                "support when n_clones is comparable to a group's sampleable "
                "constituency count"
            ),
            **{
                area_type: _support_summary(collision_free, area_type)
                for area_type in ("constituency", "la")
            },
        },
        "source_lineage": _source_lineage_report(
            household,
            modulus=args.source_lineage_modulus,
        ),
        "coverage": coverage.to_dict("records") if not coverage.empty else [],
    }


def _run_ladder_route(
    args: argparse.Namespace,
    *,
    input_h5: Path,
    input_artifact: dict[str, Any],
    output_h5: Path,
    base_summary: dict[str, Any],
    source_year: int,
    attempt: dict[str, object] | None,
) -> int:
    """Build (or dry-run plan) the rowwise dataset through the OA ladder."""

    ladder_path = args.ladder.resolve()
    ladder_artifact = _artifact_info(ladder_path)
    _verify_artifact_pin(
        ladder_artifact,
        pinned_sha256=args.ladder_sha256,
        option="--ladder",
    )
    _annotate_ladder_crosswalk_pin(ladder_artifact)
    ladder = load_uk_oa_ladder(ladder_path)

    if args.dry_run:
        plan = _ladder_dry_run_plan(
            args,
            input_h5=input_h5,
            input_artifact=input_artifact,
            output_h5=output_h5,
            base_summary=base_summary,
            source_year=source_year,
            ladder=ladder,
            ladder_artifact=ladder_artifact,
        )
        plan_path = args.out / DRY_RUN_PLAN_FILENAME
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if attempt is not None:
        state = attempt["state"]
        assert isinstance(state, AttemptState)
        pins = {
            "dataset": _pin_from_artifact(input_artifact),
            "ladder": _pin_from_artifact(ladder_artifact),
        }
        attempt["code_pin"] = git_code_pin(_REPOSITORY)
        state.build_id = _new_rowwise_build_id(
            route="ladder",
            seed=args.seed,
            timestamp=attempt["started_ts"],
        )
        state.input_pins_digest = role_pins_digest(pins)
        state.identity_digest = _rowwise_identity_digest(
            route="ladder",
            pins=pins,
            args=args,
            source_year=source_year,
        )
        append_phase(state, "configured")
        append_phase(state, "inputs_pinned")

    result = clone_uk_dataset_with_ladder_geography(
        input_h5,
        ladder,
        output_path=output_h5,
        n_clones=args.n_clones,
        seed=args.seed,
        source_year=source_year,
        expected_constituency_vintage=args.expected_constituency_vintage,
        source_lineage_modulus=args.source_lineage_modulus,
    )
    if attempt is not None:
        state = attempt["state"]
        assert isinstance(state, AttemptState)
        append_phase(state, "cloned")
    rowwise_summary = _rowwise_summary(
        result,
        base_summary=base_summary,
        source_lineage_modulus=args.source_lineage_modulus,
        geo_columns=UK_GEOGRAPHY_LADDER_COLUMNS,
        constituency_column="constituency_code",
        la_column="local_authority_code",
    )
    rowwise_summary["gate"] = {
        "name": "uk_geography_ladder",
        "passed": bool(result.gate.passed),
        "details": dict(result.gate.details),
    }
    manifest = {
        "schema_version": 1,
        "build_kind": "uk_rowwise_local_geography_dataset",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "parameters": _parameters(args, source_year=source_year),
        "inputs": {
            "dataset": input_artifact,
            "ladder": ladder_artifact,
        },
        "outputs": {
            "dataset": _artifact_info(output_h5),
            "crosswalk": None,
            "coverage_summary": None,
        },
        "base_dataset": base_summary,
        "rowwise_dataset": rowwise_summary,
        "coverage": [],
    }
    manifest_path = args.out / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if attempt is not None:
        state = attempt["state"]
        assert isinstance(state, AttemptState)
        append_phase(state, "manifest_written")
        state.gate_verdicts = {
            "uk_geography_ladder": {
                "verdict": "passed" if result.gate.passed else "failed",
                "receipt": (
                    f"{local_artifact_reference(manifest_path, repository_hint=_REPOSITORY)}"
                    "#/rowwise_dataset/gate"
                ),
            }
        }
        state.artifact_location = local_artifact_reference(
            output_h5,
            repository_hint=_REPOSITORY,
        )
        spool_path = _record_rowwise_attempt(
            state=state,
            started_at=attempt["started_at"],
            started_ts=attempt["started_ts"],
            seed=args.seed,
            code_pin=str(attempt["code_pin"]),
            disposition="iterating",
            predecessor=attempt["predecessor"],
            spool_dir=args.out / "logbook-spool",
        )
        print(f"Wrote Logbook row: {spool_path}", file=sys.stderr)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _ladder_dry_run_plan(
    args: argparse.Namespace,
    *,
    input_h5: Path,
    input_artifact: dict[str, Any],
    output_h5: Path,
    base_summary: dict[str, Any],
    source_year: int,
    ladder: Any,
    ladder_artifact: dict[str, Any],
) -> dict[str, Any]:
    """Exact ladder-route plan: the real cloned assignment at the build seed."""

    with pd.HDFStore(input_h5, mode="r") as store:
        household = store["household"]
        person_ids = _select_h5_columns(store, "person", list(PERSON_ID_COLUMNS))
        benunit_ids = _select_h5_columns(store, "benunit", ["benunit_id"])
    _validate_dry_run_input(
        input_h5,
        household=household,
        person_ids=person_ids,
        benunit_ids=benunit_ids,
    )
    _refuse_preassigned_geography(household, label="household")
    id_multiplier = id_multiplier_for_values(
        household["household_id"],
        person_ids["person_id"],
        person_ids["person_household_id"],
        person_ids["person_benunit_id"],
        benunit_ids["benunit_id"],
    )
    if args.source_lineage_modulus is not None:
        household = apply_uk_source_lineage_modulus(
            household,
            modulus=args.source_lineage_modulus,
        )
    # Fence parity with the real build (a plan must never bless a build that
    # would raise): weight validity, mass-chain currency, then the release
    # gate on the divided-weight assignment.
    from microcosm.build.uk_runtime.rowwise_dataset import _assert_mass_log_current

    weight_values = pd.to_numeric(
        household["household_weight"], errors="raise"
    ).to_numpy(dtype=float)
    if not (weight_values >= 0).all() or not np.isfinite(weight_values).all():
        raise ValueError("household weights must be finite and non-negative.")
    if float(weight_values.sum()) <= 0.0:
        raise ValueError("household weights must carry positive total mass.")
    _kind, mass_log = read_uk_single_year_weight_metadata(input_h5)
    _assert_mass_log_current(mass_log, float(weight_values.sum()))
    cloned = clone_entity_frame(
        household,
        id_columns=("household_id",),
        n_clones=args.n_clones,
        id_multiplier=id_multiplier,
        clone_index_column="clone_index",
    ).reset_index(drop=True)
    cloned["household_weight"] = (
        pd.to_numeric(cloned["household_weight"], errors="raise").to_numpy(dtype=float)
        / args.n_clones
    )
    assigned = assign_uk_geography_ladder(
        cloned,
        ladder,
        seed=args.seed,
        expected_constituency_vintage=args.expected_constituency_vintage,
    )
    gate = uk_geography_ladder_gate(
        assigned,
        assigned["household_weight"].to_numpy(dtype=float),
    )
    if not gate.passed:
        raise ValueError(
            "UK geography ladder gate would fail this build: "
            + "; ".join(gate.failures)
        )
    realized = _ladder_realized_support(assigned, ladder)
    table_rows = {
        name: base_summary["tables"][name][0]
        for name in ("person", "benunit", "household")
    }
    input_bytes = input_h5.stat().st_size
    return {
        "schema_version": 1,
        "build_kind": "uk_rowwise_local_geography_dry_run",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "parameters": _parameters(args, source_year=source_year),
        "input": {
            "dataset": input_artifact,
            "ladder": ladder_artifact,
            "tables": base_summary["tables"],
            "household_weight_sum": base_summary["household_weight_sum"],
            "time_period": base_summary["time_period"],
            "household_weight_kind": base_summary["household_weight_kind"],
            "mass_log_records": base_summary["mass_log_records"],
        },
        "plan": {
            "n_clones": args.n_clones,
            "id_multiplier": id_multiplier,
            "output_h5": str(output_h5),
            "rows": {name: rows * args.n_clones for name, rows in table_rows.items()},
            "output_bytes_estimate": input_bytes * args.n_clones,
            "output_bytes_estimate_basis": (
                "lower-bound estimate: linear scaling of the input H5 byte "
                "size by n_clones; added geography/lineage columns and HDF "
                "table overhead increase the actual size"
            ),
        },
        "realized_support": {
            "basis": (
                f"realized ladder assignment at seed {args.seed} — identical "
                "draws to the real build under these parameters; zero-row "
                "ladder areas included"
            ),
            **{
                area_type: _support_summary(realized, area_type)
                for area_type in ("constituency", "la")
            },
        },
        "source_lineage": _source_lineage_report(
            household,
            modulus=args.source_lineage_modulus,
        ),
        "coverage": [],
    }


def _ladder_realized_support(
    assigned_household: pd.DataFrame,
    ladder: Any,
) -> pd.DataFrame:
    """Realized rows per ladder area, zeros included for unassigned areas."""

    import numpy as _np

    rows: list[dict[str, Any]] = []
    for assigned_column, ladder_codes, area_type in (
        ("constituency_code", ladder.constituency_code, "constituency"),
        ("local_authority_code", ladder.local_authority_code, "la"),
    ):
        assigned = assigned_household[assigned_column].astype(str).str.strip()
        counts = assigned[assigned != ""].value_counts()
        codes = sorted(set(_np.unique(ladder_codes).tolist()) | set(counts.index))
        rows.extend(
            {
                "area_type": area_type,
                "area_code": code,
                "expected_rows": float(counts.get(code, 0)),
            }
            for code in codes
        )
    return pd.DataFrame(rows)


def _validate_dry_run_input(
    input_h5: Path,
    *,
    household: pd.DataFrame,
    person_ids: pd.DataFrame,
    benunit_ids: pd.DataFrame,
) -> None:
    """Mirror the real build's input refusals so a plan cannot bless an
    input the build would reject."""

    if input_h5.suffix != ".h5":
        raise ValueError("UK single-year dataset path must end with '.h5'.")
    for frame, column, label in (
        (household, "household_id", "household"),
        (person_ids, "person_id", "person"),
        (benunit_ids, "benunit_id", "benunit"),
    ):
        if frame[column].isna().any():
            raise ValueError(f"{label}.{column} contains missing values.")
        if frame[column].duplicated().any():
            duplicates = frame.loc[frame[column].duplicated(), column].unique()
            raise ValueError(
                f"{label}.{column} must be unique; duplicate value(s): "
                f"{list(map(str, duplicates[:5]))}."
            )
    household_ids = set(household["household_id"])
    missing_households = sorted(set(person_ids["person_household_id"]) - household_ids)
    if missing_households:
        raise ValueError(
            "person.person_household_id contains value(s) absent from "
            f"household: {missing_households[:5]}."
        )
    missing_benunits = sorted(
        set(person_ids["person_benunit_id"]) - set(benunit_ids["benunit_id"])
    )
    if missing_benunits:
        raise ValueError(
            "person.person_benunit_id contains value(s) absent from benunit: "
            f"{missing_benunits[:5]}."
        )


def _realized_area_support(
    assigned_household: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Realized rows per sampleable area, zeros included."""

    sampleable = crosswalk[pd.to_numeric(crosswalk["population"]) > 0]
    rows: list[dict[str, Any]] = []
    for assigned_column, crosswalk_column, area_type in (
        ("constituency_code_oa", "constituency_code", "constituency"),
        ("la_code_oa", "la_code", "la"),
    ):
        assigned = assigned_household[assigned_column].astype(str).str.strip()
        counts = assigned[assigned != ""].value_counts()
        codes = sorted(
            {
                str(code).strip()
                for code in sampleable[crosswalk_column]
                if str(code).strip()
            }
            | set(counts.index)
        )
        rows.extend(
            {
                "area_type": area_type,
                "area_code": code,
                "expected_rows": float(counts.get(code, 0)),
            }
            for code in codes
        )
    return pd.DataFrame(rows)


def _select_h5_columns(
    store: pd.HDFStore,
    key: str,
    columns: list[str],
) -> pd.DataFrame:
    try:
        return store.select(key, columns=columns)
    except (TypeError, ValueError, KeyError):
        return store[key][columns]


def _support_summary(
    support: pd.DataFrame,
    area_type: str,
    *,
    bottom: int = EXPECTED_SUPPORT_BOTTOM_AREAS,
) -> dict[str, Any]:
    subset = support[support["area_type"] == area_type]
    if subset.empty:
        return {
            "n_areas": 0,
            "min_rows": 0.0,
            "median_rows": 0.0,
            "mean_rows": 0.0,
            "max_rows": 0.0,
            "bottom": [],
        }
    values = subset["expected_rows"]
    ordered = subset.sort_values(
        ["expected_rows", "area_code"],
        kind="mergesort",
    ).head(bottom)
    return {
        "n_areas": int(len(subset)),
        "min_rows": float(values.min()),
        "median_rows": float(values.median()),
        "mean_rows": float(values.mean()),
        "max_rows": float(values.max()),
        "bottom": [
            {
                "area_code": str(row.area_code),
                "rows": float(row.expected_rows),
            }
            for row in ordered.itertuples(index=False)
        ],
    }


def _pool_lineage_block(household: pd.DataFrame) -> dict[str, Any] | None:
    if POOL_SOURCE_LINEAGE_COLUMN not in household.columns:
        return None
    counts = household.groupby(POOL_SOURCE_LINEAGE_COLUMN).size()
    block = {
        "distinct_pool_source_households": int(counts.size),
        "pool_copies_per_source": {
            "min": int(counts.min()),
            "median": float(counts.median()),
            "max": int(counts.max()),
        },
    }
    if "household_support_channel" in household.columns:
        block["distinct_by_support_channel"] = {
            str(channel): int(group[POOL_SOURCE_LINEAGE_COLUMN].nunique())
            for channel, group in household.groupby("household_support_channel")
        }
    return block


def _explicit_lineage_block(household: pd.DataFrame) -> dict[str, Any] | None:
    spine_columns = [
        column for column in UK_SPINE_LINEAGE_COLUMNS if column in household.columns
    ]
    if not spine_columns:
        return None

    columns_present = [
        column
        for column in ("source_household_id", *UK_SPINE_LINEAGE_COLUMNS)
        if column in household.columns
    ]
    block: dict[str, Any] = {
        "basis": "explicit_lineage_columns",
        "columns_present": columns_present,
        "flag_counts": {
            column: int(household[column].fillna(False).astype(bool).sum())
            for column in UK_SPINE_LINEAGE_COLUMNS
            if column.startswith("household_is_") and column in household.columns
        },
    }
    if "source_household_id" in household.columns:
        block["distinct_source_households"] = int(
            household["source_household_id"].nunique()
        )
        if "household_support_channel" in household.columns:
            block["distinct_by_support_channel"] = {
                str(channel): int(group["source_household_id"].nunique())
                for channel, group in household.groupby("household_support_channel")
            }
    return block


def _source_lineage_report(
    household: pd.DataFrame,
    *,
    modulus: int | None,
) -> dict[str, Any]:
    """Report pool, immediate, and explicit spine lineage when supported."""

    pool = _pool_lineage_block(household)
    immediate = None
    if "source_household_id" in household.columns:
        immediate = {
            "distinct_source_households": int(
                household["source_household_id"].nunique()
            ),
        }
    return {
        "pool_modulus": modulus,
        "pool": pool,
        "immediate": immediate,
        "explicit": _explicit_lineage_block(household),
    }


def _rowwise_summary(
    result,
    *,
    base_summary: dict[str, Any],
    source_lineage_modulus: int | None = None,
    geo_columns: tuple[str, ...] = (
        "oa_code",
        "lsoa_code",
        "msoa_code",
        "la_code_oa",
        "constituency_code_oa",
        "region_code_oa",
    ),
    constituency_column: str = "constituency_code_oa",
    la_column: str = "la_code_oa",
) -> dict[str, Any]:
    if isinstance(result, UKLadderRowwiseDatasetResult):
        person = result.frame.table("person")
        benunit = result.frame.table("benunit")
        household = engine_tables(result.frame, weighted_entities=("household",))[
            "household"
        ]
        weight_kind = uk_household_weight_kind(result.frame)
        mass_log = result.frame.mass_log
        time_period = uk_time_period(result.frame)
        clone_column = ladder_clone_index_column("household")
    else:
        person = result.person
        benunit = result.benunit
        household = result.household
        weight_kind = result.household_weight_kind
        mass_log = result.mass_log
        time_period = result.time_period
        clone_column = "clone_index"
    missing_geography = household[list(geo_columns)].isna().any(axis=1)
    for column in geo_columns:
        missing_geography |= household[column].fillna("").astype(str).str.strip().eq("")
    assigned_constituencies = household.loc[
        _nonblank_string_mask(household[constituency_column]),
        constituency_column,
    ]
    assigned_las = household.loc[
        _nonblank_string_mask(household[la_column]),
        la_column,
    ]
    by_constituency = assigned_constituencies.groupby(assigned_constituencies).size()
    by_la = assigned_las.groupby(assigned_las).size()
    weight_sum = float(household["household_weight"].sum())
    constituency_rows = _area_row_summary(by_constituency)
    la_rows = _area_row_summary(by_la)
    input_total = float(base_summary["household_weight_sum"])
    abs_delta = abs(weight_sum - input_total)
    clone0 = household
    if clone_column in household.columns:
        clone0 = household[household[clone_column] == 0]
    lineage = {
        "pool_modulus": source_lineage_modulus,
        "pool": _pool_lineage_block(clone0),
        "immediate": (
            {
                "distinct_source_households": base_summary[
                    "distinct_source_households"
                ],
            }
            if base_summary.get("distinct_source_households") is not None
            else None
        ),
        "explicit": _explicit_lineage_block(clone0),
    }
    return {
        "weights": {
            "household_weight_kind": weight_kind.value,
            "mass_log_records": len(mass_log),
            "mass_conservation": {
                "input_total": input_total,
                "output_total": weight_sum,
                "abs_delta": abs_delta,
                "relative_tolerance": MASS_CONSERVATION_RELATIVE_TOLERANCE,
                "passed": bool(
                    abs_delta <= MASS_CONSERVATION_RELATIVE_TOLERANCE * abs(input_total)
                ),
            },
        },
        "source_lineage": lineage,
        "tables": {
            "person": list(person.shape),
            "benunit": list(benunit.shape),
            "household": list(household.shape),
        },
        "time_period": time_period,
        "n_clones": result.n_clones,
        "id_multiplier": result.id_multiplier,
        "household_weight_sum": weight_sum,
        "household_weight_delta": weight_sum - base_summary["household_weight_sum"],
        "missing_geography_rows": int(missing_geography.sum()),
        "assigned_constituencies": int(by_constituency.size),
        "assigned_local_authorities": int(by_la.size),
        "min_household_rows_by_constituency": constituency_rows["min"],
        "min_household_rows_by_local_authority": la_rows["min"],
        "median_household_rows_by_constituency": constituency_rows["median"],
        "median_household_rows_by_local_authority": la_rows["median"],
        "duplicate_source_household_constituency_pairs": (
            _duplicate_source_household_constituency_pairs(
                household,
                constituency_column=constituency_column,
            )
        ),
    }


def _nonblank_string_mask(values: pd.Series) -> pd.Series:
    return values.notna() & values.astype(str).str.strip().ne("")


def _area_row_summary(counts: pd.Series) -> dict[str, int | float]:
    if counts.empty:
        return {"min": 0, "median": 0.0}
    return {"min": int(counts.min()), "median": float(counts.median())}


def _duplicate_source_household_constituency_pairs(
    household: pd.DataFrame,
    *,
    constituency_column: str = "constituency_code_oa",
) -> int:
    if "source_household_id" not in household.columns:
        return 0
    assigned = household[_nonblank_string_mask(household[constituency_column])]
    return int(assigned.duplicated(["source_household_id", constituency_column]).sum())


def _artifact_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _verify_artifact_pin(
    artifact: dict[str, Any],
    *,
    pinned_sha256: str | None,
    option: str,
) -> None:
    measured_sha256 = str(artifact["sha256"])
    if pinned_sha256 is not None and measured_sha256 != pinned_sha256:
        raise SystemExit(
            f"error: {option} sha mismatch: "
            f"measured {measured_sha256}, pinned {pinned_sha256}"
        )
    artifact["pin_verified"] = pinned_sha256 is not None


def _annotate_ladder_crosswalk_pin(ladder_artifact: dict[str, Any]) -> None:
    try:
        crosswalk = load_uk_local_area_crosswalk()
        expected_sha256 = str(crosswalk["ladder_artifact_sha256"])
    except Exception as error:
        ladder_artifact["matches_local_area_crosswalk_pin"] = None
        ladder_artifact["local_area_crosswalk_pin_error"] = (
            f"{type(error).__name__}: {error}"
        )
        return
    ladder_artifact["matches_local_area_crosswalk_pin"] = (
        str(ladder_artifact["sha256"]) == expected_sha256
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
