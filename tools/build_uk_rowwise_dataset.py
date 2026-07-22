"""Build a Populace UK row-wise local-geography dataset.

This is the narrow build driver for the UK local replacement path. It starts
from an existing compact Populace UK single-year H5, builds or loads the
official-source geography crosswalk, clones the entity tables, assigns each
household a finest available geography row, and writes diagnostics that prove
coverage and weight preservation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from populace.build.uk_runtime import (
    MASS_CONSERVATION_RELATIVE_TOLERANCE,
    PERSON_ID_COLUMNS,
    apply_uk_source_lineage_modulus,
    build_official_uk_geography_crosswalk,
    clone_uk_dataset_with_rowwise_geography,
    expected_uk_rowwise_area_support,
    geography_coverage_summary,
    id_multiplier_for_values,
    read_uk_single_year_weight_metadata,
    validate_geography_coverage,
    write_geography_crosswalk,
)

CROSSWALK_FILENAME = "uk_official_geography_crosswalk.csv.gz"
DATASET_FILENAME_TEMPLATE = "populace_uk_{source_year}_rowwise.h5"
MANIFEST_FILENAME = "rowwise_build_manifest.json"
COVERAGE_FILENAME = "geography_coverage_summary.csv"
DRY_RUN_PLAN_FILENAME = "rowwise_dry_run_plan.json"
EXPECTED_SUPPORT_BOTTOM_AREAS = 15


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-h5",
        type=Path,
        required=True,
        help="Compact Populace UK single-year H5 to clone.",
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
            f"Output H5 filename within --out. Defaults to {DATASET_FILENAME_TEMPLATE}."
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
            "Derive source_household_id as household_id mod this value before "
            "cloning, recovering pool-grain lineage (the certified UK pool "
            "encodes its 10x clone tiers with a 10**8 id offset). Refused when "
            "the input already carries source_household_id or the modulus "
            "would be an identity mapping."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Compute and write the clone plan (row/byte math, weight-kind "
            "chain, exact expected per-area support of the sampler) as "
            f"{DRY_RUN_PLAN_FILENAME} without cloning or writing a dataset. "
            "When no --crosswalk is supplied, the freshly built crosswalk "
            "cache is still written to --out."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    input_h5 = args.input_h5.resolve()
    base_summary = _h5_summary(input_h5)
    source_year = _source_year(args.source_year, base_summary=base_summary)
    output_h5 = _dataset_output_path(
        args.out,
        dataset_filename=args.dataset_filename,
        source_year=source_year,
    )
    _validate_output_paths(input_h5=input_h5, output_h5=output_h5, args=args)
    crosswalk_source = _load_or_build_crosswalk(args)
    crosswalk = crosswalk_source.frame
    crosswalk_path = crosswalk_source.path
    area_codes_by_type = _area_codes_by_type(args)
    coverage = _validate_optional_coverage(crosswalk, area_codes_by_type)

    if args.dry_run:
        plan = _dry_run_plan(
            args,
            input_h5=input_h5,
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
            "dataset": _artifact_info(input_h5),
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
    source_year: int,
) -> Path:
    filename = dataset_filename or DATASET_FILENAME_TEMPLATE.format(
        source_year=source_year
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
        if path != generated_crosswalk_path.resolve():
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
    return {
        "n_clones": args.n_clones,
        "seed": args.seed,
        "source_year": source_year,
        "require_all_countries": not args.allow_missing_country,
        "require_constituency": not args.allow_blank_constituency,
        "constrain_to_region": not args.allow_cross_region_assignment,
        "avoid_constituency_collisions": not args.allow_constituency_collisions,
        "source_lineage_modulus": args.source_lineage_modulus,
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
        }


def _dry_run_plan(
    args: argparse.Namespace,
    *,
    input_h5: Path,
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
    id_multiplier = id_multiplier_for_values(
        household["household_id"],
        person_ids["person_id"],
        person_ids["person_household_id"],
        person_ids["person_benunit_id"],
        benunit_ids["benunit_id"],
    )
    lineage = None
    if args.source_lineage_modulus is not None:
        with_lineage = apply_uk_source_lineage_modulus(
            household,
            modulus=args.source_lineage_modulus,
        )
        lineage = _lineage_summary(
            with_lineage,
            modulus=args.source_lineage_modulus,
        )
    support = expected_uk_rowwise_area_support(
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
            "dataset": _artifact_info(input_h5),
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
                "linear scaling of the input H5 byte size by n_clones; the "
                "added geography and lineage columns contribute marginal "
                "extra width"
            ),
        },
        "expected_support": {
            area_type: _expected_support_summary(support, area_type)
            for area_type in ("constituency", "la")
        },
        "source_lineage": lineage,
        "coverage": coverage.to_dict("records") if not coverage.empty else [],
    }


def _select_h5_columns(
    store: pd.HDFStore,
    key: str,
    columns: list[str],
) -> pd.DataFrame:
    try:
        return store.select(key, columns=columns)
    except (TypeError, ValueError, KeyError):
        return store[key][columns]


def _expected_support_summary(
    support: pd.DataFrame,
    area_type: str,
    *,
    bottom: int = EXPECTED_SUPPORT_BOTTOM_AREAS,
) -> dict[str, Any]:
    subset = support[support["area_type"] == area_type]
    if subset.empty:
        return {
            "n_areas": 0,
            "min_expected_rows": 0.0,
            "median_expected_rows": 0.0,
            "mean_expected_rows": 0.0,
            "max_expected_rows": 0.0,
            "bottom": [],
        }
    values = subset["expected_rows"]
    ordered = subset.sort_values(
        ["expected_rows", "area_code"],
        kind="mergesort",
    ).head(bottom)
    return {
        "n_areas": int(len(subset)),
        "min_expected_rows": float(values.min()),
        "median_expected_rows": float(values.median()),
        "mean_expected_rows": float(values.mean()),
        "max_expected_rows": float(values.max()),
        "bottom": [
            {
                "area_code": str(row.area_code),
                "expected_rows": float(row.expected_rows),
            }
            for row in ordered.itertuples(index=False)
        ],
    }


def _lineage_summary(
    household: pd.DataFrame,
    *,
    modulus: int,
) -> dict[str, Any]:
    counts = household.groupby("source_household_id").size()
    return {
        "modulus": modulus,
        "distinct_source_households": int(counts.size),
        "pool_copies_per_source": {
            "min": int(counts.min()),
            "median": float(counts.median()),
            "max": int(counts.max()),
        },
    }


def _rowwise_summary(
    result,
    *,
    base_summary: dict[str, Any],
    source_lineage_modulus: int | None = None,
) -> dict[str, Any]:
    household = result.household
    geo_columns = (
        "oa_code",
        "lsoa_code",
        "msoa_code",
        "la_code_oa",
        "constituency_code_oa",
        "region_code_oa",
    )
    missing_geography = household[list(geo_columns)].isna().any(axis=1)
    for column in geo_columns:
        missing_geography |= household[column].fillna("").astype(str).str.strip().eq("")
    assigned_constituencies = household.loc[
        _nonblank_string_mask(household["constituency_code_oa"]),
        "constituency_code_oa",
    ]
    assigned_las = household.loc[
        _nonblank_string_mask(household["la_code_oa"]),
        "la_code_oa",
    ]
    by_constituency = assigned_constituencies.groupby(assigned_constituencies).size()
    by_la = assigned_las.groupby(assigned_las).size()
    weight_sum = float(household["household_weight"].sum())
    constituency_rows = _area_row_summary(by_constituency)
    la_rows = _area_row_summary(by_la)
    input_total = float(base_summary["household_weight_sum"])
    lineage = None
    if source_lineage_modulus is not None:
        clone0 = household
        if "clone_index" in household.columns:
            clone0 = household[household["clone_index"] == 0]
        lineage = _lineage_summary(clone0, modulus=source_lineage_modulus)
    return {
        "weights": {
            "household_weight_kind": result.household_weight_kind.value,
            "mass_log_records": len(result.mass_log),
            "mass_conservation": {
                "input_total": input_total,
                "output_total": weight_sum,
                "abs_delta": abs(weight_sum - input_total),
                "relative_tolerance": MASS_CONSERVATION_RELATIVE_TOLERANCE,
                "passed": True,
            },
        },
        "source_lineage": lineage,
        "tables": {
            "person": list(result.person.shape),
            "benunit": list(result.benunit.shape),
            "household": list(household.shape),
        },
        "time_period": result.time_period,
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
            _duplicate_source_household_constituency_pairs(household)
        ),
    }


def _nonblank_string_mask(values: pd.Series) -> pd.Series:
    return values.notna() & values.astype(str).str.strip().ne("")


def _area_row_summary(counts: pd.Series) -> dict[str, int | float]:
    if counts.empty:
        return {"min": 0, "median": 0.0}
    return {"min": int(counts.min()), "median": float(counts.median())}


def _duplicate_source_household_constituency_pairs(household: pd.DataFrame) -> int:
    if "source_household_id" not in household.columns:
        return 0
    assigned = household[_nonblank_string_mask(household["constituency_code_oa"])]
    return int(
        assigned.duplicated(["source_household_id", "constituency_code_oa"]).sum()
    )


def _artifact_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


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
