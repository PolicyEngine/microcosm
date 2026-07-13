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
    build_official_uk_geography_crosswalk,
    clone_uk_dataset_with_rowwise_geography,
    geography_coverage_summary,
    validate_geography_coverage,
    write_geography_crosswalk,
)

CROSSWALK_FILENAME = "uk_official_geography_crosswalk.csv.gz"
DATASET_FILENAME_TEMPLATE = "populace_uk_{source_year}_rowwise.h5"
MANIFEST_FILENAME = "rowwise_build_manifest.json"
COVERAGE_FILENAME = "geography_coverage_summary.csv"


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
    )
    rowwise_summary = _rowwise_summary(result, base_summary=base_summary)
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
        "parameters": {
            "n_clones": args.n_clones,
            "seed": args.seed,
            "source_year": source_year,
            "require_all_countries": not args.allow_missing_country,
            "require_constituency": not args.allow_blank_constituency,
            "constrain_to_region": not args.allow_cross_region_assignment,
            "avoid_constituency_collisions": not args.allow_constituency_collisions,
        },
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
    reserved = {CROSSWALK_FILENAME, MANIFEST_FILENAME, COVERAGE_FILENAME}
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


def _h5_summary(path: Path) -> dict[str, Any]:
    with pd.HDFStore(path, mode="r") as store:
        household = store["household"]
        return {
            "path": str(path),
            "tables": {key.strip("/"): list(store[key].shape) for key in store.keys()},
            "household_weight_sum": float(household["household_weight"].sum()),
            "time_period": str(store["time_period"].iloc[0]),
        }


def _rowwise_summary(result, *, base_summary: dict[str, Any]) -> dict[str, Any]:
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
    return {
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
