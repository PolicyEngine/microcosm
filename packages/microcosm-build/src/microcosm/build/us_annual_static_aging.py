"""Build local, pinned annual US H5 candidates without publishing a release.

Each projection starts from the same base frame. Only one projected year is
held in memory at a time. The original H5 is copied byte for byte, and all later
files retain its entity/column/row layout with an explicit ``_time_period``.
Run ``python -m microcosm.build.us_annual_static_aging --help``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd

from microcosm.calibrate import (
    SeriesProjection,
    ssa_population_projection,
    static_aging,
)
from microcosm.frame import US_SCHEMA, Frame, SignedScale, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import multi_year_dataset, uprating_series
from microcosm.frame.materialize import (
    engine_tables,
    materialize_nullable_booleans_for_pytables,
    put_frame_table,
    read_frame_table,
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _check_pin(path: Path, expected: str) -> None:
    if len(expected) != 64 or _sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch for {path}")


def _model_provenance(*, source: Path, commit: str, version: str) -> dict:
    import policyengine_us

    source = source.resolve()
    if (
        not Path(policyengine_us.__file__)
        .resolve()
        .is_relative_to(source / "policyengine_us")
    ):
        raise ValueError("Imported policyengine_us does not come from model_source")
    actual_version = importlib.metadata.version("policyengine-us")
    if actual_version != version:
        raise ValueError(
            f"Model version mismatch: expected {version}, got {actual_version}"
        )

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True
        ).strip()

    head = git("rev-parse", "HEAD")
    if len(commit) != 40 or head != commit:
        raise ValueError(f"Model commit mismatch: expected {commit}, got {head}")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Model source must be a clean, committed checkout")
    files = {
        name: _sha256(source / name)
        for name in git(
            "ls-files", "--", "policyengine_us", "pyproject.toml", "uv.lock"
        ).splitlines()
        if (source / name).is_file()
    }
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "version": actual_version,
        "commit": head,
        "source": str(source),
        "source_tree_sha256": digest,
        "source_files": files,
    }


def _runtime_provenance() -> dict:
    # Hash the installed implementation, including solver/Frame helpers, rather
    # than inferring a source identity from an editable package version.
    import microcosm.calibrate
    import microcosm.frame

    sources = {}
    for package in (microcosm.calibrate, microcosm.frame):
        root = Path(package.__file__).resolve().parent
        sources.update(
            {
                f"{package.__name__}/{p.relative_to(root)}": _sha256(p)
                for p in sorted(root.rglob("*.py"))
            }
        )
    sources["annual_static_aging.py"] = _sha256(Path(__file__))
    return {
        "source_sha256": sources,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "policyengine-us",
                "policyengine-core",
                "spm-calculator",
                "microdf-python",
                "numpy",
                "pandas",
                "scipy",
                "torch",
                "tables",
                "microcosm-frame",
                "microcosm-calibrate",
                "microcosm-build",
            )
        },
    }


def _period(store: pd.HDFStore, year: int) -> None:
    if "/_time_period" not in store.keys():
        raise ValueError("Annual H5 must have explicit _time_period metadata")
    if not store.get_storer("_time_period").is_table:
        raise ValueError("Annual H5 requires table-format _time_period metadata")
    values = store["_time_period"]
    if (
        len(values) != 1
        or not isinstance(values.iloc[0], Integral)
        or values.iloc[0] != year
    ):
        raise ValueError(f"Annual H5 _time_period must be exactly {year}")


def _verify(path: Path, tables: Mapping[str, pd.DataFrame], year: int) -> dict:
    from policyengine_us.data import USSingleYearDataset

    native = USSingleYearDataset(file_path=str(path))
    if str(native.time_period) != str(year):
        raise ValueError(f"Native loader did not retain year {year}")
    with pd.HDFStore(path, "r") as store:
        _period(store, year)
        for entity, expected in tables.items():
            if not store.get_storer(entity).is_table:
                raise ValueError(
                    f"Annual native layout requires table-format storage for {entity}; "
                    "fixed-format nullable inputs need a separately certified contract"
                )
            if set(expected.columns) - set(store.get_storer(entity).table.colnames):
                raise ValueError(
                    f"Annual native layout requires direct HDF fields for {entity}; "
                    "packed value blocks need a separately certified contract"
                )
            actual = read_frame_table(store, entity)
            pd.testing.assert_frame_equal(
                actual, expected, check_dtype=False, check_exact=True
            )
            external = materialize_nullable_booleans_for_pytables(expected).table
            pd.testing.assert_frame_equal(
                getattr(native, entity), external, check_dtype=False, check_exact=True
            )
    return {"logical_tables": True, "native_loader": True, "time_period": year}


def _write_year(path: Path, tables: Mapping[str, pd.DataFrame], year: int) -> dict:
    temporary = path.with_suffix(".tmp.h5")
    with pd.HDFStore(temporary, "w") as store:
        for entity, table in tables.items():
            put_frame_table(
                store, entity, table, preferred_format="table", data_columns=True
            )
        store.put("_time_period", pd.Series([year]), format="table")
    receipt = _verify(temporary, tables, year)
    temporary.replace(path)
    return receipt


def build_annual_static_aging(
    *,
    base_h5: str | Path,
    base_sha256: str,
    parent_release: str,
    ssa_csv: str | Path,
    ssa_sha256: str,
    model_source: str | Path,
    model_commit: str,
    model_version: str,
    output_dir: str | Path,
    base_year: int = 2024,
    end_year: int = 2035,
    epochs: int = 300,
    seed: int = 0,
    max_weight_ratio: float = 5.0,
) -> dict:
    """Build a fresh local candidate, requiring explicit source and input pins.

    The caller attests ``parent_release``; certification must independently
    verify that release's base SHA. A complete candidate is not a certified
    release. Existing output directories are never reused, including failed
    attempts. A failure preserves its partial artifacts and status receipt.
    """
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")
    if (
        any(
            not isinstance(y, Integral) or isinstance(y, bool)
            for y in (base_year, end_year)
        )
        or end_year <= base_year
    ):
        raise ValueError("end_year must be an integer after integer base_year")
    if not parent_release.strip():
        raise ValueError("parent_release must identify the pinned base release")
    # Hugging Face snapshot paths are .h5 symlinks to extensionless blobs.
    # Keep the logical suffix required by the native loader; hashing and reads
    # still follow the link to authenticate the actual bytes.
    base = Path(base_h5).expanduser().absolute()
    ssa = Path(ssa_csv).expanduser().absolute()
    _check_pin(base, base_sha256)
    _check_pin(ssa, ssa_sha256)
    model_kwargs = {
        "source": Path(model_source),
        "commit": model_commit,
        "version": model_version,
    }
    model = _model_provenance(**model_kwargs)
    runtime = _runtime_provenance()
    with pd.HDFStore(base, "r") as store:
        _period(store, base_year)
        tables = {
            entity: read_frame_table(store, entity) for entity in US_SCHEMA.entities
        }
    columns = {entity: list(table.columns) for entity, table in tables.items()}
    weights = tables["household"].pop("household_weight").to_numpy(dtype=float)
    if any(
        f"{entity}_weight" in table
        for entity in US_SCHEMA.entities
        for table in tables.values()
    ):
        raise ValueError("Base must store only household weights")
    frame = Frame(
        tables, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)}
    )
    base_tables = {
        entity: table.loc[:, columns[entity]]
        for entity, table in engine_tables(
            frame, weighted_entities=("household",)
        ).items()
    }
    _verify(base, base_tables, base_year)
    del tables, base_tables
    demographics = ssa_population_projection(ssa, age_top=85, age_bands={80: 84})
    for year in range(base_year, end_year + 1):
        demographics.for_year(year)
    cells = list(demographics.cells)
    known_cells = pd.MultiIndex.from_frame(demographics.for_year(base_year)[cells])
    observed_cells = pd.MultiIndex.from_frame(frame.person[cells].drop_duplicates())
    if not observed_cells.isin(known_cells).all():
        raise ValueError("Base demographic cells do not match SSA age/sex coding")

    from policyengine_us import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()
    totals, indices, mapping = uprating_series(
        [column for entity in frame.entities for column in frame.table(entity)],
        tuple(range(base_year, end_year + 1)),
        system=system,
    )
    family = f"populace_us_{base_year}"
    manifest = {
        "schema_version": 1,
        "kind": "us_annual_static_aging_candidate",
        "status": "complete",
        "base": {
            "dataset": family,
            "year": base_year,
            "parent_release": parent_release,
            "path": str(base),
            "sha256": base_sha256,
        },
        "inputs": {"ssa": {"path": str(ssa), "sha256": ssa_sha256}},
        "model": model,
        "runtime": runtime,
        "settings": {
            "base_year": base_year,
            "end_year": end_year,
            "age_top": 85,
            "age_bands": {"80": 84},
            "epochs": epochs,
            "seed": seed,
            "max_weight_ratio": max_weight_ratio,
            "anchor": "frame",
            "learning_rate": 0.02,
            "l2_lambda": 0.0,
        },
        "metadata": {
            "dataset_years": {
                family: {
                    str(year): f"populace_us_{year}"
                    for year in range(base_year, end_year + 1)
                }
            }
        },
        "artifacts": {},
    }
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "build_status.json", {"status": "running", "completed_years": []})
    try:
        for year in range(base_year, end_year + 1):
            name = f"populace_us_{year}"
            path = output / f"{name}.h5"
            if year == base_year:
                shutil.copyfile(base, path)
                _check_pin(path, base_sha256)
                round_trip = {
                    "logical_tables": True,
                    "native_loader": True,
                    "time_period": year,
                }
                projection_receipt = None
            else:
                result = static_aging(
                    frame,
                    base_year=base_year,
                    years=(year,),
                    demographics=demographics,
                    series=SeriesProjection(totals=totals, indices=indices),
                    column_series=mapping,
                    epochs=epochs,
                    seed=seed,
                    max_weight_ratio=max_weight_ratio,
                    anchor="frame",
                    learning_rate=0.02,
                    l2_lambda=0.0,
                )
                projection = result.year(year)
                dataset = multi_year_dataset(
                    frame,
                    base_year,
                    {year: (projection.weights.values, projection.factors)},
                ).datasets[year]
                projected_tables = {
                    entity: getattr(dataset, entity).loc[:, columns[entity]]
                    for entity in frame.entities
                }
                round_trip = _write_year(path, projected_tables, year)
                receipt_path = output / f"projection_{year}.json"
                _json(
                    receipt_path,
                    {
                        "year": year,
                        "base_year": base_year,
                        "column_series": mapping,
                        "totals": totals,
                        "indices": indices,
                        "factors": {
                            column: {
                                "positive": factor.positive,
                                "negative": factor.negative,
                            }
                            if isinstance(factor, SignedScale)
                            else float(factor)
                            for column, factor in projection.factors.items()
                        },
                        "demographic_fit": projection.demographic_fit.to_dict(
                            orient="records"
                        ),
                        "fraction_within_10pct": float(
                            projection.fraction_within_10pct
                        ),
                    },
                )
                projection_receipt = {
                    "path": receipt_path.name,
                    "sha256": _sha256(receipt_path),
                }
                del projected_tables, dataset, projection, result
            artifact = {
                "path": path.name,
                "sha256": _sha256(path),
                "year": year,
                "rows": {entity: frame.n(entity) for entity in frame.entities},
                "columns": columns,
                "round_trip": round_trip,
            }
            if projection_receipt:
                artifact["projection_receipt"] = projection_receipt
            manifest["artifacts"][name] = artifact
            _json(
                output / "build_status.json",
                {
                    "status": "running",
                    "completed_years": list(range(base_year, year + 1)),
                },
            )
        _check_pin(base, base_sha256)
        _check_pin(ssa, ssa_sha256)
        if (
            _model_provenance(**model_kwargs) != model
            or _runtime_provenance() != runtime
        ):
            raise ValueError("Implementation or model source changed during the build")
        _json(
            output / "build_status.json",
            {
                "status": "complete",
                "completed_years": list(range(base_year, end_year + 1)),
            },
        )
        # The manifest is the completion marker and is written last.
        _json(output / "annual_manifest.json", manifest)
    except BaseException as exc:
        _json(
            output / "build_status.json",
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "completed_years": [
                    item["year"] for item in manifest["artifacts"].values()
                ],
            },
        )
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in (
        "base-h5",
        "base-sha256",
        "parent-release",
        "ssa-csv",
        "ssa-sha256",
        "model-source",
        "model-commit",
        "model-version",
        "output-dir",
    ):
        parser.add_argument(f"--{argument}", required=True)
    parser.add_argument("--base-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2035)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-weight-ratio", type=float, default=5.0)
    build_annual_static_aging(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
