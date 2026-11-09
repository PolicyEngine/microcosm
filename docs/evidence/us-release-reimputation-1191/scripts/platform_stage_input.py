"""Export, then replay, the retirement-contribution stage input on any platform.

``export`` loads the base H5 with the release builder's ``_load_frame`` and
builds the person table exactly as ``with_us_retirement_contribution_inputs``
hands it to ``run_source_stage`` (person columns, the resolved person weight,
and the eight prefixed QRF predictors). It keeps only the columns the stage's
operations read, adds the base and release stored outputs for comparison, and
pickles the table.

``replay`` loads that pickle, runs the packaged stage manifest through
``run_source_stage`` with the production handlers (seed 0, period 2024), and
compares the five outputs with the stored base and release values. Run it on
the platform under test; nothing in it reads the H5 files. ``--weight-factor``
multiplies every person weight (the release's mass repair), ``--threads``
limits BLAS/OpenMP pools, forest fit jobs and draw workers, and
``--compare-outputs`` compares the output values exactly with an earlier run.

``compare`` compares two saved output files (``--outputs``) from different
platforms, row by row. ``verify-export`` runs the stage on the full person
table and checks that it gives exactly the output values a replay of the
exported column-reduced table saved.

Usage:
    python platform_stage_input.py export --builder-root ROOT --base-h5 BASE \
        --release-h5 RELEASE --out INPUT.pkl
    python platform_stage_input.py replay --input INPUT.pkl --out RECEIPT.json \
        [--outputs OUTPUTS.npz] [--weight-factor F] [--threads N] \
        [--compare-outputs EARLIER.npz]
    python platform_stage_input.py compare A.npz B.npz --out RECEIPT.json
    python platform_stage_input.py verify-export --builder-root ROOT \
        --base-h5 BASE --outputs OUTPUTS.npz --out RECEIPT.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUTS = (
    "traditional_401k_contributions_desired",
    "roth_401k_contributions_desired",
    "traditional_ira_contributions_desired",
    "roth_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
)
SOURCE = ("RETCB_VAL", "WSAL_VAL", "SEMP_VAL")
PROVENANCE = ("person_id", "person_support_channel", "person_support_clone_index")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def full_stage_person(builder_root: Path, base_h5: Path):
    """The person table as with_us_retirement_contribution_inputs builds it."""

    sys.path.insert(0, str(builder_root / "tools"))
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release",
        builder_root / "tools" / "build_us_fiscal_refresh_release.py",
    )
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    from microcosm.build.us_runtime import retirement_contributions as stage

    base_sha = sha256(base_h5)
    frame = builder._load_frame(base_h5, expected_sha256=base_sha)
    person = frame.table("person")
    # Mirror with_us_retirement_contribution_inputs up to run_source_stage.
    stage_person = person.copy(deep=True)
    stage_person[stage._PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    predictors = stage._person_retirement_predictors(frame)
    predictor_columns = []
    for column in stage._PUF_PREDICTORS:
        name = stage._PUF_PREDICTOR_PREFIX + column
        stage_person[name] = predictors[column].to_numpy()
        predictor_columns.append(name)
    return stage, stage_person, predictor_columns, base_sha


def run_stage(stage, table: pd.DataFrame, seed: int, period: int) -> pd.DataFrame:
    from microcosm.build.source_runtime import SourceRuntimeConfig, run_source_stage

    return run_source_stage(
        stage.us_retirement_contributions_stage_spec(),
        tables={"person": table},
        operation_handlers={
            "derive_retirement_contributions": (
                stage.derive_us_retirement_contributions_from_manifest
            ),
            "impute_retirement_contributions_to_puf_support": (
                stage.impute_us_retirement_contributions_to_puf_support_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=seed, target_year=period),
    )


def verify_export(args: argparse.Namespace) -> None:
    """Run the stage on the full person table and compare its output values
    exactly with those a replay of the exported (column-reduced) table saved."""

    stage, stage_person, _, base_sha = full_stage_person(
        args.builder_root, args.base_h5
    )
    output = run_stage(stage, stage_person, args.seed, args.period)
    aligned = output.set_index("person_id").reindex(
        stage_person["person_id"].to_numpy()
    )
    saved = np.load(args.outputs)
    assert np.array_equal(saved["person_id"], stage_person["person_id"].to_numpy())
    receipt = {
        "base_h5": Path(args.base_h5).name,
        "base_h5_sha256": base_sha,
        "full_table_columns": int(stage_person.shape[1]),
        "exported_table_outputs": Path(args.outputs).name,
        "exported_table_outputs_sha256": sha256(args.outputs),
        "rows": int(len(stage_person)),
        "rows_differing_full_table_vs_exported_table": {
            column: int(
                (aligned[column].to_numpy(dtype=np.float64) != saved[column]).sum()
            )
            for column in OUTPUTS
        },
    }
    Path(args.out).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps(receipt["rows_differing_full_table_vs_exported_table"]))


def export(args: argparse.Namespace) -> None:
    stage, stage_person, predictor_columns, base_sha = full_stage_person(
        args.builder_root, args.base_h5
    )
    keep = [
        *PROVENANCE,
        *SOURCE,
        stage._PERSON_WEIGHT_COLUMN,
        *predictor_columns,
        *OUTPUTS,
    ]
    slim = stage_person.loc[:, keep].copy()
    release = (
        pd.read_hdf(args.release_h5, "person", columns=["person_id", *OUTPUTS])
        .set_index("person_id")
        .reindex(slim["person_id"].to_numpy())
    )
    for column in OUTPUTS:
        slim[f"stored_base__{column}"] = slim[column].to_numpy(dtype=np.float64)
        slim[f"stored_release__{column}"] = release[column].to_numpy(dtype=np.float64)
    slim.attrs = {
        "base_h5": Path(args.base_h5).name,
        "base_h5_sha256": base_sha,
        "release_h5": Path(args.release_h5).name,
        "release_h5_sha256": sha256(args.release_h5),
    }
    slim.to_pickle(args.out)
    print(f"wrote {args.out} rows={len(slim)} columns={len(slim.columns)}")


def replay(args: argparse.Namespace) -> None:
    from microcosm.build.source_runtime import SourceRuntimeConfig, run_source_stage
    from microcosm.build.us_runtime import retirement_contributions as stage

    slim = pd.read_pickle(args.input)
    if args.weight_factor != 1.0:
        # The release's mass repair multiplies every household weight by one
        # factor; person weights resolve from household weights.
        slim[stage._PERSON_WEIGHT_COLUMN] = (
            slim[stage._PERSON_WEIGHT_COLUMN].to_numpy(dtype=np.float64)
            * args.weight_factor
        )
    if args.threads is not None:
        os.environ["POPULACE_FIT_PREDICT_WORKERS"] = str(args.threads)
        os.environ["POPULACE_FIT_N_JOBS"] = str(args.threads)
    stored = {
        kind: {
            column: slim.pop(f"stored_{kind}__{column}").to_numpy()
            for column in OUTPUTS
        }
        for kind in ("base", "release")
    }
    labels = (
        slim["person_support_channel"].map(
            lambda v: v.decode() if isinstance(v, bytes) else str(v)
        )
        + "/clone"
        + slim["person_support_clone_index"].astype(int).astype(str)
    ).to_numpy()
    start = time.perf_counter()
    if args.threads is not None:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=args.threads)
    output = run_source_stage(
        stage.us_retirement_contributions_stage_spec(),
        tables={"person": slim},
        operation_handlers={
            "derive_retirement_contributions": (
                stage.derive_us_retirement_contributions_from_manifest
            ),
            "impute_retirement_contributions_to_puf_support": (
                stage.impute_us_retirement_contributions_to_puf_support_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=args.seed, target_year=args.period),
    )
    seconds = time.perf_counter() - start
    if args.threads is not None:
        limiter.restore_original_limits()
    aligned = output.set_index("person_id").reindex(slim["person_id"].to_numpy())
    receipt: dict[str, object] = {
        "input": Path(args.input).name,
        "input_sha256": sha256(Path(args.input)),
        "input_attrs": {
            key: Path(value).name if key.endswith("_h5") else value
            for key, value in slim.attrs.items()
        },
        "seed": args.seed,
        "period": args.period,
        "weight_factor": args.weight_factor,
        "threads": args.threads,
        "seconds": round(seconds, 1),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "quantile_forest": __import__("quantile_forest").__version__,
            "cpu_count": os.cpu_count(),
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "POPULACE_FIT_N_JOBS",
                    "POPULACE_FIT_PREDICT_WORKERS",
                    "PYTHONHASHSEED",
                )
            },
        },
        "columns": {},
    }
    replayed_columns = {}
    for column in OUTPUTS:
        replayed = aligned[column].to_numpy(dtype=np.float64)
        replayed_columns[column] = replayed
        entry = {}
        for kind in ("base", "release"):
            reference = stored[kind][column]
            changed = replayed != reference
            entry[f"vs_stored_{kind}"] = {
                "rows_differing": int(changed.sum()),
                "rows_differing_by_channel": {
                    str(k): int(v)
                    for k, v in pd.Series(changed).groupby(labels).sum().items()
                },
                "rows_differing_by_more_than_1e-9": int(
                    (np.abs(replayed - reference) > 1e-9).sum()
                ),
                "rows_differing_by_more_than_1_dollar": int(
                    (np.abs(replayed - reference) > 1.0).sum()
                ),
                "max_abs_difference": float(np.max(np.abs(replayed - reference))),
            }
        receipt["columns"][column] = entry
    if args.compare_outputs is not None:
        previous = np.load(args.compare_outputs)
        assert np.array_equal(previous["person_id"], slim["person_id"].to_numpy())
        receipt["compared_with"] = Path(args.compare_outputs).name
        receipt["compared_with_sha256"] = sha256(Path(args.compare_outputs))
        receipt["rows_differing_from_compared"] = {
            column: int((replayed_columns[column] != previous[column]).sum())
            for column in OUTPUTS
        }
    if args.outputs is not None:
        np.savez_compressed(
            args.outputs, person_id=slim["person_id"].to_numpy(), **replayed_columns
        )
        receipt["outputs"] = Path(args.outputs).name
    Path(args.out).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                c: (
                    v["vs_stored_base"]["rows_differing"],
                    v["vs_stored_release"]["rows_differing"],
                )
                for c, v in receipt["columns"].items()
            }
        )
    )


def compare(args: argparse.Namespace) -> None:
    left, right = np.load(args.left), np.load(args.right)
    assert np.array_equal(left["person_id"], right["person_id"])
    receipt: dict[str, object] = {
        "left": Path(args.left).name,
        "left_sha256": sha256(args.left),
        "right": Path(args.right).name,
        "right_sha256": sha256(args.right),
        "rows": int(len(left["person_id"])),
        "columns": {},
    }
    for column in OUTPUTS:
        a, b = left[column], right[column]
        difference = np.abs(a - b)
        receipt["columns"][column] = {
            "rows_differing": int((a != b).sum()),
            "rows_differing_by_more_than_1e-9": int((difference > 1e-9).sum()),
            "rows_differing_by_more_than_1_dollar": int((difference > 1.0).sum()),
            "max_abs_difference": float(difference.max()),
            "zero_nonzero_flips": int(((a == 0) != (b == 0)).sum()),
        }
    Path(args.out).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps(receipt["columns"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--builder-root", type=Path, required=True)
    ex.add_argument("--base-h5", type=Path, required=True)
    ex.add_argument("--release-h5", type=Path, required=True)
    ex.add_argument("--out", type=Path, required=True)
    rp = sub.add_parser("replay")
    rp.add_argument("--input", type=Path, required=True)
    rp.add_argument("--out", type=Path, required=True)
    rp.add_argument("--outputs", type=Path)
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--period", type=int, default=2024)
    rp.add_argument("--weight-factor", type=float, default=1.0)
    rp.add_argument("--threads", type=int)
    rp.add_argument("--compare-outputs", type=Path)
    ve = sub.add_parser("verify-export")
    ve.add_argument("--builder-root", type=Path, required=True)
    ve.add_argument("--base-h5", type=Path, required=True)
    ve.add_argument("--outputs", type=Path, required=True)
    ve.add_argument("--out", type=Path, required=True)
    ve.add_argument("--seed", type=int, default=0)
    ve.add_argument("--period", type=int, default=2024)
    cp = sub.add_parser("compare")
    cp.add_argument("left", type=Path)
    cp.add_argument("right", type=Path)
    cp.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {
        "export": export,
        "replay": replay,
        "compare": compare,
        "verify-export": verify_export,
    }[args.command](args)


if __name__ == "__main__":
    main()
