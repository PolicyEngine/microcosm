"""Export, then replay on any platform, the other three re-run stage inputs.

The companion of ``platform_stage_input.py`` (retirement contributions) for
weeks unemployed, childcare and energy subsidy. ``export`` loads the base H5
once with the release builder's ``_load_frame`` and, for each stage, builds
the person table as the stage's ``with_us_*`` function hands it to
``run_source_stage``. It then finds the columns the stage's operations read
(adding any column a slimmed run asks for), checks that the slimmed table
gives the same output as the full table bit for bit, and pickles the slimmed
table with the base and release stored outputs.

``replay`` runs one pickled stage input through the packaged stage manifest
with the production handlers and compares the output with both stored files.
Childcare and energy subsidy are compared at SPM-unit grain, after the
stage's own first-person reduction.

Usage:
    python platform_other_stages.py export --builder-root ROOT --base-h5 BASE \
        --release-h5 RELEASE --asec-2023-source PPPUB23 --out-dir DIR
    python platform_other_stages.py replay --stage NAME --input DIR/NAME.pkl \
        --out RECEIPT.json [--outputs OUTPUTS.npz]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROVENANCE = ("person_id", "person_support_channel", "person_support_clone_index")

# stage -> how its with_us_* function prepares and runs the source stage.
STAGES: dict[str, dict[str, object]] = {
    "weeks_unemployed": {
        "module": "weeks_unemployed",
        "predictors": "_person_weeks_unemployed_predictors",
        "prefix": "_PREDICTOR_PREFIX",
        "spec": "us_weeks_unemployed_stage_spec",
        "handlers": {
            "derive_weeks_unemployed": "derive_us_weeks_unemployed_from_manifest",
            "impute_weeks_unemployed_to_puf_support": (
                "impute_us_weeks_unemployed_to_puf_support_from_manifest"
            ),
        },
        "output": "weeks_unemployed",
        "stored_entity": "person",
        "group": None,
    },
    "childcare": {
        "module": "childcare",
        "predictors": "_person_childcare_predictors",
        "prefix": "_PUF_PREDICTOR_PREFIX",
        "spec": "us_childcare_stage_spec",
        "handlers": {
            "derive_childcare_inputs": "derive_us_childcare_from_manifest",
            "impute_childcare_to_puf_support": (
                "impute_us_childcare_to_puf_support_from_manifest"
            ),
        },
        "output": "spm_unit_pre_subsidy_childcare_expenses",
        "stored_entity": "spm_unit",
        "group": "person_spm_unit_id",
    },
    "energy_subsidy": {
        "module": "energy_subsidy",
        "predictors": "_person_energy_subsidy_predictors",
        "prefix": "_PUF_PREDICTOR_PREFIX",
        "spec": "us_energy_subsidy_stage_spec",
        "handlers": {
            "derive_energy_subsidy": "derive_us_energy_subsidy_from_manifest",
            "impute_energy_subsidy_to_puf_support": (
                "impute_us_energy_subsidy_to_puf_support_from_manifest"
            ),
        },
        "output": "spm_unit_energy_subsidy",
        "stored_entity": "spm_unit",
        "group": "person_spm_unit_id",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_module(name: str):
    return importlib.import_module(
        f"microcosm.build.us_runtime.{STAGES[name]['module']}"
    )


def run_stage(name: str, table: pd.DataFrame, seed: int, period: int) -> pd.DataFrame:
    from microcosm.build.source_runtime import SourceRuntimeConfig, run_source_stage

    module = stage_module(name)
    config = STAGES[name]
    return run_source_stage(
        getattr(module, config["spec"])(),
        tables={"person": table},
        operation_handlers={
            kind: getattr(module, handler)
            for kind, handler in config["handlers"].items()
        },
        config=SourceRuntimeConfig(seed=seed, target_year=period),
    )


def stage_values(name: str, output: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """The stage's stored value per stored row, indexed by that row's id."""

    column, group = STAGES[name]["output"], STAGES[name]["group"]
    aligned = output.set_index("person_id").reindex(table["person_id"].to_numpy())
    values = aligned[column].to_numpy(dtype=np.float64)
    if group is None:
        return pd.Series(values, index=table["person_id"].to_numpy())
    # with_us_childcare_inputs / with_us_energy_subsidy_input: first person.
    return (
        pd.DataFrame({"unit": table[group].to_numpy(), "value": values})
        .groupby("unit", sort=False)["value"]
        .first()
    )


def export(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.builder_root / "tools"))
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release",
        args.builder_root / "tools" / "build_us_fiscal_refresh_release.py",
    )
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)

    base_sha = sha256(args.base_h5)
    release_sha = sha256(args.release_h5)
    frame = builder._load_frame(args.base_h5, expected_sha256=base_sha)
    person = frame.table("person")
    weights = frame.resolve_weights("person").values
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, object] = {
        "base_h5": Path(args.base_h5).name,
        "base_h5_sha256": base_sha,
        "release_h5": Path(args.release_h5).name,
        "release_h5_sha256": release_sha,
        "stages": {},
    }
    for name in args.stage:
        config = STAGES[name]
        module = stage_module(name)
        stage_person = person.copy(deep=True)
        if name == "weeks_unemployed":
            # with_us_weeks_unemployed: fill missing 2022 LKWEEKS first, when
            # the frame holds source-year-2022 rows.
            source_year = pd.to_numeric(stage_person["source_year"], errors="coerce")
            if bool(
                source_year.eq(module.ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_YEAR).any()
            ):
                source = builder.load_asec_2023_weeks_unemployed_source(
                    args.asec_2023_source
                )
                stage_person = module.fill_asec_2022_weeks_unemployed_source(
                    stage_person, source
                )
        stage_person[module._PERSON_WEIGHT_COLUMN] = weights
        predictors = getattr(module, config["predictors"])(frame)
        prefix = getattr(module, config["prefix"])
        predictor_columns = []
        for column in predictors:
            stage_person[prefix + column] = predictors[column].to_numpy()
            predictor_columns.append(prefix + column)
        full = run_stage(name, stage_person, args.seed, args.period)
        full_values = stage_values(name, full, stage_person)

        keep = [*PROVENANCE, module._PERSON_WEIGHT_COLUMN, *predictor_columns]
        if config["group"] is not None:
            keep.append(config["group"])
        keep += [
            column
            for column in getattr(
                module,
                {
                    "weeks_unemployed": "US_WEEKS_UNEMPLOYED_REQUIRED_SOURCE_COLUMNS",
                    "childcare": "US_CHILDCARE_REQUIRED_SOURCE_COLUMNS",
                    "energy_subsidy": "US_ENERGY_SUBSIDY_REQUIRED_SOURCE_COLUMNS",
                }[name],
            )
            if column in stage_person and column not in keep
        ]
        if config["output"] in stage_person and config["output"] not in keep:
            keep.append(config["output"])
        added: list[str] = []
        for _ in range(40):
            try:
                slim_output = run_stage(
                    name, stage_person.loc[:, keep].copy(), args.seed, args.period
                )
                break
            except Exception as error:  # noqa: BLE001 - any missing-column error
                asked = [
                    token
                    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(error))
                    if token in stage_person.columns and token not in keep
                ]
                if not asked:
                    raise
                keep += asked
                added += asked
        else:
            raise RuntimeError(f"{name}: slim column search did not converge")
        slim = stage_person.loc[:, keep].copy()
        slim_values = stage_values(name, slim_output, slim)
        identical = bool(
            slim_values.index.equals(full_values.index)
            and np.array_equal(slim_values.to_numpy(), full_values.to_numpy())
        )
        if not identical:
            raise RuntimeError(
                f"{name}: the slimmed input does not reproduce the full-table "
                "output; a column the stage reads without failing is missing."
            )

        entity = config["stored_entity"]
        key = f"{entity}_id"
        ids = full_values.index.to_numpy()
        stored = {}
        for kind, path in (("base", args.base_h5), ("release", args.release_h5)):
            table = pd.read_hdf(path, entity, columns=[key, config["output"]])
            stored[kind] = (
                table.set_index(key)[config["output"]]
                .reindex(ids)
                .to_numpy(dtype=np.float64)
            )
        if entity == "person":
            labels_source = slim
        else:
            labels_source = pd.read_hdf(
                args.base_h5,
                entity,
                columns=[
                    key,
                    f"{entity}_support_channel",
                    f"{entity}_support_clone_index",
                ],
            ).set_index(key)
            labels_source = labels_source.reindex(ids)
        channel_column = f"{entity}_support_channel"
        clone_column = f"{entity}_support_clone_index"
        labels = (
            labels_source[channel_column].map(
                lambda v: v.decode() if isinstance(v, bytes) else str(v)
            )
            + "/clone"
            + labels_source[clone_column].astype(int).astype(str)
        ).to_numpy()
        payload = {
            "stage": name,
            "table": slim,
            "ids": ids,
            "labels": labels,
            "stored_base": stored["base"],
            "stored_release": stored["release"],
            "base_h5_sha256": base_sha,
            "release_h5_sha256": release_sha,
        }
        out = args.out_dir / f"{name}.pkl"
        pd.to_pickle(payload, out)
        receipt["stages"][name] = {
            "pickle": Path(out).name,
            "pickle_sha256": sha256(out),
            "rows": int(len(slim)),
            "columns": keep,
            "columns_added_on_request": added,
            "slim_reproduces_full_table_output": identical,
            "stored_rows": int(len(ids)),
            "full_table_run_vs_stored_base_rows_differing": int(
                (full_values.to_numpy() != stored["base"]).sum()
            ),
            "full_table_run_vs_stored_release_rows_differing": int(
                (full_values.to_numpy() != stored["release"]).sum()
            ),
        }
        print(name, json.dumps(receipt["stages"][name])[:400], flush=True)
        del stage_person, full, slim, slim_output
    (args.out_dir / "export_receipt.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n"
    )


def replay(args: argparse.Namespace) -> None:
    payload = pd.read_pickle(args.input)
    name = payload["stage"]
    assert name == args.stage, (name, args.stage)
    table = payload["table"]
    start = time.perf_counter()
    output = run_stage(name, table, args.seed, args.period)
    seconds = time.perf_counter() - start
    values = stage_values(name, output, table)
    assert np.array_equal(values.index.to_numpy(), payload["ids"])
    replayed = values.to_numpy()
    labels = payload["labels"]
    receipt: dict[str, object] = {
        "stage": name,
        "input": Path(args.input).name,
        "input_sha256": sha256(Path(args.input)),
        "base_h5_sha256": payload["base_h5_sha256"],
        "release_h5_sha256": payload["release_h5_sha256"],
        "seed": args.seed,
        "period": args.period,
        "seconds": round(seconds, 1),
        "rows": int(len(replayed)),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "quantile_forest": __import__("quantile_forest").__version__,
            "cpu_count": os.cpu_count(),
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED")
            },
        },
    }
    for kind in ("base", "release"):
        reference = payload[f"stored_{kind}"]
        changed = replayed != reference
        difference = np.abs(replayed - reference)
        receipt[f"vs_stored_{kind}"] = {
            "rows_differing": int(changed.sum()),
            "rows_differing_by_channel": {
                str(k): int(v)
                for k, v in pd.Series(changed).groupby(labels).sum().items()
            },
            "rows_differing_by_more_than_1e-9": int((difference > 1e-9).sum()),
            "max_abs_difference": float(difference.max()),
        }
    if args.outputs is not None:
        np.savez_compressed(args.outputs, ids=payload["ids"], values=replayed)
    Path(args.out).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(
        name,
        receipt["vs_stored_base"]["rows_differing"],
        receipt["vs_stored_release"]["rows_differing"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--builder-root", type=Path, required=True)
    ex.add_argument("--base-h5", type=Path, required=True)
    ex.add_argument("--release-h5", type=Path, required=True)
    ex.add_argument("--asec-2023-source", type=Path, required=True)
    ex.add_argument("--out-dir", type=Path, required=True)
    ex.add_argument("--stage", action="append", choices=sorted(STAGES))
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--period", type=int, default=2024)
    rp = sub.add_parser("replay")
    rp.add_argument("--stage", required=True, choices=sorted(STAGES))
    rp.add_argument("--input", type=Path, required=True)
    rp.add_argument("--out", type=Path, required=True)
    rp.add_argument("--outputs", type=Path)
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--period", type=int, default=2024)
    args = parser.parse_args()
    if args.command == "export" and not args.stage:
        args.stage = sorted(STAGES)
    {"export": export, "replay": replay}[args.command](args)


if __name__ == "__main__":
    main()
