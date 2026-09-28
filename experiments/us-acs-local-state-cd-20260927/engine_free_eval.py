"""Engine-free full-scale evaluation of the sparse ACS local path and state_cd.

Inputs (read-only): the 2026-09-23 ACS local release run
(``populace-us-2024-buildo-acs-local-767312d60``) -- its dense ``state``-mode
lean checkpoint (1,588,854 households x 4,459 float32 measures), its
calibrated weights, and its staging H5 -- plus the pinned Chronicle feed and
the PUMA ladder.

Stages (each writes under ``--out`` and can be re-run alone):

``convert``
    Stream the dense checkpoint into the sparse checkpoint format the tool
    now writes (structure H5 + CSR + targets.json).
``state``
    Re-calibrate the ``state`` surface from the sparse checkpoint with the
    09-23 settings through the tool's ``calibrate_surface`` and compare the
    weights with the ones the dense path produced on 09-23.
``state_cd``
    Add the ``state_cd`` district rows whose state parent is a Historic
    Table 2 row. Each such row is exactly its parent's CSR row restricted to
    the district's households, because the parent and the district row
    materialize identically apart from geography
    (``state_cd_soi_surface`` refuses otherwise). Rows whose parent is a
    district-file state total (charitable, interest paid, QBI deduction) have
    no engine column in the 09-23 checkpoint and are left out. Draw the CD
    holdout, calibrate, and score the held-out district targets under design,
    ``state``-calibrated and ``state_cd``-calibrated weights against the
    pro-rata baseline.

This is development evidence, not a release: the engine columns are the
09-23 engine pass, and the surface omits the three district-file-only
concept families.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[2]
RUN = Path(
    "/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/overnight-20260923/run"
)
DENSE_CHECKPOINT = RUN / "release" / "checkpoints"
STAGING_H5 = RUN / "staging" / "acs_multispine_staging.h5"
FEED = Path(
    "/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_c5e5bf8.jsonl"
)
LADDER = Path(
    "/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/build/us/us_puma_ladder_2020.npz"
)
SETTINGS = dict(
    epochs=800,
    epoch_batch=400,
    max_weight_ratio=5.0,
    target_loss_cap=1.0,
    l2_lambda=0.0,
    seed=0,
)


def load_tool():
    path = REPO / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location("build_us_acs_local_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dense_to_csr(path: Path, n_rows_block: int = 25_000):
    """Stream the dense float32 household block into a CSR (targets x hh)."""

    import h5py

    with h5py.File(path, "r") as handle:
        group = handle["household"]
        names = [item.decode() for item in group["block2_items"][:]]
        values = group["block2_values"]
        n_households, n_targets = values.shape
        rows, cols, data = [], [], []
        for low in range(0, n_households, n_rows_block):
            block = values[low : low + n_rows_block]
            household, target = np.nonzero(block)
            rows.append(target.astype(np.int32))
            cols.append((household + low).astype(np.int32))
            data.append(block[household, target].astype(np.float32))
        struct = pd.DataFrame(
            group["block0_values"][:],
            columns=[item.decode() for item in group["block0_items"][:]],
        )
        weights = group["block3_values"][:, 0].astype(np.float64)
    matrix = sparse.csr_array(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_targets, n_households),
    )
    matrix.sort_indices()
    return names, matrix, struct, weights


def staging_origin(household_ids: np.ndarray) -> pd.DataFrame:
    with pd.HDFStore(STAGING_H5, mode="r") as store:
        households = (
            store.select(
                "household",
                columns=["household_id", "household_spine", "household_source_id"],
            )
            if store.get_storer("household").is_table
            else store["household"][
                ["household_id", "household_spine", "household_source_id"]
            ]
        )
    if not np.array_equal(households["household_id"].to_numpy(), household_ids):
        raise SystemExit("staging household ids differ from the lean checkpoint's")
    return households


def roles_and_specs_for(tool, names, values, specs_by_name):
    """The 09-23 targets as registry specs (the recompiled feed's) and roles."""

    roles, specs, pop_names, pop_values = [], [], [], []
    for name, value in zip(names, values, strict=True):
        spec = specs_by_name.get(name)
        if spec is not None:
            if abs(spec.value - value) > 1e-9 * max(1.0, abs(value)):
                raise SystemExit(f"{name}: 09-23 value {value} != compile {spec.value}")
            roles.append(tool.cd_surface.target_record(spec))
            specs.append(spec)
        elif name.startswith(("pop_state_", "pop_cd_")):
            pop_names.append(name)
            pop_values.append(float(value))
            is_cd = name.startswith("pop_cd_")
            roles.append(
                {
                    "name": name,
                    "value": float(value),
                    "source": "us_puma_ladder_2020",
                    "family": "census_population_ladder",
                    "geography_level": "congressional_district" if is_cd else "state",
                    "state_fips": name[-4:-2] if is_cd else name[-2:],
                    "congressional_district_geoid": name[-4:] if is_cd else None,
                }
            )
            specs.append(None)
        else:
            raise SystemExit(f"09-23 target {name} is not on the recompiled surface")
    pop_specs = iter(tool.population_target_specs(pop_names, pop_values))
    specs = [spec if spec is not None else next(pop_specs) for spec in specs]
    return roles, specs


def stage_convert(tool, out: Path) -> None:
    started = time.time()
    dense_targets = json.loads((DENSE_CHECKPOINT / "targets.json").read_text())
    names, matrix, struct, weights = dense_to_csr(
        DENSE_CHECKPOINT / "target_frame_lean.h5"
    )
    if names != [target["measure"] for target in dense_targets]:
        raise SystemExit("dense block order differs from targets.json")
    print(
        f"dense -> CSR: {matrix.shape} nnz {matrix.nnz:,} ({time.time() - started:.0f}s)"
    )
    origin = staging_origin(struct["household_id"].to_numpy())
    struct = struct.assign(
        household_spine=origin["household_spine"].astype(str).to_numpy(),
        household_source_id=origin["household_source_id"].to_numpy(),
    )
    with pd.HDFStore(DENSE_CHECKPOINT / "target_frame_lean.h5", mode="r") as store:
        person = store["person"]
        groups = {group: store[group] for group in tool.GROUP_IDS}
    surface = tool.state_admin_surface(
        FEED, ["snap", "medicaid", "soi"], soi_mode="state_cd"
    )
    specs_by_name = {spec.name: spec for spec in surface.registry.specs}
    roles, specs = roles_and_specs_for(
        tool,
        [target["name"] for target in dense_targets],
        [target["value"] for target in dense_targets],
        specs_by_name,
    )
    tool.cd_surface.assign_target_roles(roles, fraction=0.0)
    struct_tables = {
        "household_struct": struct,
        "person": person,
        "groups": groups,
        "weights": weights,
    }
    _path, _registry, digests = tool.write_lean_checkpoint(
        struct_tables, matrix, specs, roles, out / "state"
    )
    import pickle

    with open(out / "state_cd_surface.pkl", "wb") as handle:
        pickle.dump(
            {"specs": list(surface.registry.specs), "receipt": surface.soi_receipt},
            handle,
        )
    (out / "state" / "identity.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(DENSE_CHECKPOINT),
                "dense_matrix_nnz": int(matrix.nnz),
                "published_matrix_nnz": 24_773_532,
                **digests,
                "households": int(len(struct)),
            },
            indent=2,
        )
    )
    print(f"convert done ({time.time() - started:.0f}s, peak {tool.rss():.1f} GB)")


def calibrate_and_save(tool, checkpoint: Path, label: str):
    """Calibrate a checkpoint through the tool's own functions and record it."""

    from microcosm.calibrate import write_calibration_diagnostics

    frame, design, registry, roles, matrix = tool.load_checkpoint_surface(checkpoint)
    target_set = tool.cd_surface.calibration_target_set(
        roles, matrix, frame.n("household"), specs=registry.specs
    )
    started = time.time()
    result, _done = tool.calibrate_surface(frame, target_set, **SETTINGS)
    weights = np.asarray(result.weights, dtype=np.float64)
    summary = {
        "soi_mode": label,
        **SETTINGS,
        "n_targets": result.problem.n_targets,
        "matrix_format": result.options["matrix_format"],
        "matrix_nnz": int(result.problem.matrix.nnz),
        "initial_loss": round(result.initial_loss, 6),
        "final_loss": round(result.final_loss, 6),
        "fraction_within_10pct": round(result.fraction_within_10pct, 4),
        "effective_sample_size": round(result.effective_sample_size, 1),
        "realized_max_weight_ratio": round(result.realized_max_weight_ratio, 4),
        "mass_conserved_ratio": round(float(weights.sum()) / float(design.sum()), 6),
        **tool.calibration_evidence(
            frame=frame,
            roles=roles,
            matrix=matrix,
            design_weights=design,
            weights=weights,
            target_loss_cap=SETTINGS["target_loss_cap"],
        ),
        "total_wall_seconds": round(time.time() - started, 1),
        "peak_rss_gb": round(tool.rss(), 3),
    }
    outcome = write_calibration_diagnostics(
        result,
        checkpoint / "calibration_diagnostics.json",
        target_registry=registry,
        build={"soi_mode": label, "experiment": "engine_free_eval"},
    )
    summary["calibration_diagnostics"] = {
        "status": outcome.status,
        "schema_version": getattr(outcome, "schema_version", None),
        "message": getattr(outcome, "message", None),
    }
    np.savez(checkpoint / "weights.npz", weights=weights, design=design)
    (checkpoint / "calibration_summary.json").write_text(json.dumps(summary, indent=1))
    return summary, weights, design, roles, matrix, frame


def stage_state(tool, out: Path) -> None:
    diagnostics, weights, _design, _records, _matrix, _frame = calibrate_and_save(
        tool, out / "state", "state"
    )
    dense = np.load(DENSE_CHECKPOINT / "weights_latest.npz")["weights"]
    difference = np.abs(weights - dense) / np.maximum(np.abs(dense), 1e-12)
    comparison = {
        "identical": bool(np.array_equal(weights, dense)),
        "max_abs_diff": float(np.abs(weights - dense).max()),
        "max_rel_diff": float(difference.max()),
        "median_rel_diff": float(np.median(difference)),
        "dense_final_loss_0923": 0.015491,
        "sparse_final_loss": diagnostics["final_loss"],
        "dense_ess_0923": 13631.3,
        "sparse_ess": diagnostics["effective_sample_size"],
        "torch_threads": int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
    }
    (out / "state" / "dense_comparison.json").write_text(
        json.dumps(comparison, indent=2)
    )
    print(json.dumps(comparison, indent=2))


def stage_state_cd(
    tool, out: Path, fraction: float, state_weights_path: Path | None = None
) -> None:
    import pickle

    cd = tool.cd_surface
    frame, design, state_registry, state_records, state_matrix = (
        tool.load_checkpoint_surface(out / "state")
    )
    households = frame.table("household")
    hh_state = pd.to_numeric(households["state_fips"]).to_numpy(np.int64)
    hh_cd = pd.to_numeric(households["congressional_district_geoid"]).to_numpy(np.int64)
    with open(out / "state_cd_surface.pkl", "rb") as handle:
        surface = pickle.load(handle)
    row_of = {record["name"]: index for index, record in enumerate(state_records)}
    district_specs, skipped = [], []
    for spec in surface["specs"]:
        if spec.metadata.get("ledger_geography_level") != "congressional_district":
            continue
        if spec.metadata.get("state_cd_parent_basis") != "historic_table_2":
            skipped.append(spec.name)
            continue
        district_specs.append(spec)
    rows, cols, data = [], [], []
    records = list(state_records)
    for offset, spec in enumerate(district_specs):
        parent_row = row_of[spec.metadata["state_cd_parent_target_name"]]
        start, stop = (
            state_matrix.indptr[parent_row],
            state_matrix.indptr[parent_row + 1],
        )
        indices = state_matrix.indices[start:stop]
        values = state_matrix.data[start:stop]
        keep = (
            hh_cd[indices] == int(spec.metadata["congressional_district_geoid"])
        ) & (hh_state[indices] == int(spec.metadata["state_fips"]))
        row = len(state_records) + offset
        rows.append(np.full(int(keep.sum()), row, dtype=np.int32))
        cols.append(indices[keep].astype(np.int32))
        data.append(values[keep])
        records.append(cd.target_record(spec))
    district_matrix = sparse.csr_array(
        (
            np.concatenate(data),
            (np.concatenate(rows) - len(state_records), np.concatenate(cols)),
        ),
        shape=(len(district_specs), state_matrix.shape[1]),
    )
    matrix = sparse.vstack([state_matrix, district_matrix], format="csr")
    matrix.sort_indices()
    populations = tool.ladder_population(LADDER, ["state", "cd"])
    tool._attach_pro_rata_populations(records, populations["cd"])
    holdout = cd.assign_target_roles(records, fraction=fraction)
    struct_tables = tool.extract_struct_tables(frame)
    _path, _registry, digests = tool.write_lean_checkpoint(
        struct_tables,
        matrix,
        (*state_registry.specs, *district_specs),
        records,
        out / "state_cd",
    )
    (out / "state_cd" / "surface.json").write_text(
        json.dumps(
            {
                "district_rows": len(district_specs),
                "district_rows_nnz": int(district_matrix.nnz),
                "state_rows_nnz": int(state_matrix.nnz),
                "total_nnz": int(matrix.nnz),
                "skipped_district_rows_without_engine_column": len(skipped),
                "skipped_examples": skipped[:5],
                "holdout": holdout,
                "surface_receipt_counts": surface["receipt"]["counts"],
                **digests,
            },
            indent=1,
        )
    )
    del frame, state_matrix, district_matrix, matrix
    gc.collect()
    diagnostics, weights, design, records, matrix, _frame = calibrate_and_save(
        tool, out / "state_cd", "state_cd"
    )
    state_weights = np.load(state_weights_path or out / "state" / "weights.npz")[
        "weights"
    ]
    same_held = cd.score_cd_holdout(
        records,
        matrix,
        design_weights=design,
        final_weights=state_weights,
        cap=SETTINGS["target_loss_cap"],
    )
    (out / "state_cd" / "holdout_under_state_weights.json").write_text(
        json.dumps({k: v for k, v in same_held.items() if k != "targets"}, indent=1)
    )
    summary = {
        "state_cd": {
            k: diagnostics[k]
            for k in (
                "n_targets",
                "n_holdout_targets",
                "final_loss",
                "fraction_within_10pct",
                "effective_sample_size",
                "matrix_nnz",
            )
        },
        "holdout_state_cd_weights": {
            k: v
            for k, v in diagnostics["cd_holdout"].items()
            if k not in ("targets", "by_family")
        },
        "holdout_state_weights": {
            k: v for k, v in same_held.items() if k not in ("targets", "by_family")
        },
        "weight_origin": diagnostics["weight_origin"],
    }
    print(json.dumps(summary, indent=1, default=str)[:6000])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["convert", "state", "state_cd", "all"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--state-weights",
        type=Path,
        default=None,
        help="state-only calibrated weights to score the holdout under "
        "(default <out>/state/weights.npz)",
    )
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    tool = load_tool()
    stages = ["convert", "state", "state_cd"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"=== {stage}", flush=True)
        if stage == "convert":
            stage_convert(tool, args.out)
        elif stage == "state":
            stage_state(tool, args.out)
        else:
            stage_state_cd(tool, args.out, args.holdout_fraction, args.state_weights)
        print(f"peak RSS {tool.rss():.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
